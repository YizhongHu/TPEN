"""Public-entry readback for the typed operational restore request (S2, item
4023abbb, review round 1 finding "required by the contract").

The unit tests in ``tests/unit/test_run_operational_restore.py`` exercise
``tpen._apply_operational_restore`` directly against a duck-typed runner. This
module instead goes through the real public entry
(``tpen.run.run_from_config``), constructs a real ``Train`` runner, and reads
back the live optimizer/trainer/sampler restore report -- per the S2
acceptance contract's requirement that config readback alone is insufficient.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tpen.checkpoint import CheckpointRef, CheckpointRestored
from tpen.execution.recovery import (
    CheckpointEvidence, OperationalRestoreRequest, RecoveryPhase, ScienceIntent, select_recovery,
)
from tpen.run import run_from_config
from tpen.runner import Train
from tpen.training.trainer import VMCTrainer


def _build_source_and_request(tmp_path):
    """Train a real source run to a checkpoint and build the admissible
    operational restore request a successor would select against it."""

    fixture = Path("tests/integration/artifacts/training/vmc_smoke.yaml")
    source = OmegaConf.load(fixture)
    source.run.root = str(tmp_path / "source")
    source.run.run_id = "source"
    source.trainer.max_steps = 2
    source.sampler.n_walkers = 8
    source.sampler.burn_in = 1
    source.sampler.n_steps = 1
    for callback in source.callbacks:
        if callback.get("_target_") == "tpen.callback.Checkpoint":
            callback.schedule.every_n = 1
    assert run_from_config(source, raise_exceptions=True) == 0
    checkpoint, = (tmp_path / "source").rglob("step_000001")
    intent = ScienceIntent({"fixture": "s2-vmc-smoke"}, target_iterations=2)
    evidence = CheckpointEvidence.from_ref(
        CheckpointRef.from_directory(checkpoint),
        science_input_digest=intent.science_input_digest, target_iterations=2,
    )
    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
    assert decision.refusal is None
    request = OperationalRestoreRequest(decision)
    target = OmegaConf.create(OmegaConf.to_container(source, resolve=False))
    target.run.root = str(tmp_path / "successor")
    target.run.run_id = "successor"
    return checkpoint, request, target


@pytest.mark.parametrize("case", ["matching-H", "wrong-live-H", "explicit-no-load"])
def test_public_operational_restore(tmp_path, monkeypatch, case):
    fixture = Path("tests/integration/artifacts/training/vmc_smoke.yaml")
    source = OmegaConf.load(fixture)
    source.run.root = str(tmp_path / "source")
    source.run.run_id = "source"
    source.trainer.max_steps = 2
    source.sampler.n_walkers = 8
    source.sampler.burn_in = 1
    source.sampler.n_steps = 1
    for callback in source.callbacks:
        if callback.get("_target_") == "tpen.callback.Checkpoint":
            callback.schedule.every_n = 1
    assert run_from_config(source, raise_exceptions=True) == 0
    checkpoint, = (tmp_path / "source").rglob("step_000001")
    predecessor = checkpoint.parent.parent

    def snapshot():
        return {
            str(p.relative_to(predecessor)):
                (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in predecessor.rglob("*") if p.is_file()
        }

    before_leaf = snapshot()
    intent = ScienceIntent({"fixture": "s2-vmc-smoke"}, target_iterations=2)
    evidence = CheckpointEvidence.from_ref(
        CheckpointRef.from_directory(checkpoint),
        science_input_digest=intent.science_input_digest, target_iterations=2,
    )
    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
    assert decision.refusal is None
    request = OperationalRestoreRequest(decision)
    target = OmegaConf.create(OmegaConf.to_container(source, resolve=False))
    target.run.root = str(tmp_path / "successor")
    target.run.run_id = "successor"
    if case == "wrong-live-H":
        target.runner.trainer = OmegaConf.to_container(target.trainer, resolve=True)
        target.runner.trainer.max_steps = 3  # Top-level trainer remains H=2.
    if case == "explicit-no-load":
        target.runner.load = {"mode": "none", "path": None}
    before_config = OmegaConf.to_yaml(target, resolve=True)
    reports, runners, fit_entries = [], [], []

    class Readback:
        def handle_occurrence(self, occurrence, context):
            if isinstance(occurrence.event, CheckpointRestored):
                reports.append(occurrence.event.report)

    original_run, original_fit = Train.run, VMCTrainer.fit

    def observe_run(self, context):
        runners.append(self)
        return original_run(self, context)

    def observe_fit(self, **kwargs):
        fit_entries.append(self.next_iteration)
        assert case != "wrong-live-H", "mismatched H reached fit"
        assert self.max_steps == 2
        assert (self.next_iteration, self.completed_updates) == (1, 1)
        assert runners[-1].load == request.load
        assert runners[-1].trainer is self
        optimizer, model = kwargs["optimizer"], kwargs["model"]
        assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
            id(p) for p in model.parameters()
        }
        saved_model = torch.load(checkpoint / "model.pt", weights_only=False)
        assert all(torch.equal(v, saved_model[k]) for k, v in model.state_dict().items())
        saved_optimizer = torch.load(checkpoint / "optimizer.pt", weights_only=False)
        live_state = optimizer.state_dict()["state"]
        assert live_state and set(live_state) == set(saved_optimizer["state"])
        for key, value in live_state.items():
            assert torch.equal(value["step"], saved_optimizer["state"][key]["step"])
        saved_sampler = torch.load(checkpoint / "sampler.pt", weights_only=False)
        assert torch.equal(
            kwargs["sampler"].mcmc_state_dict()["generator_state"],
            saved_sampler["generator_state"],
        )
        return original_fit(self, **kwargs)

    monkeypatch.setattr(Train, "run", observe_run)
    monkeypatch.setattr(VMCTrainer, "fit", observe_fit)
    try:
        if case == "wrong-live-H":
            with pytest.raises(ValueError, match="horizon|max_steps"):
                run_from_config(target, restore_request=request, raise_exceptions=True)
            assert not fit_entries
        else:
            assert run_from_config(
                target, restore_request=request, runtime_callbacks=(Readback(),),
                raise_exceptions=True,
            ) == 0
            assert fit_entries == [1] and len(reports) == 1
            assert reports[0].next_iteration == 1
            assert all(getattr(reports[0], "loaded_" + name) for name in (
                "model", "optimizer", "trainer", "sampler", "rng",
            ))
    finally:
        assert OmegaConf.to_yaml(target, resolve=True) == before_config
        assert snapshot() == before_leaf


def test_checkpoint_replaced_after_request_construction_is_refused_before_fit(tmp_path):
    """R2.3: a compatible replacement landing on the selected checkpoint's
    directory after the request was built must be refused by ``Train.run``'s
    pre-restore ``ref.validate()``, not silently loaded."""

    checkpoint, request, target = _build_source_and_request(tmp_path)
    model_path = checkpoint / "model.pt"
    mutated = bytearray(model_path.read_bytes())
    mutated[0] ^= 0xFF
    model_path.write_bytes(bytes(mutated))

    fit_entries = []
    original_fit = VMCTrainer.fit

    def observe_fit(self, **kwargs):
        fit_entries.append(self.next_iteration)
        return original_fit(self, **kwargs)

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(VMCTrainer, "fit", observe_fit)
        with pytest.raises(ValueError, match="no longer match"):
            run_from_config(target, restore_request=request, raise_exceptions=True)
    assert not fit_entries


def test_disagreeing_restored_trainer_counters_are_refused_before_fit(tmp_path):
    """R2.3: ``RestoreReport`` counters come from the manifest, while
    ``_load_trainer`` loads whatever ``trainer.json`` says; nothing upstream
    cross-checks the two (``validate_manifest_state`` only checks the trainer
    state's shape, not its agreement with the manifest). A checkpoint whose
    ``trainer.json`` already disagrees with its own manifest -- internally
    consistent by every existing digest, so neither ``ref.validate()`` nor
    the component-file digest check catches it -- must still be refused
    before ``fit`` once selected."""

    fixture = Path("tests/integration/artifacts/training/vmc_smoke.yaml")
    source = OmegaConf.load(fixture)
    source.run.root = str(tmp_path / "source")
    source.run.run_id = "source"
    source.trainer.max_steps = 2
    source.sampler.n_walkers = 8
    source.sampler.burn_in = 1
    source.sampler.n_steps = 1
    for callback in source.callbacks:
        if callback.get("_target_") == "tpen.callback.Checkpoint":
            callback.schedule.every_n = 1
    assert run_from_config(source, raise_exceptions=True) == 0
    checkpoint, = (tmp_path / "source").rglob("step_000001")

    # Corrupt the predecessor leaf *before* anything selects it: rewrite
    # trainer.json's counters and recompute its own content digest so the
    # component-file check (which only verifies trainer.json was not
    # tampered with after the manifest was written) still passes. The
    # manifest's own ``next_iteration``/``completed_updates`` -- the values
    # ``CheckpointRef``, ``select_recovery``, and ``OperationalRestoreRequest``
    # all trust -- are left untouched, so this models a checkpoint that was
    # internally inconsistent at save time, not a post-selection replacement.
    trainer_path = checkpoint / "trainer.json"
    trainer_state = json.loads(trainer_path.read_text(encoding="utf-8"))
    trainer_state["next_iteration"] += 1
    trainer_state["completed_updates"] += 1
    trainer_path.write_text(json.dumps(trainer_state), encoding="utf-8")
    manifest_path = checkpoint / "manifest.json"
    manifest_state = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_state["hashes"]["trainer_sha256"] = hashlib.sha256(trainer_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest_state), encoding="utf-8")

    intent = ScienceIntent({"fixture": "s2-vmc-smoke"}, target_iterations=2)
    evidence = CheckpointEvidence.from_ref(
        CheckpointRef.from_directory(checkpoint),
        science_input_digest=intent.science_input_digest, target_iterations=2,
    )
    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
    assert decision.refusal is None
    request = OperationalRestoreRequest(decision)
    target = OmegaConf.create(OmegaConf.to_container(source, resolve=False))
    target.run.root = str(tmp_path / "successor")
    target.run.run_id = "successor"

    fit_entries = []
    original_fit = VMCTrainer.fit

    def observe_fit(self, **kwargs):
        fit_entries.append(self.next_iteration)
        return original_fit(self, **kwargs)

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(VMCTrainer, "fit", observe_fit)
        with pytest.raises(ValueError, match="restored live trainer counters"):
            run_from_config(target, restore_request=request, raise_exceptions=True)
    assert not fit_entries
