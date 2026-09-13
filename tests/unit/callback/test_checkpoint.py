"""Tests for package-owned checkpoint restore helpers."""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import tpen.checkpoint.restore as restore_module
import tpen.checkpoint.save as save_module
from tpen.accelerator import current_accelerator_type, device_module
from tpen.callback.checkpoint import Checkpoint
from tpen.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointReplaySemantics,
    CuspDistanceSemantics,
    checkpoint_hashes,
    list_complete_checkpoints,
    read_latest,
    read_publications,
    restore_checkpoint,
    restore_checkpoint_with_events,
    save_checkpoint,
    stable_config_hash,
)
from tpen.checkpoint.catalog import reconcile_publication
from tpen.checkpoint.receipt import publication_receipt_path
from tpen.checkpoint.events import LoadStarted, LoadSucceeded
from tpen.checkpoint.hashing import file_sha256
from tpen.checkpoint.manifest import LEGACY_CHECKPOINT_KIND, LEGACY_CHECKPOINT_SCHEMA_VERSION
from tpen.checkpoint.rng import (
    ACCELERATOR_STATE_KEY,
    BACKEND_KEY,
    DEVICE_KEY,
    DEVICES_KEY,
    apply_rng_state,
    draws_from_accelerator,
    require_restorable_rng_state,
    rng_state_dict,
)
from tpen.checkpoint.schema import read_manifest
from tpen.nn import ElectronElectronCusp, HookeOrbitalBasis
from tests.unit.callback.support import RecordingContext, training_state


def _cfg(*, model_out: int = 2):
    return OmegaConf.create(
        {
            "model": {"_target_": "torch.nn.Linear", "in_features": 3, "out_features": model_out},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.01},
            "trainer": {"_target_": "tests.Trainer", "max_steps": 2},
            "sampler": {"_target_": "tests.Sampler", "n_steps": 5},
            "hamiltonian_terms": {"constant": {"_target_": "tests.ConstantHamiltonian"}},
            "run": {"run_id": "run", "dir": "/tmp/run"},
            "study": {"name": "unit", "config_id": "cfg"},
        }
    )


def _context(cfg=None):
    return SimpleNamespace(
        cfg=_cfg() if cfg is None else cfg,
        metadata=SimpleNamespace(
            run_id="run",
            device="cpu",
            dtype="float64",
            git_commit="deadbeef",
            git_branch="codex/checkpoint",
            dirty_worktree=False,
            command="pytest",
            extra={"slurm": {}},
        ),
        run_dir=Path("/tmp/run"),
    )


class _Trainer:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self) -> dict[str, int]:
        return {"next_iteration": 3, "completed_updates": 3}

    def load_state_dict(self, state) -> None:
        self.loaded = dict(state)


class _Sampler:
    def __init__(self) -> None:
        self.loaded = None

    def mcmc_state_dict(self) -> dict[str, object]:
        return {"has_burned_in": True, "position": torch.ones(1)}

    def load_mcmc_state_dict(self, state) -> None:
        self.loaded = dict(state)


def _write_checkpoint(tmp_path: Path, model: torch.nn.Module | None = None, **kwargs) -> Path:
    model = torch.nn.Linear(3, 2).double() if model is None else model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=3,
        completed_updates=3,
        model=model,
        optimizer=optimizer,
        trainer=_Trainer(),
        sampler=_Sampler(),
        context=_context(),
        **kwargs,
    )


def test_post_commit_latest_failure_keeps_publication(tmp_path: Path, monkeypatch) -> None:
    """A committed checkpoint stays published when the pointer write fails."""

    def fail_write_latest(*args, **kwargs) -> None:
        raise OSError("latest pointer write failed")

    monkeypatch.setattr(save_module, "write_latest", fail_write_latest)
    root = tmp_path / "checkpoints"
    model = torch.nn.Linear(3, 2).double()

    with pytest.raises(OSError, match="latest pointer write failed"):
        save_checkpoint(
            output_dir=root,
            next_iteration=3,
            completed_updates=3,
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            trainer=_Trainer(),
            sampler=_Sampler(),
            context=_context(),
        )

    final_dir = root / "step_000003"
    assert (final_dir / "COMPLETE").is_file()
    publication_rows = [
        json.loads(line)
        for line in (root / "publications.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(publication_rows) == 1
    assert publication_rows[0]["ref"]["checkpoint_dir"] == str(final_dir)
    assert not (root / "latest.json").exists()


class _ReplayModel(torch.nn.Module):
    """Small explicit-factor model used to prove replay gates precede loading."""

    def __init__(self) -> None:
        super().__init__()
        self.factors = torch.nn.ModuleList([ElectronElectronCusp(trainable_range=True)])
        self.projection = torch.nn.Linear(3, 2)


def _replay_context() -> SimpleNamespace:
    context = _context()
    context.metadata.git_commit = "a" * 40
    OmegaConf.update(
        context.cfg,
        "trajectory_identity.config_sha256",
        "b" * 64,
        force_add=True,
    )
    OmegaConf.update(
        context.cfg,
        "hamiltonian_terms.electron_nucleus",
        # Pin the historical checkpoint semantics explicitly; this fixture is
        # testing replay compatibility, not the production default.
        {"_target_": "tests.ElectronNucleus", "eps": 1.0e-12},
        force_add=True,
    )
    return context


def _write_replay_checkpoint(
    tmp_path: Path,
) -> tuple[Path, _ReplayModel, CheckpointReplaySemantics, SimpleNamespace]:
    context = _replay_context()
    trained = _ReplayModel().double()
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=3,
        completed_updates=3,
        model=trained,
        context=context,
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )
    manifest = read_manifest(checkpoint_dir / "manifest.json", mode="model_only")
    cusp = trained.factors[0]
    semantics = CheckpointReplaySemantics(
        source_git_sha=manifest.provenance["git_sha"],
        source_tpen_version=manifest.provenance["tpen_version"],
        checkpoint_schema_version=manifest.schema_version,
        checkpoint_kind=manifest.kind,
        checkpoint_model_sha256=file_sha256(checkpoint_dir / manifest.files["model"]),
        evaluation_config_sha256=context.cfg.trajectory_identity.config_sha256,
        runtime_dtype=context.metadata.dtype,
        cusp_distance=CuspDistanceSemantics(
            electron_electron_distance_form="sqrt_squared_distance_plus_eps_squared",
            electron_electron_distance_eps=cusp.eps,
            electron_electron_range_offset_form="softplus_plus_eps",
            electron_electron_range_offset_eps=cusp.range_eps,
            electron_nucleus_coulomb_distance_form="euclidean_norm_clamp_min_eps",
            # Match the historical explicit value recorded by this fixture.
            electron_nucleus_coulomb_distance_eps=1.0e-12,
        ),
    )
    return checkpoint_dir, trained, semantics, context


def _rewrite_manifest_as_v1(checkpoint_dir: Path) -> None:
    """Rewrite a written manifest into the retired v1 shape.

    v1 recorded one ambiguous ``step`` -- which was in fact the resume cursor
    -- under the pre-rename kind, and never recorded ``completed_updates``.
    Nothing writes that shape any more, so tests synthesize it to pin read-side
    acceptance for archived artifacts.
    """

    manifest_path = checkpoint_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["schema_version"] = LEGACY_CHECKPOINT_SCHEMA_VERSION
    data["kind"] = LEGACY_CHECKPOINT_KIND
    data["step"] = data.pop("next_iteration")
    data.pop("completed_updates")
    data["provenance"]["spenn_version"] = data["provenance"].pop("tpen_version")
    manifest_path.write_text(json.dumps(data), encoding="utf-8")


def test_legacy_checkpoint_kind_is_pinned_to_historical_manifest_literal() -> None:
    # Archived checkpoint manifests carry this exact kind; changing the
    # spelling would orphan them from the legacy restore path.
    assert LEGACY_CHECKPOINT_KIND == "spenn.checkpoint"


def test_model_only_restore_loads_weights_into_configured_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trained = torch.nn.Linear(3, 2).double()
    root = _write_checkpoint(tmp_path, model=trained).parent

    torch.manual_seed(1)
    fresh = torch.nn.Linear(3, 2).double()
    assert not torch.equal(fresh.weight, trained.weight)

    report = restore_checkpoint(
        load={"path": str(root), "mode": "model_only", "strict": True},
        model=fresh,
        context=_context(),
    )

    assert torch.equal(fresh.weight, trained.weight)
    assert report.loaded_model is True
    assert report.loaded_optimizer is False
    assert report.loaded_sampler is False


def test_replay_semantics_strict_restore_succeeds_and_is_reported(tmp_path: Path) -> None:
    """The matching record is non-vacuous: it permits and records a real load."""

    checkpoint_dir, trained, semantics, context = _write_replay_checkpoint(tmp_path)
    fresh = _ReplayModel().double()
    before_restore = {
        name: value.detach().clone() for name, value in fresh.state_dict().items()
    }

    report = restore_checkpoint(
        load={
            "path": str(checkpoint_dir),
            "mode": "model_only",
            "strict": True,
            "replay_semantics": semantics.to_dict(),
        },
        model=fresh,
        context=context,
    )

    assert report.replay_semantics == semantics
    assert report.to_dict()["replay_semantics"] == semantics.to_dict()
    assert any(
        not torch.equal(value, before_restore[name])
        for name, value in fresh.state_dict().items()
    )
    for name, value in fresh.state_dict().items():
        assert torch.equal(value, trained.state_dict()[name])


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("source_git_sha", lambda value: replace(value, source_git_sha="c" * 40)),
        ("source_tpen_version", lambda value: replace(value, source_tpen_version="9.9.9")),
        (
            "checkpoint_schema_version",
            lambda value: replace(value, checkpoint_schema_version=99),
        ),
        ("checkpoint_kind", lambda value: replace(value, checkpoint_kind="other.checkpoint")),
        (
            "checkpoint_model_sha256",
            lambda value: replace(value, checkpoint_model_sha256="c" * 64),
        ),
        (
            "evaluation_config_sha256",
            lambda value: replace(value, evaluation_config_sha256="c" * 64),
        ),
        ("runtime_dtype", lambda value: replace(value, runtime_dtype="float32")),
        (
            "electron_electron_distance_eps",
            lambda value: replace(
                value,
                cusp_distance=replace(
                    value.cusp_distance, electron_electron_distance_eps=2.0e-12
                ),
            ),
        ),
        (
            "electron_electron_range_offset_eps",
            lambda value: replace(
                value,
                cusp_distance=replace(
                    value.cusp_distance,
                    electron_electron_range_offset_eps=2.0e-12,
                ),
            ),
        ),
        (
            "electron_nucleus_coulomb_distance_eps",
            lambda value: replace(
                value,
                cusp_distance=replace(
                    value.cusp_distance,
                    electron_nucleus_coulomb_distance_eps=0.0,
                ),
            ),
        ),
    ],
)
def test_replay_semantics_mismatches_refuse_before_model_mutation(
    tmp_path: Path, field: str, mutate
) -> None:
    """Every replay gate has a negative arm and leaves its target unmodified."""

    checkpoint_dir, _, semantics, context = _write_replay_checkpoint(tmp_path)
    fresh = _ReplayModel().double()
    before_restore = {
        name: value.detach().clone() for name, value in fresh.state_dict().items()
    }

    with pytest.raises(ValueError, match=field):
        restore_checkpoint(
            load={
                "path": str(checkpoint_dir),
                "mode": "model_only",
                "strict": True,
                "replay_semantics": mutate(semantics).to_dict(),
            },
            model=fresh,
            context=context,
        )

    for name, value in fresh.state_dict().items():
        assert torch.equal(value, before_restore[name])


def test_replay_semantics_content_identity_mismatch_refuses_before_model_mutation(
    tmp_path: Path,
) -> None:
    """A serialized record cannot claim an identity different from its fields."""

    checkpoint_dir, _, semantics, context = _write_replay_checkpoint(tmp_path)
    fresh = _ReplayModel().double()
    before_restore = {
        name: value.detach().clone() for name, value in fresh.state_dict().items()
    }
    serialized = semantics.to_dict()
    serialized["content_id"] = "c" * 64

    with pytest.raises(ValueError, match="content_id mismatch"):
        restore_checkpoint(
            load={
                "path": str(checkpoint_dir),
                "mode": "model_only",
                "strict": True,
                "replay_semantics": serialized,
            },
            model=fresh,
            context=context,
        )

    for name, value in fresh.state_dict().items():
        assert torch.equal(value, before_restore[name])


def test_restore_runtime_device_check_accepts_this_hosts_accelerator() -> None:
    # THE case the old CUDA-only canonicalization could not express, and the
    # reason the bug survived: metadata carries the index-free device string a
    # config declares, while tensors report an indexed accelerator device
    # (`xpu:0`, `cuda:0`). A check that resolves an index for CUDA alone raises
    # on every other backend. Runs on whatever this host has, so it is CPU on
    # CI, `cuda` on Polaris, and `xpu` on Aurora -- where it would have failed.
    device_type = current_accelerator_type()
    if device_type != "cpu" and not device_module(device_type).is_available():
        pytest.skip("needs a live accelerator")

    context = _context()
    context.metadata.device = device_type
    model = torch.nn.Linear(3, 2).double().to(device_type)

    restore_module._assert_model_runtime(model, context)


def test_restore_runtime_device_check_still_rejects_a_different_device() -> None:
    # Discriminates the test above from one that passes because the check was
    # weakened: canonicalizing both sides must not make mismatches compare equal.
    context = _context()
    context.metadata.device = "meta"
    model = torch.nn.Linear(3, 2).double()

    with pytest.raises(RuntimeError, match="expected meta"):
        restore_module._assert_model_runtime(model, context)


def test_restore_runtime_device_check_still_rejects_a_different_dtype() -> None:
    context = _context()
    model = torch.nn.Linear(3, 2).float()

    with pytest.raises(RuntimeError, match="torch.float64"):
        restore_module._assert_model_runtime(model, context)


def test_model_only_restore_emits_load_lifecycle_events(tmp_path: Path) -> None:
    trained = torch.nn.Linear(3, 2).double()
    root = _write_checkpoint(tmp_path, model=trained).parent
    fresh = torch.nn.Linear(3, 2).double()
    events = []

    def emit(event, *, state=None):
        del state
        events.append(event)

    report = restore_checkpoint_with_events(
        load={"path": str(root), "mode": "model_only", "strict": True},
        model=fresh,
        context=_context(),
        emit=emit,
    )

    assert report.loaded_model is True
    assert [type(event) for event in events] == [LoadStarted, LoadSucceeded]
    assert events[0].path == str(root)
    assert events[0].mode == "model_only"
    assert events[0].strict is True
    assert events[1].path == str(root)
    assert events[1].report.to_dict() == {
        "mode": "model_only",
        "checkpoint_dir": str(root / "step_000003"),
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "next_iteration": 3,
        "completed_updates": 3,
        "loaded_model": True,
        "loaded_optimizer": False,
        "loaded_trainer": False,
        "loaded_sampler": False,
        "loaded_rng": False,
    }


def test_restore_rejects_checkpoint_without_complete_marker(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path)
    (checkpoint_dir / "COMPLETE").unlink()

    with pytest.raises(ValueError, match="COMPLETE"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "model_only"},
            model=torch.nn.Linear(3, 2).double(),
            context=_context(),
        )


def test_restore_rejects_model_config_hash_mismatch(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="model_config"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "model_only"},
            model=torch.nn.Linear(3, 4).double(),
            context=_context(_cfg(model_out=4)),
        )


# axiswise_v1 must be selected explicitly since the D11 default flip; the
# deprecation warning it emits is expected and irrelevant to this contract.
@pytest.mark.filterwarnings("ignore:HookeOrbitalBasis basis_semantics:FutureWarning")
def test_restore_rejects_equal_width_legacy_and_product_basis_semantics(tmp_path: Path) -> None:
    """Equal parameter shapes cannot permit an axiswise/product checkpoint swap."""

    legacy_basis_kwargs = {
        "omega": 0.5,
        "spatial_dim": 2,
        "basis_semantics": "axiswise_v1",
        "max_shell": 2,
        "include_gaussian_factor": True,
        "include_spin": False,
    }
    product_basis_kwargs = {
        "omega": 0.5,
        "spatial_dim": 2,
        "basis_semantics": "product_v2",
        "truncation": "total_shell",
        "max_total_shell": 2,
        "include_gaussian_factor": True,
        "include_spin": False,
    }
    legacy_basis_config = {"_target_": "tpen.nn.HookeOrbitalBasis", **legacy_basis_kwargs}
    product_basis_config = {"_target_": "tpen.nn.HookeOrbitalBasis", **product_basis_kwargs}
    legacy_basis = HookeOrbitalBasis(**legacy_basis_kwargs)
    product_basis = HookeOrbitalBasis(**product_basis_kwargs)

    # d * (S + 1) == binomial(S + d, d) == 6 here, but channel meanings differ.
    assert legacy_basis.out_features == product_basis.out_features == 6
    legacy_config = OmegaConf.merge(
        _cfg(),
        {"model": {"in_features": legacy_basis.out_features, "basis": legacy_basis_config}},
    )
    product_config = OmegaConf.merge(
        _cfg(),
        {"model": {"in_features": product_basis.out_features, "basis": product_basis_config}},
    )
    assert "max_shell" not in product_config.model.basis
    trained = torch.nn.Linear(legacy_basis.out_features, 2).double()
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=1,
        completed_updates=1,
        model=trained,
        context=_context(legacy_config),
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )

    same_semantics = torch.nn.Linear(legacy_basis.out_features, 2).double()
    restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "model_only", "strict": True},
        model=same_semantics,
        context=_context(legacy_config),
    )
    assert torch.equal(same_semantics.weight, trained.weight)

    different_semantics = torch.nn.Linear(product_basis.out_features, 2).double()
    before_restore = {name: value.detach().clone() for name, value in different_semantics.state_dict().items()}
    assert {name: value.shape for name, value in trained.state_dict().items()} == {
        name: value.shape for name, value in different_semantics.state_dict().items()
    }

    with pytest.raises(ValueError, match="model_config"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "model_only", "strict": True},
            model=different_semantics,
            context=_context(product_config),
        )

    for name, value in different_semantics.state_dict().items():
        assert torch.equal(value, before_restore[name])


def test_model_only_does_not_require_train_resume_files(tmp_path: Path) -> None:
    trained = torch.nn.Linear(3, 2).double()
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=1,
        completed_updates=1,
        model=trained,
        context=_context(),
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )

    fresh = torch.nn.Linear(3, 2).double()
    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "model_only"},
        model=fresh,
        context=_context(),
    )

    assert torch.equal(fresh.weight, trained.weight)
    assert report.loaded_model is True
    assert report.loaded_optimizer is False


def test_train_resume_restores_all_train_state(tmp_path: Path) -> None:
    trained = torch.nn.Linear(3, 2).double()
    checkpoint_dir = _write_checkpoint(tmp_path, model=trained)
    fresh = torch.nn.Linear(3, 2).double()
    optimizer = torch.optim.Adam(fresh.parameters(), lr=0.01)
    trainer = _Trainer()
    sampler = _Sampler()

    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "train_resume"},
        model=fresh,
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
        context=_context(),
    )

    assert torch.equal(fresh.weight, trained.weight)
    assert trainer.loaded == {"next_iteration": 3, "completed_updates": 3}
    assert sampler.loaded["has_burned_in"] is True
    assert report.loaded_optimizer is True
    assert report.loaded_trainer is True
    assert report.loaded_sampler is True
    assert report.loaded_rng is True


def test_train_resume_fails_when_required_file_is_missing(tmp_path: Path) -> None:
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=1,
        completed_updates=1,
        model=torch.nn.Linear(3, 2).double(),
        optimizer=torch.optim.Adam(torch.nn.Linear(3, 2).double().parameters(), lr=0.01),
        trainer=_Trainer(),
        sampler=_Sampler(),
        context=_context(),
        save_optimizer=False,
    )

    with pytest.raises(FileNotFoundError, match="optimizer"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "train_resume"},
            model=torch.nn.Linear(3, 2).double(),
            optimizer=torch.optim.Adam(torch.nn.Linear(3, 2).double().parameters(), lr=0.01),
            trainer=_Trainer(),
            sampler=_Sampler(),
            context=_context(),
        )


# --- accelerator RNG device contract -----------------------------------------
#
# `train_resume` must refuse rather than silently continue on a different random
# stream. These drive `tpen.checkpoint.rng` directly with fabricated device
# provenance, so every case runs on a CPU-only host: `canonical_device` preserves
# an explicit index without consulting the backend, and the payload's recorded
# device is plain data.


def _accelerator_rng_state(device: str, devices: list[str]) -> dict[str, object]:
    """Return a CPU-written RNG payload relabelled as an accelerator write."""

    state = dict(rng_state_dict("cpu"))
    recorded = torch.device(device)
    state[BACKEND_KEY] = recorded.type
    state[DEVICE_KEY] = str(recorded)
    state[DEVICES_KEY] = list(devices)
    if devices:
        # `get_rng_state_all()` returns one state per visible device, in order.
        state[ACCELERATOR_STATE_KEY] = [torch.zeros(16, dtype=torch.uint8) for _ in devices]
    return state


def _numpy_states_equal(left, right) -> bool:
    """Compare two `numpy.random.get_state()` tuples elementwise."""

    kind, keys, position, has_gauss, cached_gauss = left
    other_kind, other_keys, other_position, other_has_gauss, other_cached_gauss = right
    return bool(
        kind == other_kind
        and np.array_equal(keys, other_keys)
        and position == other_position
        and has_gauss == other_has_gauss
        and cached_gauss == other_cached_gauss
    )


def test_cpu_rng_state_round_trips_unchanged(tmp_path: Path) -> None:
    """The common path: a CPU run resuming a CPU checkpoint reproduces every stream.

    Fidelity is asserted per stream, not just "restore did not raise". Dropping
    any single ``set_*`` call in `tpen.checkpoint.rng.apply_rng_state` has to
    fail here, because nothing else in the suite compares restored RNG state
    against saved RNG state.
    """

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    saved_torch = torch.get_rng_state().clone()
    saved_python = random.getstate()
    saved_numpy = np.random.get_state()

    state = rng_state_dict("cpu")

    assert state[BACKEND_KEY] == "cpu"
    assert state[DEVICE_KEY] == "cpu"
    assert state[DEVICES_KEY] == []
    # CPU RNG lives in `torch_cpu`; no accelerator state exists to persist.
    assert ACCELERATOR_STATE_KEY not in state

    # Advance every stream so an unrestored one cannot coincidentally match.
    torch.rand(8)
    random.random()
    np.random.random(8)
    assert not torch.equal(torch.get_rng_state(), saved_torch)
    assert random.getstate() != saved_python
    assert not _numpy_states_equal(np.random.get_state(), saved_numpy)

    require_restorable_rng_state(state, "cpu", tmp_path)
    apply_rng_state(state, "cpu")

    assert torch.equal(torch.get_rng_state(), saved_torch)
    assert random.getstate() == saved_python
    assert _numpy_states_equal(np.random.get_state(), saved_numpy)


def test_rng_restore_refuses_when_accelerator_state_is_absent(tmp_path: Path) -> None:
    """A live non-CUDA accelerator persists nothing, so resume must refuse."""

    # What Aurora's XPU produces today: the device is recorded, the state is not.
    state = _accelerator_rng_state("xpu:0", devices=[])

    with pytest.raises(ValueError, match=r"carries no accelerator RNG state") as refusal:
        require_restorable_rng_state(state, "xpu:0", tmp_path)

    # Absence is reported as absence. Both mismatch messages contrast the two
    # devices with "but this run is on"; the absence message must not, or the
    # two failures would be indistinguishable to whoever reads the traceback.
    assert "but this run is on" not in str(refusal.value)
    assert "xpu:0" in str(refusal.value)


def test_rng_restore_refuses_an_empty_accelerator_state_list(tmp_path: Path) -> None:
    """An empty per-device list restores nothing, so it must not count as present.

    ``set_rng_state_all([])`` is a silent no-op, which would turn a claimed
    successful resume into no accelerator restore at all.
    """

    state = _accelerator_rng_state("cuda:0", devices=["cuda:0"])
    state[ACCELERATOR_STATE_KEY] = []

    with pytest.raises(ValueError, match=r"carries no accelerator RNG state"):
        require_restorable_rng_state(state, "cuda:0", tmp_path)


def test_rng_restore_refuses_a_payload_without_device_provenance(tmp_path: Path) -> None:
    """A pre-guard payload cannot prove its device, so it cannot be resumed.

    Refused on every device, not just accelerators: the exemption would
    otherwise apply precisely to the artifacts written before the guard existed.
    Mirrors C1's refusal of a v1 manifest for ``train_resume`` -- unprovable
    provenance is rejected rather than guessed. ``model_only`` is unaffected,
    since it restores no RNG at all.
    """

    legacy = dict(rng_state_dict("cpu"))
    for key in (BACKEND_KEY, DEVICE_KEY, DEVICES_KEY):
        legacy.pop(key)

    with pytest.raises(ValueError, match=r"records no RNG device provenance"):
        require_restorable_rng_state(legacy, "cpu", tmp_path)

    # Same refusal on an accelerator, where a legacy CUDA payload could
    # otherwise have restored `cuda:1`'s streams onto `cuda:0` silently.
    legacy_cuda = dict(legacy)
    legacy_cuda[ACCELERATOR_STATE_KEY] = [torch.zeros(16, dtype=torch.uint8)]
    with pytest.raises(ValueError, match=r"records no RNG device provenance"):
        require_restorable_rng_state(legacy_cuda, "cuda:0", tmp_path)


def test_rng_restore_refuses_a_different_accelerator_backend(tmp_path: Path) -> None:
    state = _accelerator_rng_state("cuda:0", devices=["cuda:0"])

    with pytest.raises(ValueError, match=r"backend 'cuda'.*backend 'xpu'"):
        require_restorable_rng_state(state, "xpu:0", tmp_path)


def test_rng_restore_refuses_a_different_device_index(tmp_path: Path) -> None:
    """Same backend is not enough: the guard is same-device.

    `set_rng_state_all` assigns positionally, so moving from `cuda:0` to
    `cuda:1` does not by itself rebind streams -- that is the visible-device-set
    case. It is refused because reproducibility requires the resumed run to draw
    from the device that wrote the state, and nothing in torch raises otherwise.
    This mirrors `MetropolisSampler._require_device`, which has always rejected
    an index mismatch after canonicalization.
    """

    state = _accelerator_rng_state("cuda:0", devices=["cuda:0", "cuda:1"])

    with pytest.raises(ValueError, match=r"device cuda:0.*device cuda:1"):
        require_restorable_rng_state(state, "cuda:1", tmp_path)


def test_rng_restore_refuses_a_changed_visible_device_set(tmp_path: Path) -> None:
    """The per-device state list is positional, so its length is part of the contract."""

    # One more device than this host exposes, whatever host this is.
    visible = device_module("cuda").device_count()
    recorded = [f"cuda:{index}" for index in range(visible + 1)]
    state = _accelerator_rng_state("cuda:0", devices=recorded)

    with pytest.raises(ValueError, match=r"visible device set"):
        require_restorable_rng_state(state, "cuda:0", tmp_path)


def test_rng_contract_passes_through_a_device_without_a_backend_module(tmp_path: Path) -> None:
    """`meta` must never reach an unconditional `get_device_module` lookup.

    A device type with no accelerator module is valid to construct. Resolving a
    backend module for it raises a torch-internal ``RuntimeError``, which would
    replace a caller's own clear error -- exactly what
    `tests/unit/sampling/test_metropolis.py`'s ``device="meta"`` case exists to
    prevent on the sampler side. `canonical_device` leaves such a device
    index-free, and an index is what this module treats as the
    live-accelerator signal, so no lookup happens here either.
    """

    assert draws_from_accelerator(torch.device("meta")) is False

    state = rng_state_dict("meta")
    assert state[DEVICE_KEY] == "meta"
    assert state[DEVICES_KEY] == []
    assert ACCELERATOR_STATE_KEY not in state

    # A regression that looked up the backend module would surface in either
    # entry point as a torch RuntimeError rather than a checkpoint-level message.
    require_restorable_rng_state(state, "meta", tmp_path)
    apply_rng_state(state, "meta")


def _rewrite_rng_as_foreign_device(checkpoint_dir: Path) -> None:
    """Relabel a written ``rng.pt`` as an XPU write carrying no accelerator state."""

    rng_path = checkpoint_dir / "rng.pt"
    state = torch.load(rng_path, weights_only=False)
    state[BACKEND_KEY] = "xpu"
    state[DEVICE_KEY] = "xpu:0"
    state[DEVICES_KEY] = []
    state.pop(ACCELERATOR_STATE_KEY, None)
    torch.save(state, rng_path)
    _refresh_component_digest(checkpoint_dir, "rng")


def _refresh_component_digest(checkpoint_dir: Path, component: str) -> None:
    """Keep a deliberate test mutation a valid artifact for semantic checks."""

    manifest_path = checkpoint_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hashes"][f"{component}_sha256"] = file_sha256(
        checkpoint_dir / manifest["files"][component]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_train_resume_refuses_a_checkpoint_written_on_another_device(tmp_path: Path) -> None:
    """End to end: the refusal reaches `restore_checkpoint`, it is not a silent skip.

    The refusal must also arrive before anything is mutated. `_load_sampler`
    recreates `MetropolisSampler`'s generator on a device mismatch -- reseeding
    it, or leaving it unseeded when no seed is configured -- so a guard that ran
    after the component loads would refuse the lesser hazard having already
    reset the run's dominant RNG source.
    """

    trained = torch.nn.Linear(3, 2).double()
    checkpoint_dir = _write_checkpoint(tmp_path, model=trained)
    _rewrite_rng_as_foreign_device(checkpoint_dir)

    fresh = torch.nn.Linear(3, 2).double()
    before_restore = {name: value.detach().clone() for name, value in fresh.state_dict().items()}
    trainer = _Trainer()
    sampler = _Sampler()

    with pytest.raises(ValueError, match=r"backend 'xpu'.*backend 'cpu'"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "train_resume"},
            model=fresh,
            optimizer=torch.optim.Adam(fresh.parameters(), lr=0.01),
            trainer=trainer,
            sampler=sampler,
            context=_context(),
        )

    # Nothing was restored: not the sampler chain, not the trainer counters, not
    # the weights.
    assert sampler.loaded is None
    assert trainer.loaded is None
    for name, value in fresh.state_dict().items():
        assert torch.equal(value, before_restore[name])


def test_model_only_restores_a_checkpoint_that_cannot_be_resumed(tmp_path: Path) -> None:
    """`model_only` restores no RNG, so the device guard must not reach it."""

    trained = torch.nn.Linear(3, 2).double()
    checkpoint_dir = _write_checkpoint(tmp_path, model=trained)
    _rewrite_rng_as_foreign_device(checkpoint_dir)
    fresh = torch.nn.Linear(3, 2).double()

    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "model_only", "strict": True},
        model=fresh,
        context=_context(),
    )

    assert torch.equal(fresh.weight, trained.weight)
    assert report.loaded_rng is False


def test_restore_strict_load_fails_on_unexpected_keys(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path)
    state = torch.load(checkpoint_dir / "model.pt", weights_only=False)
    state["ghost"] = torch.zeros(1)
    torch.save(state, checkpoint_dir / "model.pt")
    _refresh_component_digest(checkpoint_dir, "model")

    with pytest.raises(RuntimeError, match="ghost"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "model_only", "strict": True},
            model=torch.nn.Linear(3, 2).double(),
            context=_context(),
        )


def test_v1_manifest_restores_model_only_without_a_completed_update_count(tmp_path: Path) -> None:
    """An archived v1 artifact still restores weights; it just cannot resume."""

    trained = torch.nn.Linear(3, 2).double()
    checkpoint_dir = _write_checkpoint(tmp_path, model=trained)
    _rewrite_manifest_as_v1(checkpoint_dir)
    fresh = torch.nn.Linear(3, 2).double()

    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "model_only"},
        model=fresh,
        context=_context(),
    )

    assert torch.equal(fresh.weight, trained.weight)
    assert report.schema_version == LEGACY_CHECKPOINT_SCHEMA_VERSION
    # v1's lone `step` was the resume cursor, so it maps onto `next_iteration`.
    assert report.next_iteration == 3
    # v1 never recorded the update count, and there is no upgrade path.
    assert report.completed_updates is None
    assert report.to_dict()["completed_updates"] is None


def test_v1_manifest_is_refused_for_train_resume_at_the_schema_gate(tmp_path: Path) -> None:
    """Rejection names version and mode, and precedes any trainer load.

    Note what this pins and what it does not. The refusal is *not* because the
    restore path needs `manifest.completed_updates` -- ``train_resume`` reads
    trainer state from ``trainer.json``, never from the manifest. It is because
    `schema_version 1` spans two incompatible ``trainer.json`` key sets (B1
    renamed the trainer keys without bumping the manifest schema), so the
    version cannot prove the artifact is resumable. Hence this checkpoint --
    synthesized from a v2 write and therefore carrying *post*-B1 trainer keys
    that would in fact load -- is still refused.
    """

    checkpoint_dir = _write_checkpoint(tmp_path)
    _rewrite_manifest_as_v1(checkpoint_dir)
    trainer = _Trainer()

    with pytest.raises(
        ValueError,
        match=r"schema_version 1 is not supported for restore mode 'train_resume'",
    ):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "train_resume"},
            model=torch.nn.Linear(3, 2).double(),
            optimizer=torch.optim.Adam(torch.nn.Linear(3, 2).double().parameters(), lr=0.01),
            trainer=trainer,
            sampler=_Sampler(),
            context=_context(),
        )

    # The gate refuses before any component is touched. `load_state_dict`'s own
    # KeyError remains a backstop for a genuinely pre-B1 artifact, but it is
    # never the first failure for a v1 manifest.
    assert trainer.loaded is None


def test_restore_report_counters_for_mode_none() -> None:
    report = restore_checkpoint(
        load={"mode": "none"},
        model=torch.nn.Linear(3, 2).double(),
        context=_context(),
    )

    assert (report.next_iteration, report.completed_updates) == (None, None)
    assert report.to_dict()["next_iteration"] is None
    assert report.to_dict()["completed_updates"] is None


def test_restore_report_counters_for_model_only(tmp_path: Path) -> None:
    """Both counters come from the manifest and are reported separately."""

    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        # Diverged on purpose: one iteration skipped its optimizer update, so a
        # report that conflated the two counters could not pass this.
        next_iteration=4,
        completed_updates=3,
        model=torch.nn.Linear(3, 2).double(),
        context=_context(),
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )

    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "model_only"},
        model=torch.nn.Linear(3, 2).double(),
        context=_context(),
    )

    assert report.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert (report.next_iteration, report.completed_updates) == (4, 3)


def test_restore_report_counters_for_train_resume(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2).double()
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=4,
        completed_updates=3,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        trainer=_Trainer(),
        sampler=_Sampler(),
        context=_context(),
    )

    fresh = torch.nn.Linear(3, 2).double()
    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "train_resume"},
        model=fresh,
        optimizer=torch.optim.Adam(fresh.parameters(), lr=0.01),
        trainer=_Trainer(),
        sampler=_Sampler(),
        context=_context(),
    )

    # `train_resume` admits only v2, so neither counter can be `None` here.
    assert (report.next_iteration, report.completed_updates) == (4, 3)
    assert report.to_dict()["next_iteration"] == 4
    assert report.to_dict()["completed_updates"] == 3


def test_keep_last_compatibility_retains_all_checkpoints_and_publications(tmp_path: Path) -> None:
    """The compatibility input is a strict superset of the old keep window."""

    root = tmp_path / "checkpoints"
    for step in (1, 2, 3):
        model = torch.nn.Linear(3, 2).double()
        save_checkpoint(
            output_dir=root,
            next_iteration=step,
            completed_updates=step,
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            trainer=_Trainer(),
            sampler=_Sampler(),
            context=_context(),
            keep_last=1,
        )
    assert [path.name for path in list_complete_checkpoints(root)] == [
        "step_000001", "step_000002", "step_000003"
    ]
    assert [ref.checkpoint_dir.name for ref in read_publications(root / "publications.jsonl")] == [
        "step_000001", "step_000002", "step_000003"
    ]
    assert read_latest(root)["checkpoint_dir"] == "step_000003"


def test_save_repairs_missing_latest_without_deleting_committed_checkpoints(tmp_path: Path) -> None:
    """A damaged pointer cannot cause committed checkpoints to be removed."""

    root = tmp_path / "checkpoints"
    for step in (1, 2):
        model = torch.nn.Linear(3, 2).double()
        save_checkpoint(
            output_dir=root,
            next_iteration=step,
            completed_updates=step,
            model=model,
            context=_context(),
            save_optimizer=False,
            save_trainer=False,
            save_sampler=False,
            save_rng=False,
            keep_last=1,
        )
    (root / "latest.json").unlink()
    save_checkpoint(
        output_dir=root,
        next_iteration=3,
        completed_updates=3,
        model=torch.nn.Linear(3, 2).double(),
        context=_context(),
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
        keep_last=1,
    )
    assert [path.name for path in list_complete_checkpoints(root)] == [
        "step_000001", "step_000002", "step_000003"
    ]
    assert read_latest(root)["checkpoint_dir"] == "step_000003"


def test_save_checkpoint_appends_one_publication_receipt_with_positive_durations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    final_dir = _write_checkpoint(tmp_path)
    receipt_path = publication_receipt_path(root)

    rows = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["summary"]["checkpoint_dir"] == final_dir.name
    assert row["summary"]["write_duration_sec"] > 0.0
    assert row["summary"]["publish_duration_sec"] > 0.0
    assert row["summary"]["total_bytes"] == sum(entry["size_bytes"] for entry in row["files"])
    assert (
        row["summary"]["total_bytes"]
        == row["summary"]["payload_bytes"] + row["summary"]["metadata_bytes"]
    )


def test_publication_receipt_total_matches_an_independent_stat_oracle(tmp_path: Path) -> None:
    """Cross-check the typed sum against a test-only, non-production full enumeration.

    Production code never lists the directory (see ``measure_checkpoint_files``);
    this oracle does, but only here, to prove the typed sum is complete rather
    than merely self-consistent.
    """

    root = tmp_path / "checkpoints"
    final_dir = _write_checkpoint(tmp_path)
    receipt_path = publication_receipt_path(root)
    row = json.loads(receipt_path.read_text(encoding="utf-8").splitlines()[0])

    oracle_total = sum(
        (final_dir / name).stat().st_size for name in os.listdir(final_dir)
    )
    assert row["summary"]["total_bytes"] == oracle_total


def test_reconcile_publication_does_not_append_another_receipt(tmp_path: Path) -> None:
    """Repairing the catalog/latest index for an existing checkpoint is not a new write."""

    root = tmp_path / "checkpoints"
    final_dir = _write_checkpoint(tmp_path)
    receipt_path = publication_receipt_path(root)
    rows_before = receipt_path.read_text(encoding="utf-8").splitlines()
    assert len(rows_before) == 1

    (root / "latest.json").unlink()
    reconcile_publication(root, final_dir)

    rows_after = receipt_path.read_text(encoding="utf-8").splitlines()
    assert rows_after == rows_before


def test_checkpoint_callback_reconcile_preserves_newer_latest_target(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    older_model = torch.nn.Linear(3, 2).double()
    older = save_checkpoint(
        output_dir=root,
        next_iteration=7,
        completed_updates=7,
        model=older_model,
        optimizer=torch.optim.Adam(older_model.parameters(), lr=0.01),
        trainer=_Trainer(),
        sampler=_Sampler(),
        context=_context(),
    )
    newer_model = torch.nn.Linear(3, 2).double()
    newer = save_checkpoint(
        output_dir=root,
        next_iteration=9,
        completed_updates=9,
        model=newer_model,
        optimizer=torch.optim.Adam(newer_model.parameters(), lr=0.01),
        trainer=_Trainer(),
        sampler=_Sampler(),
        context=_context(),
    )
    latest_path = root / "latest.json"
    newer_latest_bytes = latest_path.read_bytes()

    callback = Checkpoint(root)
    callback._save(RecordingContext(), training_state(), 7, 7)

    assert older.name == "step_000007"
    assert newer.name == "step_000009"
    assert latest_path.read_bytes() == newer_latest_bytes


def test_stable_config_hash_is_canonical_and_strict() -> None:
    config = {"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 2}
    reordered = {"out_features": 2, "_target_": "torch.nn.Linear", "in_features": 3}

    assert stable_config_hash(config) == stable_config_hash(reordered)
    assert stable_config_hash(config) == stable_config_hash(OmegaConf.create(config))
    assert stable_config_hash(config) != stable_config_hash({**config, "out_features": 4})

    with pytest.raises(TypeError, match="JSON/YAML-safe"):
        stable_config_hash({"path": Path("not-json-safe")})


def test_checkpoint_hashes_resolve_interpolations_and_track_components() -> None:
    cfg = OmegaConf.create(
        {
            "width": 4,
            "steps": 5,
            "omega": 0.5,
            "run": {"run_id": "a", "dir": "/tmp/a"},
            "model": {"channels": "${width}"},
            "sampler": {"n_steps": "${steps}"},
            "hamiltonian_terms": {"trap": {"omega": "${omega}"}},
        }
    )
    same = OmegaConf.create(
        {
            "width": 4,
            "steps": 5,
            "omega": 0.5,
            "run": {"run_id": "b", "dir": "/tmp/b"},
            "model": {"channels": 4},
            "sampler": {"n_steps": 5},
            "hamiltonian_terms": {"trap": {"omega": 0.5}},
        }
    )
    changed_sampler = OmegaConf.merge(same, {"sampler": {"n_steps": 6}})
    changed_hamiltonian = OmegaConf.merge(same, {"hamiltonian_terms": {"trap": {"omega": 0.7}}})

    hashes = checkpoint_hashes(cfg)

    assert hashes["model_config"] == checkpoint_hashes(same)["model_config"]
    assert hashes["resolved_config"] == checkpoint_hashes(same)["resolved_config"]
    assert hashes["sampler_config"] != checkpoint_hashes(changed_sampler)["sampler_config"]
    assert hashes["hamiltonian_config"] != checkpoint_hashes(changed_hamiltonian)["hamiltonian_config"]
