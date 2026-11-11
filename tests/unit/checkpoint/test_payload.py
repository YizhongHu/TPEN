"""Independent tests for checkpoint payload profiles and manifests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tpen.checkpoint import (
    MODEL_ONLY_PAYLOAD,
    PAYLOAD_MANIFEST_SCHEMA,
    TRAIN_RESUME_PAYLOAD,
    CheckpointPayload,
    CheckpointCatalog,
    ModelOnly,
    TrainResume,
    restore_checkpoint,
    save_checkpoint,
)
from tpen.checkpoint.manifest import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManifest,
)
from tpen.checkpoint.hashing import file_sha256
from tpen.checkpoint.schema import read_manifest
from tpen.sampling import MetropolisSampler
from tpen.training import VMCTrainer


def test_model_only_profile_has_a_stable_evaluation_manifest() -> None:
    payload = ModelOnly()

    assert payload == MODEL_ONLY_PAYLOAD
    assert payload.profile == "model_only"
    assert payload.required_files == ("model",)
    assert payload.required_state == ()
    assert payload.allowed_restore_modes == ("model_only",)
    assert payload.to_manifest() == {
        "schema": PAYLOAD_MANIFEST_SCHEMA,
        "profile": "model_only",
        "required_files": ["model"],
        "required_state": [],
        "restore_intents": ["model_only"],
    }


def test_train_resume_profile_names_every_recovery_component() -> None:
    payload = TrainResume()

    assert payload == TRAIN_RESUME_PAYLOAD
    assert payload.profile == "train_resume"
    assert payload.required_files == (
        "model",
        "optimizer",
        "trainer",
        "sampler",
        "rng",
    )
    assert payload.required_state == ("next_iteration", "completed_updates")
    assert payload.allowed_restore_modes == ("model_only", "train_resume")


@pytest.mark.parametrize("payload", [ModelOnly(), TrainResume()])
def test_profile_manifest_round_trips_to_the_canonical_profile(
    payload: CheckpointPayload,
) -> None:
    manifest = payload.to_manifest()

    assert CheckpointPayload.from_manifest(manifest) == payload
    assert json.dumps(manifest, sort_keys=True, separators=(",", ":")) == json.dumps(
        payload.to_manifest(), sort_keys=True, separators=(",", ":")
    )


def test_model_only_is_admitted_for_evaluation_and_rejected_for_training_resume() -> None:
    payload = ModelOnly()

    payload.validate_restore_intent("model_only")
    with pytest.raises(ValueError, match="cannot satisfy restore mode 'train_resume'"):
        payload.validate_restore_intent("train_resume")


def test_train_resume_admits_both_restore_intents() -> None:
    payload = TrainResume()

    payload.validate_restore_intent("model_only")
    payload.validate_restore_intent("train_resume")


def test_train_resume_rejects_each_missing_required_file() -> None:
    payload = TrainResume()
    complete = {name: f"{name}.payload" for name in payload.required_files}

    for missing in payload.required_files:
        files = {name: path for name, path in complete.items() if name != missing}
        with pytest.raises(ValueError, match=missing):
            payload.validate_files(files)


def test_train_resume_rejects_missing_progress_state() -> None:
    payload = TrainResume()

    with pytest.raises(ValueError, match="completed_updates"):
        payload.validate_state({"next_iteration": 3})


def test_profile_value_validation_rejects_empty_or_duplicate_contract_entries() -> None:
    with pytest.raises(ValueError, match="required_files must not be empty"):
        CheckpointPayload(
            profile="model_only",
            required_files=(),
            required_state=(),
            restore_intents=("model_only",),
        )

    with pytest.raises(ValueError, match="required_files entries must be unique"):
        CheckpointPayload(
            profile="model_only",
            required_files=("model", "model"),
            required_state=("state",),
            restore_intents=("model_only",),
        )


def test_manifest_restore_gate_reads_payload_intent_without_callback_state(
    tmp_path,
) -> None:
    path = tmp_path / "manifest.json"
    CheckpointManifest(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        kind=CHECKPOINT_KIND,
        next_iteration=3,
        completed_updates=3,
        created_at_unix=0.0,
        files={"model": "model.pt"},
        hashes={},
        runtime={},
        provenance={},
        payload=ModelOnly().to_manifest(),
    ).write(path)

    assert read_manifest(path, mode="model_only").payload == ModelOnly().to_manifest()
    with pytest.raises(ValueError, match="cannot satisfy restore mode 'train_resume'"):
        read_manifest(path, mode="train_resume")


def test_manifest_rejects_a_noncanonical_payload_contract(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    payload = TrainResume().to_manifest()
    payload["required_files"] = list(reversed(payload["required_files"]))
    CheckpointManifest(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        kind=CHECKPOINT_KIND,
        next_iteration=3,
        completed_updates=3,
        created_at_unix=0.0,
        files={name: f"{name}.payload" for name in TrainResume().required_files},
        hashes={},
        runtime={},
        provenance={},
        payload=payload,
    ).write(path)

    with pytest.raises(ValueError, match="not canonical"):
        read_manifest(path, mode="train_resume")


def test_explicit_model_only_payload_owns_save_defaults_and_manifest(tmp_path: Path) -> None:
    payload = ModelOnly()
    model = torch.nn.Linear(2, 1).double()
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=2,
        completed_updates=2,
        model=model,
        context=_context(),
        payload=payload,
    )

    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    assert manifest["payload"] == payload.to_manifest()
    assert read_manifest(checkpoint_dir / "manifest.json", mode="model_only").payload == (
        payload.to_manifest()
    )
    assert payload.required_files == ("model",)
    assert payload.required_state == ()
    assert "model" in manifest["files"]
    assert manifest["hashes"]["model_sha256"] == file_sha256(checkpoint_dir / "model.pt")
    assert set(manifest["files"]).isdisjoint(
        {"optimizer", "trainer", "sampler", "rng"}
    )
    assert (checkpoint_dir / "model.pt").is_file()
    assert (checkpoint_dir / "manifest.json").is_file()
    assert (checkpoint_dir / "COMPLETE").is_file()
    for train_state_file in ("optimizer.pt", "trainer.json", "sampler.pt", "rng.pt"):
        assert not (checkpoint_dir / train_state_file).exists()


def test_save_rejects_noncanonical_payload_before_complete_publish(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    payload = CheckpointPayload(
        profile="train_resume",
        required_files=("model", "optimizer", "trainer", "sampler", "rng"),
        required_state=("completed_updates", "next_iteration"),
        restore_intents=("model_only", "train_resume"),
    )
    output_dir = tmp_path / "checkpoints"

    with pytest.raises(ValueError, match="not canonical"):
        save_checkpoint(
            output_dir=output_dir,
            next_iteration=2,
            completed_updates=2,
            model=model,
            context=_context(),
            optimizer=optimizer,
            trainer=VMCTrainer(max_steps=1),
            sampler=MetropolisSampler(
                n_walkers=2,
                burn_in=0,
                n_steps=1,
                n_electrons=1,
                spatial_dim=1,
                seed=7,
                dtype=torch.float64,
            ),
            payload=payload,
        )

    complete_dirs = tuple(
        path
        for path in output_dir.glob("step_*")
        if (path / "COMPLETE").is_file()
    ) if output_dir.exists() else ()
    catalog_records = CheckpointCatalog(output_dir / "publications.jsonl").records()
    assert not complete_dirs and not catalog_records, (
        "rejected payload left a complete-but-unpublished artifact: "
        f"complete_dirs={complete_dirs!r}, catalog_records={catalog_records!r}"
    )


@pytest.mark.parametrize("corruption", ["missing", "negative"])
def test_real_restore_rejects_corrupt_progress_before_mutating_model(
    tmp_path: Path, corruption: str
) -> None:
    source_model = torch.nn.Linear(2, 1).double()
    with torch.no_grad():
        source_model.weight.fill_(1.0)
        source_model.bias.fill_(2.0)
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.01)
    source_trainer = VMCTrainer(max_steps=1)
    source_trainer.next_iteration = 2
    source_trainer.completed_updates = 2
    source_sampler = MetropolisSampler(
        n_walkers=2,
        burn_in=0,
        n_steps=1,
        n_electrons=1,
        spatial_dim=1,
        seed=7,
        dtype=torch.float64,
    )
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=2,
        completed_updates=2,
        model=source_model,
        context=_context(),
        optimizer=source_optimizer,
        trainer=source_trainer,
        sampler=source_sampler,
        payload=TrainResume(),
    )

    trainer_path = checkpoint_dir / "trainer.json"
    trainer_state = json.loads(trainer_path.read_text())
    if corruption == "missing":
        del trainer_state["completed_updates"]
    else:
        trainer_state["completed_updates"] = -1
    trainer_path.write_text(json.dumps(trainer_state), encoding="utf-8")
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hashes"]["trainer_sha256"] = file_sha256(trainer_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    target_model = torch.nn.Linear(2, 1).double()
    with torch.no_grad():
        target_model.weight.zero_()
        target_model.bias.zero_()
    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.01)
    target_trainer = VMCTrainer(max_steps=1)
    target_sampler = MetropolisSampler(
        n_walkers=2,
        burn_in=0,
        n_steps=1,
        n_electrons=1,
        spatial_dim=1,
        seed=11,
        dtype=torch.float64,
    )
    before_model = {
        name: value.detach().clone() for name, value in target_model.state_dict().items()
    }

    with pytest.raises(ValueError, match="completed_updates"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(checkpoint_dir)},
            model=target_model,
            context=_context(),
            optimizer=target_optimizer,
            trainer=target_trainer,
            sampler=target_sampler,
        )

    for name, value in before_model.items():
        torch.testing.assert_close(target_model.state_dict()[name], value, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("component", ["optimizer", "sampler", "rng"])
def test_real_restore_rejects_corrupt_component_before_mutating_model(
    tmp_path: Path, component: str
) -> None:
    """A late component digest failure cannot leave the model half-restored."""

    source_model = torch.nn.Linear(2, 1).double()
    with torch.no_grad():
        source_model.weight.fill_(1.0)
        source_model.bias.fill_(2.0)
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=2,
        completed_updates=2,
        model=source_model,
        context=_context(),
        optimizer=torch.optim.Adam(source_model.parameters(), lr=0.01),
        trainer=VMCTrainer(max_steps=1),
        sampler=MetropolisSampler(
            n_walkers=2,
            burn_in=0,
            n_steps=1,
            n_electrons=1,
            spatial_dim=1,
            seed=7,
            dtype=torch.float64,
        ),
        payload=TrainResume(),
    )

    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    (checkpoint_dir / manifest["files"][component]).write_bytes(b"corrupt checkpoint bytes")

    target_model = torch.nn.Linear(2, 1).double()
    with torch.no_grad():
        target_model.weight.zero_()
        target_model.bias.zero_()
    before_model = {
        name: value.detach().clone() for name, value in target_model.state_dict().items()
    }

    # The intact implementation raises its owned digest ValueError.  Keep the
    # outer assertion broad so the mutation proof reaches the no-live-mutation
    # oracle when the pre-pass is removed and torch.load fails late instead.
    with pytest.raises(Exception):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(checkpoint_dir)},
            model=target_model,
            context=_context(),
            optimizer=torch.optim.Adam(target_model.parameters(), lr=0.01),
            trainer=VMCTrainer(max_steps=1),
            sampler=MetropolisSampler(
                n_walkers=2,
                burn_in=0,
                n_steps=1,
                n_electrons=1,
                spatial_dim=1,
                seed=11,
                dtype=torch.float64,
            ),
        )

    for name, value in before_model.items():
        torch.testing.assert_close(target_model.state_dict()[name], value, rtol=0.0, atol=0.0)


def test_explicit_model_only_payload_rejects_train_state_flags(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires save_optimizer=False"):
        save_checkpoint(
            output_dir=tmp_path / "checkpoints",
            next_iteration=2,
            completed_updates=2,
            model=torch.nn.Linear(2, 1).double(),
            context=_context(),
            payload=ModelOnly(),
            save_optimizer=True,
        )
    assert not (tmp_path / "checkpoints").exists()


def test_legacy_save_defaults_write_the_full_train_resume_payload(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1).double()
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=2,
        completed_updates=2,
        model=model,
        context=_context(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        trainer=SimpleNamespace(
            state_dict=lambda: {"next_iteration": 2, "completed_updates": 2}
        ),
        sampler=SimpleNamespace(mcmc_state_dict=lambda: {"seed": 0}),
    )

    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    assert manifest["payload"] == TrainResume().to_manifest()
    assert set(manifest["files"]) == {
        "resolved_config",
        "model",
        "optimizer",
        "trainer",
        "sampler",
        "rng",
    }
    assert json.loads((checkpoint_dir / "trainer.json").read_text()) == {
        "next_iteration": 2,
        "completed_updates": 2,
    }


def _context() -> SimpleNamespace:
    """Minimal direct-save context; no callback or scheduling object is used."""

    return SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "model": {"name": "linear"},
                "optimizer": {"name": "adam"},
                "trainer": {"name": "trainer"},
                "sampler": {"name": "sampler"},
                "hamiltonian_terms": {"name": "constant"},
                "run": {"run_id": "payload-test", "dir": "/tmp/payload-test"},
                "study": {"name": "unit", "config_id": "payload"},
            }
        ),
        metadata=SimpleNamespace(
            run_id="payload-test",
            device="cpu",
            dtype="float64",
            git_commit="deadbeef",
            git_branch="codex/checkpoint",
            dirty_worktree=False,
            command="pytest",
            extra={"slurm": {}},
        ),
        run_dir=Path("/tmp/payload-test"),
    )
