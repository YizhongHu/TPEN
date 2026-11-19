"""Contracts for the hash-bound Stage-1 diagnostics overlay."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1 import protocol as protocol_module
from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.launch import _binding_token, row_overrides
from experiments.atomistic.stage1.protocol import (
    CONFIG_IDENTITY_RULE,
    Protocol,
    ProtocolError,
    build_manifest,
    bind_evaluation_manifest,
    compose_train_overlay,
    config_identities,
    manifest_hash,
    overlay_sha256,
    physical_system_id_from_config,
    validate_gateway_binding,
)
from experiments.atomistic.stage1.collect import collect
from experiments.atomistic.stage1.launch import command_for_row
from experiments.toolkit.run_paths import resolve_run_dir
from tpen.run import load_config
from tpen.run import run_from_config


ROOT = Path(__file__).parents[1]
CONFIG_ROOT = ROOT / "configs"
OVERLAY = CONFIG_ROOT / "overlays" / "diagnostics-v1.yaml"
APPROVED_PROPOSAL_SHA256 = "8bbaf291e90ec6c90b5a8afbe664bded96dcb9743ad0c24cda1f66ddcb4634c1"
COMMITTED_OVERLAY_SHA256 = "fd80c4d7beb0a2c075e3ac85015ea975c0a505352e7dca58be951350b21fed7c"


def _structured_diff(before: object, after: object, prefix: str = "") -> set[str]:
    """Return changed leaf paths, including paths added or removed."""

    if isinstance(before, Mapping) and isinstance(after, Mapping):
        paths: set[str] = set()
        for key in before.keys() | after.keys():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.add(dotted)
            else:
                paths.update(_structured_diff(before[key], after[key], dotted))
        return paths
    if isinstance(before, list) and isinstance(after, list):
        return set() if before == after else {prefix}
    return set() if before == after else {prefix}


def _mapping_leaf_paths(value: Mapping[str, object], prefix: str) -> set[str]:
    """Return dotted paths for scalar leaves in one overlay section."""

    paths: set[str] = set()
    for key, nested in value.items():
        dotted = f"{prefix}.{key}"
        if isinstance(nested, Mapping):
            paths.update(_mapping_leaf_paths(nested, dotted))
        else:
            paths.add(dotted)
    return paths


def _protocol() -> Protocol:
    """Use a small manifest shape while retaining Stage-1's row contract."""

    return Protocol(1, 2, 1, 2, 3, 4, 2, 2)


def _phase1(*, overlay: bool) -> list[dict[str, object]]:
    """Build the smallest He phase-1 manifest used by identity tests."""

    return build_manifest(
        protocol=_protocol(),
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=CONFIG_ROOT, systems=("he",)),
        checkpoints=[0, 1],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1-overlay-test",
        systems=("he",),
        phase=1,
        device="cpu",
        overlay=str(OVERLAY) if overlay else None,
        overlay_sha256=overlay_sha256(OVERLAY) if overlay else None,
    )


def test_repository_overlay_is_versioned_and_checkpoint_free() -> None:
    """The committed variant records its deliberate checkpoint deviation."""

    assert overlay_sha256(OVERLAY) == COMMITTED_OVERLAY_SHA256
    assert COMMITTED_OVERLAY_SHA256 != APPROVED_PROPOSAL_SHA256
    overlay_cfg = OmegaConf.load(OVERLAY)
    assert "checkpoint_steps" not in overlay_cfg
    targets = [entry["_target_"] for entry in overlay_cfg.callbacks]
    assert "tpen.callback.InitialCheckpoint" not in targets
    assert "tpen.callback.Checkpoint" not in targets


def test_no_overlay_manifest_keeps_existing_identity_shape() -> None:
    """Rows without an overlay retain the pre-overlay manifest identity."""

    rows = _phase1(overlay=False)
    assert all("overlay" not in row and "overlay_sha256" not in row for row in rows)
    assert {row["config_identity_rule"] for row in rows} == {CONFIG_IDENTITY_RULE}


def test_no_overlay_manifest_matches_pre_overlay_golden_bytes() -> None:
    """The overlay feature leaves the pre-change no-overlay manifest untouched."""

    from experiments.atomistic.stage1.tests.test_step_zero import (
        NO_ZERO_FIXTURE,
        _no_zero_manifest_bytes,
    )

    fixture = json.loads(NO_ZERO_FIXTURE.read_text(encoding="utf-8"))
    rendered = _no_zero_manifest_bytes()
    assert len(rendered) == fixture["manifest_bytes"]
    assert hashlib.sha256(rendered).hexdigest() == fixture["manifest_sha256"]


def test_declared_overlay_is_hash_bound_to_train_rows() -> None:
    """A declared overlay appears only on train rows and carries its digest."""

    rows = _phase1(overlay=True)
    train_rows = [row for row in rows if row["kind"] == "train"]
    assert len(train_rows) == 3
    assert {row["overlay"] for row in train_rows} == {protocol_module.OVERLAY_RELATIVE_PATH}
    assert {row["overlay_sha256"] for row in train_rows} == {overlay_sha256(OVERLAY)}
    assert all("overlay" not in row for row in rows if row["kind"] == "smoke")


def _overlay_row() -> dict[str, object]:
    """Return one minimal train binding carrying the committed overlay."""

    row: dict[str, object] = {
        "kind": "train",
        "config": "configs/he/train.yaml",
        "system": "he",
        "row_id": "he-train-0",
        "seed": 123,
        "updates": 50000,
        "walkers": 4096,
        "source_sha": "c" * 40,
        "config_sha256": config_identities(config_root=CONFIG_ROOT, systems=("he",))["he"],
        "config_identity_rule": CONFIG_IDENTITY_RULE,
        "results_root": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1",
        "experiment_name": "stage1",
        "sector": "he",
        "run_id": "train-lineage-0",
        "overlay": protocol_module.OVERLAY_RELATIVE_PATH,
        "overlay_sha256": overlay_sha256(OVERLAY),
    }
    return row


def test_gateway_accepts_declared_overlay_and_rejects_hash_or_undeclared_use() -> None:
    """Gateway admission covers accepted, mismatched, and undeclared paths."""

    row = _overlay_row()
    overrides = row_overrides(row)
    token = _binding_token(row)
    assert validate_gateway_binding(
        binding=token,
        config=str(row["config"]),
        overrides=overrides,
        kind="train",
        overlay=str(row["overlay"]),
        overlay_sha256=str(row["overlay_sha256"]),
    ) == row

    with pytest.raises(ProtocolError, match="gateway overlay does not match binding"):
        validate_gateway_binding(
            binding=token,
            config=str(row["config"]),
            overrides=overrides,
            kind="train",
            overlay=str(row["overlay"]),
            overlay_sha256="d" * 64,
        )

    undeclared = {key: value for key, value in row.items() if key not in {"overlay", "overlay_sha256"}}
    with pytest.raises(ProtocolError, match="without being declared"):
        validate_gateway_binding(
            binding=_binding_token(undeclared),
            config=str(undeclared["config"]),
            overrides=row_overrides(undeclared),
            kind="train",
            overlay=protocol_module.OVERLAY_RELATIVE_PATH,
            overlay_sha256=overlay_sha256(OVERLAY),
        )


@pytest.mark.parametrize(
    "owned_override",
    (
        "loggers=[]",
        "callbacks=[{_target_:tpen.callback.Status}]",
        "stage1_overlay.sha256=" + "d" * 64,
        "[loggers]=[]",
        "[callbacks]=[]",
        "stage1_overlay[sha256]=" + "d" * 64,
        "[stage1_overlay].sha256=" + "d" * 64,
        "callbacks[0]._target_=tpen.callback.Status",
        ".loggers=[]",
    ),
)
def test_actual_train_gateway_refuses_overlay_owned_overrides(
    owned_override: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public train gateway refuses unbound overlay-owned overrides."""

    row = _overlay_row()
    overrides = row_overrides(row) + [owned_override]
    split_calls: list[str] = []
    original_split_key = protocol_module.split_key
    monkeypatch.setattr(
        protocol_module,
        "split_key",
        lambda key: split_calls.append(key) or original_split_key(key),
    )
    monkeypatch.setattr(
        "tpen.run.load_config",
        lambda *_args, **_kwargs: pytest.fail("rejected gateway loaded a config"),
    )
    monkeypatch.setattr(
        stage1_train,
        "run_from_config",
        lambda *_args, **_kwargs: pytest.fail("rejected gateway ran a config"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1-train",
            "--config",
            str(CONFIG_ROOT / "he" / "train.yaml"),
            "--kind",
            "train",
            *overrides,
            "--binding",
            _binding_token(row),
            "--overlay",
            str(row["overlay"]),
            "--overlay-sha256",
            str(row["overlay_sha256"]),
        ],
    )
    with pytest.raises(ProtocolError):
        stage1_train.main()
    assert owned_override.partition("=")[0] in split_calls


def test_actual_train_gateway_accepts_bound_overlay_without_owned_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public train gateway keeps the manifest-bound overlay path green."""

    row = _overlay_row()
    cfg = load_config(str(CONFIG_ROOT / "he" / "train.yaml"))
    observed: dict[str, object] = {}
    monkeypatch.setattr("tpen.run.load_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        stage1_train,
        "run_from_config",
        lambda resolved, **_kwargs: observed.setdefault("cfg", resolved) and 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1-train",
            "--config",
            str(CONFIG_ROOT / "he" / "train.yaml"),
            "--kind",
            "train",
            *row_overrides(row),
            "--binding",
            _binding_token(row),
            "--overlay",
            str(row["overlay"]),
            "--overlay-sha256",
            str(row["overlay_sha256"]),
        ],
    )
    assert stage1_train.main() == 0
    resolved = observed["cfg"]
    assert resolved.stage1_overlay.sha256 == row["overlay_sha256"]


def test_overlay_gateway_zero_checkpoint_preserves_base_checkpoint_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlay composition and zero routing preserve manifest checkpointing."""

    row = _overlay_row()
    row["checkpoint_steps"] = [0, 7]
    cfg = load_config(str(CONFIG_ROOT / "he" / "train.yaml"))
    observed: dict[str, object] = {}
    monkeypatch.setattr("tpen.run.load_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(
        stage1_train,
        "run_from_config",
        lambda resolved, **_kwargs: observed.setdefault("cfg", resolved) and 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1-train",
            "--config",
            str(CONFIG_ROOT / "he" / "train.yaml"),
            "--kind",
            "train",
            *row_overrides(row),
            "--binding",
            _binding_token(row),
            "--overlay",
            str(row["overlay"]),
            "--overlay-sha256",
            str(row["overlay_sha256"]),
        ],
    )
    assert stage1_train.main() == 0
    resolved = observed["cfg"]
    assert list(resolved.checkpoint_steps) == [7]
    callback_targets = [entry["_target_"] for entry in resolved.callbacks]
    assert callback_targets.count("tpen.callback.Checkpoint") == 1
    assert callback_targets.count("tpen.callback.InitialCheckpoint") == 1
    assert resolved.stage1_overlay.path == protocol_module.OVERLAY_RELATIVE_PATH
    assert resolved.loggers[0]._target_ == "tpen.logging.JSONL"


def test_train_composition_appends_diagnostics_and_preserves_base_callbacks() -> None:
    """Composition records provenance without replacing checkpoint ownership."""

    cfg = load_config(str(CONFIG_ROOT / "he" / "train.yaml"))
    original_model = OmegaConf.to_container(cfg.model, resolve=False)
    original_terms = OmegaConf.to_container(cfg.hamiltonian_terms, resolve=False)
    base_callbacks = OmegaConf.to_container(cfg.callbacks, resolve=False)
    base_targets = [entry["_target_"] for entry in base_callbacks]
    overlay_cfg = OmegaConf.load(OVERLAY)
    diagnostic_callbacks = OmegaConf.to_container(overlay_cfg.callbacks, resolve=False)
    compose_train_overlay(cfg, overlay=str(OVERLAY), expected_sha256=overlay_sha256(OVERLAY))
    assert OmegaConf.to_container(cfg.model, resolve=False) == original_model
    assert OmegaConf.to_container(cfg.hamiltonian_terms, resolve=False) == original_terms
    assert len(cfg.callbacks) == len(base_callbacks) + len(diagnostic_callbacks)
    assert [entry["_target_"] for entry in cfg.callbacks[:len(base_callbacks)]] == base_targets
    assert "tpen.callback.Checkpoint" in [entry["_target_"] for entry in cfg.callbacks]
    assert "tpen.callback.InitialCheckpoint" not in [entry["_target_"] for entry in cfg.callbacks]
    assert cfg.stage1_overlay.sha256 == overlay_sha256(OVERLAY)
    assert cfg.stage1_overlay.composition_order == protocol_module.OVERLAY_COMPOSITION_ORDER
    assert any(entry.get("_target_") == "tpen.callback.ModuleNorms" for entry in cfg.callbacks)
    assert cfg.loggers[0]._target_ == "tpen.logging.JSONL"


def test_train_composition_changes_only_overlay_owned_leaves() -> None:
    """A mapping-replace composition mutant must fail this structured diff."""

    cfg = load_config(str(CONFIG_ROOT / "he" / "train.yaml"))
    base = OmegaConf.to_container(cfg, resolve=False)
    overlay = OmegaConf.to_container(OmegaConf.load(OVERLAY), resolve=False)
    assert isinstance(base, Mapping)
    assert isinstance(overlay, Mapping)
    compose_train_overlay(cfg, overlay=str(OVERLAY), expected_sha256=overlay_sha256(OVERLAY))
    composed = OmegaConf.to_container(cfg, resolve=False)

    owned = {
        path
        for section in ("runtime", "sampler", "optimizer", "trainer")
        for path in _mapping_leaf_paths(overlay[section], section)
    }
    owned.update({"callbacks", "loggers", "stage1_overlay"})
    assert _structured_diff(base, composed) <= owned
    assert composed["trainer"]["_target_"] == base["trainer"]["_target_"]
    assert composed["runner"] == base["runner"]


def test_overlay_hash_is_exact_file_bytes() -> None:
    """The identity helper hashes bytes, not parsed YAML semantics."""

    assert overlay_sha256(OVERLAY) == hashlib.sha256(OVERLAY.read_bytes()).hexdigest()


@pytest.mark.integration
@pytest.mark.stage1_real_producer
def test_real_overlay_train_eval_collect_chain(tmp_path: Path) -> None:
    """The producer chain writes overlay diagnostics and collects eval rows."""

    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parents[2], text=True,
    ).strip()
    protocol = _protocol()
    phase1_rows = build_manifest(
        protocol=protocol,
        source_sha=source_sha,
        config_sha256=config_identities(config_root=CONFIG_ROOT, systems=("he",)),
        checkpoints=[0, 1],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path),
        systems=("he",),
        phase=1,
        device="cpu",
        overlay=str(OVERLAY),
        overlay_sha256=overlay_sha256(OVERLAY),
    )
    train_row = next(row for row in phase1_rows if row["kind"] == "train" and row["lineage"] == 0)
    train_overrides = row_overrides(train_row) + ["sampler.burn_in=1", "sampler.n_steps=1"]
    validate_gateway_binding(
        binding=_binding_token(train_row),
        config=str(train_row["config"]),
        overrides=train_overrides,
        kind="train",
        overlay=str(train_row["overlay"]),
        overlay_sha256=str(train_row["overlay_sha256"]),
    )
    assert str(train_row["config"]) in command_for_row(
        train_row, python="/usr/bin/python3", driver="experiments/atomistic/stage1/train.py",
    )
    rendered_command = command_for_row(
        train_row, python="/usr/bin/python3", driver="experiments/atomistic/stage1/train.py",
    )
    assert f"--overlay {train_row['overlay']}" in rendered_command
    assert f"--overlay-sha256 {train_row['overlay_sha256']}" in rendered_command
    cfg = load_config(str(CONFIG_ROOT / "he" / "train.yaml"))
    compose_train_overlay(
        cfg, overlay=str(OVERLAY), expected_sha256=overlay_sha256(OVERLAY),
    )
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(train_overrides))
    assert run_from_config(
        cfg, config_path=str(CONFIG_ROOT / "he" / "train.yaml"),
        command="stage1 overlay train", raise_exceptions=True,
    ) == 0
    train_dir = resolve_run_dir(train_row)
    resolved_train = OmegaConf.load(train_dir / "resolved_config.yaml")
    assert resolved_train.stage1_overlay.path == protocol_module.OVERLAY_RELATIVE_PATH
    assert resolved_train.stage1_overlay.sha256 == overlay_sha256(OVERLAY)
    assert resolved_train.stage1_overlay.composition_order == protocol_module.OVERLAY_COMPOSITION_ORDER
    status = json.loads((train_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["current_event"] == "run_end"
    metrics_path = train_dir / "metrics.jsonl"
    assert metrics_path.is_file()
    metric_records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert metric_records
    assert any(
        record.get("namespace") == "train"
        and "energy" in record.get("metrics", {})
        for record in metric_records
    )
    observed_namespaces = {
        record.get("namespace")
        for record in metric_records
        if isinstance(record.get("namespace"), str)
    }
    for namespace in (
        "checks/data_integrity",
        "checks/gradient",
        "train/perf",
        "train/factors",
        "train/training_history",
    ):
        assert namespace in observed_namespaces, namespace
    for prefix in ("train/module_norms/", "train/optimizer_state/"):
        matching_namespaces = {
            namespace for namespace in observed_namespaces if namespace.startswith(prefix)
        }
        assert matching_namespaces, prefix
        assert all(namespace.removeprefix(prefix) for namespace in matching_namespaces)
    checkpoint_file = train_dir / "checkpoints" / "step_000001" / "model.pt"
    checkpoint_sha = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()

    evidence = {
        "schema": protocol_module.CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": manifest_hash(phase1_rows),
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {
            f"{row['system']}:{row['lineage']}:1": checkpoint_sha
            for row in phase1_rows
            if row["kind"] == "train"
        },
    }
    eval_rows = bind_evaluation_manifest(
        phase1_rows=phase1_rows,
        checkpoints=[1],
        checkpoint_evidence=evidence,
        expected_physical_system_id={
            "he": physical_system_id_from_config(CONFIG_ROOT / "he" / "eval.yaml"),
        },
        protocol=protocol,
        source_sha=source_sha,
        device="cpu",
    )
    produced_eval_rows = [row for row in eval_rows if row["lineage"] == 0]
    for eval_row in produced_eval_rows:
        eval_overrides = row_overrides(eval_row) + [
            "evaluation_sampler.burn_in=1", "evaluation_sampler.n_steps=1",
        ]
        validate_gateway_binding(
            binding=_binding_token(eval_row),
            config=str(eval_row["config"]),
            overrides=eval_overrides,
            kind="eval",
            expected_checkpoint_sha256=str(eval_row["checkpoint_sha256"]),
        )
        eval_cfg = load_config(str(CONFIG_ROOT / "he" / "eval.yaml"), eval_overrides)
        assert run_from_config(
            eval_cfg, config_path=str(CONFIG_ROOT / "he" / "eval.yaml"),
            command="stage1 overlay eval", raise_exceptions=True,
        ) == 0
    # The tiny producer executes only lineage 0; the other lineages remain
    # bound above so phase-2 admission covers every planned train row, but
    # their unproduced eval rows must not be handed to the collector.
    document_rows = phase1_rows + produced_eval_rows
    collected = collect({
        "schema": "stage1/v3",
        "manifest_hash": manifest_hash(document_rows),
        "rows": document_rows,
    })
    assert len(collected["records"]) == protocol.eval_streams
    assert collected["aggregates"][0]["stream_count"] == protocol.eval_streams
