"""Focused admission tests for materialized Stage-1 rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml
from omegaconf import OmegaConf

from experiments.atomistic.stage1 import eval as stage1_eval
from experiments.atomistic.stage1 import protocol
from experiments.atomistic.stage1 import restart
from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.launch import _binding_token, command_for_row
from experiments.atomistic.stage1.protocol import ProtocolError


ROOT = Path(__file__).parents[1]
NATIVE_GOLDEN = ROOT / "tests" / "fixtures" / "native-golden-origin-dev-r2.json"


def _write_rendered(path: Path, payload: dict[str, object]) -> str:
    """Write one deterministic YAML mapping and return its exact digest."""

    raw = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _train_row(tmp_path: Path) -> tuple[dict[str, object], Path]:
    """Build a small valid v1 materialized train row and its input file."""

    results_root = tmp_path / "results"
    rendered_path = results_root / "inputs" / "train.yaml"
    rendered_path.parent.mkdir(parents=True)
    overlay = ROOT / protocol.HOOKE_OVERLAY_RELATIVE_PATH
    overlay_digest = protocol.overlay_sha256(overlay)
    row: dict[str, object] = {
        "row_id": "hooke2-materialized-train",
        "system": "hooke2", "system_id": "hooke_pair_singlet_omega0.5",
        "kind": "train", "seed": 17, "source_sha": "a" * 40,
        "config_sha256": protocol.config_identities(
            config_root=ROOT / "configs", systems=("hooke2",)
        )["hooke2"], "config_identity_rule": protocol.CONFIG_IDENTITY_RULE,
        "config": "configs/hooke2/train.yaml", "results_root": str(results_root),
        "experiment_name": "stage1", "sector": "hooke2", "run_id": "materialized-train",
        "updates": 10, "walkers": 2, "device": "cpu", "lineage": 0,
        "sampler_seed": 23, "checkpoint_steps": [0, 1, 10],
        "overlay": protocol.HOOKE_OVERLAY_RELATIVE_PATH, "overlay_sha256": overlay_digest,
        "row_extension": {
            "version": protocol.ROW_EXTENSION_VERSION,
            "rendered_config": {"path": str(rendered_path), "sha256": ""},
            "diagnostics_profile": protocol.OVERLAY_PROFILE_V1,
        },
    }
    run_dir = protocol.resolve_run_dir(row)
    payload: dict[str, object] = {
        "run": {"root": str(results_root), "run_id": row["run_id"], "dir": str(run_dir), "layout": "nested"},
        "experiment": {"name": "stage1", "sector": "hooke2"},
        "runtime": {"seed": 17, "device": "cpu", "dtype": "float64"},
        "system": {"id": row["system_id"]},
        "stage1_overlay": {
            "schema": "stage1-overlay/v1",
            "path": protocol.HOOKE_OVERLAY_RELATIVE_PATH,
            "sha256": overlay_digest,
            "composition_order": protocol.OVERLAY_COMPOSITION_ORDER,
            "stream": "train",
        },
        "model": {"_target_": "example.Model"}, "hamiltonian_terms": {},
        "optimizer": {"_target_": "torch.optim.Adam"},
        "sampler": {"_target_": "example.Sampler", "n_walkers": 2, "seed": 23},
        "trainer": {"_target_": "example.Trainer", "max_steps": 10},
        "checkpoint_steps": [1, 10],
        "runner": {
            "_target_": "tpen.runner.Train", "model": {"_target_": "example.Model"},
            "sampler": {"_target_": "example.Sampler", "n_walkers": 2, "seed": 23},
            "hamiltonian_terms": {}, "optimizer": {"_target_": "torch.optim.Adam"},
            "trainer": {"_target_": "example.Trainer", "max_steps": 10},
        },
        "callbacks": [
            {"_target_": "tpen.callback.Checkpoint", "terminal": False,
             "schedule": {"_target_": "tpen.checkpoint.ExplicitUpdates", "updates": [1, 10]},
             "payload": {"_target_": "tpen.checkpoint.TrainResume"},
             "output_dir": str(run_dir / "checkpoints")},
            {"_target_": "tpen.callback.InitialCheckpoint", "output_dir": str(run_dir / "checkpoints")},
        ],
    }
    row["row_extension"]["rendered_config"]["sha256"] = _write_rendered(rendered_path, payload)  # type: ignore[index]
    return row, rendered_path


def test_materialized_train_reads_one_checked_mapping_and_retains_identity(tmp_path: Path) -> None:
    row, rendered_path = _train_row(tmp_path)
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    assert admission.rendered_path == rendered_path.resolve()
    assert admission.rendered_file_sha256 == row["row_extension"]["rendered_config"]["sha256"]  # type: ignore[index]
    protocol.validate_materialized_admission(cfg, row, admission)


def test_materialized_train_rejects_runner_copy_mismatch(tmp_path: Path) -> None:
    row, _ = _train_row(tmp_path)
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    cfg.runner.sampler.n_walkers = 3
    with pytest.raises(protocol.ProtocolError, match="runner.sampler"):
        protocol.validate_materialized_admission(cfg, row, admission)


def test_materialized_train_rejects_native_smoke_derivation(tmp_path: Path) -> None:
    row, _ = _train_row(tmp_path)
    with pytest.raises(protocol.ProtocolError, match="native smoke adapter"):
        protocol.derive_smoke_request(row, results_root=str(tmp_path / "other"), stop_at=1)


def _canonical_digest(value: object) -> dict[str, object]:
    """Hash exact canonical JSON bytes for the immutable native comparison."""

    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _small_native_rows(system: str, results_root: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build one small native phase-1/phase-2 plan without any extension."""

    plan = protocol.Protocol(1, 2, 1, 2, 3, 4, 2, 2)
    phase1 = protocol.build_manifest(
        protocol=plan,
        source_sha="a" * 40,
        config_sha256=protocol.config_identities(
            config_root=ROOT / "configs", systems=(system,)
        ),
        checkpoints=[0, 1],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=results_root,
        systems=(system,),
        phase=1,
        device="cpu",
    )
    phase1_hash = protocol.manifest_hash(phase1)
    evidence = {
        "schema": protocol.CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": protocol.manifest_hash([row])
            for row in phase1 if row["kind"] == "train"
        },
        "checkpoints": {
            f"{system}:{lineage}:1": "c" * 64
            for lineage in range(plan.lineages)
        },
    }
    eval_rows = protocol.bind_evaluation_manifest(
        phase1_rows=phase1,
        checkpoints=[1],
        checkpoint_evidence=evidence,
        expected_physical_system_id={system: "c" * 64},
        protocol=plan,
        device="cpu",
    )
    return phase1, eval_rows


@pytest.mark.parametrize("system", ("hooke1", "hooke2", "qho2e", "he", "h2"))
def test_native_rows_and_gateways_match_exact_post_884_golden(system: str) -> None:
    """The no-extension path stays byte-identical to an exact merged-dev base."""

    golden = json.loads(NATIVE_GOLDEN.read_text(encoding="utf-8"))
    assert golden["base_sha"]
    phase1, eval_rows = _small_native_rows(system, golden["results_root"])
    train = next(row for row in phase1 if row["kind"] == "train")
    train_cfg = stage1_train.build_train_config(train)
    eval_cfgs = [stage1_eval.build_eval_config(row) for row in eval_rows]
    bindings = [
        {"token": _binding_token(row),
         "command": command_for_row(row, python="/usr/bin/python3", driver="driver.py")}
        for row in phase1 + eval_rows
    ]
    expected = golden["systems"][system]
    assert expected["config_sha256"] == protocol.config_identities(
        config_root=ROOT / "configs", systems=(system,)
    )
    assert expected["phase1"] == _canonical_digest(phase1)
    assert expected["phase1_manifest_hash"] == protocol.manifest_hash(phase1)
    assert expected["combined"] == _canonical_digest(phase1 + eval_rows)
    assert expected["combined_manifest_hash"] == protocol.manifest_hash(phase1 + eval_rows)
    assert expected["phase1_row_ids"] == [row["row_id"] for row in phase1]
    assert expected["eval_row_ids"] == [row["row_id"] for row in eval_rows]
    assert expected["train_row_bytes"] == _canonical_digest(train)
    assert expected["eval_row_bytes"] == [_canonical_digest(row) for row in eval_rows]
    assert expected["train_config"] == _canonical_digest(
        OmegaConf.to_container(train_cfg, resolve=True)
    )
    assert expected["train_callback"] == _canonical_digest(
        OmegaConf.to_container(train_cfg.callbacks, resolve=True)
    )
    assert expected["eval_configs"] == [
        _canonical_digest(OmegaConf.to_container(cfg, resolve=True))
        for cfg in eval_cfgs
    ]
    assert expected["bindings"] == _canonical_digest(bindings)
    assert all("row_extension" not in row for row in phase1 + eval_rows)


@pytest.mark.parametrize(
    "extension",
    ({}, None, {"version": protocol.ROW_EXTENSION_VERSION}),
)
def test_adding_extension_marker_to_native_row_refuses_without_fallback(
    extension: object,
) -> None:
    """Old rows cannot opt into the materialized path by marker injection."""

    phase1, _ = _small_native_rows("he", "/n/netscratch/kozinsky_lab/Lab/r2-native")
    row = dict(next(row for row in phase1 if row["kind"] == "train"))
    row["row_extension"] = extension
    with pytest.raises(ProtocolError):
        protocol.validate_gateway_binding(
            binding=_binding_token(row), expected_kinds={"train"}
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("row_extension", {"version": protocol.ROW_EXTENSION_VERSION}, "missing"),
        ("row_extension", {"version": protocol.ROW_EXTENSION_VERSION,
                            "rendered_config": {}, "diagnostics_profile": "diagnostics-v1",
                            "extra": 1}, "unsupported"),
        ("row_extension", {"version": "stage1-rendered-config/v99",
                            "rendered_config": {"path": "/x", "sha256": "0" * 64},
                            "diagnostics_profile": "diagnostics-v1"}, "unsupported"),
        ("row_extension", {"version": protocol.ROW_EXTENSION_VERSION,
                            "rendered_config": {"path": "/x", "sha256": "0" * 64},
                            "diagnostics_profile": "diagnostics-v99"}, "unsupported"),
        ("row_extension", {"version": protocol.ROW_EXTENSION_VERSION,
                            "rendered_config": {"path": "/x", "sha256": "0" * 63},
                            "diagnostics_profile": "diagnostics-v1"}, "sha256"),
    ),
)
def test_materialized_wire_shape_fails_closed(
    field: str, value: object, message: str, tmp_path: Path,
) -> None:
    """Partial, extra, unknown, and malformed extension shapes do not fall back."""

    row, _ = _train_row(tmp_path)
    row[field] = value
    with pytest.raises(ProtocolError, match=message):
        protocol.validate_gateway_binding(
            binding=_binding_token(row), expected_kinds={"train"}
        )


def _materialized_eval_row(tmp_path: Path) -> tuple[dict[str, object], Path]:
    """Create one valid strict fresh eval row with a real checkpoint byte."""

    results = tmp_path / "results"
    _, eval_rows = _small_native_rows("hooke2", str(results))
    row = dict(eval_rows[0])
    checkpoint_dir = Path(str(row["checkpoint_dir"]))
    checkpoint_dir.mkdir(parents=True)
    checkpoint_file = checkpoint_dir / "model.pt"
    checkpoint_file.write_bytes(b"checkpoint-bytes")
    row["checkpoint_sha256"] = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    native_cfg = stage1_eval.build_eval_config(row)
    rendered_path = results / "inputs" / "eval.yaml"
    rendered_path.parent.mkdir(parents=True)
    rendered = OmegaConf.to_container(native_cfg, resolve=True)
    rendered["run"]["dir"] = str(protocol.resolve_run_dir(row))  # type: ignore[index]
    eval_output_dir = f"{protocol.resolve_run_dir(row)}/mcmc_energy"
    rendered["evaluation_tasks"]["mcmc_energy"]["output_dir"] = eval_output_dir  # type: ignore[index]
    rendered["evaluator"]["tasks"][0]["output_dir"] = eval_output_dir  # type: ignore[index]
    row["row_extension"] = {
        "version": protocol.ROW_EXTENSION_VERSION,
        "rendered_config": {"path": str(rendered_path), "sha256": ""},
        "diagnostics_profile": protocol.OVERLAY_PROFILE_V1,
    }
    row["row_extension"]["rendered_config"]["sha256"] = _write_rendered(  # type: ignore[index]
        rendered_path, rendered,
    )
    return row, rendered_path


def test_real_train_and_eval_mains_reach_one_core_call_with_consumed_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise wrapper → main → gateway with one final core call each."""

    train_row, train_path = _train_row(tmp_path / "train")
    train_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        stage1_train, "run_from_config",
        lambda cfg, **kwargs: train_calls.append({
            "cfg": OmegaConf.to_container(cfg, resolve=True), **kwargs
        }) or 0,
    )
    monkeypatch.setattr(sys, "argv", ["stage1-train", "--binding", _binding_token(train_row)])
    assert stage1_train.main() == 0
    assert len(train_calls) == 1
    assert train_calls[0]["config_path"] == str(train_path.resolve())
    assert train_calls[0]["cfg"]["runner"]["sampler"]["n_walkers"] == train_row["walkers"]  # type: ignore[index]

    eval_row, eval_path = _materialized_eval_row(tmp_path / "eval")
    eval_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        stage1_eval, "run_from_config",
        lambda cfg, **kwargs: eval_calls.append({
            "cfg": OmegaConf.to_container(cfg, resolve=True), **kwargs
        }) or 0,
    )
    monkeypatch.setattr(sys, "argv", ["stage1-eval", "--binding", _binding_token(eval_row)])
    assert stage1_eval.main() == 0
    assert len(eval_calls) == 1
    assert eval_calls[0]["config_path"] == str(eval_path.resolve())
    assert eval_calls[0]["cfg"]["runner"]["load"]["mode"] == "model_only"  # type: ignore[index]


@pytest.mark.parametrize(
    "mutator",
    (
        lambda cfg: setattr(cfg.runner.sampler, "n_walkers", 3),
        lambda cfg: setattr(cfg.runner.trainer, "max_steps", 3),
        lambda cfg: setattr(cfg, "load", {"mode": "model_only"}),
    ),
)
def test_consumed_runner_mismatch_refuses_before_core(
    tmp_path: Path, mutator,
) -> None:
    """A rehashed root config cannot hide a divergent runner copy."""

    row, rendered_path = _train_row(tmp_path)
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    mutator(cfg)
    calls: list[object] = []
    with pytest.raises(restart.RestartRefusal, match="materialized"):
        restart.execute_train(
            cfg=cfg, row=row, config_path=rendered_path, admission=admission,
            run=lambda *_args, **_kwargs: calls.append(True) or 0,
        )
    assert calls == []


def test_rendered_bytes_replaced_after_checked_read_refuse_without_output_mutation(
    tmp_path: Path,
) -> None:
    """The gateway never executes bytes substituted after admission."""

    row, rendered_path = _train_row(tmp_path)
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    rendered_path.write_bytes(rendered_path.read_bytes() + b"\n# replacement\n")
    with pytest.raises(restart.RestartRefusal, match="bytes changed"):
        restart.execute_train(
            cfg=cfg, row=row, config_path=rendered_path, admission=admission,
            run=lambda *_args, **_kwargs: pytest.fail("replacement reached core"),
        )
    assert before == sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))


@pytest.mark.parametrize("path_kind", ("outside", "symlink", "lexical"))
def test_rendered_input_scope_rejects_escape_paths(tmp_path: Path, path_kind: str) -> None:
    """Rendered inputs are confined to the canonical results_root/inputs tree."""

    row, rendered_path = _train_row(tmp_path)
    outside = tmp_path / "outside.yaml"
    outside.write_bytes(rendered_path.read_bytes())
    if path_kind == "outside":
        replacement = outside
    elif path_kind == "symlink":
        replacement = rendered_path.parent / "link.yaml"
        replacement.symlink_to(outside)
    else:
        replacement = rendered_path.parent / ".." / "outside.yaml"
    row["row_extension"]["rendered_config"]["path"] = str(replacement)  # type: ignore[index]
    with pytest.raises(ProtocolError, match="canonical|beneath|traversal"):
        protocol.load_materialized_config(row, expected_kind="train")


@pytest.mark.parametrize(
    ("path", "message"),
    (
        ("load.path", "load binding"),
        ("run.root", "run identity"),
        ("evaluation_tasks.mcmc_energy.generator.n_draws", "generator bindings"),
        ("trajectory_identity.checkpoint_file", "checkpoint identity"),
    ),
)
def test_eval_load_root_and_task_mismatches_refuse_before_core(
    tmp_path: Path, path: str, message: str,
) -> None:
    """Eval checks the actual load/task mapping, not only displayed scalars."""

    row, _ = _materialized_eval_row(tmp_path)
    cfg, admission = protocol.load_materialized_config(row, expected_kind="eval")
    if path == "load.path":
        cfg.load.path = str(tmp_path / "foreign-checkpoint")
    elif path == "run.root":
        cfg.run.root = str(tmp_path / "foreign-root")
    elif path.endswith("n_draws"):
        cfg.evaluation_tasks.mcmc_energy.generator.n_draws = 99
    else:
        cfg.trajectory_identity.checkpoint_file = str(tmp_path / "foreign-model.pt")
    with pytest.raises(ProtocolError, match=message):
        protocol.validate_materialized_admission(cfg, row, admission)


def test_extended_stop_preserves_full_horizon_and_returns_planned_yield(
    tmp_path: Path,
) -> None:
    """A bounded materialized request changes only the runtime stop callback."""

    row, rendered_path = _train_row(tmp_path)
    row["execution"] = {"stop_at": 1}
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    observed: dict[str, object] = {}

    def fake_run(run_cfg, **kwargs):
        observed["cfg"] = OmegaConf.to_container(run_cfg, resolve=True)
        observed.update(kwargs)
        return 2

    assert restart.execute_train(
        cfg=cfg, row=row, config_path=rendered_path, admission=admission, run=fake_run,
    ) == 2
    assert observed["cfg"]["trainer"]["max_steps"] == row["updates"]  # type: ignore[index]
    callback = observed["runtime_callbacks"][0]  # type: ignore[index]
    assert type(callback).__name__ == "PlannedYield"
    assert callback.target_iteration == 1
    assert callback.target_iterations == row["updates"]


def test_extended_resume_and_direct_restart_refuse_before_effect(tmp_path: Path) -> None:
    """Fresh-only materialized rows cannot enter native resume adapters."""

    row, rendered_path = _train_row(tmp_path)
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    with pytest.raises(restart.RestartRefusal, match="fresh-only"):
        restart.execute_restart(
            cfg=cfg, row=row, config_path=rendered_path, run=lambda *_a, **_k: pytest.fail(),
        )
    with pytest.raises(ValueError, match="do not accept native"):
        stage1_train.build_train_config(row, config_path=ROOT / "configs/hooke2/train.yaml")
    assert admission.rendered_path == rendered_path.resolve()


def test_extended_v2_basis_provenance_and_recorder_order_are_strict(tmp_path: Path) -> None:
    """Diagnostics-v2 remains an explicit basis with recorder-before-publish order."""

    row, rendered_path = _train_row(tmp_path)
    v2_overlay = ROOT / protocol.OVERLAY_V2_PATH_BY_SYSTEM["hooke2"]
    row["overlay"] = protocol.HOOKE_OVERLAY_V2_RELATIVE_PATH
    row["overlay_sha256"] = protocol.overlay_sha256(v2_overlay)
    row["overlay_profile"] = protocol.OVERLAY_PROFILE_V2
    row["row_extension"]["diagnostics_profile"] = protocol.OVERLAY_PROFILE_V2  # type: ignore[index]
    payload = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))
    payload["stage1_overlay"] = {
        "schema": "stage1-overlay/v2",
        "path": protocol.HOOKE_OVERLAY_V2_RELATIVE_PATH,
        "sha256": row["overlay_sha256"],
        "composition_order": protocol.OVERLAY_V2_COMPOSITION_ORDER,
        "stream": "train",
        "profile": protocol.OVERLAY_PROFILE_V2,
    }
    callbacks = payload["callbacks"]
    checkpoint_index = next(
        i for i, entry in enumerate(callbacks)
        if entry.get("_target_") == "tpen.callback.Checkpoint"
    )
    callbacks.insert(checkpoint_index, {"_target_": "tpen.callback.RawTrainingBatchRecorder"})
    row["row_extension"]["rendered_config"]["sha256"] = _write_rendered(  # type: ignore[index]
        rendered_path, payload,
    )
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    protocol.validate_materialized_admission(cfg, row, admission)
    cfg.stage1_overlay.path = protocol.HOOKE_OVERLAY_RELATIVE_PATH
    with pytest.raises(ProtocolError, match="provenance"):
        protocol.validate_materialized_admission(cfg, row, admission)


def test_lr4_row_boundary_stays_single_binding_without_override_channels() -> None:
    """The row consumer remains one selected row with no fan-out parameters."""

    phase1, _ = _small_native_rows("he", "/n/netscratch/kozinsky_lab/Lab/lr4")
    row = next(row for row in phase1 if row["kind"] == "train")
    command = command_for_row(row, python="/usr/bin/python3", driver="driver.py")
    assert command.count("--binding") == 1
    assert all(marker not in command for marker in ("qsub", "--array", "fan-out", "optimizer"))
