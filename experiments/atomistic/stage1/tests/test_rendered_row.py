"""Focused admission tests for materialized Stage-1 rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

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
        "row_id": "",
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
        "sampler": {"_target_": "example.Sampler", "n_walkers": 2, "seed": 23, "dtype": "float64"},
        "trainer": {"_target_": "example.Trainer", "max_steps": 10},
        "checkpoint_steps": [1, 10],
        "runner": {
            "_target_": "tpen.runner.Train", "model": {"_target_": "example.Model"},
            "sampler": {"_target_": "example.Sampler", "n_walkers": 2, "seed": 23, "dtype": "float64"},
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
    row["row_id"] = protocol.manifest_hash([
        {key: value for key, value in row.items() if key != "row_id"}
    ])
    return row, rendered_path


def _v3_materialized_train_row(
    tmp_path: Path, *, resume: bool = True,
) -> tuple[dict[str, object], Path]:
    """Build a compact rendered row with the committed v3 recorder contract."""

    row, rendered_path = _train_row(tmp_path)
    overlay = ROOT / protocol.HOOKE_OVERLAY_V3_RELATIVE_PATH
    row.update({
        "overlay": protocol.HOOKE_OVERLAY_V3_RELATIVE_PATH,
        "overlay_sha256": protocol.overlay_sha256(overlay),
        "overlay_profile": protocol.OVERLAY_PROFILE_V3,
        "science_profile": protocol.HOOKE2_ACT50_SEGMENTED_PROFILES[0],
        "variant": "sigmoid",
    })
    row["config_sha256"] = protocol.config_identities(
        config_root=ROOT / "configs", systems=("hooke2",),
        science_profile=protocol.HOOKE2_ACT50_SEGMENTED_PROFILES[0],
    )["hooke2"]
    if resume:
        row["resume_profile"] = protocol.CHAIN_RESUME_PROFILE
    row["row_extension"]["diagnostics_profile"] = protocol.OVERLAY_PROFILE_V3  # type: ignore[index]
    payload = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))
    payload["stage1_overlay"] = {
        "schema": "stage1-overlay/v3",
        "path": row["overlay"],
        "sha256": row["overlay_sha256"],
        "composition_order": protocol.OVERLAY_COMPOSITION_ORDER,
        "stream": "train",
        "profile": protocol.OVERLAY_PROFILE_V3,
    }
    payload["callbacks"].append({
        "_target_": "tpen.callback.RawTrainingBatchRecorder",
        "format_version": 3,
        "path": str(protocol.resolve_run_dir(row) / "training_history"),
        "every_n_steps": 50,
        "partition_size_steps": 1000,
    })
    row["row_extension"]["rendered_config"]["sha256"] = _write_rendered(  # type: ignore[index]
        rendered_path, payload,
    )
    _refreeze_row_id(row)
    return row, rendered_path


def _real_v3_materialized_train_row(tmp_path: Path) -> tuple[dict[str, object], Path]:
    """Materialize one producer-built v3 row with resolved callback paths."""

    profile = protocol.HOOKE2_ACT50_SEGMENTED_PROFILES[0]
    source_row = protocol.build_manifest(
        protocol=protocol.STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=protocol.config_identities(
            config_root=ROOT / "configs", systems=("hooke2",),
            science_profile=profile,
        ),
        checkpoints=list(range(0, 50_001, 1_000)),
        lineage_seeds=[11, 12, 13], checkpoint_sha256={},
        results_root=str(tmp_path / "results"), systems=("hooke2",), phase=1,
        device="cuda", science_profile=profile, include_smoke=False,
        auto_overlay=True,
    )[0]
    cfg = stage1_train.build_train_config(source_row)
    run_dir = protocol.resolve_run_dir(source_row)
    cfg.run.dir = str(run_dir)
    rendered_path = Path(str(source_row["results_root"])) / "inputs" / "train.yaml"
    rendered_path.parent.mkdir(parents=True)
    rendered_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    row = {
        key: source_row[key]
        for key in (
            "row_id", "system", "system_id", "kind", "lineage", "seed",
            "source_sha", "config_sha256", "config_identity_rule", "config",
            "results_root", "experiment_name", "sector", "run_id", "updates",
            "walkers", "device", "sampler_seed", "checkpoint_steps", "overlay",
            "overlay_sha256", "overlay_profile", "science_profile", "variant",
        )
    }
    row["resume_profile"] = protocol.CHAIN_RESUME_PROFILE
    row["row_extension"] = {
        "version": protocol.ROW_EXTENSION_VERSION,
        "rendered_config": {"path": str(rendered_path), "sha256": hashlib.sha256(
            rendered_path.read_bytes()
        ).hexdigest()},
        "diagnostics_profile": protocol.OVERLAY_PROFILE_V3,
    }
    _refreeze_row_id(row)
    return row, rendered_path


def _hooke_production_train_row(tmp_path: Path) -> dict[str, object]:
    """Build one committed Hooke science row for the real train gateway."""

    rows = protocol.build_manifest(
        protocol=protocol.STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=protocol.config_identities(
            config_root=ROOT / "configs", systems=("hooke1",)
        ),
        checkpoints=list(range(0, 100001, 1000)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path / "results"),
        systems=("hooke1",),
        phase=1,
        device="cpu",
        science_profile=protocol.HOOKE_SCIENCE_PROFILE,
        evaluation_selection=protocol.HOOKE_EVAL_SELECTION,
        include_smoke=False,
        auto_overlay=True,
    )
    return next(row for row in rows if row["kind"] == "train")


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

    phase1, _ = _small_native_rows("he", "/fake/results-root")
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


def _materialized_eval_row(
    tmp_path: Path, *, checkpoint: int = 1,
) -> tuple[dict[str, object], Path]:
    """Create one valid strict fresh eval row with a real checkpoint byte."""

    results = tmp_path / "results"
    _, eval_rows = _small_native_rows("hooke2", str(results))
    row = dict(eval_rows[0])
    row["checkpoint"] = checkpoint
    row["updates"] = checkpoint
    row["checkpoint_dir"] = str(results / f"checkpoint-{checkpoint}")
    row["checkpoint_file"] = str(results / f"checkpoint-{checkpoint}" / "model.pt")
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
    rendered["runner"]["evaluator"]["tasks"][0]["output_dir"] = eval_output_dir  # type: ignore[index]
    row["row_extension"] = {
        "version": protocol.ROW_EXTENSION_VERSION,
        "rendered_config": {"path": str(rendered_path), "sha256": ""},
        "diagnostics_profile": protocol.OVERLAY_PROFILE_V1,
    }
    row["row_extension"]["rendered_config"]["sha256"] = _write_rendered(  # type: ignore[index]
        rendered_path, rendered,
    )
    row["row_id"] = protocol.manifest_hash([
        {key: value for key, value in row.items() if key != "row_id"}
    ])
    return row, rendered_path


def _rewrite_materialized_row(
    row: dict[str, object], rendered_path: Path, mutate,
) -> None:
    """Change one rendered mapping and refreeze its dependent row identity."""

    payload = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))
    mutate(payload)
    row["row_extension"]["rendered_config"]["sha256"] = _write_rendered(  # type: ignore[index]
        rendered_path, payload,
    )

    _refreeze_row_id(row)


def _refreeze_row_id(row: dict[str, object]) -> None:
    """Recompute the production row identity after a deliberate test mutation."""

    row["row_id"] = protocol.manifest_hash([
        {key: value for key, value in row.items() if key != "row_id"}
    ])


def _act50_gateway_rows(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Build one native ACT50 train/eval pair for gateway admission controls."""

    results_root = tmp_path / "results"
    phase1_rows = protocol.build_manifest(
        protocol=protocol.STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=protocol.config_identities(
            config_root=ROOT / "configs", systems=("hooke2",),
            science_profile=protocol.HOOKE2_ACT50_SIGMOID_PROFILE,
        ),
        checkpoints=list(range(0, 50_001, 1_000)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(results_root),
        systems=("hooke2",),
        phase=1,
        device="cuda",
        science_profile=protocol.HOOKE2_ACT50_SIGMOID_PROFILE,
        include_smoke=False,
        auto_overlay=True,
    )
    source_train_row = next(row for row in phase1_rows if row["kind"] == "train")
    phase1_hash = protocol.manifest_hash(phase1_rows)
    evidence = {
        "schema": protocol.CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": protocol.manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {
            f"{row['system']}:{row['lineage']}:{checkpoint}": "c" * 64
            for row in phase1_rows if row["kind"] == "train"
            for checkpoint in protocol.HOOKE2_ACT50_A27
        },
    }
    eval_row = next(row for row in protocol.bind_evaluation_manifest(
        phase1_rows=phase1_rows,
        checkpoints=list(protocol.HOOKE2_ACT50_A27),
        checkpoint_evidence=evidence,
        source_sha="a" * 40,
        config_sha256=protocol.config_identities(
            config_root=ROOT / "configs", systems=("hooke2",),
            science_profile=protocol.HOOKE2_ACT50_SIGMOID_PROFILE,
        ),
        protocol=protocol.STAGE1_PROTOCOL,
        phase1_manifest_hash=phase1_hash,
    ) if row["kind"] == "eval")
    checkpoint_file = Path(str(eval_row["checkpoint_file"]))
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_bytes(b"act50-checkpoint")
    eval_row["checkpoint_sha256"] = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    # The production eval binding intentionally stores its checkpoint beneath
    # the source train row's run directory.  Give the native train control a
    # distinct test root so that checkpoint setup cannot turn it into a
    # continuation before the admission assertion is exercised.
    train_row = dict(source_train_row)
    train_row["results_root"] = str(tmp_path / "native-train-results")
    _refreeze_row_id(train_row)
    return train_row, eval_row


def _attach_materialized_config(
    row: dict[str, object], cfg: object, rendered_path: Path,
) -> None:
    """Attach a valid rendered payload while retaining native science fields."""

    rendered = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(rendered, dict)
    row["row_extension"] = {
        "version": protocol.ROW_EXTENSION_VERSION,
        "rendered_config": {"path": str(rendered_path), "sha256": ""},
        "diagnostics_profile": protocol.OVERLAY_PROFILE_V1,
    }
    rendered_path.parent.mkdir(parents=True, exist_ok=True)
    row["row_extension"]["rendered_config"]["sha256"] = _write_rendered(  # type: ignore[index]
        rendered_path, rendered,
    )
    _refreeze_row_id(row)


def _snapshot_files(root: Path) -> tuple[str, ...]:
    """Return relative files under one test-owned result root."""

    if not root.exists():
        return ()
    return tuple(sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    ))


def test_extended_row_id_is_content_addressed_with_a_refrozen_green_sibling(
    tmp_path: Path,
) -> None:
    """Admission rejects stale IDs and accepts the same row after refreezing."""

    row, _ = _train_row(tmp_path)
    row["execution"] = {"stop_at": 1}
    with pytest.raises(ProtocolError, match="row_id"):
        protocol.validate_gateway_binding(
            binding=_binding_token(row), expected_kinds={"train"},
        )
    _refreeze_row_id(row)
    protocol.validate_gateway_binding(
        binding=_binding_token(row), expected_kinds={"train"},
    )


def test_act50_materialized_rows_refuse_before_core_and_output_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ACT50 science rows stay native-only when a materialized extension appears."""

    train_row, eval_row = _act50_gateway_rows(tmp_path)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        stage1_train, "run_from_config",
        lambda cfg, **kwargs: calls.append(("train", cfg)) or 0,
    )
    monkeypatch.setattr(
        stage1_eval, "run_from_config",
        lambda cfg, **kwargs: calls.append(("eval", cfg)) or 0,
    )
    # This test isolates materialized-vs-native admission from the separate
    # ACT50 checkpoint compatibility gate.
    monkeypatch.setattr(stage1_eval, "_refuse_cross_variant_checkpoint", lambda cfg, row: None)

    for kind, row in (("train", train_row), ("eval", eval_row)):
        calls.clear()
        monkeypatch.setattr(sys, "argv", [f"stage1-{kind}", "--binding", _binding_token(row)])
        main = stage1_train.main if kind == "train" else stage1_eval.main
        assert main() == 0
        assert [entry[0] for entry in calls] == [kind]
        run_dir = protocol.resolve_run_dir(row)
        assert not run_dir.exists()

        native_cfg = (
            stage1_train.build_train_config(row)
            if kind == "train" else stage1_eval.build_eval_config(row)
        )
        materialized = dict(row)
        rendered_path = Path(str(row["results_root"])) / "inputs" / f"{kind}-act50.yaml"
        _attach_materialized_config(materialized, native_cfg, rendered_path)
        before = _snapshot_files(Path(str(row["results_root"])))
        calls.clear()
        monkeypatch.setattr(
            sys, "argv", [f"stage1-{kind}", "--binding", _binding_token(materialized)],
        )
        with pytest.raises(ProtocolError, match="unsupported top-level keys"):
            main()
        assert calls == []
        assert _snapshot_files(Path(str(row["results_root"]))) == before


def test_consumed_integer_binding_requires_exact_scalar_type(tmp_path: Path) -> None:
    """A float spelling equal to an integer row binding is refused before core."""

    row, rendered_path = _train_row(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        payload["sampler"]["n_walkers"] = 2.0  # type: ignore[index]
        payload["runner"]["sampler"]["n_walkers"] = 2.0  # type: ignore[index]

    _rewrite_materialized_row(row, rendered_path, mutate)
    with pytest.raises(ProtocolError, match="sampler bindings"):
        protocol.load_materialized_config(row, expected_kind="train")


def test_consumed_dtype_must_match_runtime_dtype_for_train_and_eval(
    tmp_path: Path,
) -> None:
    """Consistent float32 sampler copies cannot accompany a float64 runtime."""

    train_row, train_path = _train_row(tmp_path / "train")

    def mutate_train(payload: dict[str, object]) -> None:
        payload["sampler"]["dtype"] = "float32"  # type: ignore[index]
        payload["runner"]["sampler"]["dtype"] = "float32"  # type: ignore[index]

    _rewrite_materialized_row(train_row, train_path, mutate_train)
    with pytest.raises(ProtocolError, match="sampler dtype"):
        protocol.load_materialized_config(train_row, expected_kind="train")

    eval_row, eval_path = _materialized_eval_row(tmp_path / "eval")

    def mutate_eval(payload: dict[str, object]) -> None:
        payload["evaluation_sampler"]["dtype"] = "float32"  # type: ignore[index]
        for task in (
            payload["evaluation_tasks"]["mcmc_energy"],  # type: ignore[index]
            payload["evaluator"]["tasks"][0],  # type: ignore[index]
            payload["runner"]["evaluator"]["tasks"][0],  # type: ignore[index]
        ):
            task["generator"]["sampler"]["dtype"] = "float32"

    _rewrite_materialized_row(eval_row, eval_path, mutate_eval)
    with pytest.raises(ProtocolError, match="evaluation sampler dtype"):
        protocol.load_materialized_config(eval_row, expected_kind="eval")


def test_checkpoint_zero_fresh_eval_is_admitted_but_zero_train_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eval admits its zero boundary while train retains a positive horizon."""

    eval_row, eval_path = _materialized_eval_row(tmp_path / "eval", checkpoint=0)
    calls: list[object] = []
    monkeypatch.setattr(
        stage1_eval, "run_from_config",
        lambda cfg, **kwargs: calls.append((cfg, kwargs)) or 0,
    )
    monkeypatch.setattr(sys, "argv", ["stage1-eval", "--binding", _binding_token(eval_row)])
    assert stage1_eval.main() == 0
    assert len(calls) == 1

    train_row, _ = _train_row(tmp_path / "train")
    train_row["updates"] = 0
    train_row["row_id"] = protocol.manifest_hash([
        {key: value for key, value in train_row.items() if key != "row_id"}
    ])
    with pytest.raises(ProtocolError, match="updates"):
        protocol.validate_gateway_binding(
            binding=_binding_token(train_row), expected_kinds={"train"},
        )


def test_materialized_system_id_must_be_from_the_finite_roster(tmp_path: Path) -> None:
    """Changing row and config IDs together cannot invent a new system."""

    row, rendered_path = _train_row(tmp_path)
    payload = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))
    payload["system"]["id"] = "unregistered-system"
    rendered_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    row["row_extension"]["rendered_config"]["sha256"] = hashlib.sha256(  # type: ignore[index]
        rendered_path.read_bytes()
    ).hexdigest()
    row["system_id"] = "unregistered-system"
    row["row_id"] = protocol.manifest_hash([
        {key: value for key, value in row.items() if key != "row_id"}
    ])
    with pytest.raises(ProtocolError, match="system identity"):
        protocol.load_materialized_config(row, expected_kind="train")


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


@pytest.mark.parametrize(
    "override, rejects",
    (
        ("runner.optimizer._target_=torch.optim.SGD", True),
        (f"sampler.n_walkers={protocol.STAGE1_PROTOCOL.lineage_walkers}", False),
    ),
)
def test_train_build_detects_runner_wiring_but_allows_bound_leaf(
    tmp_path: Path, override: str, rejects: bool,
) -> None:
    """The production train gateway guards runner wiring, not bound leaves."""

    row = _hooke_production_train_row(tmp_path)
    if rejects:
        with pytest.raises(ProtocolError, match="overlay changed base-owned section runner"):
            stage1_train.build_train_config(row, overrides=[override])
    else:
        stage1_train.build_train_config(row, overrides=[override])


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
    _refreeze_row_id(row)
    with pytest.raises(ProtocolError, match="canonical|beneath|traversal"):
        protocol.load_materialized_config(row, expected_kind="train")


@pytest.mark.parametrize(
    ("path", "message"),
    (
        ("load.path", "load binding"),
        ("run.root", "run identity"),
        ("evaluation_tasks.mcmc_energy.generator.n_draws", "evaluator task list"),
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


def test_eval_generator_mismatch_refuses_after_consistent_task_copies(tmp_path: Path) -> None:
    """A consistent task-list mutation reaches the row generator binding check."""

    row, _ = _materialized_eval_row(tmp_path)
    cfg, admission = protocol.load_materialized_config(row, expected_kind="eval")
    cfg.evaluation_tasks.mcmc_energy.generator.n_draws = 99
    cfg.evaluator.tasks[0].generator.n_draws = 99
    cfg.runner.evaluator.tasks[0].generator.n_draws = 99
    with pytest.raises(ProtocolError, match="generator bindings"):
        protocol.validate_materialized_admission(cfg, row, admission)


def test_extended_stop_preserves_full_horizon_and_returns_planned_yield(
    tmp_path: Path,
) -> None:
    """A bounded materialized request changes only the runtime stop callback."""

    row, rendered_path = _train_row(tmp_path)
    row["execution"] = {"stop_at": 1}
    _refreeze_row_id(row)
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
    with pytest.raises(restart.RestartRefusal, match="nonexistent run directory"):
        restart.execute_restart(
            cfg=cfg, row=row, config_path=rendered_path, run=lambda *_a, **_k: pytest.fail(),
        )
    with pytest.raises(ValueError, match="do not accept native"):
        stage1_train.build_train_config(row, config_path=ROOT / "configs/hooke2/train.yaml")
    assert admission.rendered_path == rendered_path.resolve()


def test_materialized_v3_train_admission(tmp_path: Path) -> None:
    """The opt-in v3 row is admitted through the same rendered reader."""

    row, rendered_path = _v3_materialized_train_row(tmp_path)
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    protocol.validate_materialized_admission(cfg, row, admission)
    assert admission.rendered_path == rendered_path.resolve()


def test_materialized_v3_resume_requires_opt_in(tmp_path: Path) -> None:
    """A v3 recorder cannot turn on continuation by omission or mutation."""

    row, rendered_path = _v3_materialized_train_row(tmp_path, resume=False)
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    before = rendered_path.read_bytes()
    protocol.resolve_run_dir(row).mkdir(parents=True)
    with pytest.raises(restart.RestartRefusal, match="fresh-only"):
        restart.execute_train(
            cfg=cfg, row=row, config_path=rendered_path, admission=admission,
            run=lambda *_a, **_k: pytest.fail(),
        )
    assert rendered_path.read_bytes() == before


def test_materialized_v3_recorder_contract_is_bound(tmp_path: Path) -> None:
    """The reader refuses a v3 row whose recorder format or cadence drifts."""

    row, rendered_path = _v3_materialized_train_row(tmp_path)
    payload = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))
    recorder = next(
        entry for entry in payload["callbacks"]
        if entry.get("_target_") == "tpen.callback.RawTrainingBatchRecorder"
    )
    recorder["format_version"] = 2
    row["row_extension"]["rendered_config"]["sha256"] = _write_rendered(  # type: ignore[index]
        rendered_path, payload,
    )
    _refreeze_row_id(row)
    with pytest.raises(ProtocolError, match="recorder contract"):
        protocol.load_materialized_config(row, expected_kind="train")


def test_materialized_v3_requires_segmented_identity(tmp_path: Path) -> None:
    """A diagnostics-v3 row cannot borrow a non-segmented science identity."""

    row, _ = _v3_materialized_train_row(tmp_path)
    row["science_profile"] = protocol.HOOKE2_ACT50_SIGMOID_PROFILE
    _refreeze_row_id(row)
    with pytest.raises(ProtocolError, match="segmented ACT50"):
        protocol.load_materialized_config(row, expected_kind="train")


def test_real_v3_materialization_round_trips_to_continuation_planning(
    tmp_path: Path,
) -> None:
    """A resolved producer row reaches the opted-in continuation planner."""

    row, rendered_path = _real_v3_materialized_train_row(tmp_path)
    assert "${" not in rendered_path.read_text(encoding="utf-8")
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    protocol.validate_materialized_admission(cfg, row, admission)

    run_dir = protocol.resolve_run_dir(row)
    checkpoint_dir = run_dir / "checkpoints" / "step_001000"
    checkpoint_dir.mkdir(parents=True)
    observed: dict[str, object] = {}

    def inspect(path: str | Path) -> SimpleNamespace:
        observed["inspect_path"] = Path(path)
        return SimpleNamespace(
            verdict="incomplete", refusal=None,
            checkpoint={
                "checkpoint_dir": str(checkpoint_dir),
                "next_iteration": 1000,
                "completed_updates": 1000,
            },
            checkpoint_ref=None, run_dir=str(run_dir),
        )

    def run(run_cfg: object, **kwargs: object) -> int:
        observed["cfg"] = run_cfg
        observed["kwargs"] = kwargs
        return 2

    assert restart.execute_train(
        cfg=cfg, row=row, config_path=rendered_path,
        admission=admission, inspector=inspect, run=run,
    ) == 2
    assert observed["inspect_path"] == run_dir
    resumed_cfg = observed["cfg"]
    assert resumed_cfg.runner.load.mode == "train_resume"  # type: ignore[union-attr]
    assert resumed_cfg.runner.load.path == str(checkpoint_dir)  # type: ignore[union-attr]
    recorder = next(
        callback for callback in resumed_cfg.callbacks  # type: ignore[union-attr]
        if callback.get("_target_") == "tpen.callback.RawTrainingBatchRecorder"
    )
    assert recorder.path == str(run_dir / "training_history")


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
    _refreeze_row_id(row)
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    protocol.validate_materialized_admission(cfg, row, admission)
    cfg.stage1_overlay.path = protocol.HOOKE_OVERLAY_RELATIVE_PATH
    with pytest.raises(ProtocolError, match="provenance"):
        protocol.validate_materialized_admission(cfg, row, admission)


def test_lr4_row_boundary_stays_single_binding_without_override_channels(tmp_path: Path) -> None:
    """The row consumer remains one selected row with no fan-out parameters."""

    phase1, _ = _small_native_rows("he", str(tmp_path / "lr4"))
    row = next(row for row in phase1 if row["kind"] == "train")
    command = command_for_row(row, python="/usr/bin/python3", driver="driver.py")
    assert command.count("--binding") == 1
    assert all(marker not in command for marker in ("qsub", "--array", "fan-out", "optimizer"))


def _invoke_train(
    row: dict[str, object], monkeypatch: pytest.MonkeyPatch, calls: list[object],
) -> None:
    monkeypatch.setattr(
        stage1_train,
        "run_from_config",
        lambda cfg, **kwargs: calls.append((cfg, kwargs)) or 0,
    )
    monkeypatch.setattr(sys, "argv", ["stage1-train", "--binding", _binding_token(row)])
    stage1_train.main()


def _invoke_eval(
    row: dict[str, object], monkeypatch: pytest.MonkeyPatch, calls: list[object],
) -> None:
    monkeypatch.setattr(
        stage1_eval,
        "run_from_config",
        lambda cfg, **kwargs: calls.append((cfg, kwargs)) or 0,
    )
    monkeypatch.setattr(sys, "argv", ["stage1-eval", "--binding", _binding_token(row)])
    stage1_eval.main()


def assert_type_drift_is_in_consumed_copy(
    payload: dict[str, object], original: object,
) -> None:
    """Check that the root duplicate stayed intact while runner copy drifted."""

    root_value = payload["evaluator"]["tasks"][0]["generator"]["n_draws"]
    runner_value = payload["runner"]["evaluator"]["tasks"][0]["generator"]["n_draws"]
    assert root_value == original and type(root_value) is type(original) is int
    assert runner_value == float(original) and type(runner_value) is float


@pytest.mark.parametrize(
    "mutation",
    (
        pytest.param(
            lambda p: p["runner"]["sampler"].__setitem__("n_walkers", 2.0),
            id="runner-sampler-n-walkers-float",
        ),
        pytest.param(
            lambda p: p["runner"]["trainer"].__setitem__("max_steps", 10.0),
            id="runner-trainer-max-steps-float",
        ),
        pytest.param(
            lambda p: p["callbacks"][0]["schedule"]["updates"].__setitem__(0, 1.0),
            id="checkpoint-schedule-update-float",
        ),
    ),
)
def test_train_consumed_copy_type_drift_refuses_before_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation,
) -> None:
    """Equal-by-value floats in consumed train copies are not admissible."""

    row, path = _train_row(tmp_path)
    _rewrite_materialized_row(row, path, mutation)
    calls: list[object] = []
    with pytest.raises((protocol.ProtocolError, restart.RestartRefusal)):
        _invoke_train(row, monkeypatch, calls)
    assert calls == []


def test_eval_consumed_generator_draw_type_drift_refuses_before_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The evaluator task copy must preserve the row's integer draw type."""

    row, path = _materialized_eval_row(tmp_path)
    original = row["draws"]
    _rewrite_materialized_row(
        row,
        path,
        lambda p: (
            p["runner"]["evaluator"]["tasks"][0]["generator"].__setitem__(
                "n_draws", float(original),
            ),
            assert_type_drift_is_in_consumed_copy(p, original),
        )[-1],
    )
    calls: list[object] = []
    with pytest.raises((protocol.ProtocolError, restart.RestartRefusal)):
        _invoke_eval(row, monkeypatch, calls)
    assert calls == []


def test_train_root_horizon_type_guard_is_separate_from_runner_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root trainer horizon float is independently rejected."""

    row, path = _train_row(tmp_path)
    _rewrite_materialized_row(row, path, lambda p: p["trainer"].__setitem__("max_steps", 10.0))
    calls: list[object] = []
    with pytest.raises((protocol.ProtocolError, restart.RestartRefusal)):
        _invoke_train(row, monkeypatch, calls)
    assert calls == []


@pytest.mark.parametrize("kind", ("train", "eval"))
def test_pristine_materialized_rows_reach_core_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    """Controls prove each independent probe has a live green sibling."""

    if kind == "train":
        row, _ = _train_row(tmp_path)
        calls: list[object] = []
        _invoke_train(row, monkeypatch, calls)
    else:
        row, _ = _materialized_eval_row(tmp_path)
        calls = []
        _invoke_eval(row, monkeypatch, calls)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("dtype", "expected_calls"),
    ((None, 1), ("float64", 1), ("float32", 0)),
)
def test_train_sampler_dtype_is_optional_but_when_present_must_match_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: str | None,
    expected_calls: int,
) -> None:
    """The contract permits omitted sampler dtype but rejects a mismatch."""

    row, path = _train_row(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        sampler = payload["sampler"]
        runner_sampler = payload["runner"]["sampler"]
        if dtype is None:
            sampler.pop("dtype", None)
            runner_sampler.pop("dtype", None)
        else:
            sampler["dtype"] = dtype
            runner_sampler["dtype"] = dtype

    _rewrite_materialized_row(row, path, mutate)
    if expected_calls:
        calls: list[object] = []
        _invoke_train(row, monkeypatch, calls)
        assert len(calls) == expected_calls
    else:
        calls = []
        with pytest.raises((protocol.ProtocolError, restart.RestartRefusal)):
            _invoke_train(row, monkeypatch, calls)
        assert calls == []



@pytest.mark.parametrize("field", ("seed", "sampler-walkers", "sampler-seed"))
@pytest.mark.parametrize("copy_name", ("root-task", "evaluator-task", "runner-task"))
def test_eval_consumed_task_scalar_types_refuse_before_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, copy_name: str,
) -> None:
    """Each nested task copy preserves the integer seed and sampler bindings."""

    row, path = _materialized_eval_row(tmp_path)

    def mutate(payload):
        task = {
            "root-task": payload["evaluation_tasks"]["mcmc_energy"],
            "evaluator-task": payload["evaluator"]["tasks"][0],
            "runner-task": payload["runner"]["evaluator"]["tasks"][0],
        }[copy_name]
        owner = task["generator"]
        key = "seed"
        if field.startswith("sampler-"):
            owner = owner["sampler"]
            key = "n_walkers" if field == "sampler-walkers" else "seed"
        assert type(owner[key]) is int
        owner[key] = float(owner[key])

    _rewrite_materialized_row(row, path, mutate)
    calls: list[object] = []
    with pytest.raises((protocol.ProtocolError, restart.RestartRefusal)):
        _invoke_eval(row, monkeypatch, calls)
    assert calls == []


@pytest.mark.parametrize("owner", ("sampler", "generator"))
@pytest.mark.parametrize("dtype", (None, "float64", "float32"))
def test_eval_dtype_is_optional_but_explicit_conflict_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str, dtype: str | None,
) -> None:
    """Omitted and matching dtypes admit; conflicting dtypes refuse before core."""

    row, path = _materialized_eval_row(tmp_path)

    def mutate(payload):
        generators = [
            payload["evaluation_tasks"]["mcmc_energy"]["generator"],
            payload["evaluator"]["tasks"][0]["generator"],
            payload["runner"]["evaluator"]["tasks"][0]["generator"],
        ]
        targets = generators if owner == "generator" else [
            payload["evaluation_sampler"], *(g["sampler"] for g in generators),
        ]
        for target in targets:
            if dtype is None:
                target.pop("dtype", None)
            else:
                target["dtype"] = dtype

    _rewrite_materialized_row(row, path, mutate)
    calls: list[object] = []
    if dtype == "float32":
        with pytest.raises((protocol.ProtocolError, restart.RestartRefusal)):
            _invoke_eval(row, monkeypatch, calls)
        assert calls == []
    else:
        _invoke_eval(row, monkeypatch, calls)
        assert len(calls) == 1
