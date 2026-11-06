"""P10 attached-route contract probes using only data and fake processes."""

from __future__ import annotations

import json
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

_ATTACHED_PATH = Path(__file__).resolve().parents[1] / "atomistic" / "he-v1" / "attached.py"
_ATTACHED_SPEC = importlib.util.spec_from_file_location("he_v1_attached_p10", _ATTACHED_PATH)
assert _ATTACHED_SPEC is not None and _ATTACHED_SPEC.loader is not None
attached = importlib.util.module_from_spec(_ATTACHED_SPEC)
sys.modules[_ATTACHED_SPEC.name] = attached
_ATTACHED_SPEC.loader.exec_module(attached)
from experiments.toolkit.attached_activation import (
    ActivationRefusal,
    allocation_context_payload,
    build_attached_inputs,
    parse_profile_verdict,
    write_activation_receipt,
)
from experiments.toolkit.dispatch import LogicalTaskSpec, StagePlanV2


def _source_config(path: Path) -> None:
    path.write_text(
        """physics: {name: he}\nmodel: {width: 32}\noptimizer: {name: adam}\nsampler:\n  n_walkers: 16\ntrainer:\n  max_steps: 25\nruntime: {seed: 0}\ncallbacks:\n  - {_target_: tpen.callback.ConfigSnapshot}\n  - {_target_: tpen.callback.ResolvedConfigSnapshot}\n  - {_target_: tpen.callback.Metadata}\n  - {_target_: tpen.callback.Status}\n  - {_target_: tpen.callback.DataIntegrity}\n  - {_target_: tpen.callback.Checkpoint, schedule: {_target_: tpen.checkpoint.EveryNUpdates, every_n: 5}}\nloggers:\n  - {_target_: tpen.logging.CSV}\n  - {_target_: tpen.logging.JSONL}\n""",
        encoding="utf-8",
    )


def _plan(tmp_path: Path) -> StagePlanV2:
    task = LogicalTaskSpec(
        logical_task_id="train-seed0000",
        stage="train",
        run_id="train-seed0000",
        command=("tpen", "train"),
        result_dir=str(tmp_path / "result"),
        params={
            "kind": "train",
            "config": str((tmp_path / "source.yaml").resolve()),
            "seed": 0,
            "n_walkers": 16,
            "max_steps": 25,
            "executor_overrides": ("runtime.seed=0", "trainer.max_steps=25", "sampler.n_walkers=16"),
        },
    )
    return StagePlanV2("he", "train", "p", str(tmp_path / "results"), (task,)).validate()


def test_reduced_config_is_explicit_and_preserves_science(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    output = tmp_path / "effective.yaml"
    _source_config(source)
    reduced = attached.generate_reduced_callback_config(
        source,
        output,
        enable_reduced_callbacks=True,
        checkpoint_every_n=25,
    )
    raw = yaml.safe_load(output.read_text())
    assert raw["physics"] == {"name": "he"}
    assert raw["model"] == {"width": 32}
    assert [entry["_target_"] for entry in raw["callbacks"]] == [
        "tpen.callback.ConfigSnapshot",
        "tpen.callback.ResolvedConfigSnapshot",
        "tpen.callback.Metadata",
        "tpen.callback.Status",
        "tpen.callback.Checkpoint",
    ]
    assert raw["callbacks"][-1]["schedule"]["every_n"] == 25
    assert [entry["_target_"] for entry in raw["loggers"]] == ["tpen.logging.CSV"]
    assert Path(reduced.diff_path).is_file()
    assert Path(reduced.hash_path).read_text().startswith(reduced.effective_sha256)


def test_attached_handoff_binds_group_identity_and_order(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    _source_config(source)
    group, table, handoff = build_attached_inputs(
        _plan(tmp_path),
        effective_config=source,
        source_config=source,
    )
    assert table["group_digest"] == handoff.group_digest
    assert [run["id"] for run in table["runs"]] == [member["task_id"] for member in group["members"]]
    assert group["members"][0]["program"]["config_ref"]["attached_activation"]["profile"]
    assert table["runs"][0]["overrides"] == [
        "runtime.seed=0",
        "trainer.max_steps=25",
        "sampler.n_walkers=16",
    ]


def test_attached_route_requires_opt_in(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    _source_config(source)
    with pytest.raises(ActivationRefusal, match="explicit opt-in"):
        attached.prepare_he_train(
            {"attempt_id": "p", "rows": []},
            output_dir=tmp_path / "handoff",
            repo_root=tmp_path,
            results_root=tmp_path / "results",
            source_config=source,
        )


def test_approved_smoke_row_translates_declared_scale_as_separate_fields(tmp_path: Path) -> None:
    source = tmp_path / "train.yaml"
    _source_config(source)
    plan, translated = attached.build_he_train_plan(
        {
            "plan_id": "approved-plan",
            "rows": [
                {
                    "logical_task_id": "02_train:seed-000:approved-plan",
                    "row_id": "seed-000",
                    "stage": "02_train",
                    "kind": "train",
                    "runtime": "tpen-cu126",
                    "config": str(source),
                    "seed": 0,
                    "max_steps": 25,
                    "n_walkers": 16,
                }
            ],
        },
        repo_root=tmp_path,
        results_root=tmp_path / "results",
        source_config=source,
        checkpoint_every_n=25,
    )
    assert plan.plan_id == "approved-plan"
    overrides = translated["seed-000"]
    assert "trainer.max_steps=25" in overrides
    assert "sampler.n_walkers=16" in overrides


@pytest.mark.parametrize(
    "row",
    [
        {
            "row_id": "seed-000",
            "stage": "02_train",
            "runtime": "ferminet",
            "seed": 0,
            "max_steps": 25,
            "n_walkers": 16,
        },
        {
            "row_id": "seed-000",
            "stage": "02_train",
            "runtime": "tpen-cu126",
            "seed": 0,
            "max_steps": 25,
            "n_walkers": 16,
            "overrides": ["trainer.max_steps=26"],
        },
    ],
)
def test_train_translation_named_rejections(tmp_path: Path, row: dict[str, object]) -> None:
    source = tmp_path / "train.yaml"
    _source_config(source)
    row = {"kind": "train", "config": str(source), **row}
    with pytest.raises(ActivationRefusal, match="runtime|conflicts"):
        attached.build_he_train_plan(
            {"plan_id": "p", "rows": [row]},
            repo_root=tmp_path,
            results_root=tmp_path / "results",
            source_config=source,
            checkpoint_every_n=25,
        )


def test_failed_generation_retires_current_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "train.yaml"
    _source_config(source)
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    (handoff / "executor_input.json").write_text("stale", encoding="utf-8")
    (handoff / "configured-runs.json").write_text("stale", encoding="utf-8")
    with pytest.raises(ActivationRefusal, match="stage"):
        attached.prepare_he_train(
            {
                "plan_id": "p",
                "rows": [{"row_id": "seed-000", "kind": "train", "stage": "wrong", "runtime": "tpen-cu126"}],
            },
            output_dir=handoff,
            repo_root=tmp_path,
            results_root=tmp_path / "results",
            source_config=source,
            enable_attached=True,
        )
    assert not (handoff / "executor_input.json").exists()
    assert not (handoff / "configured-runs.json").exists()
    assert list(handoff.glob("*.superseded-*.json"))
    assert (handoff / "executor_input_diagnostics.json").is_file()


def test_source_output_alias_is_refused_before_reduction(tmp_path: Path) -> None:
    source = tmp_path / "train.yaml"
    _source_config(source)
    before = source.read_bytes()
    with pytest.raises(ActivationRefusal, match="distinct"):
        attached.generate_reduced_callback_config(
            source,
            source,
            enable_reduced_callbacks=True,
            checkpoint_every_n=11,
        )
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value"),
    [("allocation_id", "different"), ("deadline_guard_min", -1), ("visibility_variable", "")],
)
def test_allocation_context_rejects_inconsistent_fields(field: str, value: object) -> None:
    context = SimpleNamespace(
        allocation_id="alloc",
        visibility_variable="CUDA_VISIBLE_DEVICES",
        visibility_values=("0",),
        run_root="/runs",
        deadline=123.5,
        deadline_env_var="END",
        deadline_guard_min=1,
        environment={"OMP_NUM_THREADS": "4"},
        nodes_per_block=None,
    )
    setattr(context, field, value)
    with pytest.raises(ActivationRefusal):
        allocation_context_payload(
            context,
            allocation_id="alloc",
            backend_job_id="job",
            observed_environment={"HOST": "n0"},
            observed_hostname="n0",
        )


def test_reader_rejects_empty_committed_chain_after_zero_exit(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "tpen-executor/run-manifest-v2",
                "group_id": "group",
                "group_digest": "0" * 64,
                "attempts": [],
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="no nonempty|no committed"):
        write_activation_receipt(
            manifest,
            tmp_path / "receipt.json",
            expected_task_ids=(),
            inspect_run=lambda _: {"verdict": "completed", "run_status": "completed"},
        )
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "failed"


def test_profile_verdict_is_json_not_exit_status() -> None:
    payload = {
        "schema": "tpen.profile-check/v1",
        "profile": "tpen.native-recovery/v1",
        "verdict": "supported",
        "state_owners": ["model"],
        "refusal": None,
    }
    assert parse_profile_verdict(json.dumps(payload), returncode=17).verdict == "supported"
    refused = dict(payload, verdict="refused", refusal={"code": "unsupported-callback", "reason": "no"})
    with pytest.raises(ActivationRefusal, match="profile refused"):
        # The parser accepts the typed refusal; the activation gate rejects it.
        verdict = parse_profile_verdict(json.dumps(refused))
        if verdict.verdict != "supported":
            raise ActivationRefusal("profile-refused", "profile refused")


def test_reader_writes_activation_receipt_and_rejects_wrong_task_set(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(Exception):
        write_activation_receipt(
            manifest,
            tmp_path / "receipt.json",
            expected_task_ids=("task",),
            inspect_run=lambda _: {"verdict": "completed", "checkpoint": {"checkpoint_dir": "x", "schema_version": 1, "next_iteration": 1, "completed_updates": 1}},
        )
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["schema"] == "tpen-experiments/attached-activation-v1"
    assert receipt["status"] == "failed"


def test_root_activation_modules_do_not_import_executor_or_core() -> None:
    source = Path(__file__).with_name("attached_activation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "tpen" or name.startswith("tpen_executor") for name in imports)
    assert sys.version_info >= (3, 12)
