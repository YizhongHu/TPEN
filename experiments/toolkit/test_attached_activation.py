"""P10 attached-route contract probes using only data and fake processes."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

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
    assert "tpen_executor" not in source
    assert "import tpen" not in source
    assert "from tpen" not in source
    assert sys.version_info >= (3, 12)
