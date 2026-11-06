"""P10 golden binding between the root handoff and executor run-table wire."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from tpen_executor._codec import canonical_bytes
from tpen_executor.tasks import ConfiguredRunTable


ROOT = Path(__file__).resolve().parents[2]


_PRODUCER = r'''
from pathlib import Path
import sys
from experiments.toolkit.attached_activation import build_attached_inputs
from experiments.toolkit.dispatch import LogicalTaskSpec, StagePlanV2

out = Path(sys.argv[1]).resolve()
source = out / "source.yaml"
source.write_text("callbacks: []\nloggers: []\n", encoding="utf-8")
task = LogicalTaskSpec(
    logical_task_id="train:seed-0000",
    stage="train",
    run_id="train:seed-0000",
    command=("tpen", "train"),
    result_dir=str(out / "result"),
    params={
        "kind": "train",
        "config": str(source),
        "seed": 0,
        "n_walkers": 16,
        "max_steps": 25,
        "executor_overrides": ("runtime.seed=0",),
    },
)
plan = StagePlanV2("he", "train", "golden", str(out), (task,)).validate()
group, table, _ = build_attached_inputs(plan, effective_config=source, source_config=source)
from experiments.toolkit.executor_inputs import canonical_bytes
(out / "executor_input.json").write_bytes(canonical_bytes(group) + b"\n")
(out / "configured-runs.json").write_bytes(canonical_bytes(table) + b"\n")
'''


def test_p10_configured_runs_sidecar_matches_executor_table(tmp_path: Path) -> None:
    root_python = os.environ.get("P10_ROOT_PYTHON")
    if not root_python:
        pytest.skip("named skip: P10_ROOT_PYTHON is unset")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    produced = subprocess.run(
        [root_python, "-c", _PRODUCER, str(tmp_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert produced.returncode == 0, produced.stdout + produced.stderr
    raw_input = json.loads((tmp_path / "executor_input.json").read_text())
    raw_runs = (tmp_path / "configured-runs.json").read_bytes().rstrip(b"\n")
    table = ConfiguredRunTable.decode(raw_runs)
    group_digest = hashlib.sha256(
        json.dumps(raw_input, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert table.group_digest == group_digest
    assert [run.id for run in table.runs] == [member["task_id"] for member in raw_input["members"]]
    assert canonical_bytes(table.to_dict()) == raw_runs
