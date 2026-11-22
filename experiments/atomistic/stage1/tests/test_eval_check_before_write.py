"""Refuse Stage-1 eval reuse before the run lifecycle writes anything."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1 import eval as stage1_eval


def _config(path: Path, run_dir: Path):
    """Return the smallest config shape needed by the Stage-1 gateway."""

    return OmegaConf.create(
        {
            "experiment": {"name": "stage1", "sector": "hooke1"},
            "run": {"root": str(path), "run_id": run_dir.name, "layout": "nested"},
        }
    )


def _snapshot(root: Path) -> tuple[set[str], dict[str, str]]:
    """Record every path and hash every file below a run root."""

    paths = {str(path.relative_to(root)) for path in root.rglob("*")}
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    return paths, files


def _invoke_eval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cfg,
    run_calls: list[object],
) -> int:
    """Invoke the real gateway around a light, non-science config loader."""

    config_path = tmp_path / "eval.yaml"
    config_path.write_text("placeholder: true\n", encoding="utf-8")
    monkeypatch.setattr(
        stage1_eval,
        "validate_gateway_binding",
        lambda **_: {},
    )
    import tpen.run

    monkeypatch.setattr(tpen.run, "load_config", lambda *_: cfg)
    monkeypatch.setattr(
        stage1_eval,
        "run_from_config",
        lambda actual_cfg, **_: (run_calls.append(actual_cfg) or 0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval.py",
            "--config",
            str(config_path),
            "--binding",
            "unused",
            "--kind",
            "eval",
            "--expected-checkpoint-sha256",
            "a" * 64,
        ],
    )
    return stage1_eval.main()


@pytest.mark.parametrize("state", ["completed", "partial"])
def test_existing_eval_run_refuses_without_side_effects(
    state: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed and partial rows are both immutable once created."""

    results_root = tmp_path / "results"
    run_dir = results_root / "stage1" / "hooke1" / "eval-row"
    (run_dir / "mcmc_energy").mkdir(parents=True)
    (run_dir / "metadata.json").write_bytes(b"metadata-before\n")
    (run_dir / "status.json").write_bytes(b"status-before\n")
    (run_dir / "events.jsonl").write_bytes(b"event-before\n")
    if state == "completed":
        (run_dir / "mcmc_energy" / "sampled_eval_table.csv").write_bytes(
            b"sample_index,local_energy\n0,1.0\n"
        )
    before = _snapshot(run_dir)
    run_calls: list[object] = []

    with pytest.raises(
        stage1_eval.EvalRunRefusal,
        match=rf"existing result directory .*{run_dir.name}.*no run metadata",
    ):
        _invoke_eval(monkeypatch, tmp_path, _config(results_root, run_dir), run_calls)

    assert run_calls == []
    after = _snapshot(run_dir)
    assert after[0] == before[0]
    assert after[1] == before[1]


def test_fresh_eval_row_reaches_runner_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh row still reaches the configured runner exactly once."""

    results_root = tmp_path / "results"
    run_dir = results_root / "stage1" / "hooke1" / "fresh-row"
    run_calls: list[object] = []

    assert _invoke_eval(monkeypatch, tmp_path, _config(results_root, run_dir), run_calls) == 0
    assert len(run_calls) == 1
    assert run_calls[0].run.run_id == "fresh-row"
    assert not run_dir.exists()
