"""Narrow LR-3 review probes for planning and restart outcome boundaries."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1 import restart
from experiments.atomistic.stage1.launch import row_overrides
from experiments.atomistic.stage1.protocol import (
    HOOKE_SCIENCE_PROFILE,
    STAGE1_PROTOCOL,
    build_manifest,
    config_identities,
    derive_smoke_request,
)
from experiments.atomistic.stage1.train import build_train_config


PLAN = Path(__file__).parents[1] / "plan.py"


def _run_phase1_plan(
    tmp_path: Path, *systems: str,
) -> tuple[int, list[dict[str, object]] | None, str]:
    output = tmp_path / ("-".join(systems) if systems else "all") / "manifest.json"
    output.parent.mkdir(parents=True)
    command = [
        sys.executable,
        str(PLAN),
        "--output", str(output),
        "--source-sha", "a" * 40,
        "--results-root", str(output.parent / "results"),
        "--phase", "1",
        "--lineage-seeds", "11", "12", "13",
    ]
    if systems:
        command.extend(("--systems", *systems))
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        return completed.returncode, None, completed.stderr
    return completed.returncode, json.loads(output.read_text(encoding="utf-8"))["rows"], ""


@pytest.mark.parametrize("systems", [("hooke1",), ("he",)])
def test_production_planner_controls_preserve_expected_profile_boundary(
    tmp_path: Path, systems: tuple[str, ...],
) -> None:
    """Hooke-only and He-only planning controls remain valid populations."""

    returncode, rows, stderr = _run_phase1_plan(tmp_path, *systems)
    assert returncode == 0, stderr
    assert rows is not None
    train_rows = [row for row in rows if row["kind"] == "train"]
    assert train_rows
    assert {row["system"] for row in train_rows} == set(systems)
    if systems == ("hooke1",):
        assert all(row.get("science_profile") == "hooke-hv2-v1" for row in train_rows)
        assert all(isinstance(row.get("lineage_seed"), int) for row in train_rows)
        assert all(isinstance(row.get("sampler_seed"), int) for row in train_rows)
        assert all(row.get("training_seed_rule") == "stage1-training-seeds-v1" for row in train_rows)
    else:
        assert all("science_profile" not in row for row in train_rows)


@pytest.mark.parametrize("systems", [None, ("hooke1", "he")])
def test_mixed_or_omitted_planner_does_not_emit_unprofiled_hooke_rows(
    tmp_path: Path, systems: tuple[str, ...] | None,
) -> None:
    """Mixed/default Hooke planning refuses before writing a partial manifest."""

    returncode, rows, stderr = _run_phase1_plan(tmp_path, *(systems or ()))
    output = tmp_path / ("-".join(systems) if systems else "all") / "manifest.json"
    assert returncode != 0
    assert rows is None
    assert "refuses" in stderr
    assert "Hooke science profile" in stderr
    assert not output.exists()


def _cfg(tmp_path: Path) -> object:
    return OmegaConf.create(
        {
            "experiment": {"name": "stage1_he", "sector": "atomistic"},
            "run": {"root": str(tmp_path), "run_id": "run-0", "layout": "nested"},
            "system": {"id": "he_atom"},
            "trainer": {"max_steps": 100000},
            "runtime": {"seed": 11},
            "runner": {"_target_": "tpen.runner.Train"},
            "callbacks": [],
        }
    )


def _row(tmp_path: Path) -> dict[str, object]:
    return {
        "row_id": "he-lineage-0",
        "system": "he",
        "system_id": "he_atom",
        "lineage": 0,
        "seed": 11,
        "results_root": str(tmp_path),
        "experiment_name": "stage1_he",
        "sector": "atomistic",
        "run_id": "run-0",
        "updates": 100000,
        "config_sha256": "a" * 64,
        "checkpoint_steps": [0, 1000, 100000],
        "execution": {"stop_at": 1000},
    }


def _planned_prefix(tmp_path: Path) -> tuple[object, dict[str, object], Path, object]:
    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    run_dir = tmp_path / "stage1_he" / "atomistic" / "run-0"
    checkpoint = run_dir / "checkpoints" / "step_000000"
    checkpoint.mkdir(parents=True)
    evidence = SimpleNamespace(
        verdict="incomplete",
        outcome="planned_yield",
        refusal=None,
        checkpoint={
            "checkpoint_dir": str(checkpoint),
            "next_iteration": 0,
            "completed_updates": 0,
        },
        checkpoint_ref=None,
        run_dir=str(run_dir),
    )
    return cfg, row, run_dir, evidence


def _normalise_roots(value: object, roots: tuple[str, ...]) -> object:
    if isinstance(value, dict):
        return {key: _normalise_roots(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalise_roots(item, roots) for item in value]
    if isinstance(value, str):
        for root in roots:
            if value == root or value.startswith(root + "/"):
                return "<isolated-root>" + value[len(root):]
    return value


@pytest.mark.parametrize("system", ("hooke1", "hooke2", "he", "h2"))
def test_derived_smoke_composition_preserves_science_recipe(
    tmp_path: Path, system: str,
) -> None:
    """A derived prefix changes only destination and execution.stop_at."""

    source_root = tmp_path / "science" / system
    isolated_root = tmp_path / "smoke" / system
    hooke = system in {"hooke1", "hooke2"}
    kwargs: dict[str, object] = {}
    if hooke:
        kwargs.update({
            "science_profile": HOOKE_SCIENCE_PROFILE,
            "evaluation_selection": "A37",
        })
    rows = build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=config_identities(
            config_root=Path(__file__).parents[1] / "configs", systems=(system,)
        ),
        checkpoints=list(range(0, STAGE1_PROTOCOL.lineage_updates + 1, 1000)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(source_root),
        systems=(system,),
        phase=1,
        device="cpu",
        include_smoke=False,
        auto_overlay=True,
        **kwargs,
    )
    source = next(row for row in rows if row["kind"] == "train")
    derived = derive_smoke_request(source, results_root=str(isolated_root), stop_at=1000)
    config_root = Path(__file__).parents[1] / "configs"
    source_cfg = build_train_config(
        source,
        config_path=config_root / system / "train.yaml",
        overrides=row_overrides(source),
    )
    derived_cfg = build_train_config(
        derived,
        config_path=config_root / system / "train.yaml",
        overrides=row_overrides(derived),
    )
    source_doc = _normalise_roots(
        OmegaConf.to_container(source_cfg, resolve=True),
        (str(source_root),),
    )
    derived_doc = _normalise_roots(
        OmegaConf.to_container(derived_cfg, resolve=True),
        (str(isolated_root),),
    )
    assert derived_doc == source_doc
    assert derived["execution"] == {"stop_at": 1000}
    assert derived["walkers"] == 4096
    assert derived["updates"] == 100000
    assert derived["checkpoint_steps"] == source["checkpoint_steps"]
    assert derived.get("overlay") == source.get("overlay")
    assert derived.get("seed") == source.get("seed")
    later = derive_smoke_request(source, results_root=str(isolated_root), stop_at=2_000)
    later_cfg = build_train_config(
        later,
        config_path=config_root / system / "train.yaml",
        overrides=row_overrides(later),
    )
    later_doc = _normalise_roots(
        OmegaConf.to_container(later_cfg, resolve=True),
        (str(isolated_root),),
    )
    assert later_doc == source_doc
    for identity_key in (
        "config_sha256", "system_id", "overlay", "overlay_sha256", "science_profile",
        "seed", "lineage_seed", "sampler_seed", "updates", "walkers", "checkpoint_steps",
    ):
        assert later.get(identity_key) == source.get(identity_key)
    assert later["execution"] == {"stop_at": 2_000}
    if hooke:
        assert derived["science_profile"] == HOOKE_SCIENCE_PROFILE
        assert derived["lineage_seed"] == source["lineage_seed"]
        assert derived["sampler_seed"] == source["sampler_seed"]


@pytest.mark.parametrize(
    ("exit_code", "expected"),
    [(0, "completed"), (1, "failed"), (2, "planned_yield")],
)
def test_execute_restart_classifies_public_runner_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int, expected: str,
) -> None:
    """The public runner's planned-yield code must remain a planned yield."""

    cfg, row, run_dir, evidence = _planned_prefix(tmp_path)
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    result = restart.execute_restart(
        cfg=cfg,
        row=row,
        config_path="train.yaml",
        inspector=lambda _path: evidence,
        run=lambda *_a, **_k: exit_code,
        run_dir=run_dir,
    )
    assert result.exit_code == exit_code
    assert result.outcome == expected
