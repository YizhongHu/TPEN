"""Acceptance tests for acceptance-contract amendment 20261112-r1 (TO 69cb2853):
the ``capacity-pilot-100k`` payload -- a fresh 0->100,000-update lineage P,
same config/seeds/module set as ``capacity-pilot``, with a generalized
stage-1/SEL ordinal window (0..100, window 3..99) and no stage 2.

Reuses ``test_review_eval_job_r1.py``'s fixture toolkit (the ``job``/``ev``/
``protocol``/``launch`` module aliases, ``_train``/``_checkpoint``/``_sha``
helpers, ``_fake_compute``) rather than re-deriving it, the same way
``test_review_eval_job_r2.py`` does.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("h_v2_review_r1_helpers", HERE / "test_review_eval_job_r1.py")
assert _spec is not None and _spec.loader is not None
helpers = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = helpers
_spec.loader.exec_module(helpers)
job, ev, protocol, launch = helpers.job, helpers.ev, helpers.protocol, helpers.launch
TRAIN_SHA, EVAL_SHA = helpers.TRAIN_SHA, helpers.EVAL_SHA

PAYLOAD = "capacity-pilot-100k"


@pytest.fixture(autouse=True)
def shared_modules(monkeypatch):
    monkeypatch.setattr(job, "protocol", protocol)


def _train_100k(root):
    return helpers._train(root, payload=PAYLOAD, updates=range(0, 101000, 1000))


def _b1_100k(root, *, train_receipt_sha256="c" * 64):
    """Mirror ``helpers._b1``, generalized to the 101-ordinal stream-0 sweep.

    A clear minimum-W triple is crafted at ordinals 49/50/51 (narrow spread);
    every other ordinal is flat and wide, so SEL's basin pick is unambiguous.
    """

    generations = []
    for ordinal in range(101):
        width = 1 if ordinal in (49, 50, 51) else 10
        path = root / f"b1-100k-{ordinal}.csv"
        path.write_text(f"local_energy\n{-width}\n0\n{width}\n")
        generations.append(
            {
                "ordinal": ordinal,
                "update": ordinal * 1000,
                "stream": 0,
                "seed": protocol.pilot_eval_seed(ordinal, 0),
                "artifacts": {"trajectory_csv": {"path": str(path), "sha256": helpers._sha(path)}},
            }
        )
    path = job._eval_receipt_path(root, f"h-v2-eval-{PAYLOAD}-stage1")
    receipt = ev.write_eval_receipt(
        path,
        eval_commit=EVAL_SHA,
        train_commit=TRAIN_SHA,
        train_receipt_path=root / "train.json",
        train_receipt_sha256=train_receipt_sha256,
        payload=PAYLOAD,
        generations=generations,
        readout_generator_enabled=True,
    )
    return path, receipt


def _env(root, monkeypatch) -> None:
    monkeypatch.setattr(job, "check_identity", lambda *a, **k: None)
    for name, value in {"RUN": str(root), "SHA": EVAL_SHA, "SCIENCE_PYTHON": sys.executable}.items():
        monkeypatch.setenv(name, value)


# --- eval-fanout: real CLI dispatch, faked compute --------------------------


def test_eval_fanout_dispatch_writes_101_generations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _train_100k(tmp_path)
    helpers._fake_compute(monkeypatch)
    _env(tmp_path, monkeypatch)
    rc = job.main(["eval-fanout", PAYLOAD, "--pilot-stage", "1", "--expected-train-commit", TRAIN_SHA])
    assert rc == 0
    receipt_path = job._eval_receipt_path(tmp_path, f"h-v2-eval-{PAYLOAD}-stage1")
    receipt = ev.read_eval_receipt(receipt_path)
    assert sorted(g["ordinal"] for g in receipt["generations"]) == list(range(101))
    assert {g["stream"] for g in receipt["generations"]} == {0}


def test_eval_fanout_requires_pilot_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _train_100k(tmp_path)
    helpers._fake_compute(monkeypatch)
    _env(tmp_path, monkeypatch)
    rc = job.main(["eval-fanout", PAYLOAD, "--expected-train-commit", TRAIN_SHA])
    assert rc == 2


@pytest.mark.parametrize("pilot_stage", ["2", "2-core"])
def test_eval_fanout_refuses_pilot_stage_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pilot_stage: str) -> None:
    """Mutant: the 100k payload has no stage 2 -- it must refuse, never silently
    reuse capacity-pilot's own stage-2/2-core route."""

    _train_100k(tmp_path)
    helpers._fake_compute(monkeypatch)
    _env(tmp_path, monkeypatch)
    rc = job.main(["eval-fanout", PAYLOAD, "--pilot-stage", pilot_stage, "--expected-train-commit", TRAIN_SHA])
    assert rc != 0


# --- sel: real CLI dispatch -------------------------------------------------


def test_sel_cli_writes_the_100k_named_selection_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _b1_100k(tmp_path)
    monkeypatch.setenv("RUN", str(tmp_path))
    rc = job.main(["sel", PAYLOAD])
    assert rc == 0
    selection_path = tmp_path / "attached" / "receipts" / f"h-v2-eval-{PAYLOAD}-selection.json"
    assert selection_path.is_file()
    # Never collides with capacity-pilot's own selection file name.
    assert not (tmp_path / "attached" / "receipts" / "h-v2-eval-capacity-pilot-selection.json").exists()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["c1"] == 50
    assert selection["interior_minimum"] is True


def test_sel_cli_refuses_an_unrecognized_payload_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN", str(tmp_path))
    assert job.main(["sel", "bogus-payload"]) == 2
    assert job.main(["sel", "capacity-pilot", "capacity-pilot-100k"]) == 2


# --- run_eval_fanout / run_pilot_sel: direct API, named refusals -----------


def test_run_eval_fanout_requires_pilot_stage_for_the_100k_payload(tmp_path: Path) -> None:
    with pytest.raises(job.JobError, match="--pilot-stage"):
        job.run_eval_fanout(
            run_root=tmp_path,
            expected_commit=EVAL_SHA,
            science_python=sys.executable,
            payload=PAYLOAD,
            lineage="P",
            expected_train_commit=TRAIN_SHA,
            pilot_stage=None,
        )


def test_run_pilot_sel_round_trips_for_the_100k_payload(tmp_path: Path) -> None:
    _, receipt = _b1_100k(tmp_path)
    job.run_pilot_sel(tmp_path, payload=PAYLOAD)
    selection_path = tmp_path / "attached" / "receipts" / f"h-v2-eval-{PAYLOAD}-selection.json"
    assert selection_path.is_file()


# --- regression: capacity-pilot's own dispatch is unaffected ---------------


def test_run_pilot_sel_default_payload_is_still_capacity_pilot(tmp_path: Path) -> None:
    helpers._b1(tmp_path)
    job.run_pilot_sel(tmp_path)
    assert (tmp_path / "attached" / "receipts" / "h-v2-eval-capacity-pilot-selection.json").is_file()
