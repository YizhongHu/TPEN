"""Reviewer probes for the public Hooke training gateway.

These tests intentionally exercise ``stage1.train.main`` with only the final
``run_from_config`` call replaced.  They are review probes, not product fixes:
the assertions encode the intended admission/execution contract and therefore
expose a failing boundary if the current implementation violates it.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.launch import _binding_token, row_overrides
from experiments.atomistic.stage1.protocol import (
    HOOKE_OVERLAY_RELATIVE_PATH,
    HOOKE_SCIENCE_PROFILE,
    Protocol,
    ProtocolError,
    STAGE1_PROTOCOL,
    build_manifest,
    config_identities,
    overlay_sha256,
)


ROOT = Path(__file__).parents[1]
OVERLAY = ROOT / HOOKE_OVERLAY_RELATIVE_PATH


def _hooke_train_row(system: str, *, protocol: Protocol,
                     checkpoints: list[int]) -> dict[str, object]:
    """Build one manifest row using the real production Hooke contract."""

    return next(
        row for row in build_manifest(
            protocol=protocol,
            source_sha="a" * 40,
            config_sha256=config_identities(config_root=ROOT / "configs", systems=(system,)),
            checkpoints=checkpoints,
            lineage_seeds=[11, 12, 13],
            checkpoint_sha256={},
            results_root="/external/stage1-review-probes",
            systems=(system,),
            phase=1,
            science_profile=HOOKE_SCIENCE_PROFILE,
            evaluation_selection="A37",
            overlay=str(OVERLAY),
            overlay_sha256=overlay_sha256(OVERLAY),
            device="cuda",
        )
        if row["kind"] == "train"
    )


def _invoke_train(monkeypatch: pytest.MonkeyPatch, row: dict[str, object],
                  *, extra_overrides: list[str] | None = None,
                  run_stub: object) -> object:
    """Invoke the public CLI gateway while replacing execution only."""

    monkeypatch.setenv("TPEN_STAGE1_RESULTS_ROOT", "/external/stage1-review-probes")
    monkeypatch.setattr(stage1_train, "run_from_config", run_stub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1-train",
            "--config", str(row["config"]),
            "--kind", "train",
            *(extra_overrides or row_overrides(row)),
            "--binding", _binding_token(row),
            "--overlay", str(row["overlay"]),
            "--overlay-sha256", str(row["overlay_sha256"]),
        ],
    )
    return stage1_train.main()


@pytest.mark.parametrize("system", ["hooke1", "hooke2"])
def test_public_production_hooke_rows_reach_execution_stub(
    monkeypatch: pytest.MonkeyPatch, system: str,
) -> None:
    """A valid full production row must pass validation and reach execution."""

    row = _hooke_train_row(
        system,
        protocol=STAGE1_PROTOCOL,
        checkpoints=list(range(0, STAGE1_PROTOCOL.lineage_updates + 1, 1000)),
    )
    captured: dict[str, object] = {}

    def capture(cfg: object, **kwargs: object) -> int:
        captured["cfg"] = cfg
        captured["kwargs"] = kwargs
        return 0

    assert _invoke_train(monkeypatch, row, run_stub=capture) == 0
    assert "cfg" in captured


def test_production_hooke_profile_refuses_shortened_protocol_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The science profile cannot be retained while budgets are shortened."""

    short_protocol = replace(
        STAGE1_PROTOCOL,
        lineage_updates=1,
        lineage_walkers=2,
        retention_cadence=1,
    )
    row = _hooke_train_row("hooke1", protocol=short_protocol, checkpoints=[0, 1])

    def reached_execution(*_args: object, **_kwargs: object) -> int:
        pytest.fail("shortened science-profile row reached execution")

    with pytest.raises(ProtocolError, match="science|profile|horizon|walker|checkpoint"):
        _invoke_train(
            monkeypatch,
            row,
            extra_overrides=row_overrides(row) + ["model.readout.trainable=false"],
            run_stub=reached_execution,
        )
