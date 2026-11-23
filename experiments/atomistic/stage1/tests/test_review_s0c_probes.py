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
    compose_train_overlay,
    config_identities,
    overlay_sha256,
    resolve_config_path,
    validate_resolved_train_config,
)
from tpen.run import load_config
from omegaconf import OmegaConf


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


def _bounded_hooke_train_row(system: str) -> dict[str, object]:
    """Build a bounded qualification row with no science-profile identity."""

    protocol = Protocol(1, 2, 1, 2, 3, 4, 2, 2, 1)
    return next(
        row for row in build_manifest(
            protocol=protocol,
            source_sha="a" * 40,
            config_sha256=config_identities(config_root=ROOT / "configs", systems=(system,)),
            checkpoints=[0, 1],
            lineage_seeds=[11, 12, 13],
            checkpoint_sha256={},
            results_root="/external/stage1-review-probes",
            systems=(system,),
            phase=1,
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


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("runner.trainer.max_steps", "1"),
        ("runner.sampler.seed", "123"),
        ("runner.sampler.n_walkers", "2"),
        ("runner.sampler.burn_in", "1"),
        ("runner.sampler.n_steps", "1"),
        ("runner.sampler.proposal_scale", "0.25"),
        ("runner.trainer.log_every_n_steps", "7"),
        ("runner.trainer.return_terms", "false"),
    ],
)
def test_public_production_hooke_rows_refuse_independent_runner_overrides(
    monkeypatch: pytest.MonkeyPatch, path: str, value: str,
) -> None:
    """Runner-owned sampler/trainer leaves cannot bypass the science guard.

    The override targets the runner ancestor directly, leaving the root
    sampler/trainer values supplied by the manifest row unchanged.  A valid
    production row must therefore refuse before the execution stub.
    """

    row = _hooke_train_row(
        "hooke1", protocol=STAGE1_PROTOCOL,
        checkpoints=list(range(0, STAGE1_PROTOCOL.lineage_updates + 1, 1000)),
    )

    def reached_execution(cfg: object, **_kwargs: object) -> int:
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(resolved, dict)
        root = resolved["sampler" if "sampler" in path else "trainer"]
        runner = resolved["runner"]["sampler" if "sampler" in path else "trainer"]
        pytest.fail(
            f"independent {path} override reached execution: "
            f"root={root!r} runner={runner!r}"
        )

    with pytest.raises(ProtocolError, match="runner|science|config|bound"):
        _invoke_train(
            monkeypatch,
            row,
            extra_overrides=row_overrides(row) + [f"{path}={value}"],
            run_stub=reached_execution,
        )


def test_final_guard_refuses_forged_short_hooke_row_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final resolved-config guard also rejects a forged short profile.

    This bypasses the manifest gateway intentionally: it checks whether the
    final guard owns the fixed production dimensions, rather than relying on
    ``validate_gateway_binding`` to reject the row first.
    """

    monkeypatch.setenv("TPEN_STAGE1_RESULTS_ROOT", "/external/stage1-review-probes")
    row = _hooke_train_row(
        "hooke1", protocol=STAGE1_PROTOCOL,
        checkpoints=list(range(0, STAGE1_PROTOCOL.lineage_updates + 1, 1000)),
    )
    config_path = resolve_config_path(str(row["config"]))
    cfg = load_config(str(config_path))
    OmegaConf.update(cfg, "run.root", row["results_root"], merge=False)
    base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    compose_train_overlay(
        cfg, overlay=str(row["overlay"]), expected_sha256=row["overlay_sha256"],
    )
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(row_overrides(row)))
    stage1_train.prepare_initial_checkpoint_config(cfg)
    # Establish that the same fully composed production configuration is
    # accepted before forging the row and matching short runner leaves.
    validate_resolved_train_config(cfg, row, base_config=base_cfg)

    forged = dict(row)
    forged.update(updates=1, walkers=2, checkpoint_steps=[0, 1])
    short_cfg = load_config(str(config_path))
    OmegaConf.update(short_cfg, "run.root", forged["results_root"], merge=False)
    short_base_cfg = OmegaConf.create(OmegaConf.to_container(short_cfg, resolve=False))
    compose_train_overlay(
        short_cfg, overlay=str(forged["overlay"]),
        expected_sha256=forged["overlay_sha256"],
    )
    short_cfg = OmegaConf.merge(
        short_cfg,
        OmegaConf.from_dotlist(row_overrides(row)),
        OmegaConf.from_dotlist([
            "sampler.n_walkers=2",
            "trainer.max_steps=1",
            "checkpoint_steps=[0,1]",
        ]),
    )
    stage1_train.prepare_initial_checkpoint_config(short_cfg)
    with pytest.raises(ProtocolError, match="science|profile|horizon|walker|checkpoint"):
        validate_resolved_train_config(short_cfg, forged, base_config=short_base_cfg)


def test_planner_refuses_shortened_hooke_science_profile() -> None:
    """The planner cannot issue a science row for reduced dimensions."""

    short_protocol = replace(
        STAGE1_PROTOCOL,
        lineage_updates=1,
        lineage_walkers=2,
        retention_cadence=1,
    )
    with pytest.raises(ProtocolError, match="exact production protocol dimensions"):
        _hooke_train_row("hooke1", protocol=short_protocol, checkpoints=[0, 1])


def test_bounded_no_profile_row_reaches_execution_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reduced qualification row is green when it carries no science profile."""

    row = _bounded_hooke_train_row("hooke1")
    captured: dict[str, object] = {}

    def capture(cfg: object, **kwargs: object) -> int:
        captured["cfg"] = cfg
        return 0

    assert _invoke_train(monkeypatch, row, run_stub=capture) == 0
    assert "cfg" in captured


def test_production_hooke_profile_refuses_shortened_forged_row_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gateway rejects a forged shortened science row before model overrides."""

    row = _hooke_train_row(
        "hooke1", protocol=STAGE1_PROTOCOL,
        checkpoints=list(range(0, STAGE1_PROTOCOL.lineage_updates + 1, 1000)),
    )
    row["updates"] = 1
    row["walkers"] = 2
    row["checkpoint_steps"] = [0, 1]

    def reached_execution(*_args: object, **_kwargs: object) -> int:
        pytest.fail("shortened science-profile row reached execution")

    with pytest.raises(ProtocolError, match="science|profile|horizon|walker|checkpoint"):
        _invoke_train(
            monkeypatch,
            row,
            extra_overrides=row_overrides(row) + ["model.readout.trainable=false"],
            run_stub=reached_execution,
        )
