"""Round-2 reviewer probes for the HI firewall's without-execution follower.

These tests live apart from the ordinary HI schema suite for the same reason
the round-1 probes do: a red result here is evidence about the reviewed
implementation rather than a production fix hidden in a test.

WHAT THIS FILE PINS.  ``tpen.hi_schema`` decides whether a configuration
belongs to the helium-importance family WITHOUT EXECUTING anything, by
following raw node references (``identity_without_execution`` and its
helpers).  The follower reads an interpolation body as a VERBATIM ABSOLUTE
dotted key path.  OmegaConf's own grammar does not: it trims whitespace around
the body and resolves a leading dot as a RELATIVE reference.  The two readings
agree on ordinary spellings and diverge on two axes -- whitespace and leading
dots -- in both directions:

* FAIL-CLOSED where the follower's verbatim path does not exist while
  OmegaConf resolves it.  The firewall refuses ``undeterminable-identity``:
  an availability bound, pinned green in Group 2 with the disclosure that
  these arms encode a bound and not desired behaviour.
* FAIL-OPEN where a key-space COLLISION makes the verbatim path exist on a
  DIFFERENT node than the one OmegaConf reaches.  The follower then determines
  a WRONG identity, the firewall returns clean, and the run itself resolves to
  the helium-importance family with zero enforcement.  Group 1 pins those
  carriers with strict xfail so the marker cannot outlive the defect.

Group 3 pins the same property on the LIVE path through ``tpen.run``, which no
earlier probe executed, with a green control that varies exactly one thing --
the firewall on or off -- so the carrier arm's silence is evidence and not an
instrument that could never have spoken.

Timestamps in test logging are UTC by repository convention; nothing here
records one.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

import tpen.config as config_module
from tpen.config_schema import ClosedSchemaError
from tpen.hi_schema import (
    HI_EXPERIMENT_NAME,
    HI_TRAIN_SCHEMA,
    identity_without_execution,
    validate_hi_train_config,
)


RESOLVER = config_module.BASIS_FEATURE_DIM_RESOLVER
FAMILY = HI_EXPERIMENT_NAME

# The strict-xfail reason is shared by every fail-open arm.  One string keeps
# the finding stated once: a marker that drifts from its finding is a marker
# nobody can act on.
_FAIL_OPEN_REASON = (
    "R2 finding: _raw_lookup reads an interpolation body as a verbatim "
    "absolute dotted path while OmegaConf trims whitespace and resolves "
    "leading dots relative; a key-space collision makes the follower "
    "determine a WRONG identity and the firewall returns clean on a "
    "config whose resolved name is the HI family. See "
    "reviewer-r2-final-delivery on item 847dfff4."
)


def _config(**sections: object) -> DictConfig:
    """Build a minimal schema-declaring config for the probes.

    Parameters
    ----------
    **sections : object
        Extra top-level sections merged over the schema-declaring base.

    Returns
    -------
    DictConfig
        A configuration that opts in to :data:`HI_TRAIN_SCHEMA`.
    """

    base: dict[str, object] = {
        "schema": HI_TRAIN_SCHEMA,
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.005},
    }
    base.update(sections)
    return OmegaConf.create(base)


def _rules(error: ClosedSchemaError) -> set[str]:
    """Return the rule names a refusal reported."""

    return {rejection.rule for rejection in error.rejections}


def _validate(cfg: DictConfig) -> None:
    """Validate with an EMPTY environment, so the probe reads the config only."""

    validate_hi_train_config(cfg, env={})


# ---------------------------------------------------------------------------
# Carriers.  Each is a configuration whose OmegaConf-resolved identity is the
# helium-importance family, and whose verbatim-path reading is a decoy.
# ---------------------------------------------------------------------------


def _carrier_relative_plus_empty_key() -> dict[str, object]:
    """Relative sibling reference collided with a top-level empty-string key.

    ``${.hi_name}`` resolves, by the grammar, to the sibling
    ``experiment.hi_name``.  Read verbatim the path is ``".hi_name"``, which
    splits into the segments ``""`` and ``"hi_name"`` -- so a top-level key
    spelled ``""`` swallows the lookup and answers with a decoy.
    """

    return {
        "experiment": {"name": "${.hi_name}", "hi_name": FAMILY},
        "": {"hi_name": "not-the-family"},
        # A real reference energy makes the hazard concrete: this is exactly
        # the surface the firewall exists to keep out of an HI run.
        "model": {"reference_energy": -2.903724377},
    }


def _carrier_whitespace_plus_verbatim_key() -> dict[str, object]:
    """Padded body collided with a literally padded single-segment key."""

    return {
        "experiment": {"name": "${ n }"},
        "n": FAMILY,
        " n ": "not-the-family",
        "model": {"reference_energy": -2.903724377},
    }


def _carrier_whitespace_plus_padded_key() -> dict[str, object]:
    """Padded body collided with padded keys on a MULTI-segment path.

    The verbatim split of ``" names.hi "`` is ``" names"`` and ``"hi "``, so
    both segments must be padded for the decoy to be reachable.  A single
    padded key would leave the follower refusing rather than answering wrong.
    """

    return {
        "experiment": {"name": "${ names.hi }"},
        "names": {"hi": FAMILY},
        " names": {"hi ": "not-the-family"},
        "model": {"reference_energy": -2.903724377},
    }


def _carrier_schema_side() -> dict[str, object]:
    """The same collision on the SCHEMA key rather than the name.

    The failure direction is the mirror image and just as bad: OmegaConf
    resolves ``schema`` to the real HI declaration, so the run IS an HI run,
    while the follower reads a decoy schema and hands it no enforcement.
    """

    return {
        "schema": "${.sk}",
        "sk": HI_TRAIN_SCHEMA,
        "": {"sk": "decoy-schema"},
        "experiment": {"name": "other-exp"},
    }


_COLLISION_CARRIERS = {
    "rel-empty-key": _carrier_relative_plus_empty_key,
    "ws-verbatim-key": _carrier_whitespace_plus_verbatim_key,
    "ws-padded-key": _carrier_whitespace_plus_padded_key,
    "schema-side": _carrier_schema_side,
}


# ---------------------------------------------------------------------------
# Group 1 -- fail-open collision carriers.
# ---------------------------------------------------------------------------


def test_hi_firewall_review_r2_collision_free_shape_is_refused() -> None:
    """GREEN CONTROL for Group 1, and the arm to read first.

    The same shape with NO collision key: ``${names.hi}`` is spelled the way
    the follower and the grammar agree on, so the family is followed correctly
    and the missing schema declaration is refused.  Without this arm a row of
    xfails would be indistinguishable from an instrument that cannot see a
    refusal at all.
    """

    cfg = OmegaConf.create(
        {
            "experiment": {"name": "${names.hi}"},
            "names": {"hi": FAMILY},
            "model": {"reference_energy": -2.903724377},
        }
    )
    with pytest.raises(ClosedSchemaError) as caught:
        _validate(cfg)
    assert "undeclared-schema" in _rules(caught.value)
    # The control is only a control if the grammar really reaches the family.
    assert OmegaConf.select(cfg, "experiment.name") == FAMILY


@pytest.mark.xfail(strict=True, reason=_FAIL_OPEN_REASON)
@pytest.mark.parametrize("carrier", sorted(_COLLISION_CARRIERS), ids=sorted(_COLLISION_CARRIERS))
def test_hi_firewall_review_r2_collision_carrier_is_refused(carrier: str) -> None:
    """A collision carrier must be refused, by ANY rule.

    RULE-AGNOSTIC ON PURPOSE.  Two repairs close this finding -- following the
    grammar's own spelling, which makes the family visible and the refusal
    ``undeclared-schema``; or refusing a body the follower cannot read the way
    OmegaConf would, which makes it ``undeterminable-identity``.  Pinning a
    rule name would fail the repair that took the other direction, so the arm
    pins the PROPERTY: this configuration does not pass.

    Both directions were measured to refuse, including the schema-side
    carrier, whose grammar-normal repair reaches the ordinary schema sweeps
    and is refused there.
    """

    cfg = OmegaConf.create(_COLLISION_CARRIERS[carrier]())
    with pytest.raises(ClosedSchemaError):
        _validate(cfg)


@pytest.mark.xfail(strict=True, reason=_FAIL_OPEN_REASON)
def test_hi_firewall_review_r2_collision_carrier_is_refused_from_a_file(
    tmp_path: Path,
) -> None:
    """The carrier is reachable from a configuration FILE, not only a dict.

    A dict-built carrier invites the reading "no real config is spelled that
    way".  The empty-string key spells as ``"":`` in ordinary YAML, so the
    carrier survives the route a launch actually takes: a file on disk read
    through :func:`tpen.run.load_config`.
    """

    import tpen.run

    carrier = tmp_path / "carrier.yaml"
    carrier.write_text(
        "experiment:\n"
        "  name: ${.hi_name}\n"
        f"  hi_name: {FAMILY}\n"
        '"":\n'
        "  hi_name: not-the-family\n"
        "model:\n"
        "  reference_energy: -2.903724377\n",
        encoding="utf-8",
    )
    cfg = tpen.run.load_config(str(carrier))
    # The file really does carry the collision, and the grammar really does
    # reach the family from it -- otherwise the arm below would be xfailing
    # for a YAML-parsing reason rather than for the finding.
    assert "" in OmegaConf.to_container(cfg, resolve=False)
    assert OmegaConf.select(cfg, "experiment.name") == FAMILY
    with pytest.raises(ClosedSchemaError):
        _validate(cfg)


@pytest.mark.xfail(strict=True, reason=_FAIL_OPEN_REASON)
def test_hi_firewall_review_r2_follower_agrees_with_the_grammar_or_refuses() -> None:
    """The property behind every Group-1 arm, stated without validation.

    IF the follower determines an identity THEN it must be the identity
    OmegaConf itself resolves.  A follower that refuses is admissible here --
    that is the fail-closed direction Group 2 pins -- so this arm constrains
    only the answering case, which is the one that can hand an HI run zero
    enforcement.
    """

    cfg = OmegaConf.create(_carrier_whitespace_plus_verbatim_key())
    identity = identity_without_execution(cfg, "experiment.name")
    if identity.determined:
        assert identity.value == OmegaConf.select(cfg, "experiment.name")


# ---------------------------------------------------------------------------
# Group 3 -- the live path through tpen.run.
# ---------------------------------------------------------------------------


def _recording_run_name_config(tmp_path: Path) -> DictConfig:
    """Build a run config whose ``experiment.run_name`` is a resolver carrier.

    ``prepare_run_context`` selects ``experiment.run_name`` with OmegaConf,
    which RESOLVES it -- so a resolver sitting on that node runs as soon as
    the live path reaches context preparation, and not before.
    """

    return _config(
        experiment={
            "name": "hi_firewall_r2",
            "sector": "atomistic",
            "run_name": f"${{{RESOLVER}:r2-live-path}}",
        },
        run={"root": str(tmp_path / "outputs"), "run_id": "hi_firewall_r2_0001"},
    )


def test_hi_firewall_review_r2_live_path_control_reaches_the_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GREEN CONTROL for the live path: with the firewall off, the carrier RUNS.

    Run this arm first.  It differs from the carrier arm below in EXACTLY one
    thing -- ``validate_hi_train_config`` is replaced by a no-op -- so the
    carrier arm's silent witness is evidence that the firewall refused, rather
    than evidence that a resolver on this node could never have fired.

    ``_instantiate_runner`` is spied so nothing is constructed even here: the
    control needs the resolver reached, not a training run.
    """

    import tpen.run as run_module

    calls: list[Any] = []

    def witness(argument: Any) -> str:
        calls.append(argument)
        return "r2-witness-ran"

    reached: list[str] = []

    class _StopBeforeRunner(Exception):
        """Raised by the spy so no runner is constructed by a control."""

    def _stop(context: Any) -> None:
        reached.append("_instantiate_runner")
        raise _StopBeforeRunner()

    # THE ONE VARIED THING.  Both call sites -- run_from_config and
    # prepare_run_context -- read this module-level name, so one patch covers
    # the whole live path.
    monkeypatch.setattr(run_module, "validate_hi_train_config", lambda cfg, **kw: None)
    monkeypatch.setattr(run_module, "_instantiate_runner", _stop)

    cfg = _recording_run_name_config(tmp_path)
    original = config_module.basis_feature_dim
    OmegaConf.register_new_resolver(RESOLVER, witness, replace=True)
    stopped: BaseException | None = None
    try:
        run_module.run_from_config(cfg, raise_exceptions=True)
    except BaseException as error:  # noqa: BLE001 - the spy and the path both raise
        stopped = error
    finally:
        OmegaConf.register_new_resolver(RESOLVER, original, replace=True)

    # The control asserts REACHABILITY, not how far the run got: whatever
    # stopped the path is reported so a red here names its own cause.
    assert calls, (
        "the carrier node was never resolved on the live path, so the carrier "
        f"arm below could not have observed anything; run stopped by {stopped!r}, "
        f"reached={reached}"
    )


def test_hi_firewall_review_r2_live_path_carrier_is_refused_before_the_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the real firewall in place, the same carrier never runs.

    The pair of arms varies exactly one thing.  Everything asserted here --
    the silent witness, the unimported manifest, the empty ``tmp_path`` -- is
    a property the control proved is observable.
    """

    import tpen.run as run_module

    calls: list[Any] = []

    def witness(argument: Any) -> str:
        calls.append(argument)
        return "r2-witness-ran"

    def _stop(context: Any) -> None:
        raise AssertionError("_instantiate_runner was reached; the firewall did not refuse")

    monkeypatch.setattr(run_module, "_instantiate_runner", _stop)
    # sys.modules is process-global and another test may legitimately have
    # imported the manifest already.  Deleting the entry for the duration of
    # this test makes the assertion a fact about THIS call; monkeypatch
    # restores whatever was there.
    monkeypatch.delitem(sys.modules, "tpen.hi_manifest", raising=False)

    cfg = _recording_run_name_config(tmp_path)
    original = config_module.basis_feature_dim
    OmegaConf.register_new_resolver(RESOLVER, witness, replace=True)
    try:
        status = run_module.run_from_config(cfg)
    finally:
        OmegaConf.register_new_resolver(RESOLVER, original, replace=True)

    assert status == 1, "a refused run must report failure through the entry point"
    assert calls == [], "the carrier ran before the firewall refused"
    assert "tpen.hi_manifest" not in sys.modules
    assert list(tmp_path.iterdir()) == [], "a refused run left something on disk"


def test_hi_firewall_review_r2_live_path_file_route_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file route refuses a schema-key trampoline before it opens anything.

    The round-1 probes reached this carrier through ``validate`` directly.
    This arm takes the route a launch takes -- YAML on disk, ``load_config``,
    ``run_from_config`` -- because a firewall that holds only for a
    dict-constructed config is a property of the test harness.
    """

    import tpen.run as run_module

    marker = tmp_path / "marker" / "trampoline-ran"
    marker.parent.mkdir()
    # The shipped round-1 carrier shape: one Hydra trampoline whose payload
    # opens a marker file.  Four closers are load-bearing; one short and
    # OmegaConf rejects the string while BUILDING the config.
    carrier = (
        "tpen.hi.train.v${"
        + RESOLVER
        + ":{_target_: hydra.utils.instantiate, _recursive_: false, "
        "config: {out_features: 1, payload: "
        + f"{{_target_: builtins.open, file: {marker}, mode: w}}"
        + "}}}"
    )
    config_file = tmp_path / "run.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "schema": carrier,
                "experiment": {"name": FAMILY, "sector": "atomistic"},
                "run": {"root": str(tmp_path / "outputs"), "run_id": "hi_firewall_r2_0002"},
                "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.005},
            }
        ),
        config_file,
    )

    def _stop(context: Any) -> None:
        raise AssertionError("_instantiate_runner was reached; the firewall did not refuse")

    monkeypatch.setattr(run_module, "_instantiate_runner", _stop)

    cfg = run_module.load_config(str(config_file))
    status = run_module.run_from_config(cfg, config_path=str(config_file))

    assert status == 1
    assert not marker.exists(), "the trampoline payload ran before the refusal"
    assert not (tmp_path / "outputs").exists()
