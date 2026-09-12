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
_FAIL_OPEN_FINDING_NOW_CLOSED = (
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


# ---------------------------------------------------------------------------
# Group 2 -- FLIPPED.  These arms shipped pinning a DISCLOSED AVAILABILITY
# BOUND: the firewall refused spellings OmegaConf resolves without difficulty,
# because its own follower could not read them.  The source file said in as
# many words that if the follower were ever made grammar-normal these arms
# MUST BE FLIPPED to assert the reference is FOLLOWED, and must not be
# deleted.  Delegation made it grammar-normal, so they are flipped here, in
# the same commit as the repair, and the suite is never knowingly red.
# ---------------------------------------------------------------------------

_NOW_FOLLOWABLE = {
    "whitespace-padded": ({"experiment": {"name": "${ names.hi }"}, "names": {"hi": FAMILY}}, FAMILY),
    "relative-sibling": ({"experiment": {"name": "${.base}", "base": FAMILY}}, FAMILY),
    "mid-segment-interpolation": (
        {"experiment": {"name": "${a.b.c}"}, "a": {"b": "${x}"}, "x": {"c": FAMILY}},
        FAMILY,
    ),
}


@pytest.mark.parametrize("spelling", sorted(_NOW_FOLLOWABLE), ids=sorted(_NOW_FOLLOWABLE))
def test_hi_firewall_review_r2_followable_spelling_is_followed(spelling: str) -> None:
    """A spelling OmegaConf resolves must be FOLLOWED, and nothing may run.

    The flip is the point.  Refusing these was the safe direction while the
    follower could not read them; it is the WRONG direction now that it can,
    because refusing a readable identity is an availability cost paid for no
    safety.  The witness half is unchanged and still load-bearing: following
    must reach the answer without invoking a resolver.
    """

    body, expected = _NOW_FOLLOWABLE[spelling]
    cfg = OmegaConf.create(body)
    calls: list[Any] = []

    original = config_module.basis_feature_dim
    OmegaConf.register_new_resolver(RESOLVER, lambda *a: calls.append(a) or 1, replace=True)
    try:
        identity = identity_without_execution(cfg, "experiment.name")
        reference = OmegaConf.select(cfg, "experiment.name")
    finally:
        OmegaConf.register_new_resolver(RESOLVER, original, replace=True)

    assert identity.determined, f"{spelling} was refused: {identity.reason}"
    assert identity.value == expected
    # EQUIVALENCE: the follower's answer is the grammar's answer, not merely a
    # safe one. This is the property that retiring the second implementation
    # bought, stated as an assertion rather than as prose.
    assert identity.value == reference
    assert calls == [], "following invoked a resolver"


def test_hi_firewall_review_r2_follower_equals_the_grammar_or_refuses() -> None:
    """BEHAVIOURAL half of acceptance, over every carrier in this file.

    Wherever the follower claims to have DETERMINED an identity, its answer
    must equal OmegaConf's.  Where it cannot, it must refuse rather than
    answer.  A census proving the reimplementation is gone cannot show this;
    equality with the delegate is what makes "no second implementation" a
    measured property rather than a structural claim.
    """

    corpus: list[tuple[str, dict[str, object], str]] = [
        (name, build(), "schema" if name == "schema-side" else "experiment.name")
        for name, build in sorted(_COLLISION_CARRIERS.items())
    ]
    corpus += [(k, v[0], "experiment.name") for k, v in sorted(_NOW_FOLLOWABLE.items())]

    disagreements: list[str] = []
    determined = 0
    for name, body, path in corpus:
        cfg = OmegaConf.create(body)
        identity = identity_without_execution(cfg, path)
        if not identity.determined:
            continue
        determined += 1
        reference = OmegaConf.select(cfg, path)
        if identity.value != reference:
            disagreements.append(f"{name}: follower={identity.value!r} grammar={reference!r}")

    assert not disagreements, disagreements
    assert determined >= len(corpus) - 1, (
        f"only {determined} of {len(corpus)} carriers were determined; an "
        "equivalence sweep that refuses almost everything proves little"
    )


def test_hi_firewall_review_r2_the_reimplementation_is_gone() -> None:
    """STATIC half of acceptance: no second path-semantics implementation.

    A census alone would pass a reimplementation that merely MOVED, which is
    why it is paired with the behavioural equivalence arm above.  This half
    catches the opposite failure: an equivalence sweep passes a module that
    still carries a dormant hand-rolled walk nothing currently calls.

    Named symbols rather than a shape, because the hazard is a SECOND
    implementation existing at all, not any particular spelling of one.
    """

    import ast
    import inspect

    import tpen.hi_schema as module

    source_path = Path(inspect.getsourcefile(module) or "")
    repo_root = Path(__file__).resolve().parents[2]
    assert repo_root in source_path.resolve().parents, source_path

    text = source_path.read_text()
    retired = ("_raw_lookup", "_expand_without_execution", "_names_a_resolver_call",
               "IDENTITY_FOLLOW_LIMIT", "_UnparsableInterpolation")
    present = sorted(name for name in retired if name in text)
    assert not present, (
        f"retired path-semantics machinery is back in {source_path.name}: {present}. "
        "Two implementations of path semantics is the class; leaving one leaves it"
    )

    tree = ast.parse(text)
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert "identity_without_execution" in defined
    assert not (defined & set(retired)), sorted(defined & set(retired))
