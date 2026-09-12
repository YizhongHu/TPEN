"""Reviewer probes for HI firewall ordering and admission.

These tests intentionally live apart from the ordinary HI schema suite.  They
exercise process-global effects (files, Torch RNG, and Hydra's target guard)
that a refusal must contain, so a red result is evidence about the reviewed
implementation rather than a production fix hidden in the test.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import hydra.utils
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

import tpen.config as config_module
import tpen.hi_schema as hi_schema_module
from tpen.config_schema import ClosedSchemaError
from tpen.hi_schema import (
    HI_TRAIN_POLICY,
    HI_TRAIN_SCHEMA,
    validate_hi_train_config,
)


RESOLVER = config_module.BASIS_FEATURE_DIM_RESOLVER


def _config(**sections: object) -> DictConfig:
    """Build a minimal schema-declaring config for the probes."""

    base: dict[str, object] = {
        "schema": HI_TRAIN_SCHEMA,
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.005},
    }
    base.update(sections)
    return OmegaConf.create(base)


def _rules(error: ClosedSchemaError) -> set[str]:
    return {rejection.rule for rejection in error.rejections}


def _validate(cfg: DictConfig) -> None:
    validate_hi_train_config(cfg, env={})


def _resolver_carrier(marker: Path) -> str:
    return (
        f"${{{RESOLVER}:{{_target_: builtins.open, file: {marker}, mode: w}}}}"
    )


def _real_preflight_carrier(marker: Path) -> str:
    """Build a live prefix carrier that resolves to the real HI schema."""

    # The outer mapping deliberately has no ``out_features`` key.  The basis
    # resolver must therefore instantiate it; the nested Hydra call returns a
    # mapping with out_features=1 after constructing the open() payload.
    #
    # Four closers are required and all four are load-bearing: the payload
    # mapping, the ``config`` mapping, the resolver argument mapping, and the
    # interpolation itself.  One short and OmegaConf rejects the string while
    # BUILDING the config, before any assertion in the probe runs.
    return (
        "tpen.hi.train.v${"
        + RESOLVER
        + ":{_target_: hydra.utils.instantiate, _recursive_: false, "
        "config: {out_features: 1, payload: "
        + f"{{_target_: builtins.open, file: {marker}, mode: w}}"
        + "}}}"
    )


def _family_entry_carrier(marker: Path) -> str:
    """Build the shipped nested-default carrier used by family entry."""

    return (
        "${oc.select:runtime.family,${"
        + RESOLVER
        + ":{_target_: hydra.utils.instantiate, _recursive_: false, "
        "config: {out_features: -1, payload: "
        + f"{{_target_: builtins.open, file: {marker}, mode: w}}"
        + "}}}}"
    )


def _embedding_resolver_carrier() -> str:
    return (
        f"${{{RESOLVER}:{{_target_: tpen.nn.Embedding, spatial_dim: 3, "
        "max_order: 1, out_channels: 2, hidden_channels: 2, "
        "num_hidden_layers: 0}}"
    )


def _embedding_resolver_trampoline() -> str:
    """Return one through Hydra while constructing an Embedding payload."""

    return (
        "${"
        + RESOLVER
        + ":{_target_: hydra.utils.instantiate, _recursive_: false, "
        "config: {out_features: 1, payload: "
        "{_target_: tpen.nn.Embedding, spatial_dim: 3, max_order: 1, "
        "out_channels: 2, hidden_channels: 2, num_hidden_layers: 0}}"
        + "}}"
    )


def _constructor_carrier(
    kind: str, *, initializer: dict[str, object] | None = None
) -> dict[str, object]:
    if kind == "embedding":
        carrier: dict[str, object] = {
            "_target_": "tpen.nn.Embedding",
            "spatial_dim": 3,
            "max_order": 1,
            "out_channels": 2,
            "hidden_channels": 2,
            "num_hidden_layers": 0,
        }
    elif kind == "path-aggregation":
        carrier = {
            "_target_": "tpen.nn.PathAggregation",
            "max_order": 1,
            "channels": 2,
            "path_counts_by_order": {1: 1},
        }
    else:
        raise AssertionError(f"unknown constructor probe: {kind}")
    if initializer is not None:
        carrier["initializer"] = initializer
    return carrier


@pytest.mark.parametrize("indirection", [False, True], ids=["direct", "plain-reference"])
def test_hi_firewall_review_r1_schema_select_precedes_refusal(
    indirection: bool, tmp_path: Path
) -> None:
    """Refusal must precede every config-named callable in validation."""

    marker = tmp_path / ("schema-indirect" if indirection else "schema-direct")
    carrier = _resolver_carrier(marker)
    if indirection:
        cfg = _config(
            experiment={"name": "tpen_he_importance"},
            runtime={"schema_ref": carrier},
            schema="${runtime.schema_ref}",
        )
    else:
        cfg = _config(
            experiment={"name": "tpen_he_importance"},
            schema=carrier,
        )

    observed: BaseException | None = None
    try:
        _validate(cfg)
    except BaseException as error:  # assert the side effect before exception shape
        observed = error

    assert not marker.exists(), (
        "schema selection executed the rejected resolver carrier before refusal; "
        f"observed={type(observed).__name__ if observed else 'no exception'}"
    )
    assert isinstance(observed, ClosedSchemaError)
    assert _rules(observed) == {"undeclared-schema"}


def test_hi_firewall_review_r1_real_resolver_control_outside_validate_fires_marker(
    tmp_path: Path,
) -> None:
    """The carrier is live: only validation's ordering should contain it."""

    marker = tmp_path / "outside-validation-control"
    cfg = _config(schema=_resolver_carrier(marker))
    selected: object | None = None
    try:
        selected = OmegaConf.select(cfg, "schema")
    except Exception:
        # The carrier is expected to fail after opening the marker: the
        # control proves that the resolver/constructor path is live even when
        # the carrier cannot produce a valid scalar schema value.
        pass
    if hasattr(selected, "close"):
        selected.close()
    assert marker.exists()


@pytest.mark.parametrize("indirection", [False, True], ids=["direct", "plain-reference"])
def test_hi_firewall_review_r1_real_preflight_carrier_stays_inert_in_validate(
    indirection: bool, tmp_path: Path
) -> None:
    """Validation must not execute a live config-named prefix carrier."""

    assert OmegaConf.has_resolver(RESOLVER)
    control_marker = tmp_path / ("real-control-indirect" if indirection else "real-control-direct")
    control_carrier = _real_preflight_carrier(control_marker)
    if indirection:
        control_cfg = _config(
            runtime={"schema_value": control_carrier}, schema="${runtime.schema_value}"
        )
    else:
        control_cfg = _config(schema=control_carrier)

    selected = OmegaConf.select(control_cfg, "schema")
    assert selected == HI_TRAIN_SCHEMA
    assert control_marker.exists(), "the actual registered resolver did not fire"

    validation_marker = tmp_path / (
        "real-validation-indirect" if indirection else "real-validation-direct"
    )
    validation_carrier = _real_preflight_carrier(validation_marker)
    if indirection:
        validation_cfg = _config(
            runtime={"schema_value": validation_carrier}, schema="${runtime.schema_value}"
        )
    else:
        validation_cfg = _config(schema=validation_carrier)

    # The carrier is literal text plus a RESOLVER CALL. Following the literal
    # part is free, but the resolver cannot be followed without running it, so
    # the schema cannot be established and validation refuses.
    with pytest.raises(ClosedSchemaError) as caught:
        _validate(validation_cfg)

    assert not validation_marker.exists(), (
        "validation executed a config-named callable while identifying its policy"
    )
    assert "undeterminable-identity" in _rules(caught.value)


def test_hi_firewall_review_r1_absent_schema_family_getter_does_not_resolve(
    tmp_path: Path,
) -> None:
    """Validation must not execute config-named callables outside the family."""

    marker = tmp_path / "family-getter"
    cfg = OmegaConf.create(
        {"experiment": {"name": _resolver_carrier(marker)}}
    )

    # The identity is a RESOLVER CALL, so it cannot be established without
    # running the very callable this refuses. That is now a refusal rather
    # than a silent "not my family" -- but the assertion that matters is
    # unchanged and comes first: the payload did not run.
    with pytest.raises(ClosedSchemaError) as caught:
        _validate(cfg)

    assert not marker.exists(), (
        "absent-schema family detection resolved an unadmitted carrier; this is "
        "an observational guard, not evidence that an absent-schema config is HI"
    )
    assert "undeterminable-identity" in _rules(caught.value)


def test_hi_firewall_review_r1_family_entry_eager_default_stays_inert(
    tmp_path: Path,
) -> None:
    """Validation must not execute a config-named eager default."""

    marker = tmp_path / "family-entry-default"
    cfg = OmegaConf.create(
        {
            "runtime": {"family": "tpen_he_importance"},
            "experiment": {"name": _family_entry_carrier(marker)},
        }
    )

    # oc.select IS a resolver call, even though it only reads another node,
    # so the identity cannot be established without execution and refuses.
    with pytest.raises(ClosedSchemaError) as caught:
        _validate(cfg)

    assert not marker.exists(), (
        "validation executed a config-named callable while identifying its policy"
    )
    assert "undeterminable-identity" in _rules(caught.value)


def test_hi_firewall_review_r1_manifest_locator_stays_out_of_process_before_refusal() -> None:
    """Refusal must keep config-named manifest data out of process memory."""

    probe = """
import json
import sys

from omegaconf import OmegaConf

import tpen.config as config_module
from tpen.hi_schema import validate_hi_train_config

resolver = config_module.BASIS_FEATURE_DIM_RESOLVER
carrier = (
    "tpen.hi.train.v${" + resolver + ":{_target_: hydra.utils.instantiate, "
    "_recursive_: false, config: {out_features: 1, payload: "
    "{_target_: hydra.utils.get_object, "
    "path: tpen.hi_manifest.reference_energy}}}}"
)
before = sorted(
    name for name in sys.modules
    if name == "tpen.hi_manifest" or name.startswith("tpen.hi_manifest.")
)
error = None
try:
    validate_hi_train_config(
        OmegaConf.create({
            "schema": carrier,
            "experiment": {"name": "tpen_he_importance"},
        }),
        env={},
    )
except BaseException as caught:
    error = {
        "type": type(caught).__name__,
        "rules": sorted(
            rejection.rule for rejection in getattr(caught, "rejections", ())
        ),
    }
after = sorted(
    name for name in sys.modules
    if name == "tpen.hi_manifest" or name.startswith("tpen.hi_manifest.")
)
print(json.dumps({
    "before": before,
    "after": after,
    "error": error,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    observations = json.loads(completed.stdout.strip())

    assert observations["before"] == [], observations
    assert observations["after"] == [], observations
    assert observations["error"] == {
        "type": "ClosedSchemaError",
        "rules": ["undeclared-schema"],
    }, observations


@pytest.mark.parametrize("position", ["schema-key", "ordinary-node"])
def test_hi_firewall_review_r1_recording_resolver_is_refused_before_witness(
    position: str,
) -> None:
    """Refusal must precede every config-named resolver witness.

    A recording witness observes resolver invocation directly rather than
    through a side effect, so it discriminates "the payload did not run"
    from "the payload ran and left nothing behind".

    The carrier is placed on two different nodes because the property is
    that no config-named callable runs before refusal ANYWHERE, not that
    one node was repaired.  The two nodes refuse by different routes -- the
    schema key cannot resolve to a declaration, so the family check refuses
    it as undeclared; an ordinary node reaches the raw resolver sweep -- and
    the witness must stay uncalled on both.

    The ordinary-node arm is also the ``unadmitted-resolver`` half of the
    per-rule pairing pin: every rule in
    :data:`~tpen.config_schema.RESOLVER_REFUSAL_RULES` must pair its refusal
    with ``resolved-sweep-skipped``.  The other two halves are
    ``uncheckable-resolver`` below and ``forbidden-resolver`` in
    ``test_hi_schema.py``.
    """

    calls: list[Any] = []

    def witness(argument: Any) -> str:
        calls.append(argument)
        return HI_TRAIN_SCHEMA

    assert RESOLVER not in HI_TRAIN_POLICY.allowed_resolvers
    original = config_module.basis_feature_dim
    carrier = f"${{{RESOLVER}:x}}"
    if position == "schema-key":
        cfg = _config(experiment={"name": "tpen_he_importance"}, schema=carrier)
        expected = {"undeclared-schema"}
    else:
        cfg = _config(runtime={"probe": carrier})
        expected = {"unadmitted-resolver", "resolved-sweep-skipped"}

    OmegaConf.register_new_resolver(RESOLVER, witness, replace=True)
    try:
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
    finally:
        OmegaConf.register_new_resolver(RESOLVER, original, replace=True)
    assert calls == [], f"the witness ran before refusal at {position}"
    assert expected <= _rules(caught.value)
    assert RESOLVER not in HI_TRAIN_POLICY.allowed_resolvers


def test_hi_firewall_review_r1_uncheckable_resolver_pairs_skip_recording() -> None:
    """A computed resolver name is uncheckable without invoking its witness."""

    calls: list[Any] = []

    def witness(argument: Any) -> int:
        calls.append(argument)
        return 1

    assert OmegaConf.has_resolver(RESOLVER)
    original = config_module.basis_feature_dim
    OmegaConf.register_new_resolver(RESOLVER, witness, replace=True)
    try:
        cfg = _config(
            runtime={
                "leaf": "basis_feature_dim",
                "probe": "${tpen.${runtime.leaf}:3}",
            }
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
    finally:
        OmegaConf.register_new_resolver(RESOLVER, original, replace=True)

    assert calls == []
    assert "uncheckable-resolver" in _rules(caught.value)
    assert "resolved-sweep-skipped" in _rules(caught.value)


def test_hi_firewall_review_r1_guard_refusal_restores_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A resolver-driven unadmitted target refuses and restores Hydra's hook."""

    from hydra._internal.instantiate import _instantiate2

    monkeypatch.setattr(
        hi_schema_module,
        "HI_TRAIN_POLICY",
        replace(HI_TRAIN_POLICY, allowed_resolvers=frozenset({RESOLVER})),
    )
    original_resolve_target = _instantiate2._resolve_target
    marker = tmp_path / "guard-refusal"
    cfg = _config(runtime={"probe": _resolver_carrier(marker)})
    with pytest.raises(ClosedSchemaError) as caught:
        _validate(cfg)

    assert not marker.exists()
    assert "unadmitted-free-form-target" in _rules(caught.value)
    assert _instantiate2._resolve_target is original_resolve_target


def test_known_residual_admitted_trampoline_effect_survives_raw_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin undesired RNG consumption under a future resolver widening.

    FOLLOW-UP ITEM: b4992cc8-dfa1-4a43-8944-a1a9e448f123.
    This does not weaken the invariant for shipped policy: no config-named
    callable may execute before refusal on any validation path.
    """

    admitted_policy = replace(HI_TRAIN_POLICY, allowed_resolvers=frozenset({RESOLVER}))
    monkeypatch.setattr(hi_schema_module, "HI_TRAIN_POLICY", admitted_policy)
    cfg = _config(
        runtime={
            "probe": _embedding_resolver_trampoline(),
            "raw_reference": {"_target_": "tpen.hi_manifest.reference_energy"},
        }
    )
    state_before = torch.get_rng_state().clone()
    state_after: torch.Tensor | None = None
    try:
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        state_after = torch.get_rng_state().clone()
    finally:
        torch.set_rng_state(state_before)

    assert "forbidden-target:reference-module" in _rules(caught.value)
    assert state_after is not None
    assert not torch.equal(state_after, state_before)


@pytest.mark.parametrize("kind", ["embedding", "path-aggregation"])
def test_hi_firewall_review_r1_admitted_constructor_consumes_rng_control(kind: str) -> None:
    """The positive controls prove both default constructor paths are RNG-live."""

    cfg = OmegaConf.create(_constructor_carrier(kind))
    state_before = torch.get_rng_state().clone()
    try:
        hydra.utils.instantiate(cfg)
        state_after = torch.get_rng_state().clone()
    finally:
        torch.set_rng_state(state_before)
    assert not torch.equal(state_after, state_before)


@pytest.mark.parametrize("kind", ["embedding", "path-aggregation"])
def test_hi_firewall_review_r1_seeded_initializer_is_rng_neutral_control(kind: str) -> None:
    """A nested admitted TorchInitializer supplies both green RNG controls."""

    cfg = OmegaConf.create(
        _constructor_carrier(
            kind,
            initializer={
                "_target_": "tpen.nn.initialization.TorchInitializer",
                "seed": 17,
            }
        )
    )
    state_before = torch.get_rng_state().clone()
    try:
        hydra.utils.instantiate(cfg)
        state_after = torch.get_rng_state().clone()
    finally:
        torch.set_rng_state(state_before)
    assert torch.equal(state_after, state_before)


def test_hi_firewall_review_r1_empty_allowlist_blocks_embedding_before_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The empty allowlist control must refuse before the resolver runs."""

    monkeypatch.setattr(hi_schema_module, "HI_TRAIN_POLICY", replace(HI_TRAIN_POLICY))
    assert hi_schema_module.HI_TRAIN_POLICY.allowed_resolvers == frozenset()
    cfg = _config(
        runtime={
            "probe": (
                _embedding_resolver_carrier()
            )
        }
    )
    state_before = torch.get_rng_state().clone()
    state_after: torch.Tensor | None = None
    try:
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        state_after = torch.get_rng_state().clone()
    finally:
        torch.set_rng_state(state_before)
    assert "unadmitted-resolver" in _rules(caught.value)
    assert state_after is not None
    assert torch.equal(state_after, state_before)


def test_hi_firewall_review_r1_sequential_green_control_restores_guard() -> None:
    """A normal validation leaves Hydra's process-global target resolver intact."""

    from hydra._internal.instantiate import _instantiate2

    original = _instantiate2._resolve_target
    _validate(_config(runtime={"probe": "plain"}))
    assert _instantiate2._resolve_target is original


def test_hi_firewall_review_r1_exception_control_restores_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception during resolution must also restore the original target hook."""

    from hydra._internal.instantiate import _instantiate2

    monkeypatch.setattr(
        hi_schema_module,
        "HI_TRAIN_POLICY",
        replace(HI_TRAIN_POLICY, allowed_resolvers=frozenset({RESOLVER})),
    )
    original = _instantiate2._resolve_target
    cfg = _config(
        runtime={
            "probe": (
                f"${{{RESOLVER}:{{_target_: tpen.nn.Embedding}}}}"
            )
        }
    )
    with pytest.raises(ClosedSchemaError) as caught:
        _validate(cfg)
    assert "unresolvable" in _rules(caught.value)
    assert _instantiate2._resolve_target is original


def test_known_residual_swallowing_admitted_resolver_erases_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pin the known residual; this is not desired behaviour.

    FOLLOW-UP ITEM: b4992cc8-dfa1-4a43-8944-a1a9e448f123.
    The trigger is a future widening that admits a resolver which catches
    ``Exception`` around its own work. It is not constructible today because
    ``allowed_resolvers`` is empty and ``tpen.basis_feature_dim`` propagates
    the guard refusal. This red pin names the known residual so the eventual
    hardening has an explicit test to close; it does not bless the behaviour.
    """

    def swallowing_resolver(argument: Any) -> int:
        try:
            return int(hydra.utils.instantiate(argument).out_features)
        except Exception:
            return 1

    admitted_policy = replace(HI_TRAIN_POLICY, allowed_resolvers=frozenset({RESOLVER}))
    monkeypatch.setattr(hi_schema_module, "HI_TRAIN_POLICY", admitted_policy)
    original = config_module.basis_feature_dim
    marker = tmp_path / "swallowed-resolver"
    OmegaConf.register_new_resolver(RESOLVER, swallowing_resolver, replace=True)
    try:
        cfg = _config(
            runtime={
                "probe": (
                    f"${{{RESOLVER}:{{_target_: builtins.open, file: {marker}, mode: w}}}}"
                )
            }
        )
        _validate(cfg)
        assert not marker.exists()
        OmegaConf.to_container(cfg, resolve=True)
        assert marker.exists()
    finally:
        OmegaConf.register_new_resolver(RESOLVER, original, replace=True)


def _position_sweep_template() -> dict[str, Any]:
    """Return a fresh representative HI configuration for the position sweep."""

    return {
        "schema": HI_TRAIN_SCHEMA,
        "experiment": {"name": "tpen_he_importance"},
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.005},
        "runtime": {
            "family": "tpen_he_importance",
            "note": "plain",
            "nested": {"deep": "plain"},
        },
        "run": {"run_id": "fixed-id"},
    }


def _scalar_leaf_paths(node: Any, prefix: str = "") -> list[str]:
    """Enumerate every scalar leaf of a plain container as a dotted path.

    A sequence raises rather than being skipped.  A census that silently
    passes over a shape it does not understand reports coverage it does not
    have, and the position it skipped is exactly where an unguarded read
    would hide.
    """

    if isinstance(node, dict):
        paths: list[str] = []
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_scalar_leaf_paths(value, child))
        return paths
    assert not isinstance(node, (list, tuple)), (
        f"the position sweep cannot address a sequence at {prefix!r}; extend the "
        "walker rather than letting the position go uncovered"
    )
    return [prefix]


def _with_carrier_at(path: str, carrier: str) -> DictConfig:
    """Build a fresh template whose node at ``path`` holds ``carrier``."""

    template = _position_sweep_template()
    *branches, leaf = path.split(".")
    cursor: Any = template
    for branch in branches:
        cursor = cursor[branch]
    cursor[leaf] = carrier
    return OmegaConf.create(template)


def test_hi_firewall_review_r1_every_config_position_stays_inert_before_refusal(
    tmp_path: Path,
) -> None:
    """No config-named callable runs before refusal, whatever node carries it.

    This states the guarantee as a property of the configuration rather than
    as a claim about particular functions.  The carrier positions are
    GENERATED by walking the template, not enumerated by hand, so a read
    added over any node in that template is covered without anyone
    remembering to extend a list -- which is how preflight came to be an
    unguarded location in the first place.

    Coverage this does and does not have.  It observes the payload itself,
    so it catches a resolving read reached by ANY route, including plain
    attribute access on a ``DictConfig``.

    Its bound is the template's EXACT PATH SET, not a class of node shapes.
    Saying "shape" would overstate it and make the residual harder to find
    than it is: ``runtime.device`` is the same scalar-string shape as
    ``runtime.note`` and is NOT covered, because it is not a path in the
    template.  A read of any config path absent from the pinned set below is
    outside this test, and widening the template is what widens the bound.

    The companion source census bounds the OmegaConf call surface instead,
    and the two gaps do not overlap.
    """

    positions = _scalar_leaf_paths(_position_sweep_template())
    assert set(positions) == {
        "schema",
        "experiment.name",
        "optimizer._target_",
        "optimizer.lr",
        "runtime.family",
        "runtime.note",
        "runtime.nested.deep",
        "run.run_id",
    }, positions

    rules_seen: set[str] = set()
    for path in positions:
        marker = tmp_path / f"position-{path.replace('.', '-')}"
        cfg = _with_carrier_at(path, _resolver_carrier(marker))
        try:
            validate_hi_train_config(cfg, env={})
        except ClosedSchemaError as refusal:
            rules_seen |= _rules(refusal)
        assert not marker.exists(), (
            f"a config-named callable executed before refusal with the carrier at "
            f"{path!r}; the guarantee is that nothing config-named runs before "
            "refusal on any path, not that a named function was repaired"
        )

    # Non-vacuity: the sweep must actually reach the resolver policy at least
    # once.  Without this, a sweep that aborted before the policy ever saw the
    # carrier would report the same clean markers.
    assert "unadmitted-resolver" in rules_seen, rules_seen

    # Discrimination: prove the carriers this sweep builds are live, so the
    # absent markers above mean containment rather than an inert payload.
    liveness_marker = tmp_path / "position-sweep-liveness"
    liveness_cfg = OmegaConf.create({"probe": _resolver_carrier(liveness_marker)})
    try:
        selected = OmegaConf.select(liveness_cfg, "probe")
    except Exception:
        selected = None
    if hasattr(selected, "close"):
        selected.close()
    assert liveness_marker.exists(), (
        "the sweep's carriers never fire even outside validation, so the sweep "
        "proves nothing about containment"
    )


def test_hi_firewall_review_r1_resolving_reads_stay_pinned() -> None:
    """Adding any OmegaConf read to the firewall module must fail this pin.

    Clause 2 of the contract: the guarantee is a property, and a property
    needs a mechanism rather than a docstring asking future readers to
    preserve it.  This pins the OmegaConf call surface of the module -- not
    a blocklist of the calls known to resolve -- so a new read of any kind
    goes red and has to be justified against the ordering rather than
    reviewed by whoever happens to notice.

    WHAT IT ACTUALLY ENFORCES, stated as the code enforces it rather than as
    the intent:

    - It walks the WHOLE module, including class bodies and module level,
      and labels each call by its nearest enclosing definition.  An earlier
      version walked only top-level functions, which made a call inside
      either of this module's two classes, or at module level, invisible.
    - It matches a call whose owner is the bare name ``OmegaConf``.  An
      ALIASED import would be invisible to that match, so a companion
      assertion pins the import form itself; the two together, not the call
      match alone, are what close the aliasing route.
    - It does NOT see resolution reached through a ``DictConfig`` value
      handed to a third module, nor any resolution that never names
      OmegaConf.  The behavioural position sweep covers those by observing
      the payload instead.
    """

    import ast
    import inspect

    from tpen import config_schema as config_schema_module

    source_path = Path(inspect.getsourcefile(hi_schema_module) or "")
    repo_root = Path(__file__).resolve().parents[2]
    assert repo_root in source_path.resolve().parents, (
        f"measuring {source_path} which is outside the checkout at {repo_root}"
    )

    module = ast.parse(source_path.read_text())
    observed: set[tuple[str, str, str]] = set()

    def visit(node: ast.AST, scope: str) -> None:
        """Walk every node, carrying the nearest enclosing definition name."""

        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "OmegaConf"
            ):
                resolve = "no-resolve-keyword"
                for keyword in node.keywords:
                    if keyword.arg == "resolve":
                        value = keyword.value
                        resolve = (
                            repr(value.value)
                            if isinstance(value, ast.Constant)
                            else "non-literal"
                        )
                observed.add((scope, func.attr, resolve))
        for child in ast.iter_child_nodes(node):
            child_scope = (
                child.name
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                else scope
            )
            visit(child, child_scope)

    visit(module, "<module>")

    # The call match keys on the bare name ``OmegaConf``, so an aliased
    # import would walk straight past it. Pin the import form rather than
    # leaving that hole described in prose.
    omegaconf_imports = {
        (node.module, alias.name, alias.asname)
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module == "omegaconf"
        for alias in node.names
    }
    assert omegaconf_imports == {
        ("omegaconf", "DictConfig", None),
        ("omegaconf", "ListConfig", None),
        ("omegaconf", "OmegaConf", None),
    }, (
        "the omegaconf import form changed; an alias would make every call "
        f"below invisible to this census. Imports: {sorted(omegaconf_imports)}"
    )
    assert not [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        and any(alias.name.split(".")[0] == "omegaconf" for alias in node.names)
    ], "omegaconf gained a plain import, which can bind any name"

    # The generated grammar contexts live deeper than the top-level package.
    # Pinned as its own entry so the internal dependency is visible in one
    # place and cannot grow without this test naming it.
    deep = {
        (node.module, alias.name, alias.asname)
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("omegaconf.")
        for alias in node.names
    }
    assert deep == {
        # INTERNAL API, pinned deliberately rather than smuggled in: the
        # resolver registry is a class attribute on BaseContainer, and
        # emptying it for the duration of a resolution is what makes the
        # follower unable to execute anything. The generated grammar parser
        # this pin used to name is GONE, retired with the hand-rolled walk.
        ("omegaconf.basecontainer", "BaseContainer", None)
    }, f"the internal OmegaConf dependency changed: {sorted(deep)}"

    # Each entry must stay justified by the ORDERING, not by its location:
    # a raw read may run at any time, and a resolving read may run only after
    # every refusal that could make resolution unsafe has been raised.
    assert observed == {
        # Raw: the two preflight reads share this one helper.
        ("_raw_config_mapping", "to_container", "False"),
        # Raw: the sweep input, collected before the resolver refusal.
        ("validate_hi_train_config", "to_container", "False"),
        # Resolving, and permitted: guarded, and after the refusal that makes
        # resolution unsafe has already been raised.
        ("validate_hi_train_config", "to_container", "True"),
        # Resolving, and permitted: a post-validation identity over a config
        # the caller has already had validated. Not on any preflight path.
        ("canonical_train_identity", "to_container", "True"),
        # RESOLVING, AND PERMITTED FOR A DIFFERENT REASON THAN THE OTHERS: the
        # identity follower resolves with the resolver registry swapped to an
        # EMPTY one, so a node reference resolves while a resolver call finds
        # nothing registered and raises instead of running. It is safe because
        # of WHAT CANNOT RUN during it, not because of when it happens.
        ("identity_without_execution", "select", "no-resolve-keyword"),
        # Normalises a plain mapping into a config so the same delegation
        # serves both caller shapes. Builds, resolves nothing.
        ("identity_without_execution", "create", "no-resolve-keyword"),
    }, observed

    # The raw sweep cannot resolve because it never holds an OmegaConf object.
    # Pinning the absence of the import keeps that structural rather than
    # incidental.  The module names OmegaConf in prose, so this reads the
    # import statements rather than the text.
    config_schema_tree = ast.parse(
        Path(inspect.getsourcefile(config_schema_module) or "").read_text()
    )
    imported: set[str] = set()
    for node in ast.walk(config_schema_tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "omegaconf" not in imported, (
        "the raw-sweep module gained an OmegaConf dependency; a raw sweep that "
        f"can resolve is not a raw sweep. Imports: {sorted(imported)}"
    )


def test_hi_firewall_review_r1_no_shipped_config_interpolates_the_schema_key() -> None:
    """No tracked configuration may put an interpolation in a schema key.

    Reading the schema key raw changes behaviour only for a config whose
    schema key interpolates: such a key now declares nothing.  The contract
    states the regression risk is nil because no shipped config does this.
    That is a measurement, and a measurement of the tree is perishable, so
    it is pinned here rather than left to the verification job that took it.

    The corpus is the tracked file set, which is what "shipped" means; a
    config that exists only as untracked run data is out of scope and is
    reported as such by the counts in the failure message.
    """

    import re

    repo_root = Path(__file__).resolve().parents[2]
    listing = subprocess.run(
        ["git", "ls-files", "-z", "*.yaml", "*.yml"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [name for name in listing.stdout.split("\0") if name]
    assert tracked, "the tracked YAML corpus is empty; this pin would be vacuous"

    declaration = re.compile(r"^\s*schema\s*:\s*(?P<value>.*)$")
    declarations = 0
    interpolating: list[str] = []
    for name in tracked:
        path = repo_root / name
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            matched = declaration.match(line)
            if matched is None:
                continue
            declarations += 1
            if "${" in matched.group("value"):
                interpolating.append(f"{name}:{number}: {matched.group('value')}")

    assert declarations, (
        f"no schema declaration found across {len(tracked)} tracked YAML files; "
        "the pin is not measuring what it claims to measure"
    )
    assert not interpolating, (
        f"{len(interpolating)} of {declarations} schema declarations across "
        f"{len(tracked)} tracked YAML files interpolate, so reading the schema "
        f"key raw is a behaviour change for them: {interpolating}"
    )


def test_hi_firewall_review_r1_every_carrier_builder_is_well_formed(tmp_path: Path) -> None:
    """Every hand-built carrier must survive config construction.

    These carriers are assembled by string concatenation, so a miscounted
    closing brace is easy to write and hard to see.  OmegaConf rejects such a
    string while BUILDING the config, which means the probe that uses it dies
    before reaching any assertion -- and a GrammarParseError deep in the
    OmegaConf stack reads like a defect in the code under test rather than a
    typo in the probe's own fixture.  One of these was in fact one closer
    short and cost a full verification cycle.

    This asserts the fixture is well formed, not that anything is contained;
    the containment probes each carry their own liveness control.
    """

    builders = {
        "_resolver_carrier": _resolver_carrier(tmp_path / "a"),
        "_real_preflight_carrier": _real_preflight_carrier(tmp_path / "b"),
        "_family_entry_carrier": _family_entry_carrier(tmp_path / "c"),
        "_embedding_resolver_carrier": _embedding_resolver_carrier(),
        "_embedding_resolver_trampoline": _embedding_resolver_trampoline(),
    }
    assert len(builders) == 5, builders

    unparsable: dict[str, str] = {}
    for name, carrier in builders.items():
        try:
            # Construction alone must succeed. Nothing here resolves, so this
            # never runs a payload.
            OmegaConf.create({"probe": carrier})
        except Exception as error:  # noqa: BLE001 - OmegaConf raises several types
            unparsable[name] = f"{type(error).__name__}: {error}"
    assert not unparsable, unparsable

    # Discrimination: a genuinely malformed carrier must be rejected, so the
    # check above is not merely asserting that OmegaConf accepts everything.
    with pytest.raises(Exception):
        OmegaConf.create({"probe": _resolver_carrier(tmp_path / "d")[:-1]})


# IDENTITY DETERMINED WITHOUT EXECUTION. A node reference names another node
# in the same tree, so following it is a dictionary lookup in the raw tree and
# nothing runs. A resolver-call interpolation cannot be followed without
# invoking a configured callable, so reaching one at ANY depth refuses.
#
# The carrier below is closed by being CORRECTLY IDENTIFIED and then properly
# validated -- not by being refused for concealment. A mechanism that refuses
# concealment punishes a shape; one that determines identity without executing
# gets the right answer.
REFERENCE_ENERGY = -2.903724


def _identity_config(
    *, schema: str | None, name: str, reference: bool, extra: dict[str, Any] | None = None
) -> DictConfig:
    """Build a config whose identity nodes may be node references."""

    body: dict[str, Any] = {
        "runtime": {"real": "tpen_he_importance", "sch": HI_TRAIN_SCHEMA},
        "experiment": {"name": name},
    }
    if schema is not None:
        body["schema"] = schema
    if reference:
        body["reference_energy"] = REFERENCE_ENERGY
    if extra:
        body.update(extra)
    return OmegaConf.create(body)


def test_hi_firewall_review_r1_node_reference_identity_is_followed_and_validated() -> None:
    """The both-interpolated carrier is identified, then refused on its payload.

    This is the escape the raw read opened: at ``cbc4c06`` both identity nodes
    read as non-literal, the config was taken for a foreign family, and a
    reference energy reached a training configuration that validated with ZERO
    rejections.  Following the references identifies it as helium-importance,
    the full policy runs, and the reference is refused on the ordinary rule.
    """

    cfg = _identity_config(schema="${runtime.sch}", name="${runtime.real}", reference=True)

    with pytest.raises(ClosedSchemaError) as caught:
        _validate(cfg)

    rules = _rules(caught.value)
    # Refused by the ORDINARY rule, because it was correctly identified.
    assert "forbidden-surface:reference" in rules, rules
    assert "undeterminable-identity" not in rules, rules


def test_hi_firewall_review_r1_node_reference_name_without_schema_is_undeclared() -> None:
    """The second escape: a followable name and no schema key.

    Refused ``undeclared-schema``, the rule that already existed for a config
    that is in the family and forgot to opt in.
    """

    cfg = _identity_config(schema=None, name="${runtime.real}", reference=True)

    with pytest.raises(ClosedSchemaError) as caught:
        _validate(cfg)

    assert _rules(caught.value) == {"undeclared-schema"}


def test_hi_firewall_review_r1_transitive_node_reference_is_followed() -> None:
    """Following is transitive: a reference to a reference resolves."""

    cfg = OmegaConf.create(
        {
            "a": "${b}",
            "b": "${c}",
            "c": "tpen_he_importance",
            "experiment": {"name": "${a}"},
            "reference_energy": REFERENCE_ENERGY,
        }
    )

    with pytest.raises(ClosedSchemaError) as caught:
        _validate(cfg)

    assert _rules(caught.value) == {"undeclared-schema"}


@pytest.mark.parametrize(
    ("body", "shape"),
    [
        ({"experiment": {"name": "${runtime.witness:1}"}}, "resolver-call-direct"),
        (
            {"hop": "${runtime.witness:1}", "experiment": {"name": "${hop}"}},
            "resolver-call-at-depth-2",
        ),
        (
            {"a": "${b}", "b": "${a}", "experiment": {"name": "${a}"}},
            "cycle",
        ),
        ({"experiment": {"name": "${nowhere.at.all}"}}, "dangling-target"),
        (
            {"container": {"inner": 1}, "experiment": {"name": "${container}"}},
            "container-target",
        ),
        ({"experiment": {"name": "${experiment.name}"}}, "self-cycle"),
        (
            {
                "choices": {"x": {"basis": "y"}},
                "experiment": {"name": "${choices.${runtime.witness:1}.basis}"},
            },
            "resolver-call-inside-a-path",
        ),
    ],
    ids=[
        "resolver-call-direct",
        "resolver-call-at-depth-2",
        "cycle",
        "dangling-target",
        "container-target",
        "self-cycle",
        "resolver-call-inside-a-path",
    ],
)
def test_hi_firewall_review_r1_undeterminable_identity_is_refused(
    body: dict[str, Any], shape: str
) -> None:
    """Every identity that cannot be established without execution refuses.

    Reaching a resolver call at ANY depth is the security-critical arm: the
    alternative to refusing is running a configured callable to find out what
    the config is, which is the defect this module exists to close.

    The rest are fail-closed branches with no security story of their own,
    and they are here because a guard that silently returns "not my family"
    for an input it did not understand is the same hole wearing a different
    hat.  The cycle arm also pins TERMINATION: a validator that hangs is not
    a validator that refuses.
    """

    cfg = OmegaConf.create(dict(body))

    with pytest.raises(ClosedSchemaError) as caught:
        _validate(cfg)

    assert "undeterminable-identity" in _rules(caught.value), (
        f"{shape} did not refuse: {sorted(_rules(caught.value))}"
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            {
                "slot": "hooke-axiswise-v1",
                "choices": {"hooke-axiswise-v1": {"basis": "tpen_he_importance"}},
                "experiment": {"name": "${choices.${slot}.basis}"},
            },
            "tpen_he_importance",
        ),
        (
            {
                "choices": {"hooke-axiswise-v1": {"basis": "tpen_he_importance"}},
                "experiment": {"name": "${choices.hooke-axiswise-v1.basis}"},
            },
            "tpen_he_importance",
        ),
        (
            {
                "runtime": {"real": "tpen_he_importance"},
                "experiment": {"name": "prefix_${runtime.real}"},
            },
            "prefix_tpen_he_importance",
        ),
    ],
    ids=["nested-path", "hyphenated-key", "mixed-literal-and-interpolation"],
)
def test_hi_firewall_review_r1_following_recurses_into_the_path_itself(
    body: dict[str, Any], expected: str
) -> None:
    """A reference's own PATH may interpolate, and must be followed too.

    ``choices.basis.${slot}.basis`` is a shape in active use in this
    repository -- 34 of its 1053 tracked interpolations -- so a follower that
    recursed only into VALUES would refuse configurations it could have read
    without executing anything.

    The hyphenated arm pins the key charset: OmegaConf keys carry hyphens, so
    lookup splits on ``.`` and matches whole keys with no pattern applied to
    the key text.  A key regex shaped like a Python identifier misclassified
    nine real node references in this repository when it was tried.
    """

    from tpen.hi_schema import identity_without_execution

    identity = identity_without_execution(OmegaConf.create(body), "experiment.name")

    assert identity.determined, identity.reason
    assert identity.value == expected


def test_hi_firewall_review_r1_a_deep_chain_is_followed_not_truncated() -> None:
    """A long reference chain resolves; the bound is OmegaConf's, not ours.

    This replaces an arm that asserted refusal AT a follow limit this module
    used to own. Delegating removed that limit along with the hand-rolled
    walk, so the honest pin is the opposite one: a chain far longer than the
    old bound is FOLLOWED, and whatever recursion limit exists is OmegaConf's
    to define and to enforce.
    """

    from tpen.hi_schema import identity_without_execution

    body: dict[str, Any] = {"experiment": {"name": "${n0}"}}
    for index in range(40):
        body[f"n{index}"] = "${n%d}" % (index + 1)
    body["n40"] = "tpen_he_importance"

    identity = identity_without_execution(OmegaConf.create(body), "experiment.name")

    assert identity.determined, identity.reason
    assert identity.value == "tpen_he_importance"


@pytest.mark.parametrize(
    ("body", "axis"),
    [
        ({"n": "spike", "experiment": {"name": "${${n}.w:1}"}}, "interpolated-resolver-name"),
        ({"s": "x", "experiment": {"name": "${oc.select:s}"}}, "builtin-oc-select"),
        ({"experiment": {"name": "${oc.env:HOME,x}"}}, "builtin-oc-env"),
        ({"experiment": {"name": "${tpen.basis_feature_dim:'a:b'}"}}, "quoted-arg-with-colon"),
        ({"experiment": {"name": "${tpen.basis_feature_dim:[1,2]}"}}, "list-in-resolver-arg"),
        ({"experiment": {"name": "${tpen.basis_feature_dim:{a:1}}"}}, "dict-in-resolver-arg"),
        ({"s": "x", "experiment": {"name": "${tpen.basis_feature_dim:${s}}"}}, "interpolation-in-arg"),
    ],
    ids=[
        "interpolated-resolver-name",
        "builtin-oc-select",
        "builtin-oc-env",
        "quoted-arg-with-colon",
        "list-in-resolver-arg",
        "dict-in-resolver-arg",
        "interpolation-in-arg",
    ],
)
def test_hi_firewall_review_r1_axis_refuses(body: dict[str, Any], axis: str) -> None:
    """Axes the follower must refuse because following would execute.

    ``quoted-arg-with-colon`` is the arm that justifies classifying by the
    GRAMMAR: a colon inside a quoted value is not a resolver separator, and a
    string test for ``:`` would misread it.  ``builtin-oc-select`` is the
    sharp one -- it takes a PATH and reaches further than a plain node
    reference, and it is still a resolver call, so it is refused rather than
    followed.
    """

    with pytest.raises(ClosedSchemaError) as caught:
        _validate(OmegaConf.create(dict(body)))
    assert "undeterminable-identity" in _rules(caught.value), (
        f"{axis}: {sorted(_rules(caught.value))}"
    )


@pytest.mark.parametrize(
    ("body", "expected", "axis"),
    [
        ({"s": 7, "experiment": {"name": "${s}"}}, 7, "int-target"),
        ({"s": None, "experiment": {"name": "${s}"}}, None, "null-target"),
        ({"items": ["x"], "experiment": {"name": "${items.0}"}}, "x", "list-index-dot"),
        ({"items": ["x"], "experiment": {"name": "${items[0]}"}}, "x", "list-index-bracket"),
        ({"a": "b", "b": "c", "c": {"d": "x"}, "experiment": {"name": "${${${a}}.d}"}}, "x", "three-level-path"),
    ],
    ids=["int-target", "null-target", "list-index-dot", "list-index-bracket", "three-level-path"],
)
def test_hi_firewall_review_r1_axis_follows(
    body: dict[str, Any], expected: object, axis: str
) -> None:
    """Axes the follower must resolve, because nothing needs to execute."""

    from tpen.hi_schema import identity_without_execution

    identity = identity_without_execution(OmegaConf.create(dict(body)), "experiment.name")
    assert identity.determined, f"{axis}: {identity.reason}"
    assert identity.value == expected, axis


def test_hi_firewall_review_r1_escaped_interpolation_is_not_followed() -> None:
    """An escaped opener is literal text and must never be followed.

    Escape-awareness is by backslash PARITY, not by presence: an odd number
    of backslashes escapes the opener, an even number is an escaped backslash
    followed by a REAL interpolation.
    """

    from tpen.hi_schema import identity_without_execution

    escaped = identity_without_execution(
        OmegaConf.create({"s": "tpen_he_importance", "experiment": {"name": r"\${s}"}}),
        "experiment.name",
    )
    assert escaped.determined
    assert escaped.value != "tpen_he_importance", "an escaped opener was followed"
    # Delegation CLOSED a deviation this module used to carry: the escape is
    # now unescaped exactly as OmegaConf unescapes it, so the follower's answer
    # is the grammar's answer rather than merely a safe one.
    assert escaped.value == OmegaConf.select(
        OmegaConf.create({"s": "tpen_he_importance", "experiment": {"name": r"\${s}"}}),
        "experiment.name",
    )

    # Even parity is a REAL interpolation and must be followed.
    real = identity_without_execution(
        OmegaConf.create({"s": "tpen_he_importance", "experiment": {"name": "\\\\${s}"}}),
        "experiment.name",
    )
    assert real.determined
    assert "tpen_he_importance" in str(real.value)


def test_hi_firewall_review_r1_omegaconf_does_not_resolve_keys() -> None:
    """PIN THE ASSUMPTION that closes the key-position axis by construction.

    An interpolation in a KEY cannot hide a section, because OmegaConf leaves
    it as a literal key even after FULL resolution.  That is a property of
    OmegaConf, not of this module, so it is pinned here: a future OmegaConf
    that DID resolve keys would turn this red rather than silently opening a
    hole in the identity lookup.
    """

    cfg = OmegaConf.create({"k": "experiment", "${k}": {"name": "tpen_he_importance"}})

    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert "${k}" in resolved, resolved
    assert "experiment" not in resolved, (
        "OmegaConf now resolves keys; the identity lookup reads literal keys only "
        "and an interpolated key could therefore hide an experiment section"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN AXIS, DISPOSED TO FOLLOW-UP ITEM "
        "1efc8552-800e-4248-9b8b-984549f0e3dc. A MISSING sentinel resolves to the "
        "literal '???', which is not the family name, so a config whose identity is "
        "??? receives no enforcement at all. PRE-EXISTING, not introduced by this "
        "layer: the reader this replaced returned None for a ??? identity node and "
        "raised InterpolationToMissingValueError for a ??? target, reaching the same "
        "not-this-family conclusion by two different routes. strict is deliberate -- "
        "closing it makes this XPASS and FAIL, so whoever turns it green is told "
        "which item they just closed and the marker cannot outlive the defect"
    ),
)
def test_hi_firewall_review_r1_missing_sentinel_identity_is_refused() -> None:
    """The MISSING sentinel should not buy a config zero enforcement.

    FOLLOW-UP ITEM: 1efc8552-800e-4248-9b8b-984549f0e3dc.

    This needs NO adversarial construction. ``???`` is the STANDARD
    PLACEHOLDER for a value a template requires its caller to supply, so an
    omitted override reaches this state by itself -- which is what makes it
    worth an item rather than a footnote.
    """

    cfg = OmegaConf.create(
        {"experiment": {"name": "???"}, "reference_energy": REFERENCE_ENERGY}
    )

    with pytest.raises(ClosedSchemaError):
        _validate(cfg)


# Deliberately empty. A tracked configuration that the firewall must refuse
# would be named here WITH ITS REASON, so adding one is a visible decision
# rather than a silent relaxation of the sweep below.
CONFIGS_EXPECTED_TO_BE_REFUSED: frozenset[str] = frozenset()


def test_hi_firewall_review_r1_every_tracked_config_survives_the_firewall() -> None:
    """No tracked configuration may be refused by the train-config firewall.

    THIS EXISTS BECAUSE THE SUITE WAS BLIND TO THE COST THAT MATTERED. A
    candidate mechanism that refused every unreadable identity would have
    refused four tracked hooke configs, two of them frozen provenance
    records -- and NO node in this suite would have gone red, because no test
    ran ``validate_hi_train_config`` over the tracked configs at all.  The
    cost was measured only by a one-off probe in a verification job, and a
    one-off probe is perishable in a way a repo test is not.

    ``validate_hi_train_config`` runs unconditionally on the live path in
    ``tpen/run.py``, for EVERY config and not only helium-importance ones, so
    the blast radius of any change to it is exactly this corpus.

    Nothing here resolves: refusal is decided on the raw tree, so loading and
    validating a config runs none of its configured callables.
    """

    repo_root = Path(__file__).resolve().parents[2]
    listing = subprocess.run(
        ["git", "ls-files", "-z", "*.yaml", "*.yml"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [name for name in listing.stdout.split("\0") if name]
    assert len(tracked) > 20, f"the tracked config corpus looks wrong: {len(tracked)}"

    refused: dict[str, list[str]] = {}
    unloadable: dict[str, str] = {}
    validated = 0
    for name in tracked:
        try:
            cfg = OmegaConf.load(repo_root / name)
        except Exception as error:  # noqa: BLE001 - a YAML this cannot load is not in scope
            unloadable[name] = f"{type(error).__name__}: {error}"
            continue
        validated += 1
        try:
            validate_hi_train_config(cfg, env={})
        except ClosedSchemaError as refusal:
            refused[name] = sorted(_rules(refusal))

    unexpected = {
        name: rules
        for name, rules in refused.items()
        if name not in CONFIGS_EXPECTED_TO_BE_REFUSED
    }
    assert not unexpected, (
        f"{len(unexpected)} of {validated} tracked configs are now refused by the "
        f"firewall: {unexpected}. tpen/run.py validates every config on the live "
        "path, so each of these is a run that stops working"
    )
    # Report coverage: a sweep that quietly skipped most of the corpus would
    # pass while proving nothing.
    assert validated >= len(tracked) - 2, (
        f"only {validated} of {len(tracked)} tracked configs could be loaded and "
        f"validated, so this sweep covers less than it appears to: {unloadable}"
    )


def test_hi_firewall_review_r1_blank_identity_is_an_ordinary_value() -> None:
    """A blank or whitespace identity must not reach the grammar at all.

    REGRESSION PIN. The classifier asked OmegaConf's grammar to parse the
    identity value BEFORE establishing that the value held an interpolation.
    The grammar's ``configValue`` rule does not accept the EMPTY STRING, so an
    ordinary blank name raised ``GrammarParseError`` out of validation and
    broke three parametrizations of a test in a file this layer never touched.

    Order is the fix: find the interpolations first, classify only if there
    are any.  These arms are the ones that were red.
    """

    for name in (None, "", "   ", "metadata"):
        cfg = OmegaConf.create(
            {"experiment": {"name": name, "run_name": None}, "run": {"run_id": None}}
        )
        _validate(cfg)


def test_hi_firewall_review_r1_unparsable_interpolation_fails_closed() -> None:
    """A value whose interpolation OmegaConf cannot parse must be refused.

    The fail-closed direction is the whole point: returning "no resolver call"
    for a value that could not be classified would treat an unclassifiable
    interpolation as safe.

    REACHABILITY, checked rather than assumed. ``OmegaConf.create`` REFUSES to
    build a ``DictConfig`` holding an unparsable interpolation, so this branch
    is unreachable by that route.  It is reachable through a plain mapping,
    which :func:`identity_without_execution` accepts, so the branch is live
    rather than vacuous -- and this arm uses that route deliberately.
    """

    from tpen.hi_schema import identity_without_execution

    with pytest.raises(Exception):
        OmegaConf.create({"experiment": {"name": "${unclosed"}})

    identity = identity_without_execution({"experiment": {"name": "${unclosed"}}, "experiment.name")

    assert not identity.determined
    assert "cannot parse" in (identity.reason or "")


def test_hi_firewall_review_r1_resolver_call_identity_runs_nothing() -> None:
    """Classifying a resolver call must not INVOKE it.

    The witness is the whole point: a classifier that decided by trying to
    resolve would give the same verdict and would already have run the
    payload.
    """

    calls: list[Any] = []

    def witness(argument: Any) -> str:
        calls.append(argument)
        return HI_TRAIN_SCHEMA

    original = config_module.basis_feature_dim
    OmegaConf.register_new_resolver(RESOLVER, witness, replace=True)
    try:
        cfg = OmegaConf.create({"experiment": {"name": f"${{{RESOLVER}:probe}}"}})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
    finally:
        OmegaConf.register_new_resolver(RESOLVER, original, replace=True)

    assert calls == [], "classification invoked the resolver it was classifying"
    assert "undeterminable-identity" in _rules(caught.value)


def test_hi_firewall_review_r1_shipped_interpolated_name_shape_still_passes() -> None:
    """The four tracked configs that interpolate experiment.name must pass.

    This is the shape of ``pair_stability.yaml``, ``pair_validation.yaml`` and
    both ``tpen-pair-scan-v1`` configs: ``experiment.name`` is ``${study.name}``
    and the target is a literal in the same file.  Following it costs nothing
    and identifies them, correctly, as not this family.  Two of the four are
    frozen provenance records whose protected property is that they still
    resolve, so this pin is what keeps that true.
    """

    cfg = OmegaConf.create(
        {"study": {"name": "pair_stability_v3"}, "experiment": {"name": "${study.name}"}}
    )

    _validate(cfg)


def test_hi_firewall_review_r1_readable_identity_paths_are_unchanged() -> None:
    """Literal identities behave exactly as before.

    Three arms that were already correct and must stay that way, so the new
    following is shown to be additive rather than a rewrite.
    """

    with pytest.raises(ClosedSchemaError) as caught:
        _validate(_identity_config(schema=None, name="tpen_he_importance", reference=True))
    assert _rules(caught.value) == {"undeclared-schema"}

    # A readable foreign name is still returned unvalidated: opt-in firewall.
    _validate(OmegaConf.create({"experiment": {"name": "some_other_study"}}))

    # No experiment section at all is ABSENT, which is a determined fact and
    # must not be confused with undeterminable.
    _validate(OmegaConf.create({"runtime": {"probe": "plain"}}))


def test_hi_firewall_review_r1_declared_schema_still_validates_an_interpolated_name() -> None:
    """Opting in literally must not be short-circuited by identity handling."""

    cfg = OmegaConf.create(
        {
            "schema": HI_TRAIN_SCHEMA,
            "runtime": {"real": "tpen_he_importance"},
            "experiment": {"name": "${runtime.real}"},
            "reference_energy": REFERENCE_ENERGY,
        }
    )

    with pytest.raises(ClosedSchemaError) as caught:
        _validate(cfg)

    rules = _rules(caught.value)
    assert "undeterminable-identity" not in rules, rules
    assert "forbidden-surface:reference" in rules, rules
