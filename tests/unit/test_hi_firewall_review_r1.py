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

    _validate(validation_cfg)

    assert not validation_marker.exists(), (
        "validation executed a config-named callable while identifying its policy"
    )


def test_hi_firewall_review_r1_absent_schema_family_getter_does_not_resolve(
    tmp_path: Path,
) -> None:
    """Validation must not execute config-named callables outside the family."""

    marker = tmp_path / "family-getter"
    cfg = OmegaConf.create(
        {"experiment": {"name": _resolver_carrier(marker)}}
    )

    _validate(cfg)

    assert not marker.exists(), (
        "absent-schema family detection resolved an unadmitted carrier; this is "
        "an observational guard, not evidence that an absent-schema config is HI"
    )


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

    _validate(cfg)

    assert not marker.exists(), (
        "validation executed a config-named callable while identifying its policy"
    )


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

    Coverage this does and does not have: it observes the payload itself, so
    it catches a resolving read reached by ANY route, including plain
    attribute access on a ``DictConfig``; it is bounded to the node shapes
    present in the template, so a node shape the template omits is not
    covered.  The companion source census bounds the OmegaConf API surface
    instead, and the two gaps do not overlap.
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
    preserve it.  This pins the entire OmegaConf API surface of the module
    -- not a blocklist of the calls known to resolve -- so a new read of any
    kind, at any location, goes red and has to be justified against the
    ordering rather than reviewed by whoever happens to notice.

    Coverage: this bounds the OmegaConf API surface of the two modules that
    implement the firewall.  It does not see resolution reached through a
    ``DictConfig`` value handed to a third module; the behavioural position
    sweep covers that route by observing the payload.
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
    for definition in module.body:
        if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(definition):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "OmegaConf"
            ):
                continue
            resolve = "no-resolve-keyword"
            for keyword in node.keywords:
                if keyword.arg == "resolve":
                    value = keyword.value
                    resolve = (
                        repr(value.value)
                        if isinstance(value, ast.Constant)
                        else "non-literal"
                    )
            observed.add((definition.name, func.attr, resolve))

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


# A config may hide its own identity behind an interpolation. Reading the
# identity nodes raw means an interpolated value matches no literal, so
# without a refusal the config is simply "not this family" and receives NO
# ENFORCEMENT AT ALL. These carriers each carry a reference energy, which is
# the thing the firewall exists to keep out of a training configuration.
REFERENCE_ENERGY = -2.903724


def _hidden_identity_config(*, schema: str | None, name: str) -> DictConfig:
    """Build a config whose identity nodes may be interpolations."""

    body: dict[str, Any] = {
        "runtime": {"real": "tpen_he_importance", "sch": HI_TRAIN_SCHEMA},
        "experiment": {"name": name},
        "reference_energy": REFERENCE_ENERGY,
    }
    if schema is not None:
        body["schema"] = schema
    return OmegaConf.create(body)


@pytest.mark.parametrize(
    ("schema", "name", "shape"),
    [
        ("${runtime.sch}", "${runtime.real}", "both-interpolated"),
        (None, "${runtime.real}", "name-interpolated-no-schema"),
        ("${runtime.sch}", "tpen_he_importance", "schema-interpolated-literal-name"),
    ],
    ids=["both-interpolated", "name-interpolated-no-schema", "schema-interpolated-literal-name"],
)
def test_hi_firewall_review_r1_interpolated_identity_is_refused(
    schema: str | None, name: str, shape: str
) -> None:
    """A configuration that hides its identity must be refused, not ignored.

    Reading the identity nodes raw is required -- resolving them is what let
    a config-named callable run before refusal.  But a raw read cannot tell
    an interpolated name from a foreign one, so falling through to "not my
    family" hands a config with a reference energy ZERO ENFORCEMENT and
    ``tpen.run`` then resolves it and trains on it.  That is worse than a
    payload running before a refusal, because no refusal happens at all.

    Fail closed on the face of the raw value.  Nothing here resolves.
    """

    cfg = _hidden_identity_config(schema=schema, name=name)

    with pytest.raises(ClosedSchemaError) as caught:
        _validate(cfg)

    assert "interpolated-identity" in _rules(caught.value), (
        f"shape {shape} was not refused on its hidden identity; "
        f"rules={sorted(_rules(caught.value))}"
    )


def test_hi_firewall_review_r1_literal_identity_paths_are_unchanged(tmp_path: Path) -> None:
    """The identity refusal must not fire on a readable identity.

    Three arms that were already correct and must stay that way, so the new
    refusal is shown to be narrow rather than merely present.
    """

    # A readable HI name with no schema is still the undeclared-schema
    # refusal, on the same rule it always used.
    with pytest.raises(ClosedSchemaError) as caught:
        _validate(OmegaConf.create({"experiment": {"name": "tpen_he_importance"}}))
    assert _rules(caught.value) == {"undeclared-schema"}

    # A readable foreign name with no schema is still returned unvalidated:
    # the firewall is opt-in and this config never opted in.
    _validate(OmegaConf.create({"experiment": {"name": "some_other_study"}}))

    # An ESCAPED opener is literal text that runs no resolver, so refusing it
    # would be a false refusal. This is the discrimination arm for the
    # interpolation predicate itself.
    _validate(OmegaConf.create({"experiment": {"name": r"\${runtime.real}"}}))


def test_hi_firewall_review_r1_declared_schema_still_validates_an_interpolated_name() -> None:
    """Opting in literally must not be short-circuited by the new refusal.

    A config that declares the schema as a LITERAL has opted in, so the full
    policy applies and an interpolated ``experiment.name`` is just another
    node the sweep sees. The identity refusal exists only for a config whose
    opt-in cannot be read at all; firing it here would replace a complete
    validation with a single early rejection.
    """

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
    assert "interpolated-identity" not in rules, rules
    assert any(rule.startswith("forbidden-surface") for rule in rules), rules
