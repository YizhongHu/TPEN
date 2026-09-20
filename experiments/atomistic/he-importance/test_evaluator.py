"""Contract tests for the mechanics-only HI ranking evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "he_importance_evaluator", Path(__file__).with_name("evaluator.py")
)
assert _SPEC is not None and _SPEC.loader is not None
evaluator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evaluator
_SPEC.loader.exec_module(evaluator)


def _rows(*values: float) -> tuple[object, ...]:
    return tuple(
        evaluator.LocalEnergyRow("checkpoint-a", f"chain-{index}", {"rank": index}, value)
        for index, value in enumerate(values)
    )


def _row(chain: str = "c", checkpoint: str = "cp", topology: object | None = None, value: float = 1.0):
    return evaluator.LocalEnergyRow(
        checkpoint,
        chain,
        {"rank": 0} if topology is None else topology,
        value,
    )


@pytest.mark.parametrize("oracle_name", ["analytic", "naive", "slow"])
def test_factor_qualification_accepts_each_required_oracle(oracle_name: str) -> None:
    oracle = lambda value: value * 2
    result = evaluator.qualify_factor("cusp", 3, oracle, **{f"{oracle_name}_oracle": oracle})
    assert result.value == 6
    assert result.oracle_values == {oracle_name: 6}


def test_factor_qualification_witnesses_disagreeing_oracle() -> None:
    with pytest.raises(
        evaluator.EvaluatorQualificationError,
        match="factor 'cusp' disagrees with slow oracle: 6 != 7",
    ):
        evaluator.qualify_factor("cusp", 3, lambda value: value * 2, slow_oracle=lambda value: 7)


@pytest.mark.parametrize(
    "route",
    [
        "factor id",
        "factor input",
        "calculator output",
        "analytic oracle output",
        "naive oracle output",
        "slow oracle output",
        "local-energy row[0] topology",
    ],
)
@pytest.mark.parametrize("forbidden_key", ["reference_error", "e_ref"])
def test_blinding_refuses_reference_field_on_every_evaluator_route(
    route: str, forbidden_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = evaluator._refuse_blinding_content
    seen: list[str] = []

    def only_target(value: object, label: str) -> None:
        seen.append(label)
        if label == route or label.startswith(route + ".") or label.startswith(route + "["):
            original(value, label)

    monkeypatch.setattr(evaluator, "_refuse_blinding_content", only_target)
    # Validate that a non-target screen is disabled before exercising the target.
    only_target({"reference_energy": 0}, "disabled control")
    benign = {"coordinate": 2}
    tainted = {forbidden_key: 0.01}

    if route == "local-energy row[0] topology":
        rows = (evaluator.LocalEnergyRow("checkpoint-a", "chain-0", tainted, 1.0),)
        with pytest.raises(evaluator.EvaluatorQualificationError):
            evaluator.evaluate_local_energy_rows("checkpoint-a", rows, expected_chain_ids=("chain-0",))
    else:
        value = tainted if route == "factor input" else benign
        factor_id = forbidden_key if route == "factor id" else "cusp"
        output = tainted if "output" in route else benign
        if route.startswith("analytic"):
            analytic = lambda item: output
            naive = slow = None
        elif route.startswith("naive"):
            analytic = None
            naive = lambda item: output
            slow = None
        elif route.startswith("slow"):
            analytic = naive = None
            slow = lambda item: output
        else:
            analytic = lambda item: output
            naive = slow = None
        calculator = lambda item: output
        with pytest.raises(evaluator.EvaluatorQualificationError):
            evaluator.qualify_factor(
                factor_id,
                value,
                calculator,
                analytic_oracle=analytic,
                naive_oracle=naive,
                slow_oracle=slow,
            )
    assert route in seen


def test_blinding_refuses_reference_label_channel() -> None:
    with pytest.raises(evaluator.EvaluatorQualificationError, match="forbidden blinding value 'reference_energy'"):
        evaluator.qualify_factor(
            "reference_energy",
            {"coordinate": 2},
            lambda item: {"coordinate": 2},
            analytic_oracle=lambda item: {"coordinate": 2},
        )


@pytest.mark.parametrize(
    "route",
    [
        "factor id",
        "factor input",
        "calculator output",
        "analytic oracle output",
        "naive oracle output",
        "slow oracle output",
        "local-energy row[0] topology",
    ],
)
def test_blinding_refuses_reference_string_values_on_every_qualified_route(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = evaluator._refuse_blinding_content
    seen: list[str] = []

    def only_target(value: object, label: str) -> None:
        seen.append(label)
        if label == route or label.startswith(route + ".") or label.startswith(route + "["):
            original(value, label)

    monkeypatch.setattr(evaluator, "_refuse_blinding_content", only_target)
    # A disabled screen must be proven inert; an unvalidated disable can strengthen coverage.
    only_target({"reference_energy": 0}, "disabled control")
    tainted = "reference_energy"

    if route == "local-energy row[0] topology":
        rows = (evaluator.LocalEnergyRow("checkpoint-a", "chain-0", {"label": tainted}, 1.0),)
        with pytest.raises(evaluator.EvaluatorQualificationError):
            evaluator.evaluate_local_energy_rows("checkpoint-a", rows, expected_chain_ids=("chain-0",))
    else:
        value = {"label": tainted} if route == "factor input" else {"coordinate": 2}
        factor_id = tainted if route == "factor id" else "cusp"
        output = {"label": tainted} if "output" in route else {"coordinate": 2}
        if route.startswith("analytic"):
            analytic = lambda item: output
            naive = slow = None
        elif route.startswith("naive"):
            analytic = None
            naive = lambda item: output
            slow = None
        elif route.startswith("slow"):
            analytic = naive = None
            slow = lambda item: output
        else:
            analytic = lambda item: output
            naive = slow = None
        calculator = lambda item: output
        with pytest.raises(evaluator.EvaluatorQualificationError):
            evaluator.qualify_factor(
                factor_id,
                value,
                calculator,
                analytic_oracle=analytic,
                naive_oracle=naive,
                slow_oracle=slow,
            )
    assert route in seen


@pytest.mark.parametrize(
    "token",
    ["reference", "reference_energy", "e_ref", "accuracy", "x-reference-y", "E-REF", "ACCURACY"],
)
@pytest.mark.parametrize("channel", ["key", "value"])
def test_common_frozen_predicate_at_visited_nodes(token: str, channel: str) -> None:
    leaf = {token: 1} if channel == "key" else {"label": token}
    nested = MappingProxyType({"outer": [MappingProxyType({"inner": (leaf,)})]})
    with pytest.raises(evaluator.EvaluatorQualificationError):
        evaluator._refuse_blinding_content(nested, "probe")


@pytest.mark.parametrize("token", ["energy", "local_energy", "some-energy-value", "LOCAL-ENERGY"])
@pytest.mark.parametrize("channel", ["key", "value"])
def test_l3a_allows_energy_tokens_in_both_channels(token: str, channel: str) -> None:
    """L3a permits energy tokens; see frozen-screen-adjudication-union-and-intersection."""

    payload = {token: 1} if channel == "key" else {"label": token}
    evaluator._refuse_blinding_content(payload, "legitimate L3a payload")


def test_a_requires_an_oracle() -> None:
    with pytest.raises(evaluator.EvaluatorQualificationError):
        evaluator.qualify_factor("cusp", 1, lambda _: 1)


@pytest.mark.parametrize("guard", ["void", "topology", "duplicate_chain", "wrong_checkpoint", "missing_chain", "extra_chain"])
def test_a_row_guard_independent_limbs(guard: str) -> None:
    rows, expected = (_row(),), ("c",)
    if guard == "void":
        rows, expected = (), ()
    elif guard == "topology":
        rows = (_row(topology={}),)
    elif guard == "duplicate_chain":
        rows = (_row(), _row())
    elif guard == "wrong_checkpoint":
        rows = (_row(checkpoint="another"),)
    elif guard == "missing_chain":
        expected = ("c", "d")
    elif guard == "extra_chain":
        rows = (_row(), _row(chain="d"))
    with pytest.raises(evaluator.EvaluatorQualificationError):
        evaluator.evaluate_local_energy_rows("cp", rows, expected_chain_ids=expected)


@pytest.mark.parametrize("case", ["duplicate", "partial", "extra", "nonfinite"])
def test_a_artifact_guard_independent_limbs(case: str) -> None:
    ready = evaluator.evaluate_local_energy_rows("cp", (_row(),), expected_chain_ids=("c",))
    states, expected = (ready,), ("cp",)
    if case == "duplicate":
        states = (ready, ready)
    elif case == "partial":
        expected = ("cp", "missing")
    elif case == "extra":
        expected = ()
    elif case == "nonfinite":
        nonfinite = evaluator.evaluate_local_energy_rows(
            "cp", (_row(value=float("nan")),), expected_chain_ids=("c",)
        )
        states = (nonfinite,)
        assert nonfinite.rows[0].chain_id == "c"
        assert nonfinite.status is evaluator.InferenceStatus.NONFINITE_LOCAL_ENERGY
    with pytest.raises(evaluator.IncompleteRankArtifactError):
        evaluator.require_complete_rank_artifact(states, expected_checkpoint_ids=expected)


def test_a_legitimate_energy_row_can_be_qualified() -> None:
    payload = _row(value=-2.8)
    result = evaluator.qualify_factor("local_energy", payload, lambda value: value, naive_oracle=lambda value: value)
    assert result.value == payload
    state = evaluator.evaluate_local_energy_rows("cp", (payload,), expected_chain_ids=("c",))
    evaluator.require_complete_rank_artifact((state,), expected_checkpoint_ids=("cp",))


def test_blinding_refuses_reference_name_in_local_energy_topology() -> None:
    rows = (evaluator.LocalEnergyRow("checkpoint-a", "chain-0", {"reference_energy": 0.0}, 1.0),)
    with pytest.raises(
        evaluator.EvaluatorQualificationError,
        match="local-energy row\\[0\\] topology contains forbidden blinding field 'reference_energy'",
    ):
        evaluator.evaluate_local_energy_rows("checkpoint-a", rows, expected_chain_ids=("chain-0",))


def test_nonfinite_row_is_visible_with_its_chain_and_topology_witness() -> None:
    rows = _rows(1.0, float("nan"))
    state = evaluator.evaluate_local_energy_rows(
        "checkpoint-a", rows, expected_chain_ids=("chain-0", "chain-1")
    )
    assert state.status is evaluator.InferenceStatus.NONFINITE_LOCAL_ENERGY
    assert state.rows == rows
    assert state.rows[1].chain_id == "chain-1"
    assert state.rows[1].topology == {"rank": 1}


def test_local_energy_requires_nonempty_topology_provenance() -> None:
    rows = (evaluator.LocalEnergyRow("checkpoint-a", "chain-0", {}, 1.0),)
    with pytest.raises(evaluator.EvaluatorQualificationError, match="require nonempty topology provenance"):
        evaluator.evaluate_local_energy_rows("checkpoint-a", rows, expected_chain_ids=("chain-0",))


def test_local_energy_rejects_mismatched_row_checkpoint_provenance() -> None:
    rows = (evaluator.LocalEnergyRow("checkpoint-b", "chain-0", {"rank": 0}, 1.0),)
    with pytest.raises(evaluator.EvaluatorQualificationError, match="checkpoint provenance"):
        evaluator.evaluate_local_energy_rows("checkpoint-a", rows, expected_chain_ids=("chain-0",))


def test_local_energy_rejects_void_corpus() -> None:
    with pytest.raises(evaluator.EvaluatorQualificationError, match="requires at least one row"):
        evaluator.evaluate_local_energy_rows("checkpoint-a", (), expected_chain_ids=())


@pytest.mark.parametrize(
    ("rows", "expected", "message"),
    [
        (_rows(1.0), ("chain-0", "chain-1"), "each expected chain exactly once"),
        (_rows(1.0, 2.0), ("chain-0",), "each expected chain exactly once"),
        (
            (
                evaluator.LocalEnergyRow("checkpoint-a", "chain-0", {"rank": 0}, 1.0),
                evaluator.LocalEnergyRow("checkpoint-a", "chain-0", {"rank": 1}, 2.0),
            ),
            ("chain-0",),
            "each expected chain exactly once",
        ),
    ],
)
def test_local_energy_chain_completeness_has_a_discriminating_witness(
    rows: tuple[object, ...], expected: tuple[str, ...], message: str
) -> None:
    with pytest.raises(evaluator.IncompleteRankArtifactError, match=message):
        evaluator.evaluate_local_energy_rows("checkpoint-a", rows, expected_chain_ids=expected)


def test_rank_artifact_rejects_nonfinite_state_instead_of_silently_dropping_it() -> None:
    ready = evaluator.evaluate_local_energy_rows("checkpoint-a", _rows(1.0), expected_chain_ids=("chain-0",))
    bad_rows = tuple(
        evaluator.LocalEnergyRow("checkpoint-b", row.chain_id, row.topology, float("inf"))
        for row in _rows(1.0)
    )
    nonfinite = evaluator.evaluate_local_energy_rows(
        "checkpoint-b", bad_rows, expected_chain_ids=("chain-0",)
    )
    with pytest.raises(
        evaluator.IncompleteRankArtifactError,
        match=r"rank artifact contains nonfinite local-energy state: \['checkpoint-b'\]",
    ):
        evaluator.require_complete_rank_artifact(
            (ready, nonfinite), expected_checkpoint_ids=("checkpoint-a", "checkpoint-b")
        )


def test_rank_artifact_rejects_duplicate_checkpoint_ids() -> None:
    ready = evaluator.InferenceState("checkpoint-a", evaluator.InferenceStatus.READY, ())
    with pytest.raises(evaluator.IncompleteRankArtifactError, match="each expected checkpoint exactly once"):
        evaluator.require_complete_rank_artifact(
            (ready, ready), expected_checkpoint_ids=("checkpoint-a",)
        )


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        ([], ("checkpoint-a",)),
        (
            [
                evaluator.InferenceState("checkpoint-a", evaluator.InferenceStatus.READY, ()),
                evaluator.InferenceState("checkpoint-a", evaluator.InferenceStatus.READY, ()),
            ],
            ("checkpoint-a",),
        ),
    ],
)
def test_rank_artifact_rejects_empty_or_duplicate_checkpoint_sets(
    states: list[object], expected: tuple[str, ...]
) -> None:
    with pytest.raises(evaluator.IncompleteRankArtifactError, match="each expected checkpoint exactly once"):
        evaluator.require_complete_rank_artifact(states, expected_checkpoint_ids=expected)
