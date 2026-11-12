from __future__ import annotations

from dataclasses import dataclass

import pytest

from tpen.physics.context import ContextKey, ReadOnlyContext, requirements
from tpen.physics.operators import OperatorId, register_operator
from tpen.physics.planner import LocalEnergyPlanner, OperatorLoweringRegistry, emit_receipt


@register_operator(OperatorId("tests.planner", "first"))
class FirstOperator:
    pass


@register_operator(OperatorId("tests.planner", "second"))
class SecondOperator:
    pass


@register_operator(OperatorId("tests.planner", "third"))
class ThirdOperator:
    pass


class UnregisteredOperator:
    pass


FIRST = FirstOperator.operator_id
SECOND = SecondOperator.operator_id
THIRD = ThirdOperator.operator_id


@dataclass(frozen=True)
class Kernel:
    claimed_operator_ids: frozenset[OperatorId]
    requirements: frozenset[ContextKey[object]] = frozenset()
    value: int = 0

    def evaluate(self, context: ReadOnlyContext) -> int:
        del context
        return self.value


@dataclass(frozen=True)
class Lowering:
    operator_type: type
    kernel: Kernel

    def lower(self, operator: object, capabilities: object) -> Kernel:
        assert type(operator) is self.operator_type
        del capabilities
        return self.kernel


@dataclass(frozen=True)
class AlternateLowering(Lowering):
    """A distinct backend identity with the same lowering behaviour."""


@dataclass(frozen=True)
class Provider:
    key: ContextKey[object]
    dependencies: frozenset[ContextKey[object]] = frozenset()
    value: object = 0
    calls: list[str] | None = None

    def build(self, context: ReadOnlyContext) -> object:
        del context
        if self.calls is not None:
            self.calls.append(self.key.name)
        return self.value


def test_registry_routes_a_registered_synthetic_operator_without_planner_edit() -> None:
    kernel = Kernel(frozenset({FIRST}), value=7)
    planner = LocalEnergyPlanner(OperatorLoweringRegistry([Lowering(FirstOperator, kernel)]))

    plan = planner.plan([FirstOperator()])

    assert plan.manifest == {FIRST: kernel}
    assert plan.kernels == (kernel,)
    assert plan.evaluate(ReadOnlyContext({})) == 7


def test_planner_rejects_unregistered_operator_by_class_name() -> None:
    planner = LocalEnergyPlanner(OperatorLoweringRegistry([]))

    with pytest.raises(ValueError, match="unregistered operator class UnregisteredOperator"):
        planner.plan([UnregisteredOperator()])


def test_planner_reports_unavailable_capability_with_operator_and_key() -> None:
    needed = ContextKey("distance", int)
    kernel = Kernel(frozenset({FIRST}), requirements=requirements(needed))
    planner = LocalEnergyPlanner(OperatorLoweringRegistry([Lowering(FirstOperator, kernel)]))

    with pytest.raises(ValueError, match="tests.planner:first.*distance"):
        planner.plan([FirstOperator()])


def test_fused_kernel_claims_two_operators_and_is_evaluated_once() -> None:
    fused = Kernel(frozenset({FIRST, SECOND}), value=11)
    planner = LocalEnergyPlanner(
        OperatorLoweringRegistry(
            [
                Lowering(FirstOperator, fused),
                Lowering(SecondOperator, fused),
            ]
        )
    )

    plan = planner.plan([SecondOperator(), FirstOperator()])

    assert plan.manifest == {FIRST: fused, SECOND: fused}
    assert plan.kernels == (fused,)
    assert plan.evaluate(ReadOnlyContext({})) == 11


def test_plan_accumulates_distinct_kernel_contributions_exactly_once() -> None:
    first = Kernel(frozenset({FIRST}), value=5)
    second = Kernel(frozenset({SECOND}), value=7)
    planner = LocalEnergyPlanner(
        OperatorLoweringRegistry(
            [Lowering(SecondOperator, second), Lowering(FirstOperator, first)]
        )
    )

    plan = planner.plan([SecondOperator(), FirstOperator()])

    assert plan.evaluate(ReadOnlyContext({})) == 12


def test_planner_rejects_a_double_claim() -> None:
    first = Kernel(frozenset({FIRST, SECOND}), value=1)
    second = Kernel(frozenset({SECOND}), value=2)
    planner = LocalEnergyPlanner(
        OperatorLoweringRegistry(
            [Lowering(FirstOperator, first), Lowering(SecondOperator, second)]
        )
    )

    with pytest.raises(ValueError, match="tests.planner:second.*more than one"):
        planner.plan([FirstOperator(), SecondOperator()])


def test_planner_rejects_a_zero_claim() -> None:
    planner = LocalEnergyPlanner(
        OperatorLoweringRegistry([Lowering(FirstOperator, Kernel(frozenset()))])
    )

    with pytest.raises(ValueError, match="FirstOperator.*not claimed"):
        planner.plan([FirstOperator()])


def test_provider_graph_is_closed_and_execution_is_topological() -> None:
    calls: list[str] = []
    source = ContextKey("source", int)
    derived = ContextKey("derived", int)
    kernel = Kernel(frozenset({FIRST}), requirements=requirements(derived), value=3)
    planner = LocalEnergyPlanner(OperatorLoweringRegistry([Lowering(FirstOperator, kernel)]))

    plan = planner.plan(
        [FirstOperator()],
        [
            Provider(derived, requirements(source), value=2, calls=calls),
            Provider(source, value=1, calls=calls),
        ],
    )

    assert tuple(provider.key for provider in plan.providers) == (source, derived)
    assert calls == []
    assert plan.evaluate(ReadOnlyContext({})) == 3
    assert calls == ["source", "derived"]


def test_plan_identity_is_order_independent_and_changes_with_operator_set() -> None:
    first = Kernel(frozenset({FIRST}), value=1)
    second = Kernel(frozenset({SECOND}), value=2)
    registry = OperatorLoweringRegistry(
        [Lowering(SecondOperator, second), Lowering(FirstOperator, first)]
    )
    planner = LocalEnergyPlanner(registry)
    source = ContextKey("source", int)
    capabilities = {
        source: 1,
    }

    one = planner.plan([FirstOperator()], capabilities)
    two = planner.plan([FirstOperator(), SecondOperator()], capabilities)
    reversed_two = planner.plan([SecondOperator(), FirstOperator()], {source: 1})

    assert two.identity == reversed_two.identity
    assert one.identity != two.identity


def test_plan_identity_changes_when_lowering_class_changes() -> None:
    kernel = Kernel(frozenset({FIRST}), value=1)
    ordinary = LocalEnergyPlanner(
        OperatorLoweringRegistry([Lowering(FirstOperator, kernel)])
    ).plan([FirstOperator()])
    alternate = LocalEnergyPlanner(
        OperatorLoweringRegistry([AlternateLowering(FirstOperator, kernel)])
    ).plan([FirstOperator()])

    assert ordinary.kernels == alternate.kernels
    assert ordinary.identity != alternate.identity


def test_registry_rejects_duplicate_lowering_types() -> None:
    kernel = Kernel(frozenset({FIRST}))
    with pytest.raises(ValueError, match="duplicate lowering.*FirstOperator"):
        OperatorLoweringRegistry(
            [Lowering(FirstOperator, kernel), Lowering(FirstOperator, kernel)]
        )


def test_plan_manifest_and_provider_graph_are_immutable() -> None:
    key = ContextKey("source", int)
    kernel = Kernel(frozenset({FIRST}), requirements=requirements(key), value=1)
    plan = LocalEnergyPlanner(
        OperatorLoweringRegistry([Lowering(FirstOperator, kernel)])
    ).plan([FirstOperator()], {key: 1})

    with pytest.raises(TypeError):
        plan.manifest[FIRST] = kernel  # type: ignore[index]
    with pytest.raises(TypeError):
        plan.provider_graph[key] = plan.providers[0]  # type: ignore[index]


# --- S1-E review-r1 finding 5: planner-to-receipt handoff --------------------


def test_emit_receipt_distinguishes_plans_by_identity_not_manifest_shape() -> None:
    # Two lowering classes produce an identical manifest shape (same single
    # kernel, same claimed operator) but a distinct plan.identity (S1-D's
    # test_plan_identity_changes_when_lowering_class_changes already proves
    # this at the plan level). A receipt recipe that only mirrored the
    # manifest, without retaining plan.identity, could not tell these two
    # lowerings apart.
    kernel = Kernel(frozenset({FIRST}), value=1)
    plans = [
        LocalEnergyPlanner(OperatorLoweringRegistry([kind(FirstOperator, kernel)])).plan(
            [FirstOperator()]
        )
        for kind in (Lowering, AlternateLowering)
    ]
    assert plans[0].identity != plans[1].identity

    receipts = [emit_receipt(plan, system_id="same-system") for plan in plans]

    assert receipts[0].method_id() != receipts[1].method_id()
    assert emit_receipt(plans[0], system_id="same-system") == receipts[0]


def test_emit_receipt_fused_kernel_shares_one_column_across_claimed_ids() -> None:
    # GREEN companion: one kernel claiming two operators reports one stable
    # column name under both ids, never a stringified kernel or an invented
    # per-operator diagnostic.
    kernel = Kernel(frozenset({FIRST, SECOND}), value=1)
    plan = LocalEnergyPlanner(
        OperatorLoweringRegistry(
            [Lowering(FirstOperator, kernel), Lowering(SecondOperator, kernel)]
        )
    ).plan([FirstOperator(), SecondOperator()])

    receipt = emit_receipt(plan, system_id="same-system")

    assert receipt.manifest[FIRST] == receipt.manifest[SECOND]
    assert len(receipt.manifest[FIRST]) == 1
    assert receipt.manifest[FIRST][0] != str(kernel)
    assert receipt.diagnostics == {}
