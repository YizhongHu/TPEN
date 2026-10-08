"""Explicit update observations, actual-method descriptions, and sink safety.

These cover the VMC-IF-A contract: that one update attempt reports its own
status, reason and detached diagnostics on the record it RETURNS; that the
trainer consumes that record rather than probing the method object; and that
what gets logged is the method and carrier actually constructed or restored,
in a form both the JSONL and the CSV sink can carry.

WHY THE SINKS ARE EXERCISED FOR REAL HERE. Both impose a hard limit that a
plausible-looking value passes in memory and fails only at the boundary:
`tpen.logging.jsonl` calls `json.dumps(..., allow_nan=False)`, so a non-finite
float RAISES in the sink; `tpen.logging.csv` writes unquoted rows, so a value
containing a comma silently becomes extra columns and the file still parses.
A test that asserted on the metrics dict alone would see neither.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tpen.logging.base import LogRecord
from tpen.logging.csv import CSV
from tpen.logging.jsonl import JSONL
from tpen.training.block_ng import BlockNGPolicy, BlockDiagonalNaturalGradientUpdate
from tpen.training.qgt import DampingPolicy
from tpen.training.score_geometry import ScoreConventions
from tpen.training.spring import SPRINGPolicy, SPRINGUpdate
from tpen.training.sr import SRPolicy, StochasticReconfigurationUpdate
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    UPDATE_REASON_APPLIED,
    UPDATE_REASON_UNREPORTED,
    UPDATE_REASON_ZERO_ELECTRON_BATCH,
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    MinimalUpdateDiagnostics,
    ModelParameterBinding,
    UpdateDiagnostics,
    UpdateMethodDescription,
    VMCUpdateMethod,
    VMCUpdateResult,
    carrier_settings,
    flatten_settings,
    json_safe_scalar,
)
from tests.helpers.hooke_models import build_tiny_hamiltonian_terms
from tests.unit.training.test_vmc_update import _batch, _output, _reevaluation
from tests.unit.training.test_sr_trainer_integration import (
    LEARNING_RATE,
    _FixedSampler,
    build_connected_model,
)
from tests.unit.training.test_vmc_trainer_tpen_smoke import (
    _ConstantWavefunction,
    _VacuumSampler,
    _StubContext,
)


# --------------------------------------------------------------------------
# json_safe_scalar: the never-fabricate-zero representation
# --------------------------------------------------------------------------


def test_a_nonfinite_observation_is_labelled_not_turned_into_zero() -> None:
    """Non-finite observations must be NAMED, never silently become 0.0.

    Zero is a legitimate value for every norm this slice reports, so a
    fabricated zero is indistinguishable from a real measurement and would
    corrupt any downstream average without ever looking wrong.
    """

    assert json_safe_scalar(float("nan")) == "nan"
    assert json_safe_scalar(float("inf")) == "inf"
    assert json_safe_scalar(float("-inf")) == "-inf"
    # The polarity that matters: NOT zero, and not a float at all, so a reader
    # can discriminate on type alone.
    for sentinel in ("nan", "inf", "-inf"):
        assert not isinstance(sentinel, float)
    assert json_safe_scalar(0.0) == 0.0
    assert isinstance(json_safe_scalar(0.0), float)


def test_an_unavailable_observation_stays_none_rather_than_becoming_zero() -> None:
    """Missing is represented as missing: JSON null, empty CSV cell."""

    assert json_safe_scalar(None) is None


def test_a_value_containing_a_comma_cannot_reach_an_unquoted_csv_row() -> None:
    """A comma in a value would split one metric into two CSV columns."""

    assert "," not in json_safe_scalar("a,b")
    assert json_safe_scalar("a,b") == "a;b"


def test_finite_scalars_and_bools_pass_through_unchanged() -> None:
    """The common path must not be perturbed by the guard."""

    assert json_safe_scalar(1.5) == 1.5
    assert json_safe_scalar(3) == 3
    assert json_safe_scalar(True) is True
    assert json_safe_scalar(False) is False


# --------------------------------------------------------------------------
# The legacy adapter: explicit reason, and the vacuum/disconnected distinction
# --------------------------------------------------------------------------


def _legacy_method(model, **kwargs) -> LegacyAutogradUpdate:
    parameters = tuple(model.parameters())
    return LegacyAutogradUpdate(
        optimizer=torch.optim.Adam(parameters, lr=0.01),
        model_parameters=ModelParameterBinding(parameters=parameters),
        **kwargs,
    )


def test_a_vacuum_skip_reports_its_own_reason_and_a_minimal_record() -> None:
    """The zero-electron skip must be self-describing, not an unexplained False."""

    context = _StubContext()
    model = _ConstantWavefunction()
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    trainer.fit(
        model=model,
        sampler=_VacuumSampler(),
        hamiltonian_terms=[],
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        context=context,
        emit=lambda **_: None,
    )

    train_records = [m for ns, m in context.records if ns == "train"]
    assert train_records, "the trainer logged no train metrics"
    metrics = train_records[-1]
    assert metrics["optimizer_step"] is False
    assert metrics["update_reason"] == UPDATE_REASON_ZERO_ELECTRON_BATCH
    # The minimal record reached the sink under the method's own prefix.
    assert metrics["update_record_applied"] is False
    assert metrics["update_record_reason"] == UPDATE_REASON_ZERO_ELECTRON_BATCH
    assert metrics["update_record_method"] == "LegacyAutogradUpdate"


def test_the_vacuum_skip_stays_distinct_from_a_disconnected_objective() -> None:
    """Same `applied=False` shape, genuinely different conditions.

    The vacuum declines and NAMES itself; a disconnected objective on a
    NONZERO-electron batch still RAISES. Collapsing the two would turn a real
    modelling bug into a silently skipped step.

    `test_vmc_update.py` owns the raise half of this contract. What is added
    here is the half the raise cannot show: that the declining half carries an
    explicit reason and a detached record, rather than being an unexplained
    `False`.
    """

    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    adapter = LegacyAutogradUpdate(
        torch.optim.SGD([parameter], lr=0.1),
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
    )
    vacuum = _batch(n_electrons=0)
    skipped = adapter.update(
        AutogradUpdateInput(
            batch=vacuum,
            wavefunction=_output(vacuum),
            local_energy=torch.zeros(1, dtype=torch.float64),
            step=0,
            objective=torch.tensor(1.0, dtype=torch.float64),
            reevaluate=_reevaluation(vacuum),
        )
    )

    assert skipped.applied is False
    assert skipped.reason == UPDATE_REASON_ZERO_ELECTRON_BATCH
    # The two reasons are different tokens, which is what makes the record
    # able to express the distinction at all.
    assert UPDATE_REASON_ZERO_ELECTRON_BATCH != UPDATE_REASON_APPLIED
    # grad_norm 0.0 here is a REAL measurement -- no backward ran, so the
    # gradient domain genuinely holds no gradient -- and it is an established
    # compatibility field, so it keeps its value and its meaning.
    assert skipped.grad_norm == 0.0
    assert parameter.grad is None

    record = skipped.diagnostics
    assert isinstance(record, MinimalUpdateDiagnostics)
    assert record.as_metrics()["update_record_reason"] == UPDATE_REASON_ZERO_ELECTRON_BATCH
    assert record.as_metrics()["update_record_applied"] is False


def test_an_applied_legacy_step_reports_the_applied_reason() -> None:
    """The ordinary path is explicit too, not merely 'not a skip'."""

    model = build_connected_model()
    method = _legacy_method(model)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)
    context = _StubContext()
    trainer.fit(
        model=model,
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=method.optimizer,
        context=context,
        emit=lambda **_: None,
    )
    metrics = [m for ns, m in context.records if ns == "train"][-1]
    assert metrics["update_reason"] == UPDATE_REASON_APPLIED
    assert metrics["update_record_applied"] is True


def test_the_legacy_adapter_remembers_its_attempt_like_any_other_method() -> None:
    """`last_result` must work on the method every default config uses.

    The legacy adapter is the DEFAULT update method, so a `last_result` that
    were permanently `None` here would be None for most runs in the project
    while appearing to work everywhere else.
    """

    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    adapter = LegacyAutogradUpdate(
        torch.optim.SGD([parameter], lr=0.1),
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
    )
    assert adapter.last_result is None

    vacuum = _batch(n_electrons=0)
    result = adapter.update(
        AutogradUpdateInput(
            batch=vacuum,
            wavefunction=_output(vacuum),
            local_energy=torch.zeros(1, dtype=torch.float64),
            step=0,
            objective=torch.tensor(1.0, dtype=torch.float64),
            reevaluate=_reevaluation(vacuum),
        )
    )
    assert adapter.last_result is result


def test_the_legacy_norm_is_described_as_post_clip() -> None:
    """With clipping configured the reported norm is bounded, and says so."""

    model = build_connected_model()
    method = _legacy_method(model, gradient_clip_norm=0.5)
    description = method.describe()
    assert description.norm_semantics == "post_clip_grad_l2_over_gradient_domain"
    assert description.settings["gradient_clip_norm"] == 0.5


# --------------------------------------------------------------------------
# Freshness: no previous attempt's record may leak into the next
# --------------------------------------------------------------------------


def _sr_method(model, **policy_kwargs) -> StochasticReconfigurationUpdate:
    parameters = tuple(model.parameters())
    policy = SRPolicy(
        damping=DampingPolicy(absolute=0.0, relative=1.0e-2, minimum=1.0e-12),
        learning_rate=LEARNING_RATE,
        **policy_kwargs,
    )
    return StochasticReconfigurationUpdate(
        torch.optim.SGD(parameters, lr=LEARNING_RATE),
        model_parameters=ModelParameterBinding(parameters=parameters),
        policy=policy,
        conventions=ScoreConventions(solve_dtype=torch.float64),
    )


def test_a_skip_after_an_applied_step_reports_the_skip_not_the_solve() -> None:
    """A REAL applied attempt then a skipped one must leave NO stale record.

    This is the leak the returned-result contract closes. While the record
    lived on the method object, a skip that produced no solver diagnostics
    left the PREVIOUS step's `SolveDiagnostics` in place, so a reader of the
    skipped step saw a solve that had not happened on it.

    The applied step is driven through the REAL trainer rather than
    stubbed, so the first record is a genuine solve and the test cannot pass
    against a fabricated one.
    """

    model = build_connected_model()
    method = _sr_method(model)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)
    trainer.fit(
        model=model,
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=method.optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )

    applied = method.last_result
    assert applied is not None
    assert applied.applied is True
    assert applied.reason == UPDATE_REASON_APPLIED
    # A real solve happened, so the record carries real solve diagnostics.
    assert applied.diagnostics is not None
    assert applied.diagnostics.diagnostics is not None

    skipped = method._skip(
        reason="insufficient_finite_samples",
        step=1,
        n_samples=4,
        n_finite=0,
        n_parameters=8,
    )
    # The NEW record, not the old one.
    assert skipped.applied is False
    assert skipped.diagnostics.reason == "insufficient_finite_samples"
    assert skipped.diagnostics.step == 1
    # The decisive assertion: the earlier solve did NOT survive into this one.
    assert skipped.diagnostics.diagnostics is None
    # And the compatibility property follows the result rather than lagging it.
    assert method.last_telemetry is skipped.diagnostics
    assert method.last_result is skipped


def test_last_telemetry_is_derived_from_the_result_not_a_second_authority() -> None:
    """The compatibility property must be unable to disagree with the result."""

    model = build_connected_model()
    method = _sr_method(model)
    assert method.last_telemetry is None
    assert method.last_result is None

    result = method._skip(
        reason="declined",
        step=3,
        n_samples=2,
        n_finite=0,
        n_parameters=4,
    )
    assert method.last_telemetry is result.diagnostics
    assert method.last_result is result


# --------------------------------------------------------------------------
# The actual method, not the carrier, and not the label
# --------------------------------------------------------------------------


def test_sr_is_not_reported_as_plain_sgd_despite_its_sgd_carrier() -> None:
    """The carrier must never be mistaken for the method.

    SR, SPRING and block NG all drive a plain `torch.optim.SGD` carrier. Any
    identification that read the optimizer would report all three as SGD --
    a baseline nobody ran.
    """

    model = build_connected_model()
    method = _sr_method(model)
    description = method.describe()

    assert description.method_class.endswith("StochasticReconfigurationUpdate")
    assert description.carrier_class.endswith("SGD")
    assert description.method_class != description.carrier_class
    # The norm's meaning is stated, because SR's `.grad` holds a
    # PRECONDITIONED DIRECTION rather than a gradient by the time it is read.
    assert description.norm_semantics == (
        "energy_gradient_l2_preconditioned_direction_in_grad"
    )
    assert description.forward_request.endswith("MaterializedParameterScoreRequest")


def test_a_misleading_external_label_cannot_change_what_is_described() -> None:
    """Description is read off the live objects, never off a caller's name."""

    model = build_connected_model()
    method = _sr_method(model)
    # A deliberately wrong label of the kind a config or run name carries.
    method.display_name = "adam-baseline"
    description = method.describe()
    assert "adam" not in description.method_class.lower()
    assert description.method_class.endswith("StochasticReconfigurationUpdate")


def test_the_description_is_emitted_even_for_a_zero_step_fit() -> None:
    """A run that takes no step must still record WHAT WOULD HAVE STEPPED.

    A zero-step fit is exactly the run one most wants to audit; describing
    nothing there would make the interesting case the silent one.
    """

    model = build_connected_model()
    method = _sr_method(model)
    context = _StubContext()
    trainer = VMCTrainer(max_steps=0, log_every_n_steps=1, update_method=method)
    trainer.fit(
        model=model,
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=method.optimizer,
        context=context,
        emit=lambda **_: None,
    )

    described = [m for ns, m in context.records if ns == "train/update_method"]
    assert len(described) == 1, "a zero-step fit emitted no method description"
    assert described[0]["update_method_class"].endswith(
        "StochasticReconfigurationUpdate"
    )
    # No training step ran, so no train metrics were produced.
    assert not [m for ns, m in context.records if ns == "train"]


def test_a_direct_fit_caller_receives_the_same_description() -> None:
    """A caller with no logging sink must still get the record."""

    model = build_connected_model()
    method = _sr_method(model)
    context = _StubContext()
    trainer = VMCTrainer(max_steps=0, log_every_n_steps=1, update_method=method)
    assert trainer.update_method_description is None
    trainer.fit(
        model=model,
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=method.optimizer,
        context=context,
        emit=lambda **_: None,
    )
    description = trainer.update_method_description
    assert isinstance(description, UpdateMethodDescription)
    logged = [m for ns, m in context.records if ns == "train/update_method"][0]
    assert description.as_metrics() == logged


# --------------------------------------------------------------------------
# Carrier settings: compound values split, never repr'd
# --------------------------------------------------------------------------


def test_adam_betas_are_split_into_named_scalars() -> None:
    """A tuple repr in the unquoted CSV sink would silently add columns."""

    model = build_connected_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    settings = carrier_settings(optimizer)

    assert settings["betas1"] == pytest.approx(0.9)
    assert settings["betas2"] == pytest.approx(0.999)
    assert "betas" not in settings
    for key, value in settings.items():
        assert not isinstance(value, (tuple, list, dict)), f"{key} is compound"
        assert "," not in str(value), f"{key} would break an unquoted CSV row"


def test_the_live_parameter_list_never_enters_a_description() -> None:
    """`params` is unbounded and holds the model's tensors."""

    model = build_connected_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    settings = carrier_settings(optimizer)
    assert "params" not in settings
    assert settings["n_param_groups"] == 1


def test_a_nested_policy_fingerprint_is_flattened_for_the_sinks() -> None:
    """Neither sink can express nesting; SR's damping sub-mapping must flatten."""

    flat = flatten_settings({"a": {"b": 1.0, "c": {"d": 2.0}}, "e": (3.0, 4.0)})
    assert flat == {"a_b": 1.0, "a_c_d": 2.0, "e1": 3.0, "e2": 4.0}
    for value in flat.values():
        assert not isinstance(value, (dict, tuple, list))


def test_the_sr_description_carries_its_resolved_policy_flat() -> None:
    """SR's damping policy must survive into the description as scalars."""

    model = build_connected_model()
    method = _sr_method(model)
    settings = method.describe().settings
    assert settings["policy_learning_rate"] == pytest.approx(LEARNING_RATE)
    assert settings["policy_damping_relative"] == pytest.approx(1.0e-2)
    for key, value in settings.items():
        assert not isinstance(value, (tuple, list, dict)), f"{key} is compound"


# --------------------------------------------------------------------------
# The real sinks
# --------------------------------------------------------------------------


def _describe_every_builtin_method(model) -> list[UpdateMethodDescription]:
    parameters = tuple(model.parameters())
    binding = ModelParameterBinding(parameters=parameters)
    sr_policy = SRPolicy(learning_rate=LEARNING_RATE)
    return [
        LegacyAutogradUpdate(
            optimizer=torch.optim.Adam(parameters, lr=0.01),
            model_parameters=binding,
            gradient_clip_norm=1.0,
        ).describe(),
        StochasticReconfigurationUpdate(
            torch.optim.SGD(parameters, lr=LEARNING_RATE),
            model_parameters=binding,
            policy=sr_policy,
        ).describe(),
        SPRINGUpdate(
            torch.optim.SGD(parameters, lr=LEARNING_RATE),
            model_parameters=binding,
            policy=SPRINGPolicy(base=sr_policy, history_decay=0.9),
        ).describe(),
        BlockDiagonalNaturalGradientUpdate(
            torch.optim.SGD(parameters, lr=LEARNING_RATE),
            model_parameters=binding,
            policy=BlockNGPolicy(damping=1.0e-3, learning_rate=LEARNING_RATE),
        ).describe(),
    ]


def test_every_builtin_description_survives_the_real_jsonl_sink(tmp_path) -> None:
    """`json.dumps(..., allow_nan=False)` RAISES on a non-finite float."""

    model = build_connected_model()
    path = tmp_path / "metrics.jsonl"
    sink = JSONL(path)
    for description in _describe_every_builtin_method(model):
        sink.log(
            LogRecord(step=0, namespace="train/update_method", metrics=description.as_metrics())
        )

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 4
    for line in lines:
        payload = json.loads(line)
        assert payload["namespace"] == "train/update_method"
        assert payload["metrics"]["update_method_class"]


def test_every_builtin_description_survives_the_real_csv_sink(tmp_path) -> None:
    """The CSV sink writes UNQUOTED rows; a comma would add a phantom column."""

    model = build_connected_model()
    path = tmp_path / "metrics.csv"
    sink = CSV(path)
    for description in _describe_every_builtin_method(model):
        sink.log(
            LogRecord(step=0, namespace="train/update_method", metrics=description.as_metrics())
        )

    rows = path.read_text().strip().splitlines()
    assert rows[0] == "step,namespace,key,value"
    for row in rows[1:]:
        # Exactly four fields: any comma inside a value would make it five.
        assert len(row.split(",")) == 4, f"row split into extra columns: {row!r}"


def test_the_sr_diagnostics_record_survives_both_real_sinks(tmp_path) -> None:
    """A skip's diagnostics are written, not just held in memory."""

    model = build_connected_model()
    method = _sr_method(model)
    result = method._skip(
        reason="insufficient_finite_samples",
        step=2,
        n_samples=4,
        n_finite=0,
        n_parameters=8,
    )
    metrics = result.diagnostics.as_metrics()

    jsonl_path = tmp_path / "m.jsonl"
    JSONL(jsonl_path).log(LogRecord(step=2, namespace="train", metrics=metrics))
    payload = json.loads(jsonl_path.read_text().strip())
    assert payload["metrics"]["sr_reason"] == "insufficient_finite_samples"
    assert payload["metrics"]["sr_applied"] is False

    csv_path = tmp_path / "m.csv"
    CSV(csv_path).log(LogRecord(step=2, namespace="train", metrics=metrics))
    for row in csv_path.read_text().strip().splitlines()[1:]:
        assert len(row.split(",")) == 4, f"row split into extra columns: {row!r}"


def test_no_method_owned_diagnostic_key_collides_with_a_trainer_owned_key() -> None:
    """A collision would silently overwrite one metric with the other.

    This is not hypothetical. `MinimalUpdateDiagnostics` first shipped with the
    prefix `update`, which collided with the trainer's generic `update_reason`.
    Both happened to carry the same value, so nothing looked wrong and nothing
    failed -- the worst shape a defect can have. The guard is mechanical so the
    next record added cannot reintroduce it by choosing a plausible prefix.
    """

    # Keys the trainer composes itself, at the metrics boundary.
    trainer_owned = {
        "grad_norm",
        "param_norm",
        "loss_has_grad",
        "optimizer_step",
        "update_reason",
    }

    model = build_connected_model()
    parameters = tuple(model.parameters())
    binding = ModelParameterBinding(parameters=parameters)
    sr_policy = SRPolicy(learning_rate=LEARNING_RATE)
    records: list[UpdateDiagnostics] = [
        MinimalUpdateDiagnostics(
            method="Any", applied=True, reason="applied", step=0, grad_norm=1.0
        ),
        StochasticReconfigurationUpdate(
            torch.optim.SGD(parameters, lr=LEARNING_RATE),
            model_parameters=binding,
            policy=sr_policy,
        )._skip(reason="declined", step=0, n_samples=1, n_finite=0, n_parameters=1).diagnostics,
    ]
    for record in records:
        overlap = trainer_owned & set(record.as_metrics())
        assert not overlap, f"{type(record).__name__} collides on {sorted(overlap)}"


# --------------------------------------------------------------------------
# Custom and stateless methods keep working
# --------------------------------------------------------------------------


class _CustomMethodPredatingTheReasonContract(VMCUpdateMethod):
    """A method written before `reason`/`diagnostics` existed on the result."""

    def update(self, update_input: Any) -> VMCUpdateResult:
        return VMCUpdateResult(applied=True, grad_norm=2.5)


def test_a_custom_method_may_still_return_the_two_field_result() -> None:
    """Making either new field required would break every such method at once."""

    result = _CustomMethodPredatingTheReasonContract().update(None)
    assert result.applied is True
    assert result.grad_norm == 2.5
    assert result.reason is None
    assert result.diagnostics is None


def test_an_absent_reason_reads_as_unreported_not_as_applied() -> None:
    """A record must never claim a reason its method did not give."""

    result = VMCUpdateResult(applied=True, grad_norm=1.0)
    assert result.reported_reason == UPDATE_REASON_UNREPORTED
    assert result.reported_reason != UPDATE_REASON_APPLIED


def test_a_stateless_method_describes_itself_without_naming_a_carrier() -> None:
    """A stateless method owns no optimizer; it must not name one."""

    description = _CustomMethodPredatingTheReasonContract().describe()
    assert description.carrier_class == "none"
    assert description.method_class.endswith(
        "_CustomMethodPredatingTheReasonContract"
    )
    assert description.forward_request == "value"
    assert description.norm_semantics == "method_defined"


def test_a_minimal_record_is_an_explicit_choice_with_honest_absence() -> None:
    """The minimal record reports what it lacks rather than inventing it."""

    record = MinimalUpdateDiagnostics(
        method="Custom",
        applied=False,
        reason="declined",
        step=7,
        grad_norm=None,
    )
    assert isinstance(record, UpdateDiagnostics)
    metrics = record.as_metrics()
    assert metrics["update_record_grad_norm"] is None
    assert metrics["update_record_reason"] == "declined"
    assert metrics["update_record_step"] == 7


# --------------------------------------------------------------------------
# Result validation
# --------------------------------------------------------------------------


def test_a_reason_containing_a_comma_is_rejected_at_construction() -> None:
    """Rejected where it is cheap, not where it silently corrupts a CSV."""

    with pytest.raises(ValueError, match="comma"):
        VMCUpdateResult(applied=True, grad_norm=0.0, reason="a,b")


def test_an_empty_reason_is_rejected_rather_than_treated_as_absent() -> None:
    """Absent is `None`; empty string would be a reason that says nothing."""

    with pytest.raises(TypeError):
        VMCUpdateResult(applied=True, grad_norm=0.0, reason="")


def test_diagnostics_must_be_the_nominal_contract() -> None:
    """Structural look-alikes are refused; the type is the contract."""

    with pytest.raises(TypeError, match="UpdateDiagnostics"):
        VMCUpdateResult(
            applied=True,
            grad_norm=0.0,
            diagnostics=SimpleNamespace(as_metrics=lambda: {}),
        )
