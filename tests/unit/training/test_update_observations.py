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

import csv
import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tpen.logging.base import LogRecord
from tpen.logging.csv import CSV
from tpen.logging.jsonl import JSONL
from tpen.training.block_ng import (
    BlockDiagonalNaturalGradientUpdate,
    BlockNGPolicy,
    BlockNGTelemetry,
)
from tpen.training.qgt import DampingPolicy, SolveDiagnostics
from tpen.training.score_geometry import ScoreConventions
from tpen.training.spring import SPRINGPolicy, SPRINGTelemetry, SPRINGUpdate
from tpen.training.sr import SRPolicy, SRTelemetry, StochasticReconfigurationUpdate
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


def test_a_newline_cannot_reach_an_unquoted_csv_row() -> None:
    """A newline TERMINATES a CSV row, which is worse than a comma.

    `tpen/logging/csv.py` writes ``f"{step},{ns},{key},{value}\n"`` with no
    quoting. A comma in a value adds a phantom column; a NEWLINE ends the row
    outright and turns the remainder into a bogus record. The first version of
    this guard escaped commas and not newlines, which left the more damaging
    half of the same failure open.

    Exception text is the realistic source: a method description can carry a
    class name or a setting whose repr spans lines.
    """

    assert "\n" not in json_safe_scalar("line one\nline two")
    assert "\r" not in json_safe_scalar("line one\rline two")
    assert json_safe_scalar("line one\nline two") == "line one line two"
    # The comma guard must survive the newline guard being added.
    assert json_safe_scalar("a,b\nc") == "a;b c"


def test_a_csv_row_built_from_a_hostile_value_still_has_four_fields(tmp_path) -> None:
    """End-to-end control at the real sink, not just on the helper."""

    path = tmp_path / "hostile.csv"
    CSV(path).log(
        LogRecord(
            step=0,
            namespace="train",
            metrics={"hostile": json_safe_scalar("a,b\nc\rd")},
        )
    )
    rows = _strict_csv_rows(path)
    # Header plus exactly ONE data row. A newline would have produced two.
    assert len(rows) == 2, f"value broke the row structure: {rows!r}"
    assert len(rows[1]) == 4, f"row did not have four fields: {rows[1]!r}"


def test_a_tensor_valued_setting_keeps_its_exact_value() -> None:
    """A tensor `lr` must not be reported as rounded display text.

    PyTorch permits a tensor learning rate (the capturable and fused optimizer
    paths use it). `str(torch.tensor(0.001))` is ``"tensor(0.0010)"`` -- display
    text, already rounded -- so a description built from `str()` silently loses
    precision on exactly the setting a reader most wants to trust.
    """

    exact = 0.0001234567890123
    reported = json_safe_scalar(torch.tensor(exact, dtype=torch.float64))
    assert isinstance(reported, float), f"got display text: {reported!r}"
    assert reported == exact
    # And it must still be representable by the sinks.
    assert "," not in str(reported)


def test_a_tensor_lr_survives_carrier_settings_with_full_precision() -> None:
    """The same property through the real carrier-settings path."""

    model = build_connected_model()
    exact = 0.0001234567890123
    optimizer = torch.optim.SGD(model.parameters(), lr=exact)
    # Mirror what a capturable/fused optimizer does: a tensor in the group.
    optimizer.param_groups[0]["lr"] = torch.tensor(exact, dtype=torch.float64)
    settings = carrier_settings(optimizer)
    assert isinstance(settings["g0_lr"], float), f"got display text: {settings['g0_lr']!r}"
    assert settings["g0_lr"] == exact


def test_a_multi_element_tensor_setting_is_named_not_dumped() -> None:
    """A pathological setting must be labelled, never serialized into a row."""

    reported = json_safe_scalar(torch.ones(3, dtype=torch.float64))
    assert isinstance(reported, str)
    assert "," not in reported
    assert "\n" not in reported
    assert "3" in reported


def test_a_method_with_no_carrier_reports_unknown_not_zero_parameters() -> None:
    """An unknown parameter count must NOT be reported as 0.

    This slice already forbids fabricating zero for an unavailable observation,
    and gives the reason: zero is a legitimate value, so a fabricated zero is
    indistinguishable from a real measurement. `describe()` violated that in
    the same module, reporting `n_parameters=0` for a method that owns no
    carrier and therefore has no parameter domain to count.

    Zero and unknown are genuinely different here: a real zero-parameter
    binding is possible, and must stay distinguishable from "not reported".
    """

    description = _CustomMethodPredatingTheReasonContract().describe()
    assert description.n_parameters is None
    assert description.n_parameter_tensors is None
    # Not merely non-zero: the metrics must carry absence, not a number.
    metrics = description.as_metrics()
    assert metrics["update_method_n_parameters"] is None
    assert metrics["update_method_n_parameter_tensors"] is None


def test_a_real_parameter_count_is_still_reported_as_a_number() -> None:
    """Control for the test above: the available case must not regress to None."""

    model = build_connected_model()
    method = _sr_method(model)
    description = method.describe()
    assert isinstance(description.n_parameters, int)
    assert description.n_parameters > 0
    assert description.as_metrics()["update_method_n_parameters"] == description.n_parameters


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

    assert settings["g0_betas1"] == pytest.approx(0.9)
    assert settings["g0_betas2"] == pytest.approx(0.999)
    assert "g0_betas" not in settings
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

    # Parsed with the STRICT reader rather than split on commas: a naive split
    # is blind to an unbalanced double quote, which is a real hazard here and
    # was missed by the first version of this assertion.
    rows = _strict_csv_rows(path)
    assert rows[0] == ["step", "namespace", "key", "value"]
    for row in rows[1:]:
        assert len(row) == 4, f"row did not have four fields: {row!r}"


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
    for row in _strict_csv_rows(csv_path)[1:]:
        assert len(row) == 4, f"row did not have four fields: {row!r}"


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
        # SPRING and block NG were NOT covered by the first version of this
        # test, which is how the description-namespace collision (R5) survived
        # it. Partial coverage of a collision check is close to no coverage:
        # it certifies the pairs it happened to look at.
        _nonfinite_sr_telemetry(1.0),
        SPRINGTelemetry(
            applied=False, reason="declined", step=0, n_samples=1, n_finite_samples=0,
            n_parameters=1, energy_gradient_norm=1.0, update_direction_norm=0.0,
            applied_update_norm=0.0, trust_scale=1.0, history_norm=0.0,
            history_decay=0.9, history_advanced=False, diagnostics=None,
        ),
        BlockNGTelemetry(
            applied=True, step=0, n_samples=1, n_blocks=1,
            energy_gradient_norm=1.0, update_direction_norm=1.0,
            solve_dtype="torch.float64",
        ),
    ]
    for record in records:
        overlap = trainer_owned & set(record.as_metrics())
        assert not overlap, f"{type(record).__name__} collides on {sorted(overlap)}"

    # The DESCRIPTION shares the metrics dict with these records too, and was
    # not checked at all before.
    description_keys = set(_sr_method(model).describe().as_metrics())
    assert not (trainer_owned & description_keys)
    for record in records:
        clash = description_keys & set(record.as_metrics())
        assert not clash, f"{type(record).__name__} collides with description on {sorted(clash)}"


def _strict_csv_rows(path) -> list[list[str]]:
    """Parse a CSV with the STRICT reader, not by splitting on commas.

    WHY THIS HELPER EXISTS. Earlier tests here asserted
    ``len(row.split(",")) == 4``. That is a hand-rolled check which agrees with
    a real reader only for the hazards its author already thought of: it is
    blind to an unbalanced DOUBLE QUOTE, which makes `csv.reader` fail with
    "unexpected end of data" while a naive split happily returns four fields.
    Parsing with the real reader means a future hostile character fails the
    READER rather than passing a check that was only ever as good as its
    author's imagination.
    """

    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle, strict=True))


@pytest.mark.parametrize("hostile", ['a"b', '"unclosed', 'a,b', "a\nb", "a\r\nb", 'mix",\n"x'])
def test_any_hostile_value_still_yields_one_four_field_csv_row(tmp_path, hostile) -> None:
    """The guard is asserted against a STRICT reader, over several hazards.

    Parameterised rather than written once, because this guard was built up one
    reported symptom at a time -- comma, then newline, then quote -- which is
    precisely how a guard ends up covering less than its name promises.
    """

    path = tmp_path / f"hostile.csv"
    CSV(path).log(
        LogRecord(step=0, namespace="train", metrics={"hostile": json_safe_scalar(hostile)})
    )
    rows = _strict_csv_rows(path)
    assert len(rows) == 2, f"expected header + one row, got {rows!r}"
    assert len(rows[1]) == 4, f"row did not have four fields: {rows[1]!r}"


def test_every_builtin_description_is_strict_csv_readable(tmp_path) -> None:
    """Whole-description control, through the real sink and a real reader."""

    model = build_connected_model()
    path = tmp_path / "descriptions.csv"
    sink = CSV(path)
    for description in _describe_every_builtin_method(model):
        sink.log(
            LogRecord(step=0, namespace="train/update_method", metrics=description.as_metrics())
        )
    rows = _strict_csv_rows(path)
    assert len(rows) > 1
    for row in rows[1:]:
        assert len(row) == 4, f"row did not have four fields: {row!r}"


# --------------------------------------------------------------------------
# Non-finite diagnostics must not crash the sink that records them
# --------------------------------------------------------------------------


def _nonfinite_sr_telemetry(value: float) -> SRTelemetry:
    return SRTelemetry(
        applied=False,
        reason="nonfinite_update_direction",
        step=1,
        n_samples=4,
        n_finite_samples=4,
        n_parameters=8,
        energy_gradient_norm=value,
        update_direction_norm=value,
        applied_update_norm=0.0,
        trust_scale=1.0,
        diagnostics=None,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_sr_diagnostics_with_a_nonfinite_norm_survive_the_real_jsonl_sink(
    tmp_path, value
) -> None:
    """The record written to EXPLAIN a bad solve must not crash the run.

    This is the sharpest form of the hazard. `tpen/logging/jsonl.py` calls
    ``json.dumps(..., allow_nan=False)``, which RAISES on a non-finite float.
    SR's `as_metrics` used a bare ``float()``, and SR has
    ``nonfinite_update_direction`` as an EXPECTED skip reason -- so the single
    path most likely to produce a non-finite norm was the one that killed the
    run recording it.
    """

    path = tmp_path / "sr.jsonl"
    JSONL(path).log(
        LogRecord(step=1, namespace="train", metrics=_nonfinite_sr_telemetry(value).as_metrics())
    )
    payload = json.loads(path.read_text().strip())
    # Named, not crashed, and NOT silently turned into a number.
    assert payload["metrics"]["sr_energy_gradient_norm"] in ("nan", "inf", "-inf")
    assert payload["metrics"]["sr_reason"] == "nonfinite_update_direction"


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_spring_diagnostics_with_a_nonfinite_norm_survive_the_real_jsonl_sink(
    tmp_path, value
) -> None:
    """SPRING carries the same hazard on the same terms."""

    telemetry = SPRINGTelemetry(
        applied=False,
        reason="nonfinite_projected_history",
        step=1,
        n_samples=4,
        n_finite_samples=4,
        n_parameters=8,
        energy_gradient_norm=value,
        update_direction_norm=0.0,
        applied_update_norm=0.0,
        trust_scale=1.0,
        history_norm=value,
        history_decay=0.9,
        history_advanced=False,
        diagnostics=None,
    )
    path = tmp_path / "spring.jsonl"
    JSONL(path).log(LogRecord(step=1, namespace="train", metrics=telemetry.as_metrics()))
    payload = json.loads(path.read_text().strip())
    assert payload["metrics"]["spring_energy_gradient_norm"] in ("nan", "inf")
    assert payload["metrics"]["spring_history_norm"] in ("nan", "inf")


def test_nested_solve_diagnostics_with_a_nonfinite_eigenvalue_survive_jsonl(tmp_path) -> None:
    """The NESTED record is emitted too, and had the same raw-float defect.

    `SolveDiagnostics.as_metrics` documented itself as returning "JSON-safe"
    keys while using a bare ``float()``. An ill-conditioned or overflowing
    solve makes these values non-finite, and they reach the sink nested under
    ``sr_qgt_*`` / ``spring_qgt_*``.
    """

    diagnostics = SolveDiagnostics(
        space="parameter",
        shift=float("inf"),
        trace=float("nan"),
        n_modes=8,
        retained_modes=8,
        max_eigenvalue=float("inf"),
        min_retained_eigenvalue=0.0,
        dtype="torch.float64",
    )
    path = tmp_path / "qgt.jsonl"
    JSONL(path).log(LogRecord(step=1, namespace="train", metrics=diagnostics.as_metrics()))
    payload = json.loads(path.read_text().strip())
    assert payload["metrics"]["qgt_shift"] == "inf"
    assert payload["metrics"]["qgt_trace"] == "nan"
    assert payload["metrics"]["qgt_max_eigenvalue"] == "inf"
    # The finite one stays a real number: the guard must not flatten everything.
    assert payload["metrics"]["qgt_min_retained_eigenvalue"] == 0.0


def test_a_finite_diagnostics_record_is_numerically_unchanged() -> None:
    """CONTROL. Routing through the guard must not alter finite values.

    Without this, the fix above could have silently stringified every norm and
    the tests would still pass. Finite values must remain floats, with their
    exact magnitudes.
    """

    telemetry = _nonfinite_sr_telemetry(2.5)
    metrics = telemetry.as_metrics()
    assert metrics["sr_energy_gradient_norm"] == 2.5
    assert isinstance(metrics["sr_energy_gradient_norm"], float)
    assert metrics["sr_step"] == 1
    assert metrics["sr_applied"] is False


# --------------------------------------------------------------------------
# Reviewer round 2 findings R2, R4, R5
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hostile", ["line one\nline two", "line one\rline two", 'quote"d'])
def test_a_hostile_returned_reason_cannot_break_the_csv_record(tmp_path, hostile) -> None:
    """A method's reason must not be able to corrupt the trainer's own record.

    `reported_reason` is logged by the trainer directly. A custom method may
    return a reason assembled from an exception message, which can carry a
    newline -- and the unquoted CSV sink TERMINATES ITS ROW on one, so the
    trainer-owned record is broken even when the method's own diagnostics are
    clean.
    """

    result = VMCUpdateResult(applied=False, grad_norm=0.0, reason=hostile)
    path = tmp_path / "reason.csv"
    sink = CSV(path)
    sink.log(LogRecord(step=0, namespace="train", metrics={"update_reason": result.reported_reason}))
    sink.log(LogRecord(step=1, namespace="sentinel", metrics={"after": "still-readable"}))

    rows = _strict_csv_rows(path)
    for row in rows[1:]:
        assert len(row) == 4, f"row did not have four fields: {row!r}"
    # The sentinel proves the reason did not swallow following records.
    assert rows[-1] == ["1", "sentinel", "after", "still-readable"]


def test_a_hostile_reason_is_sanitized_at_emission_but_stored_verbatim() -> None:
    """Sanitize the EMITTED form; leave the method's own datum untouched.

    An earlier version RAISED on a comma in a reason. That let the observation
    layer abort the run it was only supposed to describe -- the same mistake,
    in the opposite direction, as letting a non-finite float reach a sink that
    raises on it. A caller reading the result object should still see exactly
    what the method said.
    """

    result = VMCUpdateResult(applied=False, grad_norm=0.0, reason="a,b\nc")
    assert result.reason == "a,b\nc"
    assert result.reported_reason == "a;b c"


def test_a_valid_reason_is_unchanged_by_sanitization() -> None:
    """CONTROL: ordinary reasons must pass through untouched."""

    result = VMCUpdateResult(applied=True, grad_norm=1.0, reason=UPDATE_REASON_APPLIED)
    assert result.reported_reason == UPDATE_REASON_APPLIED


def _single_parameter_method(shape: tuple[int, ...], fill: float) -> LegacyAutogradUpdate:
    parameter = torch.nn.Parameter(torch.full(shape, fill, dtype=torch.float64))
    return LegacyAutogradUpdate(
        optimizer=torch.optim.SGD([parameter], lr=0.1),
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
    )


def test_the_description_distinguishes_equal_size_parameter_layouts() -> None:
    """Counts alone cannot tell (4,) from (2, 2); the description must.

    The binding design requires "parameter count/layout identity". With only
    scalar and tensor counts emitted, two genuinely different models produced
    IDENTICAL description metrics, which defeats the point of describing what
    was actually constructed.
    """

    flat = _single_parameter_method((4,), 1.0).describe().as_metrics()
    square = _single_parameter_method((2, 2), 1.0).describe().as_metrics()
    other_flat = _single_parameter_method((4,), 7.0).describe().as_metrics()

    # Different LAYOUT must differ...
    assert flat != square
    assert flat["update_method_layout_fingerprint"] != square["update_method_layout_fingerprint"]
    # ...while different VALUES at the same layout must NOT, so the fingerprint
    # is an identity of the architecture and not of the run.
    assert flat["update_method_layout_fingerprint"] == other_flat["update_method_layout_fingerprint"]


def test_the_layout_fingerprint_is_sink_safe() -> None:
    """It is emitted to both sinks, so it must carry no separator."""

    fingerprint = _single_parameter_method((2, 2), 1.0).describe().layout_fingerprint
    assert fingerprint is not None
    for hostile in (",", "\n", "\r", '"'):
        assert hostile not in fingerprint


def test_carrier_metadata_cannot_overwrite_the_authoritative_identity() -> None:
    """A param_group key must never be able to rename the method.

    param_groups are ordinary dicts that anything may add keys to, and their
    contents are read live. While settings shared the description's namespace,
    a group carrying `class` or `n_parameters` OVERWROTE the authoritative
    value: the description object stayed truthful while the emitted metrics
    lied.

    This is the exact promise the description contract exists to make. The
    earlier test of that promise passed only because it set an ATTRIBUTE on
    the method; the reachable route was through the carrier's group dict. A
    guard is only as good as the channel it watches.
    """

    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    # The hostile keys, placed exactly where they are actually reachable.
    optimizer.param_groups[0]["class"] = "misleading-run-label"
    optimizer.param_groups[0]["n_parameters"] = 999
    optimizer.param_groups[0]["layout_fingerprint"] = "not-the-real-layout"
    method = LegacyAutogradUpdate(
        optimizer=optimizer,
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
    )

    description = method.describe()
    metrics = description.as_metrics()

    # The OBJECT was already truthful; the METRICS are what regressed.
    assert description.method_class.endswith("LegacyAutogradUpdate")
    assert metrics["update_method_class"].endswith("LegacyAutogradUpdate")
    assert metrics["update_method_class"] != "misleading-run-label"
    assert metrics["update_method_n_parameters"] == 4
    assert metrics["update_method_layout_fingerprint"] != "not-the-real-layout"
    # The hostile values are still REPORTED, just under a namespace that
    # cannot impersonate identity -- suppressing them would hide real state.
    assert metrics["update_method_setting_class"] == "misleading-run-label"
    assert metrics["update_method_setting_n_parameters"] == 999


def test_no_setting_key_can_ever_reach_an_authoritative_name() -> None:
    """Structural guard, not a sample of hostile keys.

    Enumerating bad key names would only ever cover the ones imagined. Every
    authoritative name is checked to not begin with the settings sub-namespace,
    which is what makes the separation hold for ANY key a param_group carries.
    """

    model = build_connected_model()
    description = _sr_method(model).describe()
    metrics = description.as_metrics()
    authoritative = {
        "update_method_class",
        "update_method_carrier_class",
        "update_method_n_parameters",
        "update_method_n_parameter_tensors",
        "update_method_layout_fingerprint",
        "update_method_forward_request",
        "update_method_norm_semantics",
    }
    assert authoritative <= set(metrics)
    for name in authoritative:
        assert not name.startswith("update_method_setting_")
    for key in metrics:
        if key.startswith("update_method_setting_"):
            assert key not in authoritative


# --------------------------------------------------------------------------
# Reviewer round 3: setting NAMES are a channel too
# --------------------------------------------------------------------------


def test_a_hostile_setting_name_cannot_corrupt_the_csv_row(tmp_path) -> None:
    """A setting NAME becomes a metric name, so it is a channel like any other.

    An earlier version sanitized setting VALUES and not KEYS -- half a channel,
    which is the same blind spot that produced three earlier findings. A
    param_group is an ordinary dict and nothing stops a key carrying a comma or
    a newline.
    """

    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    optimizer.param_groups[0]["hostile,name"] = 1.0
    optimizer.param_groups[0]["hostile\nname"] = 2.0
    method = LegacyAutogradUpdate(
        optimizer=optimizer,
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
    )

    path = tmp_path / "names.csv"
    sink = CSV(path)
    sink.log(
        LogRecord(step=0, namespace="train/update_method", metrics=method.describe().as_metrics())
    )
    sink.log(LogRecord(step=1, namespace="sentinel", metrics={"after": "still-readable"}))

    rows = _strict_csv_rows(path)
    for row in rows[1:]:
        assert len(row) == 4, f"row did not have four fields: {row!r}"
    assert rows[-1] == ["1", "sentinel", "after", "still-readable"]
    for row in rows[1:]:
        assert "," not in row[2] and "\n" not in row[2]


def test_a_group_zero_key_cannot_impersonate_the_group_count() -> None:
    """`n_param_groups` is seeded by the function and must stay authoritative.

    With group 0 unprefixed, a group-0 key literally named `n_param_groups`
    was merged straight over the count this function had just computed.
    """

    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    optimizer.param_groups[0]["n_param_groups"] = 999
    settings = carrier_settings(optimizer)

    assert settings["n_param_groups"] == 1
    # Still reported, under a name that cannot impersonate the real count.
    assert settings["g0_n_param_groups"] == 999


def test_a_group_zero_key_cannot_impersonate_another_group_setting() -> None:
    """Group 0 being unprefixed let it collide with later groups' real names.

    A group-0 key named `group1_lr` produced exactly the name group 1's own
    learning rate produced, so a per-layer rate could be reported as a value
    from a different group entirely.
    """

    a = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    b = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = torch.optim.SGD([{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.2}])
    optimizer.param_groups[0]["group1_lr"] = 999.0
    settings = carrier_settings(optimizer)

    # Each group's real rate is reported under its own unambiguous name.
    assert settings["g0_lr"] == pytest.approx(0.1)
    assert settings["g1_lr"] == pytest.approx(0.2)
    # The impostor is reported, but cannot be mistaken for group 1's rate.
    assert settings["g0_group1_lr"] == pytest.approx(999.0)


def test_an_unavoidable_name_collision_is_reported_not_silently_resolved() -> None:
    """Structure cannot remove every collision, so the rest must be VISIBLE.

    Keys are arbitrary strings: a group-0 key named `betas1` lands on the name
    produced by splitting a `betas` pair. Letting the later write win is how a
    description comes to report a value nobody set. The first value is kept and
    the clash is named, so a reader can see the map is incomplete.
    """

    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=0.1, betas=(0.9, 0.999))
    optimizer.param_groups[0]["betas1"] = -1.0
    settings = carrier_settings(optimizer)

    # The structural value survives; the impostor does not overwrite it.
    assert settings["g0_betas1"] == pytest.approx(0.9)
    assert "setting_name_collisions" in settings
    assert "g0_betas1" in settings["setting_name_collisions"]


def test_no_collision_marker_when_there_is_no_collision() -> None:
    """CONTROL: the marker must not appear on ordinary optimizers."""

    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    settings = carrier_settings(torch.optim.Adam([parameter], lr=0.1))
    assert "setting_name_collisions" not in settings


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
    assert description.n_parameters is None
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


def test_a_hostile_reason_is_accepted_at_construction_and_fixed_at_emission() -> None:
    """DELIBERATE REVERSAL of an earlier assertion, recorded as such.

    This test previously required `VMCUpdateResult(reason="a,b")` to RAISE.
    That was wrong in a way worth naming: it let the observation layer abort
    the run it was only supposed to describe. A method returning an awkward
    string -- a reason built from an exception message, say -- crashed
    training to protect a log file's column alignment.

    The replacement is not a relaxation. The hazard is closed at the EMISSION
    boundary instead, which is strictly stronger: construction-time rejection
    only ever covered reasons that passed through this constructor, while
    `reported_reason` covers every path to a sink. The construction check that
    remains is the one that cannot be fixed downstream -- a reason that is
    empty or not a string carries no information to sanitize.
    """

    result = VMCUpdateResult(applied=True, grad_norm=0.0, reason="a,b")
    assert result.reason == "a,b"
    assert result.reported_reason == "a;b"


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
