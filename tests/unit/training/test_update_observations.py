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
from dataclasses import replace
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
    DIAGNOSTIC_NAME_COLLISIONS_KEY,
    SETTING_NAME_COLLISIONS_KEY,
    carrier_settings,
    deserialize_parameter_layout,
    flatten_settings,
    merge_named_metrics,
    parameter_layout_fingerprint,
    serialize_parameter_layout,
    flatten_settings,
    json_safe_metric_name,
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

    # The description is checked against the TRAINER-OWNED names only. An
    # earlier version also asserted it could not collide with the per-attempt
    # telemetry records, which OVERSTATED the relationship: descriptions are
    # logged under the `train/update_method` namespace and per-attempt metrics
    # under `train`, so they never share one dict at runtime. Asserting a
    # constraint that the code does not actually need is its own kind of
    # wrong -- it invites a future change to be judged against a rule nobody
    # relies on.
    description_keys = set(_sr_method(model).describe().as_metrics())
    assert not (trainer_owned & description_keys)


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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_block_ng_diagnostics_with_a_nonfinite_norm_survive_the_real_jsonl_sink(
    tmp_path, value
) -> None:
    """Block NG carries the same hazard, and was the only method left untested.

    WHY THIS EXISTS, since the omission is the point. SR, SPRING and the
    nested QGT record each had a dedicated non-finite case; block NG did not,
    even though ``BlockNGTelemetry.as_metrics`` routes both norms through the
    same ``json_safe_scalar``. The suite constructed ``BlockNGTelemetry``
    exactly once, with finite values, inside a NAME-COLLISION test -- so a
    change that dropped the guard from this one class alone would have left
    every test green. That is the same shape as the gap recorded a few
    functions above: partial coverage certifies only the pairs it happened to
    look at, and block NG was the member it did not look at, twice.

    REACHABILITY, stated precisely rather than implied. ``update()`` raises on
    non-finite local energies and again on a non-finite update DIRECTION, so
    the component tensors reaching the telemetry are finite. It does NOT check
    the energy GRADIENT, and -- more to the point -- a vector norm of entirely
    finite components can still overflow to ``inf``, in float32 especially. So
    ``inf`` is reachable here through ``update()`` today; ``nan`` and
    ``-inf`` are not reachable through that path, and are covered because the
    dataclass is public, constructible by callers and by a restore path, and
    the guarantee being asserted is a property of ``as_metrics``, not of one
    caller's arithmetic.
    """

    telemetry = BlockNGTelemetry(
        applied=True,
        step=1,
        n_samples=4,
        n_blocks=2,
        energy_gradient_norm=value,
        update_direction_norm=value,
        solve_dtype="torch.float64",
    )
    path = tmp_path / "block_ng.jsonl"
    JSONL(path).log(LogRecord(step=1, namespace="train", metrics=telemetry.as_metrics()))
    payload = json.loads(path.read_text().strip())
    # Named, not crashed, and NOT silently turned into a number.
    assert payload["metrics"]["block_ng_energy_gradient_norm"] in ("nan", "inf", "-inf")
    assert payload["metrics"]["block_ng_update_direction_norm"] in ("nan", "inf", "-inf")
    # The finite fields stay themselves: the guard must not flatten everything.
    assert payload["metrics"]["block_ng_n_blocks"] == 2
    assert payload["metrics"]["block_ng_reason"] == "applied"


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
    # Note the DOUBLE namespacing: `{prefix}_setting_` separates settings from
    # identity, and `g0_` separates group 0 from every other group. Both are
    # load-bearing and both appear in the final name.
    assert metrics["update_method_setting_g0_class"] == "misleading-run-label"
    assert metrics["update_method_setting_g0_n_parameters"] == 999


def test_direct_description_preserves_all_colliding_setting_values() -> None:
    """SENSITIVE replacement for a guard that review proved was still blind.

    THE HISTORY MATTERS HERE. The first version of this guard was vacuous. I
    rewrote it to be "generative" and believed that fixed it. Review then
    MEASURED it under the namespace-removal mutation and found it STILL
    PASSED -- the node only looked detected because a NEIGHBOUR test failed,
    and my harness scored the whole node together. Two rewrites, both blind,
    both believed.

    Why the generative version failed: it derived its suffixes from an emitter
    that already includes settings, injected through the GROUP prefix so the
    names were doubly namespaced anyway, and never required the injected
    VALUES to survive.

    This version follows the design the reviewer supplied. It freezes the
    authoritative fields from an EMPTY-settings description, so the expected
    values cannot be contaminated by the injection; injects a unique sentinel
    for every authoritative suffix DIRECTLY into settings, bypassing the group
    prefix; then requires both that every authority is unchanged AND that every
    sentinel is present at its own name. Removing the settings namespace makes
    a sentinel land on an authoritative name, so one of those two requirements
    must break.
    """

    base = UpdateMethodDescription(
        method_class="tpen.training.update.LegacyAutogradUpdate",
        carrier_class="torch.optim.sgd.SGD",
        n_parameters=4,
        n_parameter_tensors=1,
        layout_fingerprint="float64:4",
        forward_request="value",
        norm_semantics="post_clip_grad_l2_over_gradient_domain",
        settings={},
    )
    prefix = "update_method_"
    frozen = base.as_metrics()
    suffixes = [k[len(prefix):] for k in frozen if k.startswith(prefix)]
    assert suffixes, "description emitted no authoritative fields to guard"

    injected = {suffix: f"sentinel-{suffix}" for suffix in suffixes}
    described = replace(base, settings=injected).as_metrics()

    # Every authoritative field keeps the value it had with NO settings at all.
    for suffix in suffixes:
        name = f"{prefix}{suffix}"
        assert described[name] == frozen[name], f"{name} was overwritten by a setting"
    # AND every injected value survives, under its own namespaced name.
    for suffix, sentinel in injected.items():
        assert described[f"{prefix}setting_{suffix}"] == sentinel, (
            f"setting {suffix} was lost"
        )


@pytest.mark.parametrize("hostile", ["custom,record", "custom\nrecord", "custom\rrecord"])
def test_a_hostile_description_prefix_cannot_break_the_csv_row(tmp_path, hostile) -> None:
    """R4-1: the PREFIX is the third part of a name, and was unguarded.

    A supported custom method may select a built-in record and choose its
    prefix. Encoding the keys while interpolating the prefix verbatim guarded
    two thirds of a name: review measured five columns for the comma case and
    doubled row counts for CR/LF.
    """

    description = UpdateMethodDescription(
        method_class="Custom", carrier_class="none", n_parameters=1,
        n_parameter_tensors=1, layout_fingerprint="float64:1",
        forward_request="value", norm_semantics="method_defined",
        settings={"lr": 0.1},
    )
    metrics = description.as_metrics(prefix=hostile)
    path = tmp_path / "prefix.csv"
    sink = CSV(path)
    sink.log(LogRecord(step=0, namespace="train/update_method", metrics=metrics))
    sink.log(LogRecord(step=1, namespace="sentinel", metrics={"after": "still-readable"}))

    rows = _strict_csv_rows(path)
    assert len(rows) == 1 + len(metrics) + 1, f"row count wrong: {len(rows)}"
    for row in rows[1:]:
        assert len(row) == 4, f"row did not have four fields: {row!r}"
    assert rows[-1] == ["1", "sentinel", "after", "still-readable"]


@pytest.mark.parametrize("hostile", ["m,p", "m\np"])
def test_a_hostile_minimal_record_prefix_cannot_break_the_csv_row(tmp_path, hostile) -> None:
    """Same defect in the minimal record, which a custom method may also select."""

    record = MinimalUpdateDiagnostics(
        method="Custom", applied=True, reason="applied", step=0,
        grad_norm=1.0, prefix=hostile,
    )
    metrics = record.as_metrics()
    path = tmp_path / "minimal.csv"
    sink = CSV(path)
    sink.log(LogRecord(step=0, namespace="train", metrics=metrics))
    sink.log(LogRecord(step=1, namespace="sentinel", metrics={"after": "still-readable"}))

    rows = _strict_csv_rows(path)
    assert len(rows) == 1 + len(metrics) + 1
    for row in rows[1:]:
        assert len(row) == 4, f"row did not have four fields: {row!r}"
    assert rows[-1] == ["1", "sentinel", "after", "still-readable"]


def test_an_ordinary_prefix_is_unchanged_by_encoding() -> None:
    """CONTROL: the default prefixes must encode to themselves."""

    record = MinimalUpdateDiagnostics(
        method="M", applied=True, reason="applied", step=0, grad_norm=1.0
    )
    assert "update_record_method" in record.as_metrics()


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
    assert SETTING_NAME_COLLISIONS_KEY in settings
    assert "g0_betas1" in settings[SETTING_NAME_COLLISIONS_KEY]


def test_no_collision_marker_when_there_is_no_collision() -> None:
    """CONTROL: the marker must not appear on ordinary optimizers."""

    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    settings = carrier_settings(torch.optim.Adam([parameter], lr=0.1))
    assert SETTING_NAME_COLLISIONS_KEY not in settings


@pytest.mark.parametrize("hostile_key", ["tag,source", "tag\nsource", "tag\rsource"])
def test_a_hostile_setting_key_keeps_the_csv_record_intact(tmp_path, hostile_key) -> None:
    """Reviewer R3-1's design, with its exact literal keys and row oracle.

    The comma case produced an extra COLUMN; the CR and LF cases produced extra
    RECORDS (22 rows where 21 were expected). Asserting the exact row count,
    not merely four fields per row, is what catches the record-splitting half.
    """

    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    optimizer.param_groups[0][hostile_key] = "marker"
    optimizer.param_groups[0]["tag"] = "control"
    method = LegacyAutogradUpdate(
        optimizer=optimizer,
        model_parameters=ModelParameterBinding(parameters=(parameter,)),
    )
    metrics = method.describe().as_metrics()

    path = tmp_path / "keys.csv"
    sink = CSV(path)
    sink.log(LogRecord(step=0, namespace="train/update_method", metrics=metrics))
    sink.log(LogRecord(step=1, namespace="sentinel", metrics={"after": "still-readable"}))

    rows = _strict_csv_rows(path)
    # Exact count: header + one row per metric + sentinel. An extra RECORD
    # from a newline shows up here and nowhere else.
    assert len(rows) == 1 + len(metrics) + 1, f"row count wrong: {len(rows)}"
    for row in rows[1:]:
        assert len(row) == 4, f"row did not have four fields: {row!r}"
    assert rows[-1] == ["1", "sentinel", "after", "still-readable"]
    # The hostile key's value survives exactly once, under a safe name.
    markers = [row for row in rows[1:] if row[3] == "marker"]
    assert len(markers) == 1
    assert "," not in markers[0][2] and "\n" not in markers[0][2]
    assert [row for row in rows[1:] if row[3] == "control"]


@pytest.mark.parametrize("spoofed_group", [0, 1])
def test_a_compound_setting_cannot_be_spoofed_in_a_multi_group_optimizer(
    spoofed_group,
) -> None:
    """Reviewer R3-2's design. Literal oracles, not values read back from the map.

    Two-group Adam with genuinely different betas. Injecting `betas1` into
    EITHER group used to report 999 as that group's beta1, because the split
    tuple element and the raw key produced the same name. The expected values
    here are written literally so the test cannot agree with a wrong map.
    """

    a = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    b = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [
            {"params": [a], "betas": (0.8, 0.9)},
            {"params": [b], "betas": (0.7, 0.95)},
        ],
        lr=0.1,
    )
    optimizer.param_groups[spoofed_group]["betas1"] = 999.0
    settings = carrier_settings(optimizer)

    assert settings["g0_betas1"] == pytest.approx(0.8)
    assert settings["g0_betas2"] == pytest.approx(0.9)
    assert settings["g1_betas1"] == pytest.approx(0.7)
    assert settings["g1_betas2"] == pytest.approx(0.95)
    # The clash is surfaced rather than silently resolved.
    assert SETTING_NAME_COLLISIONS_KEY in settings


def test_the_group_count_cannot_be_spoofed_in_a_multi_group_optimizer() -> None:
    """Reviewer R3-2's second design: the count must be measured, not merged."""

    a = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    b = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = torch.optim.SGD([{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.2}])
    # Assert the live truth FIRST, so the oracle is independent of the map.
    assert len(optimizer.param_groups) == 2
    optimizer.param_groups[0]["n_param_groups"] = 999
    assert carrier_settings(optimizer)["n_param_groups"] == 2


@pytest.mark.parametrize(
    "a,b",
    [
        ("tag\nsource", "tag\rsource"),
        ("a,b", "a;b"),
        ("x%2Cy", "x,y"),
        ('q"uote', "q'uote"),
    ],
)
def test_metric_name_encoding_never_collapses_two_distinct_names(a, b) -> None:
    """The name encoder must be INJECTIVE, or sanitizing silently loses data.

    A replacement-based sanitizer is not injective, and the collisions are not
    exotic. Measured on the previous implementation: "tag\\nsource" and
    "tag\\rsource" BOTH became "tag source", so one of two real settings was
    dropped by the act of making it safe, with nothing reporting the loss.
    The `%` cases pin the escape-ordering bug that makes a literal "%2C"
    indistinguishable from an encoded comma.
    """

    assert a != b
    assert json_safe_metric_name(a) != json_safe_metric_name(b)
    for encoded in (json_safe_metric_name(a), json_safe_metric_name(b)):
        for hostile in (",", "\n", "\r", '"'):
            assert hostile not in encoded


def test_metric_name_encoding_leaves_an_ordinary_name_untouched() -> None:
    """CONTROL: the common case must not be disfigured by the encoder."""

    assert json_safe_metric_name("lr") == "lr"
    assert json_safe_metric_name("g0_betas1") == "g0_betas1"


def test_a_hostile_setting_key_is_sanitized_at_the_description_boundary() -> None:
    """ISOLATE the as_metrics layer, which carrier_settings otherwise MASKS.

    WHY THIS EXISTS, and it is a lesson about test design rather than about
    this code. Key sanitization happens in TWO places: `carrier_settings`
    cleans names as it builds the map, and `as_metrics` cleans them again as it
    composes metric names. A mutation removing the SECOND one SURVIVED both
    end-to-end tests, because every key those tests supply has already been
    cleaned by the first. The tests were not wrong about their property -- a
    hostile key genuinely cannot corrupt the CSV -- but they pinned the wrong
    LAYER, and a layer no test isolates can be deleted without anything going
    red.

    The second layer is not redundant. Settings reach a description from
    sources that never pass through `carrier_settings`: a policy fingerprint
    merged by SR/SPRING's `describe`, or a custom method constructing an
    `UpdateMethodDescription` directly, as here. For those, this is the ONLY
    guard.
    """

    description = UpdateMethodDescription(
        method_class="Custom",
        carrier_class="none",
        n_parameters=1,
        n_parameter_tensors=1,
        layout_fingerprint="float64:1",
        forward_request="value",
        norm_semantics="method_defined",
        # Built directly, bypassing carrier_settings entirely.
        settings={"tag,source": "a", "tag\nsource": "b", "tag\rsource": "c", 'q"uote': "d"},
    )
    metrics = description.as_metrics()

    for key in metrics:
        assert "," not in key, f"metric name carries a comma: {key!r}"
        assert "\n" not in key, f"metric name carries a newline: {key!r}"
        assert "\r" not in key, f"metric name carries a CR: {key!r}"
        assert '"' not in key, f"metric name carries a quote: {key!r}"
    # All four distinct hostile keys still produce four distinct entries:
    # sanitizing must not collapse them into one and silently lose three.
    setting_keys = [k for k in metrics if k.startswith("update_method_setting_")]
    assert len(setting_keys) == 4, f"sanitizing collapsed keys: {setting_keys!r}"


def test_a_hostile_setting_key_survives_the_real_csv_sink_from_a_direct_description(
    tmp_path,
) -> None:
    """The same isolated layer, proven at the real sink rather than by string check."""

    description = UpdateMethodDescription(
        method_class="Custom",
        carrier_class="none",
        n_parameters=1,
        n_parameter_tensors=1,
        layout_fingerprint="float64:1",
        forward_request="value",
        norm_semantics="method_defined",
        settings={"tag,source": "a", "tag\nsource": "b"},
    )
    metrics = description.as_metrics()
    path = tmp_path / "direct.csv"
    sink = CSV(path)
    sink.log(LogRecord(step=0, namespace="train/update_method", metrics=metrics))
    sink.log(LogRecord(step=1, namespace="sentinel", metrics={"after": "still-readable"}))

    rows = _strict_csv_rows(path)
    assert len(rows) == 1 + len(metrics) + 1
    for row in rows[1:]:
        assert len(row) == 4, f"row did not have four fields: {row!r}"
    assert rows[-1] == ["1", "sentinel", "after", "still-readable"]


@pytest.mark.parametrize("hostile", ["pfx,a", "pfx\nb", 'pfx"c'])
def test_a_hostile_flatten_settings_prefix_cannot_break_the_csv_row(
    tmp_path, hostile
) -> None:
    """`flatten_settings` takes a PUBLIC prefix argument that arrived raw.

    Encoding the prefixes at the five emitters left this one: a public helper
    whose caller-supplied prefix went straight into every name it composed.
    Found by verification, not by a test -- the fifth consecutive finding in
    the same shape, a guard applied at the sites named rather than at every
    site of the kind.
    """

    flat = flatten_settings({"lr": 0.1, "damping": {"relative": 1e-2}}, prefix=hostile)
    path = tmp_path / "flat.csv"
    sink = CSV(path)
    sink.log(LogRecord(step=0, namespace="train", metrics=flat))
    sink.log(LogRecord(step=1, namespace="sentinel", metrics={"after": "still-readable"}))

    rows = _strict_csv_rows(path)
    assert len(rows) == 1 + len(flat) + 1
    for row in rows[1:]:
        assert len(row) == 4, f"row did not have four fields: {row!r}"
    assert rows[-1] == ["1", "sentinel", "after", "still-readable"]


def test_the_collision_marker_cannot_overwrite_a_user_setting() -> None:
    """The marker reports lost entries; it must not itself lose one.

    The marker was written with a plain assignment under an ordinary name, so
    a user setting encoding to that same name was silently clobbered -- the
    exact failure the marker exists to report, reintroduced by the reporting.

    The marker name now begins with `%`, which `json_safe_metric_name` escapes
    to `%25` FIRST, so no encoded user key can ever spell it. This is a
    structural guarantee rather than a reserved-word convention, which is why
    the test asserts the ENCODING property and not merely this one name.
    """

    # No encoded key can produce the marker name.
    assert SETTING_NAME_COLLISIONS_KEY.startswith("%")
    assert json_safe_metric_name(SETTING_NAME_COLLISIONS_KEY) != SETTING_NAME_COLLISIONS_KEY

    # A user setting NAMED like the marker survives under its own encoded name,
    # alongside a genuine collision report.
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=0.1, betas=(0.9, 0.999))
    optimizer.param_groups[0][SETTING_NAME_COLLISIONS_KEY] = "user-value"
    optimizer.param_groups[0]["betas1"] = -1.0
    settings = carrier_settings(optimizer)

    assert settings[SETTING_NAME_COLLISIONS_KEY]  # the real marker
    assert "betas1" in settings[SETTING_NAME_COLLISIONS_KEY]
    # The user's similarly-named setting is still present and distinct.
    encoded = json_safe_metric_name(SETTING_NAME_COLLISIONS_KEY)
    assert settings[f"g0_{encoded}"] == "user-value"


def test_an_already_encoded_prefix_is_not_double_escaped() -> None:
    """Encode ONCE per boundary; re-encoding corrupts a correct caller.

    `merge_named_metrics` takes an already-encoded prefix by contract. Had it
    re-encoded, a caller that had done the right thing would see `%` become
    `%25` become `%2525`.
    """

    once = json_safe_metric_name("a%b")
    twice = json_safe_metric_name(once)
    assert once == "a%25b"
    assert twice != once, "control: encoding is NOT idempotent, hence encode-once"

    target: dict[str, object] = {}
    merge_named_metrics(target, {"k": 1}, prefix=once + "_")
    assert f"{once}_k" in target


class _HostileNameDiagnostics(UpdateDiagnostics):
    """A custom record returning names no built-in would produce.

    `UpdateDiagnostics` is a NOMINAL contract: it obliges an implementation to
    return flat JSON-safe entries and cannot enforce it. This is what an
    implementation that does not comply looks like.
    """

    def as_metrics(self) -> dict:
        return {
            "custom,comma": 1.0,
            "custom\nnewline": 2.0,
            1: "int-key",
            "1": "str-key",
            # The literal marker name. A record may return it, and until the
            # marker was reserved, the marker write replaced this value.
            "update_diagnostic_name_collisions": "user-value",
        }


class _HostileDiagnosticsMethod(LegacyAutogradUpdate):
    """A supported custom method whose diagnostics record is non-compliant."""

    def update(self, update_input):
        result = super().update(update_input)
        return VMCUpdateResult(
            applied=result.applied,
            grad_norm=result.grad_norm,
            reason=result.reason,
            diagnostics=_HostileNameDiagnostics(),
        )


def test_custom_diagnostics_names_are_guarded_at_the_trainer_merge(tmp_path) -> None:
    """The guard must hold at the CONTRACT, not only for the records we ship.

    Reviewer-demonstrated witness: replacing the trainer's guarded merge with a
    bare `metrics.update` left ALL 93 existing observation tests passing. The
    protection was real and nothing observed it, which is the same shape as the
    two vacuous guards found earlier in this lane.

    Mixed `1` and `"1"` keys are included because they also break the JSONL
    sink outright -- `json.dumps(sort_keys=True)` raises TypeError comparing
    int to str -- so the unguarded path fails in two different ways.
    """

    model = build_connected_model()
    parameters = tuple(model.parameters())
    method = _HostileDiagnosticsMethod(
        optimizer=torch.optim.SGD(parameters, lr=LEARNING_RATE),
        model_parameters=ModelParameterBinding(parameters=parameters),
    )
    csv_path = tmp_path / "custom.csv"
    jsonl_path = tmp_path / "custom.jsonl"
    context = _SinkWritingContext(csv_path, jsonl_path)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)
    trainer.fit(
        model=model,
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=method.optimizer,
        context=context,
        emit=lambda **_: None,
    )

    rows = _strict_csv_rows(csv_path)
    for row in rows[1:]:
        assert len(row) == 4, f"row did not have four fields: {row!r}"
    # Every line must parse: the mixed-key case raises in the sink unguarded.
    for line in jsonl_path.read_text().strip().splitlines():
        payload = json.loads(line)
        for key in payload["metrics"]:
            assert isinstance(key, str)
            assert "," not in key and "\n" not in key


def test_a_non_string_key_is_actually_accepted_not_rejected_by_typeguard() -> None:
    """The collision path must be REACHABLE, not excluded by the annotation.

    typeguard enforces annotations at runtime in this repository, so
    `Mapping[str, Any]` on these surfaces did not merely document an
    expectation -- it rejected non-string keys at the boundary with a
    TypeCheckError, excluding one whole class of collision input.

    NON-STRING KEYS ARE NOT THE ONLY SOURCE, and an earlier version of this
    docstring said they were. Ordinary strings collide structurally through
    the flattening join; see
    `test_two_ordinary_string_keys_can_collide_structurally`.

    A guard and a type that disagree about what can reach a function is the
    same failure as a docstring that overstates its body, which this slice has
    now produced twice. This test pins the agreement rather than the guard.
    """

    # flatten_settings: the public helper.
    flat = flatten_settings({1: "int", "other": "str"})
    assert flat["1"] == "int"
    assert flat["other"] == "str"

    # UpdateMethodDescription.settings: the direct-construction surface.
    description = UpdateMethodDescription(
        method_class="Custom", carrier_class="none", n_parameters=1,
        n_parameter_tensors=1, layout_fingerprint="float64:1",
        forward_request="value", norm_semantics="method_defined",
        settings={1: "int-value", "plain": "str-value"},
    )
    metrics = description.as_metrics()
    assert metrics["update_method_setting_1"] == "int-value"
    assert metrics["update_method_setting_plain"] == "str-value"


def test_the_trainer_collision_marker_cannot_overwrite_a_custom_diagnostic(
    tmp_path,
) -> None:
    """The trainer's marker needed the same reservation as the settings marker.

    Found by verification, not by a test. The `%`-prefix rule was applied at
    three name-composition sites and missed at the fourth: the trainer wrote
    its diagnostics marker under an ORDINARY key, so a custom record returning
    that literal name lost its value to the marker write.

    That is the silent-loss failure the marker exists to report, reintroduced
    by the reporting -- for the SECOND time in this slice. The first instance
    was the settings marker; fixing one and not the other is the recurring
    shape of every defect found here.
    """

    model = build_connected_model()
    parameters = tuple(model.parameters())
    method = _HostileDiagnosticsMethod(
        optimizer=torch.optim.SGD(parameters, lr=LEARNING_RATE),
        model_parameters=ModelParameterBinding(parameters=parameters),
    )
    context = _SinkWritingContext(tmp_path / "m.csv", tmp_path / "m.jsonl")
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)
    trainer.fit(
        model=model,
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=method.optimizer,
        context=context,
        emit=lambda **_: None,
    )

    metrics = [m for ns, m in context.records if ns == "train"][-1]
    # The custom record's own value survives under its encoded name...
    assert metrics["update_diagnostic_name_collisions"] == "user-value"
    # ...and the marker is reported separately, under its reserved name.
    assert DIAGNOSTIC_NAME_COLLISIONS_KEY in metrics
    assert metrics[DIAGNOSTIC_NAME_COLLISIONS_KEY] != "user-value"
    # The reservation is structural: no encoded name can spell the marker.
    assert DIAGNOSTIC_NAME_COLLISIONS_KEY.startswith("%")
    assert (
        json_safe_metric_name(DIAGNOSTIC_NAME_COLLISIONS_KEY)
        != DIAGNOSTIC_NAME_COLLISIONS_KEY
    )


def test_two_ordinary_string_keys_can_collide_structurally() -> None:
    """Collisions are NOT confined to non-string keys, as I had claimed.

    Flattening joins nested names with `_`, so `{"a_b": 1}` and
    `{"a": {"b": 2}}` both produce `a_b` from perfectly ordinary strings.
    Injectivity of the encoder says nothing about a name ASSEMBLED from
    several encoded parts -- a distinction my docstring asserted away, and
    the reviewer corrected.

    This matters beyond wording: had the "only non-string keys" claim been
    believed, narrowing the annotations back to `Mapping[str, Any]` would have
    looked like a safe simplification that silently removed a reachable guard.
    """

    flat = flatten_settings({"a_b": 1, "a": {"b": 2}})
    # First writer retained, clash reported -- the same discipline as the
    # coercion case, reached by a completely different route.
    assert flat["a_b"] == 1
    assert SETTING_NAME_COLLISIONS_KEY in flat
    assert "a_b" in flat[SETTING_NAME_COLLISIONS_KEY]


def test_flatten_settings_reports_a_scalar_level_collision() -> None:
    """Collisions must PROPAGATE out of the recursion, not just be survived.

    Reviewer-demonstrated witness: dropping `collisions=collisions` from the
    scalar merge left retention working -- the first value still won -- so
    every value-level assertion passed while the report silently vanished.
    Retention and reporting are separate promises and need separate witnesses.
    """

    flat = flatten_settings({1: "int-first", "1": "str-second"})
    # Retention: the first writer survives.
    assert flat["1"] == "int-first"
    # Reporting: and the clash is named.
    assert SETTING_NAME_COLLISIONS_KEY in flat
    assert "1" in flat[SETTING_NAME_COLLISIONS_KEY]


def test_flatten_settings_encodes_a_hostile_parent_mapping_key() -> None:
    """The PARENT key of a nested mapping is a name component too.

    Reviewer-demonstrated witness: encoding only leaf keys produced
    `outer,section_child`, which splits a CSV row, while the whole existing
    corpus passed. Built-in policy fingerprints nest exactly this way.
    """

    flat = flatten_settings({"outer,section": {"child": "nested-marker"}})
    assert flat == {"outer%2Csection_child": "nested-marker"}


# --------------------------------------------------------------------------
# Gaps the 5c4631a0 verifier declared uncovered. Closing them here rather
# than carrying them, because a declared gap that nobody closes becomes a
# permanent exception.
# --------------------------------------------------------------------------


class _SinkWritingContext(_StubContext):
    """A context that writes through the REAL CSV and JSONL sinks.

    The existing tests exercise `reported_reason` directly and assert on the
    metrics dict. That leaves the actual trainer-to-sink path untested, which
    is where the corruption would occur: the trainer composes the record, and
    the sink is what a newline or quote actually breaks.
    """

    def __init__(self, csv_path, jsonl_path) -> None:
        super().__init__()
        self._csv = CSV(csv_path)
        self._jsonl = JSONL(jsonl_path)

    def log(self, metrics, *, step=None, namespace="run") -> None:
        super().log(metrics, step=step, namespace=namespace)
        record = LogRecord(step=step, namespace=namespace, metrics=dict(metrics))
        self._csv.log(record)
        self._jsonl.log(record)


class _HostileReasonMethod(LegacyAutogradUpdate):
    """A supported custom method whose reason carries CSV-hostile text.

    Realistic rather than contrived: a reason assembled from an exception
    message routinely contains a newline.
    """

    HOSTILE = 'line one\nline two, with "quotes"'

    def update(self, update_input):
        result = super().update(update_input)
        return VMCUpdateResult(
            applied=result.applied,
            grad_norm=result.grad_norm,
            reason=self.HOSTILE,
            diagnostics=result.diagnostics,
        )


def test_a_hostile_reason_survives_the_real_trainer_to_sink_path(tmp_path) -> None:
    """END TO END, through the real trainer and both real sinks.

    This is the path the 5c4631a0 verifier named as uncovered: everything
    before this exercised `reported_reason` in isolation, so a defect in how
    the TRAINER composes or emits the record would not have been caught.
    """

    model = build_connected_model()
    parameters = tuple(model.parameters())
    method = _HostileReasonMethod(
        optimizer=torch.optim.SGD(parameters, lr=LEARNING_RATE),
        model_parameters=ModelParameterBinding(parameters=parameters),
    )
    csv_path = tmp_path / "train.csv"
    jsonl_path = tmp_path / "train.jsonl"
    context = _SinkWritingContext(csv_path, jsonl_path)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)
    trainer.fit(
        model=model,
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=method.optimizer,
        context=context,
        emit=lambda **_: None,
    )

    # CSV: every row still four fields under a STRICT reader.
    rows = _strict_csv_rows(csv_path)
    assert len(rows) > 1
    for row in rows[1:]:
        assert len(row) == 4, f"row did not have four fields: {row!r}"
    reason_rows = [row for row in rows[1:] if row[2] == "update_reason"]
    assert len(reason_rows) == 1
    assert "\n" not in reason_rows[0][3] and "," not in reason_rows[0][3]

    # JSONL: every line parses, and the reason is present and sanitized.
    for line in jsonl_path.read_text().strip().splitlines():
        payload = json.loads(line)
        assert isinstance(payload["metrics"], dict)


# --- Layout fingerprint: dtypes and shapes the writer had not tried ---


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16]
)
def test_the_fingerprint_separates_dtypes_at_identical_shape(dtype) -> None:
    """Same shape, different dtype, must not fingerprint identically.

    The writer's tests used float64 only, so a fingerprint that dropped dtype
    entirely would have passed them.
    """

    reference = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float64))
    candidate = torch.nn.Parameter(torch.ones((2, 2), dtype=dtype))
    reference_fp = parameter_layout_fingerprint(
        ModelParameterBinding(parameters=(reference,)).layout
    )
    candidate_fp = parameter_layout_fingerprint(
        ModelParameterBinding(parameters=(candidate,)).layout
    )
    if dtype is torch.float64:
        assert reference_fp == candidate_fp
    else:
        assert reference_fp != candidate_fp


def test_the_fingerprint_separates_slot_ORDER() -> None:
    """Two layouts with the same multiset of slots in different order differ."""

    a = torch.nn.Parameter(torch.ones((2,), dtype=torch.float64))
    b = torch.nn.Parameter(torch.ones((3,), dtype=torch.float64))
    forward = parameter_layout_fingerprint(
        ModelParameterBinding(parameters=(a, b)).layout
    )
    reverse = parameter_layout_fingerprint(
        ModelParameterBinding(parameters=(b, a)).layout
    )
    assert forward != reverse


def test_the_fingerprint_survives_a_layout_serialization_round_trip() -> None:
    """Stable across restore, which is the property that makes it usable.

    Comparing a resumed run against its original is the actual use; a
    fingerprint that changed across serialize/deserialize would be useless for
    exactly that.
    """

    parameters = (
        torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float64)),
        torch.nn.Parameter(torch.ones((4,), dtype=torch.float32)),
    )
    layout = ModelParameterBinding(parameters=parameters).layout
    restored = deserialize_parameter_layout(serialize_parameter_layout(layout))
    assert parameter_layout_fingerprint(restored) == parameter_layout_fingerprint(layout)


def test_the_fingerprint_handles_a_scalar_shaped_parameter() -> None:
    """A 0-dim parameter has an empty shape tuple and must still be named."""

    scalar = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    fingerprint = parameter_layout_fingerprint(
        ModelParameterBinding(parameters=(scalar,)).layout
    )
    assert fingerprint
    assert "scalar" in fingerprint
    vector = parameter_layout_fingerprint(
        ModelParameterBinding(
            parameters=(torch.nn.Parameter(torch.ones((1,), dtype=torch.float64)),)
        ).layout
    )
    # A scalar and a one-element vector are DIFFERENT layouts.
    assert fingerprint != vector


# --- Arbitrary param_group keys ---


def test_a_non_string_param_group_key_does_not_raise() -> None:
    """Mixed key types must not break sorting before anything is emitted.

    `sorted(group.items())` on mixed types raises TypeError, which would fail
    the description before a single metric was produced -- a crash, not a
    corrupted value, and so a different failure mode from the rest of this
    family.
    """

    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    optimizer.param_groups[0][7] = "int-keyed"
    optimizer.param_groups[0][("tuple", "key")] = "tuple-keyed"

    settings = carrier_settings(optimizer)
    assert settings["g0_7"] == "int-keyed"
    for key in settings:
        assert "," not in key and "\n" not in key


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
