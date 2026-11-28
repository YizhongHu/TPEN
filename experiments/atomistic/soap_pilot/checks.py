"""Offline, fail-closed Rung-1 and Rev3 acceptance checks.

This module consumes named evidence supplied by the producer.  It never walks
protected output trees and never treats scheduler completion as a product
verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class Verdict(str, Enum):
    """Evidence classification used by every checker."""

    PASS = "PASS"
    PRODUCT_FAILURE = "PRODUCT_FAILURE"
    HARNESS_INVALID = "HARNESS_INVALID"
    UNMEASURED = "UNMEASURED"


@dataclass(frozen=True)
class CheckResult:
    """One named check result with explicit evidence and limits."""

    name: str
    verdict: Verdict
    detail: str
    measured: bool = True
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether this check is a measured pass."""

        return self.verdict is Verdict.PASS and self.measured


def _missing(name: str, detail: str) -> CheckResult:
    return CheckResult(name, Verdict.UNMEASURED, detail, measured=False)


def _invalid(name: str, detail: str) -> CheckResult:
    return CheckResult(name, Verdict.HARNESS_INVALID, detail)


def _failure(name: str, detail: str, evidence: Mapping[str, Any] | None = None) -> CheckResult:
    return CheckResult(name, Verdict.PRODUCT_FAILURE, detail, evidence=evidence or {})


def _ok(name: str, detail: str, evidence: Mapping[str, Any] | None = None) -> CheckResult:
    return CheckResult(name, Verdict.PASS, detail, evidence=evidence or {})


def _compare(expected: Any, actual: Any, name: str, field: str) -> CheckResult | None:
    if actual is None:
        return _missing(name, f"missing {field}")
    if expected != actual:
        return _failure(name, f"{field} mismatch: expected={expected!r}, actual={actual!r}")
    return None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def check_resume_metric_keys(uninterrupted: Mapping[Any, Mapping[str, Any]] | None,
                             resumed: Mapping[Any, Mapping[str, Any]] | None) -> CheckResult:
    """Check exact logical state and metric-key equality at every overlap."""

    # Rev3 §4.1 sentence: “For every overlapping step, require exact equality
    # of the logical state and metric-key sets before comparing metric values.”
    name = "resume-logical-state-and-metric-keys"
    if uninterrupted is None or resumed is None:
        return _missing(name, "uninterrupted and resumed metric evidence is required")
    overlap = sorted(set(uninterrupted) & set(resumed))
    if not overlap:
        return _missing(name, "no overlapping steps were supplied")
    for step in overlap:
        left, right = uninterrupted[step], resumed[step]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return _invalid(name, f"step {step!r} metric record is not a mapping")
        left_keys, right_keys = set(left), set(right)
        if left_keys != right_keys:
            return _failure(name, f"step {step}: uninterrupted-only={sorted(left_keys-right_keys)!r}; resumed-only={sorted(right_keys-left_keys)!r}")
        if left.get("logical_state") != right.get("logical_state"):
            return _failure(name, f"step {step}: logical state differs")
        for key in left_keys - {"logical_state"}:
            if left[key] != right[key]:
                return _failure(name, f"step {step}: metric {key!r} differs")
    return _ok(name, f"exact logical state and metric keys at {len(overlap)} overlapping steps", {"steps": overlap})


def check_resume_parity(
    uninterrupted: Mapping[Any, Mapping[str, Any]] | None,
    resumed: Mapping[Any, Mapping[str, Any]] | None,
    *,
    tolerances: Mapping[str, tuple[float, float]] | None = None,
    discrete_fields: Sequence[str] = ("cursor", "step", "state"),
) -> CheckResult:
    """Check finite-first resume parity with exact state and field tolerances.

    # Rev3 §3.1 sentence: “Same-chunk train_resume ... with identical N,
    # chunk, max_steps and policies”; Rev3 §3.1 requires finite checks before
    # field-specific atol/rtol comparisons and exact discrete state values.
    """

    name = "resume-finite-state-cursor-tolerance"
    if uninterrupted is None or resumed is None:
        return _missing(name, "uninterrupted and resumed evidence is required")
    overlap = sorted(set(uninterrupted) & set(resumed))
    if not overlap:
        return _missing(name, "no overlapping steps were supplied")
    tolerances = tolerances or {}
    for step in overlap:
        left, right = uninterrupted[step], resumed[step]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return _invalid(name, f"step {step!r} evidence is not a mapping")
        if set(left) != set(right):
            return _failure(name, f"step {step}: metric key sets differ; uninterrupted-only={sorted(set(left)-set(right))!r}; resumed-only={sorted(set(right)-set(left))!r}")
        for key in left:
            a, b = left[key], right[key]
            if key in discrete_fields:
                if a != b:
                    return _failure(name, f"step {step}: discrete field {key!r} differs")
                continue
            if isinstance(a, (int, float)) or isinstance(b, (int, float)):
                if not (_finite(a) and _finite(b)):
                    return _failure(name, f"step {step}: metric {key!r} is non-finite")
                atol, rtol = tolerances.get(key, (0.0, 0.0))
                if abs(float(a) - float(b)) > atol + rtol * max(abs(float(a)), abs(float(b))):
                    return _failure(name, f"step {step}: metric {key!r} exceeds tolerance", {"atol": atol, "rtol": rtol})
            elif a != b:
                return _failure(name, f"step {step}: metric {key!r} differs")
    return _ok(name, f"finite-first resume parity passed at {len(overlap)} overlapping steps", {"steps": overlap, "tolerances": tolerances})


def check_parent_join(expected: Mapping[str, Any] | None, actual: Mapping[str, Any] | None) -> CheckResult:
    """Check an eval parent against the actual singleton train document.

    # P0 §6 sentence: “P2 ... recomputes its row ID/full hashes ... A foreign
    # parent ... refuses BEFORE launch.”
    """

    name = "parent-join"
    if expected is None or actual is None:
        return _missing(name, "expected and actual parent evidence are required")
    for field in ("campaign_id", "invocation_id", "manifest_hash", "train_row_hash", "row_id", "source_sha", "rendered_config_sha256"):
        mismatch = _compare(expected.get(field), actual.get(field), name, field)
        if mismatch is not None:
            return mismatch
    return _ok(name, "parent join matches its singleton source and rendered identity")


def check_profile(expected: Mapping[str, Any] | None, actual: Mapping[str, Any] | None) -> CheckResult:
    """Reject profile/configuration drift, including missing or extra fields.

    # P0 §6 sentence: “Each entry binds source, full row_id/full row hash,
    # rendered bytes, input document hashes, run_id/results_root and diagnostic
    # basis.”
    """

    name = "profile-identity"
    if expected is None or actual is None:
        return _missing(name, "expected and actual profile evidence are required")
    if set(expected) != set(actual):
        return _failure(name, f"profile key sets differ: expected-only={sorted(set(expected)-set(actual))!r}; actual-only={sorted(set(actual)-set(expected))!r}")
    for field, value in expected.items():
        mismatch = _compare(value, actual.get(field), name, field)
        if mismatch is not None:
            return mismatch
    return _ok(name, "profile identity and all derived fields match")


def check_config_identity(config: Mapping[str, Any] | None, *, rendered_bytes: bytes | None = None,
                          expected_sha256: str | None = None) -> CheckResult:
    """Verify the checked rendered bytes and resolved mapping identity.

    # P0 §4.1 sentence: “Read the named rendered file once into bytes, hash
    # those bytes and parse that same in-memory buffer.”
    """

    name = "rendered-config-identity"
    if config is None or rendered_bytes is None or expected_sha256 is None:
        return _missing(name, "config, checked bytes, and expected digest are required")
    actual = hashlib.sha256(rendered_bytes).hexdigest()
    if actual != expected_sha256:
        return _failure(name, "rendered bytes do not match the admitted digest")
    if not isinstance(config, Mapping) or not config:
        return _invalid(name, "resolved rendered config is not a non-empty mapping")
    return _ok(name, "rendered bytes and resolved mapping are admitted", {"sha256": actual})


def check_checkpoint(expected_step: int | None, evidence: Mapping[str, Any] | None) -> CheckResult:
    """Require the named complete publication and model-byte identity.

    # Rev3 §4.2 sentence: “For each checkpoint read the exact scheduled step's
    # manifest.json and the named publications.jsonl/publication_receipts.jsonl/latest.json.”
    """

    name = "checkpoint-publication"
    if expected_step is None or evidence is None:
        return _missing(name, "expected checkpoint step and publication evidence are required")
    if evidence.get("step") != expected_step:
        return _failure(name, "checkpoint step is not the scheduled step")
    if evidence.get("complete") is not True:
        return _missing(name, "checkpoint publication is not COMPLETE")
    if not evidence.get("publication_id") or not evidence.get("model_sha256"):
        return _missing(name, "checkpoint publication/model identity is missing")
    files = evidence.get("files")
    if not isinstance(files, Mapping) or "model.pt" not in files or "COMPLETE" not in files:
        return _missing(name, "checkpoint manifest does not name model.pt and COMPLETE")
    return _ok(name, "scheduled checkpoint publication is complete and named")


def check_hash(expected: str | None, actual: str | None, *, name: str = "hash-identity") -> CheckResult:
    """Compare a producer digest without treating shape-only hashes as proof."""

    # P0 §6 sentence: “A claimed matching string triplet is not proof.”
    if expected is None or actual is None:
        return _missing(name, "both expected and actual hashes are required")
    if not isinstance(expected, str) or not isinstance(actual, str):
        return _invalid(name, "hash identities must be strings")
    return _ok(name, "hash identity matches") if expected == actual else _failure(name, "hash identity mismatch")


def check_seed_domain(rows: Sequence[Mapping[str, Any]] | None, *, matched_groups: Sequence[set[str]] = ()) -> CheckResult:
    """Check seed/domain collisions while preserving explicit matched groups."""

    # P0 §6 sentence: “A repeated seed outside declared matched membership is
    # invalid, not silently accepted because documents differ.”
    name = "seed-domain"
    if rows is None:
        return _missing(name, "seed-domain rows are required")
    seen: dict[tuple[Any, Any], str] = {}
    groups = [set(group) for group in matched_groups]
    for row in rows:
        invocation = row.get("invocation_id", row.get("run_id"))
        key = (row.get("seed"), row.get("sampler_seed", row.get("stream")))
        prior = seen.get(key)
        if prior is not None and not any({prior, invocation} <= group for group in groups):
            return _failure(name, f"seed/domain collision between {prior!r} and {invocation!r}")
        seen[key] = invocation
    return _ok(name, "seed/domain membership has no undeclared collision")


def check_metric_integrity(metrics: Mapping[str, Any] | None) -> CheckResult:
    """Reject non-finite persisted JSONL metric values.

    # Rev3 §3.1 sentence: “All floating model tensors must retain the
    # configured float64 dtype. Compare EVERY non-floating model state value exactly.”
    # JSONL records contain scalar values without dtype metadata, so this
    # consumer checks finiteness only; model-tensor dtype is enforced by
    # ``check_model_float64`` below.
    """

    name = "metric-finite"
    if metrics is None:
        return _missing(name, "metric evidence is required")
    if not isinstance(metrics, Mapping):
        return _invalid(name, "metric evidence must be a mapping")

    def first_nonfinite(value: Any, path: str) -> str | None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                found = first_nonfinite(child, f"{path}.{key}")
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                found = first_nonfinite(child, f"{path}[{index}]")
                if found is not None:
                    return found
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                return path
        return None

    nonfinite = first_nonfinite(metrics, "metrics")
    if nonfinite is not None:
        return _failure(name, f"metric {nonfinite!r} is non-finite")
    return _ok(name, "JSONL metrics are finite; scalar dtype is not claimed")


def check_state_cursor_tolerance(actual: Mapping[str, Any] | None, expected: Mapping[str, Any] | None,
                                 *, tolerances: Mapping[str, tuple[float, float]] | None = None) -> CheckResult:
    """Check exact discrete state/cursor and declared numeric tolerances."""

    # Rev3 §3.1 sentence: “finite/shape/dtype/discrete checks ... no
    # tolerances may be widened after observation.”
    if actual is None or expected is None:
        return _missing("state-cursor-tolerance", "state and expected evidence are required")
    tolerances = tolerances or {}
    for key, expected_value in expected.items():
        if key not in actual:
            return _missing("state-cursor-tolerance", f"missing field {key!r}")
        actual_value = actual[key]
        if key in {"state", "cursor", "step", "dtype", "shape"}:
            if actual_value != expected_value:
                return _failure("state-cursor-tolerance", f"discrete field {key!r} drifted")
            continue
        if isinstance(expected_value, (int, float)):
            if not (_finite(actual_value) and _finite(expected_value)):
                return _failure("state-cursor-tolerance", f"field {key!r} is non-finite")
            atol, rtol = tolerances.get(key, (0.0, 0.0))
            if abs(float(actual_value)-float(expected_value)) > atol + rtol * max(abs(float(actual_value)), abs(float(expected_value))):
                return _failure("state-cursor-tolerance", f"field {key!r} exceeds tolerance")
        elif actual_value != expected_value:
            return _failure("state-cursor-tolerance", f"field {key!r} differs")
    return _ok("state-cursor-tolerance", "state/cursor exact and numeric fields within declared tolerances")


def check_callbacks(config: Mapping[str, Any] | None, *, family: str, train_timing_synchronize: bool) -> CheckResult:
    """Validate the profile-specific callback set from a rendered config.

    # P0 callback sentence: “OptimizerState Adam/SOAP ONLY; ...
    # TrainPhaseTiming/TrainStepTiming accelerator_synchronize=true for R2/100,
    # false for R1/target and R2/20; RunTiming true throughout.”
    """

    name = "profile-callbacks"
    if config is None or not isinstance(config.get("callbacks"), list):
        return _missing(name, "rendered callback list is required")
    callbacks = config["callbacks"]
    targets = [item.get("_target_") for item in callbacks if isinstance(item, Mapping)]
    has_optimizer_state = "tpen.callback.OptimizerState" in targets
    if has_optimizer_state != (family in {"adam", "soap"}):
        return _failure(name, "OptimizerState callback set does not match the optimizer family")
    for target in ("tpen.callback.TrainPhaseTiming", "tpen.callback.TrainStepTiming"):
        matching = [item for item in callbacks if isinstance(item, Mapping) and item.get("_target_") == target]
        if len(matching) != 1 or matching[0].get("accelerator_synchronize") is not train_timing_synchronize:
            return _failure(name, f"{target} synchronization policy drifted")
    run_timing = [item for item in callbacks if isinstance(item, Mapping) and item.get("_target_") == "tpen.callback.RunTiming"]
    if len(run_timing) != 1 or run_timing[0].get("accelerator_synchronize", True) is not True:
        return _failure(name, "RunTiming must synchronize throughout")
    return _ok(name, "rendered callback set matches the profile policy")


def check_eval_metrics(evidence: Mapping[str, Any] | None) -> CheckResult:
    """Reject metrics not configured by the frozen evaluation recipe.

    # P0 §4.2 sentence: “That task's generator ... contains exactly the
    # single cfg.evaluation_tasks.mcmc_energy mapping.”
    """

    name = "configured-evaluation-metrics"
    if evidence is None:
        return _missing(name, "evaluation metric evidence is required")
    configured = evidence.get("configured_metric_keys")
    observed = evidence.get("metric_keys")
    if not isinstance(configured, (list, tuple, set)) or not isinstance(observed, (list, tuple, set)):
        return _missing(name, "configured and observed metric-key sets are required")
    if set(configured) != set(observed):
        return _failure(name, f"evaluation metric keys differ: configured-only={sorted(set(configured)-set(observed))!r}; observed-only={sorted(set(observed)-set(configured))!r}")
    return _ok(name, "evaluation metrics are exactly the configured set")


def check_step_zero(values: Mapping[str, Any] | None) -> CheckResult:
    """Apply amendment-1 step-zero energy, variance, gradient, and SHA rules."""

    # Amendment-1 sentence: “Step-0 energy and energy_variance: EXACT equality
    # across all families ... grad_norm: EXACT equality within the same
    # gradient construction ... across constructions: |x-y| <= ...; model SHA
    # identity across all families.”
    name = "step-zero-amendment-1"
    if values is None:
        return _missing(name, "step-zero evidence is required")
    required = {"energy", "energy_variance", "grad_norm", "model_sha256", "construction"}
    if not required <= set(values):
        return _missing(name, f"missing step-zero fields: {sorted(required-set(values))}")
    families = values.get("energy")
    variance = values.get("energy_variance")
    gradients = values.get("grad_norm")
    shas = values.get("model_sha256")
    if not all(isinstance(x, Mapping) for x in (families, variance, gradients, shas)):
        return _invalid(name, "step-zero fields must be family mappings")
    expected_families = {"adam", "soap", "minsr64", "minsr128"}
    family_fields = {
        "energy": families,
        "energy_variance": variance,
        "grad_norm": gradients,
        "model_sha256": shas,
    }
    for field, family_values in family_fields.items():
        keys = set(family_values)
        if keys != expected_families:
            missing = sorted(expected_families - keys)
            extra = sorted(keys - expected_families)
            if missing and not extra:
                return _missing(name, f"step-zero {field} is missing families: {missing}")
            return _invalid(name, f"step-zero {field} has malformed family set: missing={missing}; extra={extra}")
    expected_constructions = {"autograd", "qgt"}
    groups = values["construction"]
    if not isinstance(groups, Mapping):
        return _invalid(name, "construction groups must name two gradient values")
    construction_keys = set(groups)
    if construction_keys != expected_constructions:
        return _invalid(name, f"construction key set must be exactly {sorted(expected_constructions)}")
    if not all(isinstance(v, (list, tuple)) and len(v) == 2 for v in groups.values()):
        return _invalid(name, "construction groups must name two gradient values")
    if not all(_finite(v) for v in [*families.values(), *variance.values(), *gradients.values()]):
        return _failure(name, "step-zero metric is non-finite")
    if len(set(families.values())) != 1 or len(set(variance.values())) != 1:
        return _failure(name, "step-zero energy or energy_variance is not exactly equal")
    if len(set(shas.values())) != 1:
        return _failure(name, "step-zero model SHA differs across families")
    for label, pair in groups.items():
        a, b = pair
        if not (_finite(a) and _finite(b)):
            return _failure(name, f"{label} gradient construction is non-finite")
        if label in {"autograd", "qgt"} and a != b:
            return _failure(name, f"{label} within-construction grad_norm differs exactly")
    all_gradients = list(gradients.values())
    scale = max(abs(float(x)) for x in all_gradients)
    limit = 1e-10 + 1e-8 * scale
    if max(all_gradients) - min(all_gradients) > limit:
        return _failure(name, f"cross-construction grad_norm exceeds atol/rtol: limit={limit}")
    return _ok(name, "step-zero amendment-1 criteria passed")


def check_model_float64(config: Mapping[str, Any] | None, checkpoints: Mapping[Any, Iterable[Any]] | None) -> CheckResult:
    """Check configured dtype and every named model checkpoint by value."""

    # Rev3 §4.2 sentence: “Derive configured dtype and checkpoint steps from
    # each rendered run; require all named checkpoint model tensors to be float64.”
    # This is the model-tensor dtype gate; JSONL scalar metrics have no dtype
    # metadata and are checked for finiteness only by ``check_metric_integrity``.
    name = "rendered-config-model-float64"
    if config is None or checkpoints is None:
        return _missing(name, "rendered config and named checkpoint tensors are required")
    runtime = config.get("runtime")
    steps = config.get("checkpoint_steps")
    if not isinstance(runtime, Mapping) or runtime.get("dtype") != "float64":
        return _failure(name, "rendered runtime.dtype is not float64")
    if not isinstance(steps, list) or not steps:
        return _missing(name, "rendered checkpoint_steps is absent")
    for step in steps:
        if step not in checkpoints:
            return _missing(name, f"named checkpoint {step!r} is absent")
        tensors = checkpoints[step]
        if not tensors:
            return _missing(name, f"checkpoint {step!r} has no model tensors")
        for tensor in tensors:
            dtype = tensor.get("dtype") if isinstance(tensor, Mapping) else getattr(tensor, "dtype", None)
            if str(dtype).replace("torch.", "") != "float64":
                return _failure(name, f"checkpoint {step!r} contains non-float64 model tensor")
    return _ok(name, "configured dtype and all named model checkpoint tensors are float64", {"steps": steps})


def check_disconnected_optimizer_state(
    records: Sequence[Any] | None,
    *,
    groups: Iterable[str] | Mapping[str, Any] | None = None,
    terminal_step: int | None = None,
    expected_tensors: int = 16,
    group_inventory: Mapping[str, Any] | None = None,
    empty_groups: Iterable[str] | None = None,
) -> CheckResult:
    """Aggregate terminal-checkpoint tensor-count evidence across groups.

    # Protocol sentence: “n_no_optimizer_state is a TENSOR count (16 tensors =
    # 492,000 elements),” not an element count.  The 492,000-element value is
    # architecture context, not a runtime assertion; the consumer therefore
    # checks only the tensor count and ignores any element-context field.

    ``terminal_step`` is the named checkpoint step, not the JSONL envelope
    step.  ``OptimizerState`` receives ``event.iteration.step`` from the
    trainer (``trainer.py:1358-1361,1525-1526``), while checkpoint publication
    uses the resume cursor ``next_iteration`` (``checkpoint.py:274-311``).
    Therefore a completed checkpoint at ``N`` emits optimizer metrics at
    ``N - 1``.  The checker accepts an exact-step legacy record only when no
    converted record exists, and refuses evidence that completes both
    coordinates as ambiguous.

    The rendered callback declaration is not itself an emitting inventory:
    ``resolve_parameter_groups`` drops empty assignments
    (``tpen/parameter_groups.py:111-118``).  A declared group may therefore be
    omitted only when ``empty_groups`` or ``group_inventory`` proves it empty;
    every other missing group remains ``UNMEASURED``.
    """

    name = "disconnected-optimizer-state"
    if records is None:
        return _missing(name, "optimizer-state JSONL evidence is required")
    if groups is None:
        return _missing(name, "rendered OptimizerState groups are required")
    if terminal_step is None or type(terminal_step) is not int:
        return _missing(name, "terminal checkpoint step is required")
    group_names = tuple(groups) if not isinstance(groups, Mapping) else tuple(groups)
    if not group_names or any(not isinstance(group, str) or not group for group in group_names):
        return _invalid(name, "rendered OptimizerState groups must be non-empty names")
    if len(set(group_names)) != len(group_names):
        return _invalid(name, "rendered OptimizerState groups must be unique")

    if isinstance(empty_groups, (str, bytes)):
        return _invalid(name, "empty-group evidence must be an iterable of group names")
    try:
        proven_empty: set[str] = set(empty_groups or ())
    except TypeError:
        return _invalid(name, "empty-group evidence must be an iterable of group names")
    if any(group not in group_names for group in proven_empty):
        return _invalid(name, "empty-group evidence names an undeclared OptimizerState group")
    if group_inventory is not None:
        if not isinstance(group_inventory, Mapping) or set(group_inventory) != set(group_names):
            return _invalid(name, "group inventory must cover every rendered OptimizerState group")
        for group, inventory in group_inventory.items():
            if not isinstance(inventory, Mapping):
                return _invalid(name, f"group inventory for {group!r} must be a mapping")
            parameters = inventory.get("parameters")
            if type(parameters) is not int or parameters < 0:
                return _invalid(name, f"group inventory for {group!r} must name a non-negative parameter count")
            if parameters == 0:
                proven_empty.add(group)
            else:
                proven_empty.discard(group)
    emitting_groups = tuple(group for group in group_names if group not in proven_empty)
    if not emitting_groups:
        return _missing(name, "all rendered OptimizerState groups are proven empty")

    def unpack(record: Any) -> tuple[int | None, str, Mapping[str, Any]] | None:
        if isinstance(record, Mapping):
            step, namespace, metrics = record.get("step"), record.get("namespace"), record.get("metrics")
        else:
            step = getattr(record, "step", None)
            namespace = getattr(record, "namespace", None)
            metrics = getattr(record, "metrics", None)
        if (step is not None and type(step) is not int) or not isinstance(namespace, str) or not isinstance(metrics, Mapping):
            return None
        return step, namespace, metrics

    unpacked_records: list[tuple[int | None, str, Mapping[str, Any]]] = []
    for record in records:
        unpacked = unpack(record)
        if unpacked is None:
            return _invalid(name, "optimizer-state record envelope is malformed")
        unpacked_records.append(unpacked)

    def matches_at(step: int) -> dict[str, list[Mapping[str, Any]]]:
        return {
            group: [
                metrics
                for record_step, record_namespace, metrics in unpacked_records
                if record_step == step
                and record_namespace == f"train/optimizer_state/{group}"
            ]
            for group in emitting_groups
        }

    # The callback coordinate is the primary, explicit contract.  The exact
    # checkpoint coordinate is retained only as a compatibility path for
    # pre-contract evidence; accepting both would make the join ambiguous.
    metric_step = terminal_step - 1
    converted = matches_at(metric_step) if terminal_step > 0 else {}
    exact = matches_at(terminal_step)
    converted_complete = all(len(matches) == 1 for matches in converted.values())
    exact_complete = all(len(matches) == 1 for matches in exact.values())
    if converted_complete and exact_complete:
        return _invalid(name, f"checkpoint step {terminal_step} has ambiguous metric coordinates {metric_step} and {terminal_step}")
    if converted_complete:
        selected_step, selected = metric_step, converted
    elif exact_complete:
        selected_step, selected = terminal_step, exact
    else:
        missing = next(group for group, matches in converted.items() if not matches) if converted else emitting_groups[0]
        return _missing(name, f"missing optimizer-state record for non-empty group {missing!r} at checkpoint step {terminal_step} (expected metric step {metric_step})")

    counts: dict[str, int] = {}
    for group, matches in selected.items():
        if len(matches) != 1:
            return _invalid(name, f"duplicate optimizer-state records for group {group!r} at metric step {selected_step}")
        tensors = matches[0].get("n_no_optimizer_state")
        if type(tensors) is not int or tensors < 0:
            return _invalid(name, f"group {group!r} tensor count must be a non-negative integer")
        counts[group] = tensors

    total = sum(counts.values())
    if total != expected_tensors:
        return _failure(name, f"expected {expected_tensors} disconnected tensors, got {total}", {"groups": counts, "checkpoint_step": terminal_step, "metric_step": selected_step})
    return _ok(name, f"checkpoint step {terminal_step} has {total} disconnected tensors across non-empty groups", {"groups": counts, "checkpoint_step": terminal_step, "metric_step": selected_step, "empty_groups": sorted(proven_empty)})


def check_rendered_diff(config: Mapping[str, Any] | None, observed: Mapping[str, Any] | None) -> CheckResult:
    """Fail closed when a rendered smoke/science diff has unapproved fields."""

    # Rev3 §3 sentence: “The R1→R2 diff is fail-closed and enumerates only the
    # approved scale/provenance changes.”
    name = "rendered-smoke-science-diff"
    if config is None or observed is None:
        return _missing(name, "both rendered configs and the enumerated diff are required")
    if observed.get("unexpected"):
        return _failure(name, f"unexpected rendered fields: {observed['unexpected']!r}")
    if not observed.get("complete", False):
        return _missing(name, "rendered diff is not complete")
    if observed.get("runtime_dtype") != config.get("runtime", {}).get("dtype"):
        return _failure(name, "diff did not derive runtime dtype from the rendered config")
    return _ok(name, "rendered diff is complete and has no unexpected fields")


def check_named_outputs(manifest: Mapping[str, Any] | None, root: str | Path | None = None) -> CheckResult:
    """Reconcile only the TPEN-declared named output files and their hashes."""

    # Rev3 §4.2 sentence: “The file inventory is NOT discovered by find, du,
    # rglob, recursive glob, os.walk, scp -r or recursive directory pulls.”
    name = "named-tpen-output-inventory"
    if manifest is None:
        return _missing(name, "declared output manifest is required")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        return _missing(name, "declared output file list is missing")
    declared_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
            return _invalid(name, "output entries must contain path and sha256")
        path = Path(entry["path"])
        declared_names.add(str(path.resolve()))
        if root is not None:
            try:
                path.resolve().relative_to(Path(root).resolve())
            except ValueError:
                return _invalid(name, f"declared output escapes protected root: {path}")
        if not path.is_file():
            return _missing(name, f"declared output is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            return _failure(name, f"declared output hash mismatch: {path}")
        if entry.get("complete") is False:
            return _missing(name, f"declared writer output is incomplete: {path}")
    if manifest.get("complete") is not True:
        return _missing(name, "output manifest is not marked COMPLETE")
    complete = manifest.get("complete_path")
    if complete is not None and not Path(complete).is_file():
        return _missing(name, "writer COMPLETE marker is missing")
    extra = manifest.get("present_undeclared")
    if extra:
        return _failure(name, f"present undeclared writer outputs: {extra!r}")
    return _ok(name, "all declared TPEN outputs exist with matching bytes and hashes")


def check_evaluation(evidence: Mapping[str, Any] | None) -> CheckResult:
    """Check complete finite frozen evaluation evidence."""

    # Protocol sentence: “Frozen evaluation must be strict, complete, finite,
    # and unclipped; missing writer output is not a pass.”
    name = "frozen-evaluation"
    if evidence is None:
        return _missing(name, "evaluation evidence is required")
    required = {"strict", "complete", "finite", "unclipped", "records"}
    if not required <= set(evidence):
        return _missing(name, f"missing evaluation evidence: {sorted(required-set(evidence))}")
    if not all(evidence[key] is True for key in ("strict", "complete", "finite", "unclipped")):
        return _failure(name, "frozen evaluation is not strict/complete/finite/unclipped")
    if type(evidence["records"]) is not int or evidence["records"] <= 0:
        return _missing(name, "evaluation record count is absent or empty")
    metric_result = check_eval_metrics(evidence)
    if metric_result.verdict is not Verdict.PASS:
        return metric_result
    return _ok(name, "frozen evaluation is strict, complete, finite, and unclipped")


def check_path_containment(path: str | Path | None, root: str | Path | None) -> CheckResult:
    """Refuse relative, missing, symlink, and sibling-prefix path escapes.

    # P0 §3 sentence: “Reject lexical '..', relative/noncanonical paths,
    # symlink escape and sibling-prefix tricks before opening/creating anything.”
    """

    name = "path-containment"
    if path is None or root is None:
        return _missing(name, "path and admitted root are required")
    candidate, admitted = Path(path), Path(root)
    if not candidate.is_absolute() or not admitted.is_absolute():
        return _invalid(name, "path and root must be absolute")
    try:
        candidate.resolve(strict=True).relative_to(admitted.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return _invalid(name, "path escapes the admitted root")
    if candidate.resolve() == admitted.resolve():
        return _invalid(name, "path must be below the admitted root")
    return _ok(name, "path is contained by the admitted root")


def run_rung1_checks(evidence: Mapping[str, Any], *, config: Mapping[str, Any] | None = None) -> list[CheckResult]:
    """Run all offline checks available from one named evidence packet.

    ``run_dir`` is a declared evidence boundary: the consumer opens exactly
    ``run_dir/metrics.jsonl`` through :mod:`.evidence`.  It does not discover a
    sibling or "latest" file.  Missing/empty evidence is ``UNMEASURED``;
    malformed JSONL is an explicit ``HARNESS_INVALID`` result.
    """

    from .evidence import EvidenceFormatError, check_metric_records, load_metric_records

    run_dir = evidence.get("run_dir")
    if run_dir is None:
        metric_result = check_metric_records(None)
        metric_records = None
    elif not isinstance(run_dir, (str, Path)):
        metric_result = CheckResult(
            "metric-jsonl-evidence",
            Verdict.HARNESS_INVALID,
            "declared run_dir must be a path",
        )
        metric_records = None
    else:
        try:
            metric_records = load_metric_records(run_dir)
        except EvidenceFormatError as exc:
            metric_result = CheckResult("metric-jsonl-evidence", Verdict.HARNESS_INVALID, str(exc))
            metric_records = None
        else:
            metric_result = check_metric_records(metric_records)

    optimizer_evidence = evidence.get("optimizer_state")
    if isinstance(optimizer_evidence, Mapping) and "records" in optimizer_evidence:
        optimizer_records = optimizer_evidence.get("records")
        optimizer_result = check_disconnected_optimizer_state(
            optimizer_records,
            groups=optimizer_evidence.get("groups"),
            terminal_step=optimizer_evidence.get("checkpoint_step", optimizer_evidence.get("terminal_step")),
            group_inventory=optimizer_evidence.get("group_inventory"),
            empty_groups=optimizer_evidence.get("empty_groups"),
        )
    elif isinstance(optimizer_evidence, Mapping):
        optimizer_result = check_disconnected_optimizer_state(
            metric_records,
            groups=optimizer_evidence.get("groups"),
            terminal_step=optimizer_evidence.get("checkpoint_step", optimizer_evidence.get("terminal_step")),
            group_inventory=optimizer_evidence.get("group_inventory"),
            empty_groups=optimizer_evidence.get("empty_groups"),
        )
    else:
        optimizer_result = check_disconnected_optimizer_state(None)
    return [
        metric_result,
        check_resume_metric_keys(evidence.get("uninterrupted"), evidence.get("resumed")),
        check_resume_parity(evidence.get("uninterrupted"), evidence.get("resumed"), tolerances=evidence.get("tolerances")),
        check_step_zero(evidence.get("step_zero")),
        check_model_float64(config or evidence.get("config"), evidence.get("checkpoints")),
        optimizer_result,
        check_rendered_diff(config or evidence.get("config"), evidence.get("rendered_diff")),
        check_named_outputs(evidence.get("outputs"), evidence.get("outputs_root")),
        check_evaluation(evidence.get("evaluation")),
        check_eval_metrics(evidence.get("evaluation")),
        check_path_containment(evidence.get("outputs_root"), evidence.get("protected_root")) if evidence.get("protected_root") else _missing("path-containment", "protected_root is required"),
    ]


def summarize(results: Iterable[CheckResult]) -> dict[str, Any]:
    """Summarize results without collapsing missing evidence into PASS."""

    results = list(results)
    counts = {verdict.value: sum(result.verdict is verdict for result in results) for verdict in Verdict}
    if results and all(result.passed for result in results):
        overall = Verdict.PASS
    else:
        # Product evidence is the strongest negative classification, followed
        # by an invalid harness and then an unmeasured gate.
        overall = next((verdict for verdict in (Verdict.PRODUCT_FAILURE, Verdict.HARNESS_INVALID, Verdict.UNMEASURED) if any(result.verdict is verdict for result in results)), Verdict.UNMEASURED)
    return {"overall": overall.value, "counts": counts, "checks": [result.__dict__ for result in results]}


__all__ = [
    "CheckResult", "Verdict", "check_callbacks", "check_checkpoint", "check_config_identity",
    "check_disconnected_optimizer_state", "check_eval_metrics", "check_evaluation", "check_hash",
    "check_metric_integrity", "check_model_float64", "check_named_outputs", "check_parent_join",
    "check_path_containment", "check_profile", "check_rendered_diff", "check_resume_metric_keys",
    "check_resume_parity", "check_seed_domain", "check_state_cursor_tolerance", "check_step_zero",
    "run_rung1_checks", "summarize",
]
