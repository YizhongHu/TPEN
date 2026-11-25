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
from typing import Any, Iterable, Mapping


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
    if not all(_finite(v) for v in [*families.values(), *variance.values(), *gradients.values()]):
        return _failure(name, "step-zero metric is non-finite")
    if len(set(families.values())) != 1 or len(set(variance.values())) != 1:
        return _failure(name, "step-zero energy or energy_variance is not exactly equal")
    if len(set(shas.values())) != 1:
        return _failure(name, "step-zero model SHA differs across families")
    groups = values["construction"]
    if not isinstance(groups, Mapping) or not all(isinstance(v, (list, tuple)) and len(v) == 2 for v in groups.values()):
        return _invalid(name, "construction groups must name two gradient values")
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


def check_disconnected_optimizer_state(state: Mapping[str, Any] | None, *, expected_tensors: int = 16, expected_elements: int = 492000) -> CheckResult:
    """Check disconnected parameters are absent from Adam/SOAP state."""

    # Protocol sentence: “n_no_optimizer_state is a TENSOR count (16 tensors =
    # 492,000 elements),” not an element count.
    name = "disconnected-optimizer-state"
    if state is None:
        return _missing(name, "optimizer-state evidence is required")
    tensors = state.get("n_no_optimizer_state")
    elements = state.get("n_no_optimizer_state_elements")
    if type(tensors) is not int or type(elements) is not int:
        return _invalid(name, "optimizer-state absence counts must be integers")
    if tensors != expected_tensors or elements != expected_elements:
        return _failure(name, f"expected {expected_tensors} tensors/{expected_elements} elements, got {tensors}/{elements}")
    return _ok(name, "disconnected optimizer state is absent with the contract tensor/element counts")


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
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
            return _invalid(name, "output entries must contain path and sha256")
        path = Path(entry["path"])
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
    if manifest.get("complete") is not True:
        return _missing(name, "output manifest is not marked COMPLETE")
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
    return _ok(name, "frozen evaluation is strict, complete, finite, and unclipped")


def run_rung1_checks(evidence: Mapping[str, Any], *, config: Mapping[str, Any] | None = None) -> list[CheckResult]:
    """Run all offline checks available from one named evidence packet."""

    return [
        check_resume_metric_keys(evidence.get("uninterrupted"), evidence.get("resumed")),
        check_step_zero(evidence.get("step_zero")),
        check_model_float64(config or evidence.get("config"), evidence.get("checkpoints")),
        check_disconnected_optimizer_state(evidence.get("optimizer_state")),
        check_rendered_diff(config or evidence.get("config"), evidence.get("rendered_diff")),
        check_named_outputs(evidence.get("outputs"), evidence.get("outputs_root")),
        check_evaluation(evidence.get("evaluation")),
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


__all__ = ["CheckResult", "Verdict", "check_disconnected_optimizer_state", "check_evaluation", "check_model_float64", "check_named_outputs", "check_rendered_diff", "check_resume_metric_keys", "check_step_zero", "run_rung1_checks", "summarize"]
