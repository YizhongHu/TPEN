"""Range-conditioned local-energy and bounded rare-event statistics.

The producer in this module consumes the typed, complete retained-trajectory
CSV.  It never accepts a terminal snapshot.  Rows are streamed in bounded
chunks, while the only arrays retained at trajectory scale are draw-level
numerator and denominator reductions for the conditional-mean ratio estimator.

For a condition ``b`` and retained draw ``d`` the producer first reduces the
parallel walker axis,

``A_d = sum_w I[b, d, w] E[d, w]`` and
``B_d = sum_w I[b, d, w]``.

The correlated series is then the one-dimensional influence series
``A_d - mu_b B_d`` over retained draws.  A dedicated single-series Geyer IPS
estimate is used; the already walker-reduced series is never handed to the
multi-chain pooled trajectory estimator.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import heapq
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from tpen.data.batch import ElectronBatch, two_electron_atomic_geometry
from tpen.statistics.producer import DEFAULT_MIN_DRAWS_PER_CHAIN

if TYPE_CHECKING:
    from tpen.evaluation.trajectory_records import TrajectoryRecordArtifact

CONDITIONED_LOCAL_ENERGY_SCHEMA = "conditioned_local_energy/v1"
"""Versioned JSON schema for conditioned statistics and rare-event records."""

DRAW_RATIO_ESTIMATOR_ID = "draw_level_ratio_geyer_ips"
"""Geyer IPS on one walker-reduced influence series indexed by draw."""

REQUIRED_RANGE_QUANTITIES: tuple[str, ...] = (
    "minimum_electron_nuclear_radius",
    "electron_electron_distance",
    "maximum_electron_nuclear_radius",
    "hyperradius",
    "cos_theta12",
    "logabs",
)
"""The six predeclared one-dimensional conditioning partitions."""

MAX_CONDITION_COUNT = 256
MAX_JOINT_STRATA = 64
MAX_QUANTILE_SAMPLE_CAP = 8192
MAX_EVENT_RECORD_CAP = 10000
MAX_CCDF_THRESHOLD_COUNT = 256
DEFAULT_MIN_OCCUPIED_DRAWS = DEFAULT_MIN_DRAWS_PER_CHAIN
"""Default support gate before a conditional-mean MCSE may be published."""


@dataclass(frozen=True)
class ConditionedStatisticsReport:
    """Validated JSON-safe conditioned-statistics artifact payload.

    Parameters
    ----------
    record : mapping
        JSON-safe payload produced by
        :func:`produce_conditioned_local_energy_statistics`.
    """

    record: Mapping[str, Any]

    def validate(self) -> "ConditionedStatisticsReport":
        """Fail loudly on source-identity or variance-attribution mismatch."""

        if self.record.get("schema") != CONDITIONED_LOCAL_ENERGY_SCHEMA:
            raise ValueError("conditioned statistics report has an unsupported schema")
        source = self.record.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("conditioned statistics report is missing source identity")
        expected_digest = source.get("csv_sha256")
        expected_bytes = source.get("byte_count")
        mismatches: list[str] = []
        for pass_name in ("statistics_pass", "rare_events_pass"):
            receipt = source.get(pass_name)
            if not isinstance(receipt, Mapping):
                mismatches.append(f"{pass_name}: missing pass receipt")
                continue
            if receipt.get("csv_sha256") != expected_digest:
                mismatches.append(
                    f"{pass_name}.csv_sha256={receipt.get('csv_sha256')!r} "
                    f"source={expected_digest!r}"
                )
            if receipt.get("byte_count") != expected_bytes:
                mismatches.append(
                    f"{pass_name}.byte_count={receipt.get('byte_count')!r} "
                    f"source={expected_bytes!r}"
                )

        global_record = self.record.get("global")
        partitions = self.record.get("range_conditioned")
        if not isinstance(global_record, Mapping) or not isinstance(partitions, Mapping):
            mismatches.append("missing global or range-conditioned statistics")
        else:
            actual_quantities = {str(quantity) for quantity in partitions}
            expected_quantities = set(REQUIRED_RANGE_QUANTITIES)
            if actual_quantities != expected_quantities:
                mismatches.append(
                    "range-conditioned quantities: "
                    f"actual={sorted(actual_quantities)} expected={sorted(expected_quantities)}"
                )
            global_count = int(global_record.get("finite_local_energy_count", -1))
            global_second_moment = global_record.get("second_moment_about_mean")
            for quantity, partition in partitions.items():
                if not isinstance(partition, Mapping):
                    mismatches.append(f"{quantity}: partition is not a mapping")
                    continue
                reconciliation = partition.get("reconciliation")
                if not isinstance(reconciliation, Mapping):
                    mismatches.append(f"{quantity}: missing reconciliation")
                    continue
                bins = partition.get("bins")
                if not isinstance(bins, Sequence):
                    mismatches.append(f"{quantity}: missing bins")
                    continue
                bin_ids = {
                    str(bin_record.get("id"))
                    for bin_record in bins
                    if isinstance(bin_record, Mapping)
                }
                missing_structural = {"underflow", "overflow", "nonfinite"} - bin_ids
                if missing_structural:
                    mismatches.append(
                        f"{quantity}: missing structural bins {sorted(missing_structural)}"
                    )
                computed_count = 0
                computed_probability = 0.0
                computed_contribution = 0.0
                for bin_record in bins:
                    if not isinstance(bin_record, Mapping):
                        mismatches.append(f"{quantity}: bin is not a mapping")
                        continue
                    observables = bin_record.get("observables")
                    attribution = bin_record.get("variance_attribution")
                    if not isinstance(observables, Mapping) or not isinstance(attribution, Mapping):
                        mismatches.append(f"{quantity}: bin lacks observables or attribution")
                        continue
                    local = observables.get("local_energy")
                    if not isinstance(local, Mapping):
                        mismatches.append(f"{quantity}: bin lacks local_energy statistics")
                        continue
                    finite_count = int(local.get("finite_count", -1))
                    probability = float(attribution.get("probability", float("nan")))
                    contribution = float(
                        attribution.get("second_moment_contribution", float("nan"))
                    )
                    computed_count += finite_count
                    computed_probability += probability
                    computed_contribution += contribution
                    expected_probability = finite_count / global_count if global_count > 0 else 0.0
                    if not _close_float(
                        probability,
                        expected_probability,
                        atol=1.0e-12,
                        rtol=1.0e-12,
                    ):
                        mismatches.append(
                            f"{quantity}.{bin_record.get('id')}.probability={probability!r} "
                            f"expected={expected_probability!r}"
                        )
                if int(reconciliation.get("finite_count_sum", -1)) != computed_count:
                    mismatches.append(
                        f"{quantity}.finite_count_sum={reconciliation.get('finite_count_sum')!r} "
                        f"computed={computed_count}"
                    )
                if computed_count != global_count:
                    mismatches.append(f"{quantity}.computed_finite_count={computed_count} global={global_count}")
                probability_sum = reconciliation.get("probability_sum")
                if not _close_float(
                    probability_sum,
                    computed_probability,
                    atol=1.0e-12,
                    rtol=1.0e-12,
                ):
                    mismatches.append(
                        f"{quantity}.probability_sum={probability_sum!r} "
                        f"computed={computed_probability!r}"
                    )
                if not _close_float(probability_sum, 1.0, atol=1.0e-12, rtol=1.0e-12):
                    mismatches.append(f"{quantity}.probability_sum={probability_sum!r} expected=1.0")
                contribution_sum = reconciliation.get("second_moment_contribution_sum")
                tolerance = max(1.0e-12, 1.0e-10 * abs(float(global_second_moment)))
                if not _close_float(
                    contribution_sum,
                    computed_contribution,
                    atol=tolerance,
                    rtol=1.0e-10,
                ):
                    mismatches.append(
                        f"{quantity}.second_moment_contribution_sum={contribution_sum!r} "
                        f"computed={computed_contribution!r}"
                    )
                if not _close_float(
                    contribution_sum,
                    global_second_moment,
                    atol=tolerance,
                    rtol=1.0e-10,
                ):
                    mismatches.append(
                        f"{quantity}.second_moment_contribution_sum={contribution_sum!r} "
                        f"global={global_second_moment!r}"
                    )
        if mismatches:
            raise ValueError(
                "conditioned statistics reconciliation failed: " + "; ".join(mismatches)
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return an owned JSON-safe copy of the payload."""

        return copy.deepcopy(dict(self.record))


@dataclass(frozen=True)
class _StreamReceipt:
    csv_sha256: str
    byte_count: int
    row_count: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "csv_sha256": self.csv_sha256,
            "byte_count": self.byte_count,
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class _ParsedChunk:
    rows: tuple[dict[str, str], ...]
    sample_index: torch.Tensor
    draw_index: torch.Tensor
    walker_index: torch.Tensor
    positions: torch.Tensor
    local_energy: torch.Tensor
    term_energies: Mapping[str, torch.Tensor]
    logabs: torch.Tensor
    sign: torch.Tensor
    quantities: Mapping[str, torch.Tensor]
    angle_defined: torch.Tensor
    angle_undefined_at_coalescence: torch.Tensor

    @property
    def row_count(self) -> int:
        return int(self.draw_index.numel())


@dataclass(frozen=True)
class _JointStratum:
    name: str
    bounds: Mapping[str, tuple[float, float]]

    def mask(self, quantities: Mapping[str, torch.Tensor]) -> torch.Tensor:
        first = next(iter(quantities.values()))
        selected = torch.ones(first.shape, dtype=torch.bool)
        for quantity, (lower, upper) in self.bounds.items():
            values = quantities[quantity]
            selected &= torch.isfinite(values) & (values >= lower) & (values < upper)
        return selected

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bounds": {
                quantity: {"lower_inclusive": lower, "upper_exclusive": upper}
                for quantity, (lower, upper) in self.bounds.items()
            },
        }


class _RangePartition:
    """One predeclared cut sequence plus structural full-support bins."""

    def __init__(self, quantity: str, edges: Sequence[float]) -> None:
        self.quantity = quantity
        self.edges = tuple(float(value) for value in edges)
        if not self.edges:
            raise ValueError(f"range edges for {quantity!r} must not be empty")
        if any(not math.isfinite(value) for value in self.edges):
            raise ValueError(
                f"range edges for {quantity!r} must be finite; underflow and overflow "
                "are structural bins"
            )
        if any(left >= right for left, right in zip(self.edges, self.edges[1:])):
            raise ValueError(f"range edges for {quantity!r} must be strictly increasing")
        self.edge_tensor = torch.tensor(self.edges, dtype=torch.float64)

    @property
    def bin_count(self) -> int:
        # One underflow, len(edges)-1 interior, one overflow, one non-finite.
        return len(self.edges) + 2

    @property
    def nonfinite_index(self) -> int:
        return len(self.edges) + 1

    def indices(self, values: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(values)
        # right=True gives: x < e0 -> 0, e_i <= x < e_(i+1) -> i+1,
        # x >= e_last -> len(edges).  No finite row can disappear.
        result = torch.bucketize(values, self.edge_tensor, right=True)
        return torch.where(finite, result, torch.full_like(result, self.nonfinite_index))

    def bin_identity(self, index: int) -> dict[str, Any]:
        if index == 0:
            return {
                "id": "underflow",
                "kind": "underflow",
                "lower": "-inf",
                "upper": self.edges[0],
                "lower_inclusive": False,
                "upper_inclusive": False,
            }
        if index == len(self.edges):
            return {
                "id": "overflow",
                "kind": "overflow",
                "lower": self.edges[-1],
                "upper": "inf",
                "lower_inclusive": True,
                "upper_inclusive": False,
            }
        if index == self.nonfinite_index:
            return {
                "id": "nonfinite",
                "kind": "nonfinite",
                "lower": None,
                "upper": None,
                "lower_inclusive": False,
                "upper_inclusive": False,
            }
        return {
            "id": f"range_{index - 1:03d}",
            "kind": "range",
            "lower": self.edges[index - 1],
            "upper": self.edges[index],
            "lower_inclusive": True,
            "upper_inclusive": False,
        }


class _DeterministicReservoir:
    """Algorithm-R reservoir with a stable, explicitly seeded RNG."""

    def __init__(self, *, cap: int, seed: int, stream_name: str) -> None:
        self.cap = cap
        digest = hashlib.sha256(f"{seed}:{stream_name}".encode("utf-8")).digest()
        self.effective_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        self.random = random.Random(self.effective_seed)
        self.values: list[float] = []
        self.seen = 0

    def extend(self, values: torch.Tensor) -> None:
        for value in values.tolist():
            self.seen += 1
            if len(self.values) < self.cap:
                self.values.append(float(value))
                continue
            replacement = self.random.randrange(self.seen)
            if replacement < self.cap:
                self.values[replacement] = float(value)

    def quantiles(self, probabilities: Sequence[float], *, configured_seed: int) -> dict[str, Any]:
        ordered = sorted(self.values)
        values = {
            _probability_label(probability): _linear_quantile(ordered, probability)
            for probability in probabilities
        }
        return {
            "descriptive_only": True,
            "method": "deterministic_seeded_algorithm_r",
            "configured_seed": configured_seed,
            "effective_stream_seed": self.effective_seed,
            "sample_cap": self.cap,
            "sample_count": len(ordered),
            "population_count": self.seen,
            "exact": self.seen <= self.cap,
            "values": values,
        }


class _ObservableAccumulator:
    def __init__(
        self,
        *,
        n_draws: int,
        quantile_sample_cap: int,
        quantile_seed: int,
        stream_name: str,
    ) -> None:
        self.finite_count = 0
        self.nonfinite_count = 0
        self.running_mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.draw_counts = torch.zeros(n_draws, dtype=torch.float64)
        self.draw_sums = torch.zeros(n_draws, dtype=torch.float64)
        self.reservoir = _DeterministicReservoir(
            cap=quantile_sample_cap,
            seed=quantile_seed,
            stream_name=stream_name,
        )

    def add(self, values: torch.Tensor, draw_indices: torch.Tensor) -> None:
        finite = torch.isfinite(values)
        n_finite = int(finite.sum().item())
        self.finite_count += n_finite
        self.nonfinite_count += int(values.numel()) - n_finite
        if not n_finite:
            return
        selected = values[finite].to(torch.float64)
        selected_draws = draw_indices[finite]
        chunk_mean = float(selected.mean().item())
        chunk_m2 = float((selected - chunk_mean).square().sum().item())
        if not math.isfinite(chunk_mean) or not math.isfinite(chunk_m2):
            raise ValueError("finite observable values overflowed the stable moment reduction")
        previous_count = self.finite_count - n_finite
        combined_count = previous_count + n_finite
        if previous_count == 0:
            self.running_mean = chunk_mean
            self.m2 = chunk_m2
        else:
            delta = chunk_mean - self.running_mean
            self.m2 += chunk_m2 + delta * delta * previous_count * n_finite / combined_count
            self.running_mean += delta * n_finite / combined_count
        if not math.isfinite(self.m2) or not math.isfinite(self.running_mean):
            raise ValueError("stable moment merge overflowed across CSV chunks")
        self.minimum = min(self.minimum, float(selected.min().item()))
        self.maximum = max(self.maximum, float(selected.max().item()))
        self.draw_counts.index_add_(
            0,
            selected_draws,
            torch.ones(selected.shape, dtype=torch.float64),
        )
        self.draw_sums.index_add_(0, selected_draws, selected)
        self.reservoir.extend(selected)

    def finish(
        self,
        *,
        support: int,
        probabilities: Sequence[float],
        quantile_seed: int,
        min_occupied_draws: int,
    ) -> dict[str, Any]:
        if support == 0:
            finite_status = "empty"
        elif self.finite_count == 0:
            finite_status = "all_nonfinite"
        elif self.nonfinite_count:
            finite_status = "partially_nonfinite"
        else:
            finite_status = "all_finite"
        if self.finite_count:
            mean = self.running_mean
            variance = self.m2 / self.finite_count
            if variance < 0.0:
                raise ValueError(f"negative conditional variance from stable reduction: {variance}")
            minimum: float | None = self.minimum
            maximum: float | None = self.maximum
            iid_stderr: float | None = math.sqrt(variance / self.finite_count)
        else:
            mean = None
            variance = None
            minimum = None
            maximum = None
            iid_stderr = None

        occupied_draws = int((self.draw_counts > 0).sum().item())
        mcse = _conditional_mean_mcse(
            draw_sums=self.draw_sums,
            draw_counts=self.draw_counts,
            mean=mean,
            support=support,
            finite_count=self.finite_count,
            occupied_draws=occupied_draws,
            min_occupied_draws=min_occupied_draws,
        )
        return {
            "finite_status": finite_status,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "minimum": minimum,
            "maximum": maximum,
            "conditional_mean": mean,
            "conditional_variance": variance,
            "flattened_iid_stderr": iid_stderr,
            "flattened_quantiles": self.reservoir.quantiles(
                probabilities,
                configured_seed=quantile_seed,
            ),
            "conditional_mean_mcse": mcse,
        }


class _ConditionAccumulator:
    def __init__(
        self,
        *,
        n_draws: int,
        n_walkers: int,
        observable_names: Sequence[str],
        quantile_sample_cap: int,
        quantile_seed: int,
        stream_name: str,
    ) -> None:
        self.support = 0
        self.draw_occupied = torch.zeros(n_draws, dtype=torch.bool)
        self.walker_occupied = torch.zeros(n_walkers, dtype=torch.bool)
        self.observables = {
            name: _ObservableAccumulator(
                n_draws=n_draws,
                quantile_sample_cap=quantile_sample_cap,
                quantile_seed=quantile_seed,
                stream_name=f"{stream_name}:{name}",
            )
            for name in observable_names
        }

    def add(
        self,
        *,
        mask: torch.Tensor,
        draw_indices: torch.Tensor,
        walker_indices: torch.Tensor,
        observables: Mapping[str, torch.Tensor],
    ) -> None:
        count = int(mask.sum().item())
        if not count:
            return
        self.support += count
        self.draw_occupied[draw_indices[mask].unique()] = True
        self.walker_occupied[walker_indices[mask].unique()] = True
        for name, accumulator in self.observables.items():
            accumulator.add(observables[name][mask], draw_indices[mask])

    def finish(
        self,
        *,
        probabilities: Sequence[float],
        quantile_seed: int,
        min_occupied_draws: int,
    ) -> dict[str, Any]:
        local = self.observables["local_energy"]
        if self.support == 0:
            finite_status = "empty"
        elif local.finite_count == 0:
            finite_status = "all_nonfinite"
        elif local.nonfinite_count:
            finite_status = "partially_nonfinite"
        else:
            finite_status = "all_finite"
        return {
            "support": self.support,
            "occupied_draws": int(self.draw_occupied.sum().item()),
            "occupied_walkers": int(self.walker_occupied.sum().item()),
            "finite_status": finite_status,
            "observables": {
                name: accumulator.finish(
                    support=self.support,
                    probabilities=probabilities,
                    quantile_seed=quantile_seed,
                    min_occupied_draws=min_occupied_draws,
                )
                for name, accumulator in self.observables.items()
            },
        }


def _conditional_mean_mcse(
    *,
    draw_sums: torch.Tensor,
    draw_counts: torch.Tensor,
    mean: float | None,
    support: int,
    finite_count: int,
    occupied_draws: int,
    min_occupied_draws: int,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "estimator_id": DRAW_RATIO_ESTIMATOR_ID,
        "correlated_axis": "retained_draw",
        "walker_reduction": "within_draw_sum_before_autocorrelation",
        "minimum_occupied_draws": min_occupied_draws,
        "occupied_draws": occupied_draws,
        "value": None,
        "tau_int": None,
        "influence_variance": None,
    }
    if support == 0:
        return {**base, "status": "empty", "reason": "condition has zero support"}
    if not finite_count or mean is None:
        return {
            **base,
            "status": "unresolved",
            "reason": "condition has no finite observable values",
        }
    if occupied_draws < min_occupied_draws:
        return {
            **base,
            "status": "unresolved",
            "reason": (
                f"condition occupies {occupied_draws} retained draw(s), minimum "
                f"{min_occupied_draws}"
            ),
        }

    influence = draw_sums - mean * draw_counts
    mean_draw_denominator = float(draw_counts.mean().item())
    if not (mean_draw_denominator > 0.0):
        return {**base, "status": "unresolved", "reason": "zero draw-level denominator"}
    estimate = _single_draw_series_geyer_ips(influence)
    if estimate["status"] != "available":
        return {**base, **estimate}
    influence_variance = float(estimate["influence_variance"])
    tau_int = estimate["tau_int"]
    if influence_variance == 0.0:
        value = 0.0
    else:
        value = math.sqrt(influence_variance * float(tau_int) / draw_sums.numel())
        value /= mean_draw_denominator
    return {**base, **estimate, "value": value}


def _single_draw_series_geyer_ips(values: torch.Tensor) -> dict[str, Any]:
    """Estimate IAT for one walker-reduced series on the retained-draw axis."""

    series = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    n_draws = int(series.numel())
    if n_draws < DEFAULT_MIN_DRAWS_PER_CHAIN:
        return {
            "status": "unresolved",
            "reason": (
                f"influence series has {n_draws} retained draws, minimum "
                f"{DEFAULT_MIN_DRAWS_PER_CHAIN}"
            ),
            "tau_int": None,
            "influence_variance": None,
        }
    if not bool(torch.isfinite(series).all()):
        return {
            "status": "unresolved",
            "reason": "draw-level influence series contains non-finite values",
            "tau_int": None,
            "influence_variance": None,
        }
    centered = series - series.mean()
    variance = float(series.var(unbiased=True).item())
    if variance == 0.0:
        return {
            "status": "available",
            "reason": "draw-level influence series has zero variance",
            "tau_int": None,
            "influence_variance": 0.0,
            "plateau_reached": True,
            "truncation_lag": 0,
        }
    if not (variance > 0.0) or not math.isfinite(variance):
        return {
            "status": "unresolved",
            "reason": f"invalid draw-level influence variance: {variance}",
            "tau_int": None,
            "influence_variance": None,
        }

    n_fft = 1 << max(1, 2 * n_draws - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=n_fft)
    autocovariance = torch.fft.irfft(
        spectrum.real.square() + spectrum.imag.square(),
        n=n_fft,
    )[:n_draws] / n_draws
    rho = autocovariance / autocovariance[0]
    n_pairs = n_draws // 2
    pairs = rho[: 2 * n_pairs].reshape(n_pairs, 2).sum(dim=1)
    monotone = torch.cummin(pairs, dim=0).values
    positive = monotone > 0.0
    if not bool(positive[0]):
        return {
            "status": "unresolved",
            "reason": "initial draw-level Geyer pair is non-positive",
            "tau_int": None,
            "influence_variance": variance,
        }
    if bool(positive.all()):
        return {
            "status": "unresolved",
            "reason": f"no draw-level plateau within {n_draws - 1} lags",
            "tau_int": None,
            "influence_variance": variance,
        }
    pair_count = int((~positive).nonzero()[0].item())
    tau_int = float(-1.0 + 2.0 * monotone[:pair_count].sum().item())
    if not math.isfinite(tau_int) or not (tau_int > 0.0):
        return {
            "status": "unresolved",
            "reason": f"non-positive draw-level tau_int estimate: {tau_int}",
            "tau_int": None,
            "influence_variance": variance,
        }
    return {
        "status": "available",
        "reason": None,
        "tau_int": tau_int,
        "influence_variance": variance,
        "plateau_reached": True,
        "truncation_lag": 2 * pair_count - 1,
    }


class _TopKRecords:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.total_count = 0
        self.heap: list[tuple[float, int, dict[str, Any]]] = []

    def add(self, *, score: float, sample_index: int, record: dict[str, Any]) -> None:
        self.total_count += 1
        candidate = (float(score), -int(sample_index), record)
        if len(self.heap) < self.cap:
            heapq.heappush(self.heap, candidate)
        elif candidate[:2] > self.heap[0][:2]:
            heapq.heapreplace(self.heap, candidate)

    def finish(self) -> dict[str, Any]:
        ordered = sorted(self.heap, key=lambda value: (-value[0], -value[1]))
        return {
            "total_count": self.total_count,
            "record_cap": self.cap,
            "record_count": len(ordered),
            "truncated": self.total_count > len(ordered),
            "records": [record for _, _, record in ordered],
        }


class _FirstRecords:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.total_count = 0
        self.records: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        self.total_count += 1
        if len(self.records) < self.cap:
            self.records.append(record)

    def finish(self) -> dict[str, Any]:
        return {
            "total_count": self.total_count,
            "record_cap": self.cap,
            "record_count": len(self.records),
            "truncated": self.total_count > len(self.records),
            "selection": "first_in_draw_major_row_order",
            "records": self.records,
        }


def produce_conditioned_local_energy_statistics(
    artifact: TrajectoryRecordArtifact,
    *,
    range_edges: Mapping[str, Sequence[float]],
    joint_strata: Sequence[Mapping[str, Any]] = (),
    quantiles: Sequence[float] = (0.01, 0.05, 0.5, 0.95, 0.99),
    quantile_sample_cap: int = 4096,
    quantile_seed: int,
    deviation_ccdf_thresholds: Sequence[float],
    top_k: int = 100,
    max_event_records: int = 100,
    cancellation_ratio_threshold: float = 100.0,
    cancellation_term_l1_threshold: float = 10.0,
    cancellation_energy_floor: float = 1.0e-12,
    low_logabs_threshold: float = -20.0,
    min_occupied_draws: int = DEFAULT_MIN_OCCUPIED_DRAWS,
    chunk_size: int = 8192,
) -> ConditionedStatisticsReport:
    """Produce conditioned distributions and bounded pathology records.

    Parameters
    ----------
    artifact : TrajectoryRecordArtifact
        Typed complete retained-trajectory artifact.  A terminal snapshot is
        not accepted by this API.
    range_edges : mapping of str to sequence of float
        Predeclared finite cut points for all six quantities in
        :data:`REQUIRED_RANGE_QUANTITIES`.  Structural underflow, overflow, and
        non-finite bins are always added, so every row belongs to exactly one
        bin in every partition.
    joint_strata : sequence of mappings, optional
        Predeclared rectangular strata.  Each mapping has ``name`` and
        ``bounds``; every bound is ``[lower_inclusive, upper_exclusive]``.
    quantiles : sequence of float, optional
        Descriptive flattened quantile probabilities.
    quantile_sample_cap : int, optional
        Hard per-condition, per-observable reservoir cap.
    quantile_seed : int
        Required deterministic reservoir seed, recorded in the artifact.
    deviation_ccdf_thresholds : sequence of float
        Predeclared absolute-deviation thresholds for the local-energy CCDF.
    top_k : int, optional
        Maximum full records retained for the largest energy deviations.
    max_event_records : int, optional
        Maximum full records retained for each nonfinite, cancellation, and
        low-amplitude event class.
    cancellation_ratio_threshold : float, optional
        Minimum ``sum(abs(term)) / max(abs(total), floor)`` event ratio.
    cancellation_term_l1_threshold : float, optional
        Minimum absolute term sum for a cancellation event.
    cancellation_energy_floor : float, optional
        Fixed denominator floor in the cancellation ratio.
    low_logabs_threshold : float, optional
        ``logabs`` at or below which an event is low-amplitude.
    min_occupied_draws : int, optional
        Minimum number of retained draws containing a finite conditioned value
        before conditional MCSE may be published.
    chunk_size : int, optional
        Maximum CSV rows materialized in memory at once.

    Returns
    -------
    ConditionedStatisticsReport
        Validated, deterministic, JSON-safe report.

    Raises
    ------
    TypeError
        If `artifact` is not a :class:`TrajectoryRecordArtifact`.
    ValueError
        If configuration, source identity, grid ordering, or variance
        attribution fails validation.
    """

    # Imported lazily so the statistics package does not import the evaluation
    # package while `tpen.evaluation.trajectory_records` is itself importing
    # the trajectory statistics primitives.
    from tpen.evaluation.trajectory_records import (
        TRAJECTORY_RECORD_SCHEMA,
        TrajectoryRecordArtifact,
    )

    if not isinstance(artifact, TrajectoryRecordArtifact):
        raise TypeError(
            "conditioned statistics require a TrajectoryRecordArtifact from the "
            "typed retained trajectory; terminal snapshots are not accepted"
        )
    artifact.validate()
    if artifact.atomic_configuration is None:
        raise ValueError(
            "conditioned local-energy statistics require nuclear geometry; "
            "trajectory records explicitly declare nucleus-free geometry as absent"
        )
    partitions = _coerce_range_partitions(range_edges)
    strata = _coerce_joint_strata(joint_strata)
    probabilities = _coerce_probabilities(quantiles)
    ccdf_thresholds = _coerce_nonnegative_sequence(
        deviation_ccdf_thresholds,
        name="deviation_ccdf_thresholds",
        require_nonempty=True,
    )
    if len(ccdf_thresholds) > MAX_CCDF_THRESHOLD_COUNT:
        raise ValueError(
            f"deviation_ccdf_thresholds defines {len(ccdf_thresholds)} thresholds, "
            f"hard cap {MAX_CCDF_THRESHOLD_COUNT}"
        )
    quantile_sample_cap = _bounded_positive_int(
        quantile_sample_cap,
        name="quantile_sample_cap",
        maximum=MAX_QUANTILE_SAMPLE_CAP,
    )
    top_k = _bounded_positive_int(top_k, name="top_k", maximum=MAX_EVENT_RECORD_CAP)
    max_event_records = _bounded_positive_int(
        max_event_records,
        name="max_event_records",
        maximum=MAX_EVENT_RECORD_CAP,
    )
    min_occupied_draws = _bounded_positive_int(
        min_occupied_draws,
        name="min_occupied_draws",
        maximum=1_000_000,
    )
    if min_occupied_draws < DEFAULT_MIN_OCCUPIED_DRAWS:
        raise ValueError(
            f"min_occupied_draws must be at least {DEFAULT_MIN_OCCUPIED_DRAWS}, "
            f"got {min_occupied_draws}"
        )
    chunk_size = _bounded_positive_int(chunk_size, name="chunk_size", maximum=1_000_000)
    quantile_seed = int(quantile_seed)
    cancellation_ratio_threshold = _positive_finite(
        cancellation_ratio_threshold,
        name="cancellation_ratio_threshold",
    )
    cancellation_term_l1_threshold = _nonnegative_finite(
        cancellation_term_l1_threshold,
        name="cancellation_term_l1_threshold",
    )
    cancellation_energy_floor = _positive_finite(
        cancellation_energy_floor,
        name="cancellation_energy_floor",
    )
    low_logabs_threshold = float(low_logabs_threshold)
    if not math.isfinite(low_logabs_threshold):
        raise ValueError("low_logabs_threshold must be finite")

    observable_names = ("local_energy", *(f"term/{name}" for name in artifact.term_names))
    global_accumulator = _ConditionAccumulator(
        n_draws=artifact.n_draws,
        n_walkers=artifact.n_walkers,
        observable_names=observable_names,
        quantile_sample_cap=quantile_sample_cap,
        quantile_seed=quantile_seed,
        stream_name="global",
    )
    range_accumulators = {
        partition.quantity: [
            _ConditionAccumulator(
                n_draws=artifact.n_draws,
                n_walkers=artifact.n_walkers,
                observable_names=observable_names,
                quantile_sample_cap=quantile_sample_cap,
                quantile_seed=quantile_seed,
                stream_name=f"range:{partition.quantity}:{index}",
            )
            for index in range(partition.bin_count)
        ]
        for partition in partitions
    }
    joint_accumulators = [
        _ConditionAccumulator(
            n_draws=artifact.n_draws,
            n_walkers=artifact.n_walkers,
            observable_names=observable_names,
            quantile_sample_cap=quantile_sample_cap,
            quantile_seed=quantile_seed,
            stream_name=f"joint:{stratum.name}",
        )
        for stratum in strata
    ]
    angle_coalescence_count = 0
    angle_nonfinite_geometry_count = 0

    def statistics_chunk(chunk: _ParsedChunk) -> None:
        nonlocal angle_coalescence_count, angle_nonfinite_geometry_count
        observables = {"local_energy": chunk.local_energy, **chunk.term_energies}
        angle_coalescence_count += int(chunk.angle_undefined_at_coalescence.sum().item())
        angle_nonfinite_geometry_count += int(
            (
                ~chunk.angle_defined
                & ~chunk.angle_undefined_at_coalescence
            ).sum().item()
        )
        all_rows = torch.ones(chunk.row_count, dtype=torch.bool)
        global_accumulator.add(
            mask=all_rows,
            draw_indices=chunk.draw_index,
            walker_indices=chunk.walker_index,
            observables=observables,
        )
        for partition in partitions:
            indices = partition.indices(chunk.quantities[partition.quantity])
            for index, accumulator in enumerate(range_accumulators[partition.quantity]):
                accumulator.add(
                    mask=indices == index,
                    draw_indices=chunk.draw_index,
                    walker_indices=chunk.walker_index,
                    observables=observables,
                )
        for stratum, accumulator in zip(strata, joint_accumulators, strict=True):
            accumulator.add(
                mask=stratum.mask(chunk.quantities),
                draw_indices=chunk.draw_index,
                walker_indices=chunk.walker_index,
                observables=observables,
            )

    statistics_pass = _stream_artifact(
        artifact,
        chunk_size=chunk_size,
        consume=statistics_chunk,
    )
    global_local = global_accumulator.observables["local_energy"]
    if global_local.finite_count == 0:
        raise ValueError("conditioned statistics require at least one finite local-energy row")
    global_mean = global_local.running_mean
    global_second_moment = global_local.m2 / global_local.finite_count
    if global_second_moment < 0.0:
        raise ValueError(
            "negative global second moment from stable reduction: "
            f"{global_second_moment}"
        )

    range_records: dict[str, Any] = {}
    for partition in partitions:
        bins: list[dict[str, Any]] = []
        finite_count_sum = 0
        contributions: list[float] = []
        for index, accumulator in enumerate(range_accumulators[partition.quantity]):
            record = {
                **partition.bin_identity(index),
                **accumulator.finish(
                    probabilities=probabilities,
                    quantile_seed=quantile_seed,
                    min_occupied_draws=min_occupied_draws,
                ),
            }
            local = accumulator.observables["local_energy"]
            probability = local.finite_count / global_local.finite_count
            if local.finite_count:
                conditional_second_moment = (
                    local.m2
                    + local.finite_count * (local.running_mean - global_mean) ** 2
                ) / local.finite_count
            else:
                conditional_second_moment = None
            contribution = (
                0.0
                if conditional_second_moment is None
                else probability * conditional_second_moment
            )
            record["variance_attribution"] = {
                "probability": probability,
                "conditional_second_moment_about_global_mean": conditional_second_moment,
                "second_moment_contribution": contribution,
            }
            finite_count_sum += local.finite_count
            contributions.append(contribution)
            bins.append(record)
        # The recorded mass is a count identity rather than a floating sum of
        # fractions, so a complete partition publishes exactly 1.0 even when
        # the denominator is not a power of two.
        probability_sum = finite_count_sum / global_local.finite_count
        contribution_sum = math.fsum(contributions)
        range_records[partition.quantity] = {
            "predeclared_edges": list(partition.edges),
            "structural_bins": ["underflow", "overflow", "nonfinite"],
            "bins": bins,
            "reconciliation": {
                "finite_count_sum": finite_count_sum,
                "global_finite_count": global_local.finite_count,
                "probability_sum": probability_sum,
                "second_moment_contribution_sum": contribution_sum,
                "global_second_moment": global_second_moment,
            },
        }

    joint_records = []
    for stratum, accumulator in zip(strata, joint_accumulators, strict=True):
        joint_records.append(
            {
                **stratum.to_dict(),
                **accumulator.finish(
                    probabilities=probabilities,
                    quantile_seed=quantile_seed,
                    min_occupied_draws=min_occupied_draws,
                ),
            }
        )

    ccdf_counts = [0 for _ in ccdf_thresholds]
    top_deviations = _TopKRecords(top_k)
    nonfinite_events = _FirstRecords(max_event_records)
    cancellation_events = _TopKRecords(max_event_records)
    low_amplitude_events = _TopKRecords(max_event_records)

    def rare_event_chunk(chunk: _ParsedChunk) -> None:
        for row_index in range(chunk.row_count):
            local_energy = float(chunk.local_energy[row_index].item())
            term_values = {
                name: float(values[row_index].item())
                for name, values in chunk.term_energies.items()
            }
            logabs = float(chunk.logabs[row_index].item())
            primitive_values = [
                local_energy,
                *term_values.values(),
                logabs,
                float(chunk.sign[row_index].item()),
                *chunk.positions[row_index].reshape(-1).tolist(),
            ]
            full_record = _full_record(chunk, row_index)
            if not all(math.isfinite(value) for value in primitive_values):
                nonfinite_events.add(copy.deepcopy(full_record))
            if math.isfinite(local_energy):
                deviation = abs(local_energy - global_mean)
                for index, threshold in enumerate(ccdf_thresholds):
                    if deviation >= threshold:
                        ccdf_counts[index] += 1
                full_record["event_score"] = deviation
                top_deviations.add(
                    score=deviation,
                    sample_index=int(chunk.sample_index[row_index].item()),
                    record=full_record,
                )
            if math.isfinite(local_energy) and all(
                math.isfinite(value) for value in term_values.values()
            ):
                term_l1 = sum(abs(value) for value in term_values.values())
                ratio = term_l1 / max(abs(local_energy), cancellation_energy_floor)
                if (
                    term_l1 >= cancellation_term_l1_threshold
                    and ratio >= cancellation_ratio_threshold
                ):
                    cancellation_record = dict(full_record)
                    cancellation_record["event_score"] = ratio
                    cancellation_record["term_l1"] = term_l1
                    cancellation_events.add(
                        score=ratio,
                        sample_index=int(chunk.sample_index[row_index].item()),
                        record=cancellation_record,
                    )
            if math.isfinite(logabs) and logabs <= low_logabs_threshold:
                low_amplitude_record = dict(full_record)
                low_amplitude_record["event_score"] = logabs
                low_amplitude_events.add(
                    score=-logabs,
                    sample_index=int(chunk.sample_index[row_index].item()),
                    record=low_amplitude_record,
                )

    rare_events_pass = _stream_artifact(
        artifact,
        chunk_size=chunk_size,
        consume=rare_event_chunk,
    )
    cos_partition = next(
        partition for partition in partitions if partition.quantity == "cos_theta12"
    )
    undefined_angle_count = range_accumulators["cos_theta12"][
        cos_partition.nonfinite_index
    ].support
    if undefined_angle_count != angle_coalescence_count + angle_nonfinite_geometry_count:
        raise ValueError(
            "cos_theta12 domain-status reconciliation failed: "
            f"undefined={undefined_angle_count}, coalescence={angle_coalescence_count}, "
            f"nonfinite_geometry={angle_nonfinite_geometry_count}"
        )
    record = {
        "schema": CONDITIONED_LOCAL_ENERGY_SCHEMA,
        "source": {
            "trajectory_record_schema": TRAJECTORY_RECORD_SCHEMA,
            "csv_sha256": artifact.csv_sha256,
            "byte_count": artifact.byte_count,
            "row_count": artifact.row_count,
            "draw_count": artifact.n_draws,
            "walker_count": artifact.n_walkers,
            "observable_values_content_id": artifact.observable_values_content_id,
            "atomic_configuration_id": artifact.atomic_configuration.content_id(),
            "statistics_pass": statistics_pass.to_dict(),
            "rare_events_pass": rare_events_pass.to_dict(),
            "two_pass_identity_confirmed": True,
        },
        "estimator": {
            "conditional_mean_mcse": DRAW_RATIO_ESTIMATOR_ID,
            "correlated_axis": "retained_draw",
            "walker_reduction": "within_draw_sum_before_autocorrelation",
            "ratio_influence_series": "A_d - conditional_mean * B_d",
            "minimum_occupied_draws": min_occupied_draws,
            "headline_estimator": False,
        },
        "configuration": {
            "range_edges": {
                partition.quantity: list(partition.edges) for partition in partitions
            },
            "joint_strata": [stratum.to_dict() for stratum in strata],
            "quantiles": list(probabilities),
            "quantile_sampling": {
                "method": "deterministic_seeded_algorithm_r",
                "seed": quantile_seed,
                "sample_cap_per_condition_observable": quantile_sample_cap,
                "descriptive_only": True,
            },
            "deviation_ccdf_thresholds": list(ccdf_thresholds),
            "top_k": top_k,
            "max_event_records": max_event_records,
            "cancellation_ratio_threshold": cancellation_ratio_threshold,
            "cancellation_term_l1_threshold": cancellation_term_l1_threshold,
            "cancellation_energy_floor": cancellation_energy_floor,
            "low_logabs_threshold": low_logabs_threshold,
            "chunk_size": chunk_size,
        },
        "global": {
            "finite_local_energy_count": global_local.finite_count,
            "nonfinite_local_energy_count": global_local.nonfinite_count,
            "finite_local_energy_mean_for_diagnostic_centering": global_mean,
            "second_moment_about_mean": global_second_moment,
            "interpretation": "diagnostic centering only; not a headline energy estimator",
            "cos_theta12_domain": {
                "defined_count": artifact.row_count - undefined_angle_count,
                "undefined_count": undefined_angle_count,
                "undefined_at_electron_nucleus_coalescence_count": angle_coalescence_count,
                "undefined_from_nonfinite_geometry_count": angle_nonfinite_geometry_count,
                "undefined_semantics": (
                    "finite coalescence has no angular direction; nonfinite input geometry "
                    "is reported separately"
                ),
            },
        },
        "range_conditioned": range_records,
        "joint_strata": joint_records,
        "rare_events": {
            "absolute_deviation_ccdf": [
                {
                    "threshold": threshold,
                    "count": count,
                    "probability_over_finite_local_energy": count / global_local.finite_count,
                }
                for threshold, count in zip(ccdf_thresholds, ccdf_counts, strict=True)
            ],
            "top_k_absolute_deviations": top_deviations.finish(),
            "nonfinite": nonfinite_events.finish(),
            "cancellation": {
                **cancellation_events.finish(),
                "definition": "sum(abs(term)) / max(abs(local_energy), cancellation_energy_floor)",
            },
            "low_amplitude": low_amplitude_events.finish(),
        },
    }
    return ConditionedStatisticsReport(record).validate()


def _coerce_range_partitions(
    raw: Mapping[str, Sequence[float]],
) -> tuple[_RangePartition, ...]:
    if not isinstance(raw, Mapping):
        raise TypeError("range_edges must be a mapping of quantity to predeclared cuts")
    actual = {str(name) for name in raw}
    required = set(REQUIRED_RANGE_QUANTITIES)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing or unknown:
        raise ValueError(
            f"range_edges must define exactly {list(REQUIRED_RANGE_QUANTITIES)!r}; "
            f"missing={missing}, unknown={unknown}"
        )
    partitions = tuple(
        _RangePartition(quantity, raw[quantity]) for quantity in REQUIRED_RANGE_QUANTITIES
    )
    condition_count = sum(partition.bin_count for partition in partitions)
    if condition_count > MAX_CONDITION_COUNT:
        raise ValueError(
            f"range partitions define {condition_count} bins, hard cap {MAX_CONDITION_COUNT}"
        )
    return partitions


def _coerce_joint_strata(raw: Sequence[Mapping[str, Any]]) -> tuple[_JointStratum, ...]:
    if len(raw) > MAX_JOINT_STRATA:
        raise ValueError(
            f"joint_strata defines {len(raw)} strata, hard cap {MAX_JOINT_STRATA}"
        )
    strata: list[_JointStratum] = []
    names: set[str] = set()
    for index, spec in enumerate(raw):
        if not isinstance(spec, Mapping):
            raise TypeError(f"joint_strata[{index}] must be a mapping")
        unknown_keys = sorted({str(key) for key in spec} - {"name", "bounds"})
        if unknown_keys:
            raise ValueError(f"joint_strata[{index}] has unknown keys {unknown_keys}")
        name = str(spec.get("name", "")).strip()
        if not name:
            raise ValueError(f"joint_strata[{index}] requires a non-empty name")
        if name in names:
            raise ValueError(f"joint stratum names must be unique, duplicate {name!r}")
        names.add(name)
        raw_bounds = spec.get("bounds")
        if not isinstance(raw_bounds, Mapping) or not raw_bounds:
            raise ValueError(f"joint stratum {name!r} requires non-empty bounds")
        bounds: dict[str, tuple[float, float]] = {}
        for raw_quantity, raw_interval in raw_bounds.items():
            quantity = str(raw_quantity)
            if quantity not in REQUIRED_RANGE_QUANTITIES:
                raise ValueError(
                    f"joint stratum {name!r} has unsupported quantity {quantity!r}"
                )
            if (
                not isinstance(raw_interval, Sequence)
                or isinstance(raw_interval, (str, bytes, bytearray))
                or len(raw_interval) != 2
            ):
                raise ValueError(
                    f"joint stratum {name!r} bound for {quantity!r} must be [lower, upper]"
                )
            lower, upper = (float(raw_interval[0]), float(raw_interval[1]))
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(
                    f"joint stratum {name!r} bound for {quantity!r} must be finite and increasing"
                )
            bounds[quantity] = (lower, upper)
        strata.append(_JointStratum(name=name, bounds=bounds))
    return tuple(strata)


def _coerce_probabilities(values: Sequence[float]) -> tuple[float, ...]:
    probabilities = tuple(float(value) for value in values)
    if not probabilities:
        raise ValueError("quantiles must not be empty")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("quantiles must be finite and lie in [0, 1]")
    if any(left >= right for left, right in zip(probabilities, probabilities[1:])):
        raise ValueError("quantiles must be strictly increasing")
    return probabilities


def _coerce_nonnegative_sequence(
    values: Sequence[float],
    *,
    name: str,
    require_nonempty: bool,
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if require_nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{name} must contain finite non-negative values")
    if any(left >= right for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def _bounded_positive_int(value: int, *, name: str, maximum: int) -> int:
    integer = int(value)
    if integer < 1 or integer > maximum:
        raise ValueError(f"{name} must lie in [1, {maximum}], got {integer}")
    return integer


def _positive_finite(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not number > 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return number


def _nonnegative_finite(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _stream_artifact(
    artifact: TrajectoryRecordArtifact,
    *,
    chunk_size: int,
    consume: Callable[[_ParsedChunk], None],
) -> _StreamReceipt:
    """Stream one identity-pinned pass and return its measured receipt."""

    if artifact.path.stat().st_size != artifact.byte_count:
        raise ValueError("trajectory CSV byte_count changed before streaming pass")
    expected_header = _expected_header(artifact)
    digest = hashlib.sha256()
    measured_bytes = 0
    row_count = 0

    with artifact.path.open("rb") as handle:

        def decoded_lines():
            nonlocal measured_bytes
            for encoded in handle:
                digest.update(encoded)
                measured_bytes += len(encoded)
                yield encoded.decode("utf-8")

        reader = csv.DictReader(decoded_lines())
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError(
                "trajectory CSV header disagrees with typed artifact schema: "
                f"actual={reader.fieldnames!r}, expected={list(expected_header)!r}"
            )
        pending: list[dict[str, str]] = []
        for row in reader:
            pending.append(row)
            if len(pending) == chunk_size:
                parsed = _parse_chunk(artifact, pending, row_offset=row_count)
                consume(parsed)
                row_count += len(pending)
                pending = []
        if pending:
            parsed = _parse_chunk(artifact, pending, row_offset=row_count)
            consume(parsed)
            row_count += len(pending)

    receipt = _StreamReceipt(
        csv_sha256=digest.hexdigest(),
        byte_count=measured_bytes,
        row_count=row_count,
    )
    mismatches: list[str] = []
    if receipt.csv_sha256 != artifact.csv_sha256:
        mismatches.append(
            f"csv_sha256 measured={receipt.csv_sha256} artifact={artifact.csv_sha256}"
        )
    if receipt.byte_count != artifact.byte_count:
        mismatches.append(
            f"byte_count measured={receipt.byte_count} artifact={artifact.byte_count}"
        )
    if receipt.row_count != artifact.row_count:
        mismatches.append(f"row_count measured={receipt.row_count} artifact={artifact.row_count}")
    if artifact.path.stat().st_size != artifact.byte_count:
        mismatches.append("trajectory CSV byte_count changed during streaming pass")
    if mismatches:
        raise ValueError("trajectory CSV identity changed during streaming pass: " + "; ".join(mismatches))
    return receipt


def _expected_header(artifact: TrajectoryRecordArtifact) -> tuple[str, ...]:
    return (
        "sample_index",
        "draw_index",
        "walker_index",
        "local_energy",
        *(f"term/{name}" for name in artifact.term_names),
        "logabs",
        "sign",
        "finite",
        *(
            f"position/electron_{electron}/axis_{axis}"
            for electron in range(artifact.n_electrons)
            for axis in range(artifact.spatial_dim)
        ),
    )


def _parse_chunk(
    artifact: TrajectoryRecordArtifact,
    rows: Sequence[dict[str, str]],
    *,
    row_offset: int,
) -> _ParsedChunk:
    sample_indices = torch.tensor([int(row["sample_index"]) for row in rows], dtype=torch.int64)
    draw_indices = torch.tensor([int(row["draw_index"]) for row in rows], dtype=torch.int64)
    walker_indices = torch.tensor([int(row["walker_index"]) for row in rows], dtype=torch.int64)
    expected_samples = torch.arange(row_offset, row_offset + len(rows), dtype=torch.int64)
    expected_draws = expected_samples // artifact.n_walkers
    expected_walkers = expected_samples % artifact.n_walkers
    if not torch.equal(sample_indices, expected_samples):
        raise ValueError("trajectory CSV sample_index must be contiguous in draw-major order")
    if not torch.equal(draw_indices, expected_draws) or not torch.equal(
        walker_indices,
        expected_walkers,
    ):
        raise ValueError("trajectory CSV must be the complete ordered [draw, walker] grid")

    positions = torch.tensor(
        [
            [
                [
                    _parse_float(row[f"position/electron_{electron}/axis_{axis}"])
                    for axis in range(artifact.spatial_dim)
                ]
                for electron in range(artifact.n_electrons)
            ]
            for row in rows
        ],
        dtype=torch.float64,
    )
    local_energy = torch.tensor(
        [_parse_float(row["local_energy"]) for row in rows],
        dtype=torch.float64,
    )
    serialized_finite = torch.tensor(
        [_parse_bool(row["finite"]) for row in rows],
        dtype=torch.bool,
    )
    if not torch.equal(serialized_finite, torch.isfinite(local_energy)):
        raise ValueError("trajectory CSV finite column must equal isfinite(local_energy)")
    term_energies = {
        f"term/{name}": torch.tensor(
            [_parse_float(row[f"term/{name}"]) for row in rows],
            dtype=torch.float64,
        )
        for name in artifact.term_names
    }
    logabs = torch.tensor([_parse_float(row["logabs"]) for row in rows], dtype=torch.float64)
    sign = torch.tensor([_parse_float(row["sign"]) for row in rows], dtype=torch.float64)
    if artifact.atomic_configuration is None:
        raise ValueError(
            "trajectory CSV geometry quantities require nuclear geometry; "
            "the record declares atomic_configuration as absent"
        )
    atoms = artifact.atomic_configuration.to(device="cpu", dtype=torch.float64)
    batch = ElectronBatch(
        positions=positions,
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges,
        atomic_configuration=atoms,
    )
    geometry = two_electron_atomic_geometry(batch)
    quantities = {
        "minimum_electron_nuclear_radius": geometry.minimum_electron_nuclear_radius,
        "electron_electron_distance": geometry.electron_electron_distance,
        "maximum_electron_nuclear_radius": geometry.maximum_electron_nuclear_radius,
        "hyperradius": geometry.hyperradius,
        "cos_theta12": geometry.cos_theta12,
        "logabs": logabs,
    }
    return _ParsedChunk(
        rows=tuple(dict(row) for row in rows),
        sample_index=sample_indices,
        draw_index=draw_indices,
        walker_index=walker_indices,
        positions=positions,
        local_energy=local_energy,
        term_energies=term_energies,
        logabs=logabs,
        sign=sign,
        quantities=quantities,
        angle_defined=geometry.angle_defined,
        angle_undefined_at_coalescence=geometry.angle_undefined_at_coalescence,
    )


def _full_record(chunk: _ParsedChunk, row_index: int) -> dict[str, Any]:
    geometry = {
        name: _json_number(float(values[row_index].item()))
        for name, values in chunk.quantities.items()
        if name != "logabs"
    }
    geometry["angle_defined"] = bool(chunk.angle_defined[row_index].item())
    if geometry["angle_defined"]:
        geometry["angle_domain_status"] = "defined"
    elif bool(chunk.angle_undefined_at_coalescence[row_index].item()):
        geometry["angle_domain_status"] = "undefined_at_electron_nucleus_coalescence"
    else:
        geometry["angle_domain_status"] = "undefined_from_nonfinite_geometry"
    return {
        "row": dict(chunk.rows[row_index]),
        "derived_geometry": geometry,
    }


def _parse_float(value: str) -> float:
    return float(value)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"trajectory CSV boolean must be True or False, got {value!r}")


def _json_number(value: float) -> float | str:
    if math.isfinite(value):
        return value
    return "inf" if value > 0 else "-inf" if value < 0 else "nan"


def _probability_label(value: float) -> str:
    return f"{value:.12g}"


def _linear_quantile(ordered: Sequence[float], probability: float) -> float | None:
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float((1.0 - weight) * ordered[lower] + weight * ordered[upper])


def _close_float(left: object, right: object, *, atol: float, rtol: float) -> bool:
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return False
    return math.isclose(left_float, right_float, abs_tol=atol, rel_tol=rtol)


__all__ = [
    "CONDITIONED_LOCAL_ENERGY_SCHEMA",
    "DEFAULT_MIN_OCCUPIED_DRAWS",
    "DRAW_RATIO_ESTIMATOR_ID",
    "MAX_CONDITION_COUNT",
    "MAX_CCDF_THRESHOLD_COUNT",
    "MAX_EVENT_RECORD_CAP",
    "MAX_JOINT_STRATA",
    "MAX_QUANTILE_SAMPLE_CAP",
    "REQUIRED_RANGE_QUANTITIES",
    "ConditionedStatisticsReport",
    "produce_conditioned_local_energy_statistics",
]
