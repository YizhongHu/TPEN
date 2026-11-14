"""Aggregate baseline energies across independent random seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from experiments.baselines.records import BaselineRecord, RecordValidationError


class SpreadError(ValueError):
    """Base error for invalid or incomparable seed-spread input."""


class DuplicateSeedError(SpreadError):
    """A group contains more than one record for the same seed."""


class MissingEnergyError(SpreadError):
    """A record cannot contribute because its energy is missing."""


@dataclass(frozen=True, order=True)
class GroupKey:
    """The dimensions that define one comparable across-seed group."""

    system_id: str
    code: str
    ansatz: str | None
    estimator: str

    def to_json_dict(self) -> dict[str, Any]:
        """Return the key as a JSON object."""

        return {"system_id": self.system_id, "code": self.code, "ansatz": self.ansatz, "estimator": self.estimator}


@dataclass(frozen=True)
class SpreadSummary:
    """Summary statistics for one comparison group."""

    key: GroupKey
    n_records: int
    n_distinct_seeds: int
    n_seedless: int
    seeds: tuple[int, ...]
    run_ids: tuple[str | None, ...]
    seedless_run_ids: tuple[str | None, ...]
    mean_hartree: float | None
    min_hartree: float | None
    max_hartree: float | None
    range_hartree: float | None
    sample_std_hartree: float | None
    sem_hartree: float | None
    insufficient_seeds: bool

    def to_json_dict(self) -> dict[str, Any]:
        """Return a stable, machine-readable representation."""

        result = self.key.to_json_dict()
        result.update({
            "n_records": self.n_records, "n_distinct_seeds": self.n_distinct_seeds,
            "n_seedless": self.n_seedless, "seeds": list(self.seeds), "run_ids": list(self.run_ids),
            "seedless_run_ids": list(self.seedless_run_ids),
            "mean_hartree": self.mean_hartree, "min_hartree": self.min_hartree,
            "max_hartree": self.max_hartree, "range_hartree": self.range_hartree,
            "sample_std_hartree": self.sample_std_hartree, "sem_hartree": self.sem_hartree,
            "insufficient_seeds": self.insufficient_seeds,
        })
        return result


def summarize(records: Iterable[BaselineRecord]) -> list[SpreadSummary]:
    """Summarize energy spread for each ``(system, code, ansatz, estimator)``.

    Seedless records are counted but excluded from the statistics. Duplicate
    seeded records are rejected because averaging re-emitted measurements would
    hide a collection error.

    Raises
    ------
    MissingEnergyError
        If any input record has no energy estimate.
    DuplicateSeedError
        If a group contains more than one record for a seed.
    """

    groups: dict[GroupKey, list[BaselineRecord]] = {}
    for record in records:
        if record.energy_hartree is None:
            raise MissingEnergyError(f"missing energy for {record.system_id}/{record.code} (run_id={record.run_id!r})")
        key = GroupKey(record.system_id, record.code, record.ansatz, record.estimator)
        groups.setdefault(key, []).append(record)

    summaries: list[SpreadSummary] = []
    for key in sorted(
        groups,
        key=lambda item: (
            item.system_id,
            item.code,
            item.ansatz is not None,
            item.ansatz or "",
            item.estimator,
        ),
    ):
        group = groups[key]
        seeded: dict[int, BaselineRecord] = {}
        seedless = 0
        seedless_run_ids: list[str | None] = []
        for record in group:
            if record.seed is None:
                seedless += 1
                seedless_run_ids.append(record.run_id)
            elif record.seed in seeded:
                raise DuplicateSeedError(f"duplicate seed {record.seed} in {key}")
            else:
                seeded[record.seed] = record
        numeric = [float(seeded[seed].energy_hartree) for seed in sorted(seeded)]
        n = len(numeric)
        sample_std = statistics.stdev(numeric) if n >= 2 else None
        summaries.append(SpreadSummary(
            key=key, n_records=len(group), n_distinct_seeds=n, n_seedless=seedless,
            seeds=tuple(sorted(seeded)), run_ids=tuple(record.run_id for record in group),
            seedless_run_ids=tuple(seedless_run_ids),
            mean_hartree=statistics.fmean(numeric) if numeric else None,
            min_hartree=min(numeric) if numeric else None, max_hartree=max(numeric) if numeric else None,
            range_hartree=(max(numeric) - min(numeric)) if numeric else None,
            sample_std_hartree=sample_std,
            sem_hartree=sample_std / math.sqrt(n) if sample_std is not None else None,
            insufficient_seeds=n < 3,
        ))
    return summaries


def read_results(path: Path) -> list[BaselineRecord]:
    """Read and validate a collector-produced ``results.jsonl`` file."""

    records: list[BaselineRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(BaselineRecord.from_json_dict(json.loads(line)))
        except (json.JSONDecodeError, RecordValidationError) as error:
            raise SpreadError(f"invalid record at {path}:{line_number}: {error}") from error
    return records


def records_from_inputs(results: Sequence[Path], run_roots: Sequence[Path]) -> list[BaselineRecord]:
    """Read results files and/or collect validated records from run roots."""

    records = [record for path in results for record in read_results(path)]
    for root in run_roots:
        # Keep the JSONL-only path stdlib-only.  The existing collector is
        # imported only when its registry-backed run-root reader is requested.
        from experiments.baselines.collect import collect

        report = collect(root)
        if report.failures:
            details = "; ".join(f"{path}: {reason}" for path, reason in report.failures)
            raise SpreadError(f"invalid records under {root}: {details}")
        records.extend(report.records)
    return records


def markdown_table(summaries: Sequence[SpreadSummary]) -> str:
    """Render a compact Markdown table for human review."""

    lines = [
        "| system | code | ansatz | estimator | seeds | mean (Ha) | range (Ha) | sample std (Ha) | SEM (Ha) | flag |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        data = summary.to_json_dict()
        data["ansatz"] = summary.key.ansatz or ""
        data["flag"] = "insufficient_seeds" if summary.insufficient_seeds else ""
        lines.append("| {system_id} | {code} | {ansatz} | {estimator} | {n_distinct_seeds} | {mean_hartree} | {range_hartree} | {sample_std_hartree} | {sem_hartree} | {flag} |".format(
            **data
        ))
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", default=[], help="results.jsonl (repeatable)")
    parser.add_argument("--run-root", type=Path, action="append", default=[], help="run tree (repeatable)")
    parser.add_argument("--output", type=Path, required=True, help="JSON output")
    parser.add_argument("--markdown-output", type=Path, help="optional Markdown table output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the seed-spread aggregator and write its outputs."""

    args = _parser().parse_args(argv)
    if not args.results and not args.run_root:
        raise SystemExit("one of --results or --run-root is required")
    summaries = summarize(records_from_inputs(args.results, args.run_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([item.to_json_dict() for item in summaries], indent=2) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown_table(summaries), encoding="utf-8")
    print(f"summarized {len(summaries)} group(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DuplicateSeedError", "GroupKey", "MissingEnergyError", "SpreadError", "SpreadSummary", "main", "markdown_table", "read_results", "records_from_inputs", "summarize"]
