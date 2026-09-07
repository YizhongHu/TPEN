"""Render one collected He-v1 attempt as a receipt a reader can audit.

The report states what it does not know as loudly as what it does:

- an absent value renders as ``absent``; it is never blank and never ``0``;
- every aggregate carries its coverage (``n/N rows``), so a median over two of
  nine rows cannot be read as a median over nine;
- ``local_energy_stderr`` is labelled an IID standard error, because pairing an
  IID bar against a correlation-corrected one is the comparison the
  estimator-pairing rule forbids;
- the canonical energy is the whole-trajectory estimate with its MCSE. The
  final-draw snapshot is retained beside it and labelled a snapshot, because an
  unqualified ``local_energy_mean`` is a slice of ~0.4% of the sampled records
  and was previously indistinguishable from an estimate over all of them;
- a trajectory statistic that never resolved renders ``absent`` and is never
  backfilled from the snapshot, and ``plateau_reached`` travels beside the MCSE
  because a truncated sequence understates it; and
- a row whose delivered GPU did not match its requested stratum is listed as a
  failure, not a footnote.

This module imports no ``tpen`` (``experiments/README.md``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

# Siblings are loaded study-scoped, not by bare import: experiments/ has several
# same-named modules and the first study loaded would otherwise own the bare name
# for every study after it. See experiments/toolkit/study_imports.py.
#
# The loader is reached BY PATH rather than by putting the repository root on
# sys.path. A study directory that mutates sys.path is the mechanism behind the
# very defect this import exists to fix, and he-cutover's gateway test forbids it
# outright -- so the fix must not reintroduce it in order to install itself.
import importlib.util as _tpen_importlib  # noqa: E402
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

if "_tpen_study_imports" not in _tpen_sys.modules:
    _tpen_spec = _tpen_importlib.spec_from_file_location(
        "_tpen_study_imports",
        _TpenPath(__file__).resolve().parents[3] / "experiments" / "toolkit" / "study_imports.py",
    )
    _tpen_module = _tpen_importlib.module_from_spec(_tpen_spec)
    _tpen_sys.modules["_tpen_study_imports"] = _tpen_module
    _tpen_spec.loader.exec_module(_tpen_module)
sibling = _tpen_sys.modules["_tpen_study_imports"].sibling

absence = sibling(__file__, 'absence')

collect_stage = sibling(__file__, 'collect')
layout = sibling(__file__, 'layout')
plan_stage = sibling(__file__, 'plan')

REPORT_FILENAME = "report.md"

#: Columns whose estimator has to be named wherever the number appears.
#:
#: The snapshot entries are labelled for the same reason the IID bar always was:
#: an unqualified name is what let a final-draw slice be read as an estimate over
#: the whole chain. Naming the estimand is cheaper than expecting every reader to
#: know which ``local_energy_mean`` they are looking at.
ESTIMATOR_LABELS: Mapping[str, str] = {
    "local_energy_mean": "final-draw snapshot, not the trajectory estimate",
    "local_energy_stderr": "IID stderr (not an MCSE)",
    "local_energy_variance": "final-draw snapshot",
    "local_energy_trajectory_mean": "whole-trajectory estimate",
    "local_energy_mcse": "MCSE, correlation-corrected",
    "local_energy_stderr_iid": "IID stderr over the trajectory",
    "local_energy_mcse_inflation": "MCSE / IID stderr",
    "local_energy_plateau_reached": "false ⇒ the MCSE beside it is UNDERSTATED",
}

#: The pair a reader should quote. Named once, here, so the row table and the
#: prose below cannot drift apart about which column is canonical.
HEADLINE_ENERGY_KEY = "local_energy_trajectory_mean"
HEADLINE_ERROR_KEY = "local_energy_mcse"


def read_collected(results_root: str | Path, collect_attempt_id: str) -> dict[str, Any]:
    """Read one collected attempt."""

    path = layout.collect_attempt_dir(results_root, collect_attempt_id) / collect_stage.COLLECTED_FILENAME
    payload = layout.read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"collected table {path} is not a mapping")
    return payload


def render(collected: Mapping[str, Any]) -> str:
    """Render the markdown report for one collected attempt."""

    lines: list[str] = []
    lines += _header_lines(collected)
    lines += _row_lines(collected)
    lines += _aggregate_lines(collected)
    lines += _gate_lines(collected)
    lines += _failure_lines(collected)
    return "\n".join(lines) + "\n"


def _header_lines(collected: Mapping[str, Any]) -> list[str]:
    declared = bool(collected.get("gate_spec_declared"))
    lines = [
        f"# He-v1 study report — {collected['study']}",
        "",
        f"- plan attempt: `{collected['plan_attempt_id']}` (plan hash `{collected['plan_hash']}`)",
        f"- collect attempt: `{collected['collect_attempt_id']}`",
        f"- rows: {collected['n_rows']} ({collected['n_pass']} pass, {collected['n_fail']} fail)",
        f"- gate spec source: `{collected.get('gate_spec_source', 'absent')}`",
        f"- checkpoint hashing: {'on' if collected.get('checkpoint_hashing') else 'off'}",
        "- resume/restart: forbidden; rows are sized to finish",
        "",
    ]
    bindings = collected.get("metric_namespaces") or {}
    if bindings:
        lines += [
            "Metric namespace bindings. Each metric below was read from one task's",
            "namespace and from nowhere else, for the reported column and for the gates",
            "alike, so a tolerance cannot land on a different task that happens to emit",
            "the same metric name.",
            "",
        ]
        lines += [
            f"- `{metric}` read from namespace `{bindings[metric]}`" for metric in sorted(bindings)
        ]
        lines.append("")
    if not declared:
        lines += [
            "> No tolerance was predeclared for this attempt, so every value gate reports",
            "> `absent` with its observed value retained. That is the honest state of an",
            "> ungated run: the thresholds are predeclared in H-F1, and an absent gate is",
            "> never a pass.",
            "",
        ]
    return lines


def _row_lines(collected: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Rows",
        "",
        "One line per planned row, in manifest order.",
        "",
        "The canonical energy is the whole-trajectory estimate and its MCSE. The",
        "final-draw snapshot is kept beside it for comparison and is labelled as a",
        "snapshot; it is not the study's answer. `plateau` is the truncation flag",
        "that qualifies the MCSE: `False` means Geyer's sequence was cut at the",
        "window edge and the bar is UNDERSTATED. A row whose trajectory statistics",
        "never resolved shows `absent` in the trajectory columns -- it never borrows",
        "the snapshot to fill the gap.",
        "",
    ]
    header = [
        "row_id",
        "kind",
        "status",
        "seed",
        "step",
        "chain",
        "stratum",
        "delivered",
        f"energy ({ESTIMATOR_LABELS[HEADLINE_ENERGY_KEY]})",
        f"mcse ({ESTIMATOR_LABELS[HEADLINE_ERROR_KEY]})",
        "plateau",
        f"snapshot energy ({ESTIMATOR_LABELS['local_energy_mean']})",
        f"snapshot stderr ({ESTIMATOR_LABELS['local_energy_stderr']})",
        "checkpoint_sha256",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in collected["rows"]:
        identity = row["identity"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{identity['row_id']}`",
                    str(identity["kind"]),
                    str(row["status"]),
                    str(identity["seed"]),
                    _cell(identity["checkpoint_step"]),
                    _cell(identity["chain"]),
                    str(identity["requested_stratum"]),
                    _cell(identity["delivered_device"]),
                    # Headline first. These read `absent` on a row whose receipt
                    # was unresolved, and no branch here substitutes the snapshot
                    # columns to the right of them.
                    _metric(row, HEADLINE_ENERGY_KEY),
                    _metric(row, HEADLINE_ERROR_KEY),
                    _metric(row, "local_energy_plateau_reached"),
                    _metric(row, "local_energy_mean"),
                    _metric(row, "local_energy_stderr"),
                    _short_hash(_cell(identity["checkpoint_sha256"])),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _aggregate_lines(collected: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Aggregates over evaluation rows",
        "",
        "Every aggregate names how many rows supplied a value. Absent rows are excluded",
        "from the statistic and counted, never treated as zero. A metric written",
        "`<namespace>.<key>` was collected from that task alone: several tasks can share",
        "a summary class and therefore a metric name while measuring different things.",
        "",
        "| metric | coverage | mean | median | min | max |",
        "|---|---|---|---|---|---|",
    ]
    summaries = collected.get("summaries", {})
    for key in collected["metric_keys"]:
        summary = summaries.get(key)
        if not isinstance(summary, Mapping):
            continue
        label = key if key not in ESTIMATOR_LABELS else f"{key} — {ESTIMATOR_LABELS[key]}"
        coverage = f"{summary['n_present']}/{summary['n_rows']} rows"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{label}`",
                    coverage,
                    _cell(summary["mean"]),
                    _cell(summary["median"]),
                    _cell(summary["min"]),
                    _cell(summary["max"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _gate_lines(collected: Mapping[str, Any]) -> list[str]:
    counts: dict[str, dict[str, int]] = {}
    for row in collected["rows"]:
        for gate_row in row["gates"]:
            bucket = counts.setdefault(str(gate_row["name"]), {"pass": 0, "fail": 0, "absent": 0})
            bucket[str(gate_row["status"])] = bucket.get(str(gate_row["status"]), 0) + 1
    lines = ["## Gates", ""]
    if not counts:
        lines += ["No evaluation row produced a gate outcome.", ""]
        return lines
    lines += [
        "`absent` means the gate could not be decided — the metric was not emitted, the",
        "fit was unavailable, or no threshold was declared. It never means it passed.",
        "",
        "| gate | pass | fail | absent |",
        "|---|---|---|---|",
    ]
    for name in sorted(counts):
        bucket = counts[name]
        lines.append(
            f"| `{name}` | {bucket.get('pass', 0)} | {bucket.get('fail', 0)} | "
            f"{bucket.get('absent', 0)} |"
        )
    lines.append("")
    return lines


def _failure_lines(collected: Mapping[str, Any]) -> list[str]:
    failed = [row for row in collected["rows"] if row["status"] != "pass"]
    lines = ["## Failures", ""]
    if not failed:
        lines += ["None.", ""]
        return lines
    for row in failed:
        lines.append(f"- `{row['identity']['row_id']}`")
        for reason in row["reasons"]:
            lines.append(f"  - {reason}")
    lines.append("")
    return lines


def _metric(row: Mapping[str, Any], key: str) -> str:
    return _cell(row["metrics"].get(key, absence.cell(None)))


def _cell(cell: Any) -> str:
    return absence.render(absence.cell_value(cell))


def _short_hash(text: str) -> str:
    if text == absence.ABSENT_TEXT:
        return text
    return f"`{text[:12]}`"


def write_report(collected: Mapping[str, Any], *, results_root: Path, attempt_id: str) -> Path:
    """Write the rendered report for one collected attempt."""

    directory = layout.report_attempt_dir(results_root, attempt_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REPORT_FILENAME
    path.write_text(render(collected), encoding="utf-8")
    layout.write_latest(layout.stage_dir(results_root, layout.STAGE_REPORT), attempt_id)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse report command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, help="Durable study results root.")
    parser.add_argument(
        "--collect-attempt-id", default=None, help="Collect attempt (defaults to latest)."
    )
    parser.add_argument("--report-attempt-id", default=None, help="Explicit report attempt id.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render one collected attempt."""

    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    collect_attempt_id = layout.resolve_attempt_id(
        results_root, layout.STAGE_COLLECT, args.collect_attempt_id
    )
    collected = read_collected(results_root, collect_attempt_id)
    path = write_report(
        collected,
        results_root=results_root,
        attempt_id=args.report_attempt_id or plan_stage.now_attempt_id(),
    )
    print(f"[he-v1] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
