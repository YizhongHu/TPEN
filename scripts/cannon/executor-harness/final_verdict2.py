#!/usr/bin/env python3
"""Fail-closed adapter for the S2b-1 full exact-head final matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from helpers.path_guard import assert_write_path

ARMS = ("c1-py312-parsl", "c2-py314-parsl", "c3-py312-none", "c4-py314-none")


def expected_suite(killer: str) -> str:
    path = killer.split("::", 1)[0].replace("\\", "/")
    return "executor" if path == "executor/tests" or path.startswith("executor/tests/") else "root"


def load_spec() -> dict[str, object]:
    raw = json.loads(Path(os.environ["MUTANT_SPEC"]).read_text(encoding="utf-8"))
    rows = [dict(row) for row in raw.get("mutants", [])]
    legacy_reader = raw.get("reader_mutant")
    if legacy_reader is not None and not any(row.get("id") == legacy_reader.get("id") for row in rows):
        reader = dict(legacy_reader)
        reader.setdefault("suite", "root")
        rows.append(reader)
    for row in rows:
        mutant_id, killer, suite = row.get("id"), row.get("killer"), row.get("suite")
        if not isinstance(mutant_id, str) or not isinstance(killer, str):
            raise ValueError("each mutant requires string id and killer")
        if suite not in {"root", "executor"}:
            raise ValueError(f"{mutant_id}: suite must be root or executor")
        required = expected_suite(killer)
        if suite != required:
            raise ValueError(f"{mutant_id}: killer {killer!r} requires suite={required}, got {suite}")
    return {**raw, "mutants": rows}


SPEC = load_spec()
MUTANT_ROWS = tuple(SPEC["mutants"])
REAL_CORE_IDS = (
    "tests/test_review_p5b_r1.py::test_r1_t0_real_core_module_help_prints_usage",
    "tests/test_review_p5d_r1.py::test_r1_t7_real_core_pinned_overrides_win_and_evidence_names_actual_run",
    "tests/test_review_p5d_r1.py::test_r1_t11_parent_write_trace_includes_log_opens_and_successful_members",
)

def kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if not sep or not key or key in values:
            raise ValueError(f"invalid key/value file: {path}")
        values[key] = value
    return values

def counts(path: Path) -> dict[str, int]:
    return {key: int(value) for key, value in kv(path).items()}

def junit_skip_count(path: Path) -> int:
    root = ET.parse(path).getroot()
    return sum(1 for case in root.iter("testcase") if case.find("skipped") is not None)


def selector_for_run(row: dict[str, object]) -> str:
    killer = str(row["killer"])
    if row["suite"] == "executor" and killer.startswith("executor/"):
        return killer.removeprefix("executor/")
    return killer


def _matches_killer(nodeid: str, killer: str) -> bool:
    """Exact id, or a parametrization of it (``killer[...]``)."""

    return nodeid == killer or nodeid.startswith(killer + "[")


def baseline_green(root: Path, suite_key: str, killer: str) -> bool:
    try:
        payload = json.loads((root / "evidence" / f"{suite_key}-collect.json").read_text(encoding="utf-8"))
        # A parametrized killer is collected as ``killer[param]`` (Cannon 59466935: N1 was falsely
        # NOT_GREEN because only the bare id was matched). Every parametrization must pass.
        rows = [row for row in payload.get("reports", []) if row.get("when") == "call" and _matches_killer(str(row.get("nodeid", "")), killer)]
        return bool(rows) and all(row.get("outcome") == "passed" for row in rows)
    except (OSError, ValueError, TypeError):
        return False


def mutant_failed(root: Path, prefix: str, killer: str) -> bool:
    try:
        junit = ET.parse(root / f"{prefix}.xml").getroot()
        rows = []
        for case in junit.iter("testcase"):
            properties = {item.attrib.get("name"): item.attrib.get("value", "") for item in case.findall("./properties/property")}
            if _matches_killer(properties.get("review_nodeid") or "", killer) or _matches_killer(case.attrib.get("name", ""), killer.rsplit("::", 1)[-1]) or killer == case.attrib.get("name", ""):
                rows.append(case)
        return bool(rows) and any(case.find("failure") is not None or case.find("error") is not None for case in rows)
    except (OSError, ET.ParseError):
        return False

def check_run(root: Path, prefix: str, *, require_pass: bool) -> tuple[bool, dict[str, object], list[str]]:
    reasons: list[str] = []
    detail: dict[str, object] = {}
    try:
        run_counts = counts(root / f"{prefix}-counts.txt")
        step = kv(root / f"{prefix}-step.rc")
    except (OSError, ValueError) as error:
        return False, detail, [f"{prefix}:unreadable={error}"]
    derived = run_counts.get("tests", 0) - run_counts.get("failures", 0) - run_counts.get("errors", 0) - run_counts.get("skips", 0)
    run_counts["derived_passed"] = derived
    if run_counts.get("passed", derived) != derived:
        reasons.append(f"{prefix}:passed-mismatch")
    if step.get("collection-errors") != "0" or step.get("reconcile-counts") != "0":
        reasons.append(f"{prefix}:harness-step")
    expected_tests = run_counts.get("expected_tests", -1)
    junit_extra = run_counts.get("subtests_passed", 0) + run_counts.get("collection_extra", 0)
    if run_counts.get("tests", 0) <= 0 or (
        expected_tests >= 0 and run_counts.get("tests") - expected_tests != junit_extra
    ):
        reasons.append(f"{prefix}:collection-population")
    if run_counts.get("tests") != run_counts.get("passed", 0) + run_counts.get("failures", 0) + run_counts.get("errors", 0) + run_counts.get("skips", 0):
        reasons.append(f"{prefix}:count-partition")
    if require_pass and (step.get("pytest") != "0" or run_counts.get("failures", 0) != 0 or run_counts.get("errors", 0) != 0):
        reasons.append(f"{prefix}:not-pass")
    if not require_pass and (run_counts.get("failures", 0) + run_counts.get("errors", 0) <= 0 or step.get("pytest") not in {"1", "2"}):
        reasons.append(f"{prefix}:not-killed")
    try:
        skip_path = root / f"{prefix}-skip-manifest.txt"
        skip_count = len([line for line in skip_path.read_text(encoding="utf-8").splitlines() if line.strip()]) if skip_path.is_file() else junit_skip_count(root / f"{prefix}.xml")
        if skip_count != run_counts.get("skips"):
            reasons.append(f"{prefix}:skip-manifest")
    except (OSError, ET.ParseError) as error:
        reasons.append(f"{prefix}:skip-manifest={error}")
    detail["counts"] = run_counts
    detail["step"] = step
    return not reasons, detail, reasons

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: final_verdict2.py RUN_ROOT", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    reasons: list[str] = []
    detail: dict[str, object] = {}
    verdict = "PASS"
    for arm in ARMS:
        good, info, rows = check_run(root, arm, require_pass=True)
        detail[arm] = info
        reasons.extend(rows)
        if not good:
            verdict = "FAIL" if not any("unreadable" in row for row in rows) else "INVALID"
        if arm in ARMS[:2]:
            try:
                real = json.loads((root / "evidence" / f"{arm}-real-core.json").read_text(encoding="utf-8"))
                detail[f"{arm}:real_core"] = real
                rows = real.get("required", [])
                if len(rows) != len(REAL_CORE_IDS) or any(not row.get("collected") or row.get("call_outcome") != "passed" for row in rows):
                    reasons.append(f"{arm}:real-core-not-measured")
                    verdict = "FAIL"
            except (OSError, ValueError, TypeError) as error:
                reasons.append(f"{arm}:real-core-unreadable={error}")
                verdict = "INVALID"
    good, info, rows = check_run(root, "root", require_pass=True)
    detail["root"] = info
    reasons.extend(rows)
    if not good:
        verdict = "FAIL" if not any("unreadable" in row for row in rows) else "INVALID"
    mutant_detail: dict[str, object] = {}
    for row in MUTANT_ROWS:
        mutant = str(row["id"])
        suite = str(row["suite"])
        killer = selector_for_run(row)
        run_keys = ARMS if suite == "executor" else ("root",)
        mutant_rows: dict[str, object] = {}
        mutant_ok = True
        for arm in run_keys:
            prefix = f"mutant-{mutant}-{arm}"
            good, info, rows = check_run(root, prefix, require_pass=False)
            mutant_rows[arm] = {"result": "NOT_KILLED", **info, "reasons": rows}
            reasons.extend(rows)
            mutant_ok &= good
            try:
                edit = kv(root / "evidence" / f"{prefix}-edit.rc")
                needle = kv(root / "evidence" / f"{prefix}-needle.txt")
                if edit.get("edit_rc") != "0" or needle.get("needle_count") != "1":
                    reasons.append(f"{prefix}:edit-or-needle")
                    mutant_ok = False
                cache = kv(root / "evidence" / f"{prefix}-pycache-before.txt")
                if cache.get("pycache_count") != "0":
                    reasons.append(f"{prefix}:stale-pycache")
                    mutant_ok = False
                compile_rc = kv(root / "evidence" / f"{prefix}-compile.rc")
                if compile_rc.get("compile_rc") != "0":
                    reasons.append(f"{prefix}:offline-compile")
                    mutant_ok = False
                baseline_key = arm if suite == "executor" else "root"
                base_green = baseline_green(root, baseline_key, killer)
                mutant_red = mutant_failed(root, prefix, killer)
                mutant_rows[arm].update({"killing_test": killer, "baseline": "GREEN" if base_green else "NOT_GREEN", "mutant": "RED" if mutant_red else "NOT_RED"})
                if not base_green or not mutant_red:
                    reasons.append(f"{prefix}:baseline-green-rule")
                    mutant_ok = False
                elif good:
                    mutant_rows[arm]["result"] = "KILLED"
            except (OSError, ValueError) as error:
                reasons.append(f"{prefix}:edit-evidence={error}")
                mutant_ok = False
        mutant_detail[mutant] = {"suite": suite, "result": "KILLED" if mutant_ok else "NOT_KILLED", "runs": mutant_rows}
        if not mutant_ok:
            verdict = "FAIL"
    detail["mutants"] = mutant_detail
    if any("unreadable" in row or "edit-evidence" in row for row in reasons):
        verdict = "INVALID"
    payload = {"verdict": verdict, "reasons": reasons, **detail}
    verdict_path = root / "final-verdict.json"
    assert_write_path(verdict_path)
    verdict_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if verdict == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
