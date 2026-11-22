#!/usr/bin/env python3
"""Reconcile JUnit counts with pytest's terminal summary and list skips."""

from __future__ import annotations

import json
import sys
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from path_guard import assert_write_path


_SUMMARY_RE = re.compile(r"(?P<count>\d+)\s+(?P<kind>passed|failed|errors?|skipped)\b")
_SUBTEST_SUMMARY_RE = re.compile(
    r"(?P<count>\d+)\s+subtests?\s+(?P<kind>passed|failed|errors?|skipped)\b"
)


def main() -> int:
    if len(sys.argv) not in {4, 5, 6}:
        raise SystemExit(
            "usage: reconcile_counts.py JUNIT PYTEST_LOG OUTPUT [SKIP_MANIFEST] [COLLECTION_JSON]"
        )
    junit_path, log_path, output_path = map(Path, sys.argv[1:4])
    skip_path = Path(sys.argv[4]) if len(sys.argv) >= 5 else None
    collection_path = Path(sys.argv[5]) if len(sys.argv) == 6 else None
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    keys = ("tests", "failures", "errors", "skipped")
    totals = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in keys}
    terminal_lines = [
        line.strip()
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if _SUMMARY_RE.search(line)
    ]
    if not terminal_lines:
        raise SystemExit(f"pytest terminal summary not found: {log_path}")
    summary = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for match in _SUMMARY_RE.finditer(terminal_lines[-1]):
        kind = match.group("kind")
        if kind == "error":
            kind = "errors"
        elif kind == "errors":
            kind = "errors"
        summary[kind] += int(match.group("count"))
    # Pytest reports successful subtests separately from the test-case count,
    # while JUnit records each subtest as a testcase.  Include that explicit
    # terminal-summary population in the observed totals before reconciling.
    for match in _SUBTEST_SUMMARY_RE.finditer(terminal_lines[-1]):
        kind = match.group("kind")
        if kind == "error":
            kind = "errors"
        elif kind == "errors":
            kind = "errors"
        summary[kind] += int(match.group("count"))
    expected_passed = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    observed = {
        "passed": summary["passed"],
        "failed": summary["failed"],
        "errors": summary["errors"],
        "skipped": summary["skipped"],
    }
    expected = {
        "passed": expected_passed,
        "failed": totals["failures"],
        "errors": totals["errors"],
        "skipped": totals["skipped"],
    }
    if observed != expected:
        raise SystemExit(
            f"pytest/JUnit summary mismatch expected={expected} observed={observed} "
            f"summary_line={terminal_lines[-1]!r}"
        )
    skips = []
    for case in root.iter("testcase"):
        skipped = case.find("skipped")
        if skipped is None:
            continue
        props = {
            item.attrib.get("name"): item.attrib.get("value", "")
            for item in case.findall("./properties/property")
        }
        nodeid = props.get("review_nodeid") or (
            f"{case.attrib.get('classname')}::{case.attrib.get('name')}"
        )
        reason = skipped.attrib.get("message", "").strip()
        detail = " ".join("".join(skipped.itertext()).split()).strip()
        if detail and detail != reason:
            reason = f"{reason}: {detail}" if reason else detail
        if not reason:
            raise SystemExit(f"skip reason missing: {nodeid}")
        skips.append(f"{nodeid}\t{reason}")
    if len(skips) != totals["skipped"]:
        raise SystemExit(
            f"skip manifest mismatch junit={totals['skipped']} rows={len(skips)}"
        )
    if len(sys.argv) >= 5:
        assert_write_path(skip_path)
        skip_path.write_text(
            "".join(f"{line}\n" for line in skips), encoding="utf-8"
        )
    expected_tests = -1
    collection_extra = 0
    if collection_path is not None:
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
        expected_tests = len(collection.get("collected", []))
        collection_extra = len(list(root.iter("testcase"))) - expected_tests
    assert_write_path(output_path)
    with output_path.open("w", encoding="utf-8") as stream:
        # Keep the stable key name used by the mutation population comparator.
        for key in keys:
            output_key = "skips" if key == "skipped" else key
            stream.write(f"{output_key}={totals[key]}\n")
        stream.write(f"passed={summary['passed']}\n")
        stream.write(f"subtests_passed={sum(int(match.group('count')) for match in _SUBTEST_SUMMARY_RE.finditer(terminal_lines[-1]))}\n")
        stream.write(f"junit_cases={len(list(root.iter('testcase')))}\n")
        stream.write(f"collection_extra={collection_extra}\n")
        stream.write(f"expected_tests={expected_tests}\n")
    print(
        f"junit={junit_path} summary={terminal_lines[-1]} "
        f"tests={totals['tests']} failures={totals['failures']} "
        f"errors={totals['errors']} skipped={totals['skipped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
