"""Run selected pytest arms and make JUnit counts the verification verdict."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def junit_counts(path: Path) -> dict[str, int]:
    """Return reconciled testcase counts from one JUnit document."""
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failed = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    return {"total": len(cases), "passed": len(cases) - failed - errors - skipped, "failed": failed, "errors": errors, "skipped": skipped}


def run_arm(name: str, repo: Path, arm: str, evidence: Path) -> dict:
    xml = evidence / f"{name}.xml"
    log = evidence / f"{name}.log"
    command = [sys.executable, "-B", "-m", "pytest", arm, "-q", "--junitxml", str(xml)]
    with log.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": command, "cwd": str(repo)}) + "\n")
        rc = subprocess.run(command, cwd=repo, env={**os.environ, "PYTHONPATH": str(repo), "PYTEST_ADDOPTS": ""}, stdout=stream, stderr=subprocess.STDOUT).returncode
    result = {"arm": name, "selection": arm, "rc": rc, "junit": str(xml), "log": str(log)}
    if not xml.is_file():
        result.update(status="FAIL", reason="missing JUnit")
    else:
        try:
            result["counts"] = junit_counts(xml)
            counts = result["counts"]
            result["status"] = "PASS" if rc == 0 and counts["total"] > 0 and counts["failed"] == 0 and counts["errors"] == 0 else "FAIL"
        except (ET.ParseError, OSError) as exc:
            result.update(status="FAIL", reason=f"unparseable JUnit: {exc}")
    (evidence / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    repo = Path(manifest["target_repo"])
    evidence = Path(args.evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    results = [run_arm(f"arm-{index}", repo, arm, evidence) for index, arm in enumerate(manifest["arms"])]
    output = {"manifest": str(args.manifest), "target_sha": manifest["target_sha"], "arms": results, "all_pass": all(row["status"] == "PASS" for row in results)}
    (evidence / "results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
