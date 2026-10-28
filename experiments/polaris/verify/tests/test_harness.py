"""CPU-only tests for the versioned Polaris harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import re


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("polaris_stage", ROOT / "stage.py")
stage = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(stage)
driver_spec = importlib.util.spec_from_file_location("polaris_driver", ROOT / "driver.py")
driver = importlib.util.module_from_spec(driver_spec)
assert driver_spec.loader is not None
driver_spec.loader.exec_module(driver)
job_spec = importlib.util.spec_from_file_location("polaris_job", ROOT / "job.py")
job = importlib.util.module_from_spec(job_spec)
assert job_spec.loader is not None
job_spec.loader.exec_module(job)


def test_valid_arm_and_override_control() -> None:
    with tempfile.TemporaryDirectory() as raw:
        repo = Path(raw)
        (repo / "tests").mkdir()
        (repo / "tests" / "test_one.py").write_text("def test_one(): pass\n")
        assert stage.validate_arm("tests/test_one.py", repo) == "tests/test_one.py"
        for bad in ("runner.trainer.max_steps=1", "../outside.py", "-k test_one"):
            try:
                stage.validate_arm(bad, repo)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted invalid arm: {bad}")


def test_literal_env_round_trip_and_single_manifest_source() -> None:
    with tempfile.TemporaryDirectory() as raw:
        manifest = Path(raw) / "manifest with $X.json"
        env = Path(raw) / "run.env"
        manifest.write_text(json.dumps({"literal": "$X `quoted`"}))
        stage._write_literal_env(env, manifest, "/facility/python")
        assert str(manifest) in env.read_text()
        assert "EXPECTED_HEAD" not in (ROOT / "job.pbs").read_text()


def test_harness_has_no_embedded_commit_or_run_path() -> None:
    source = "\n".join(path.read_text() for path in ROOT.iterdir() if path.is_file())
    assert not re.search(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", source)
    assert "/eagle/" not in source


def test_junit_counts_reconcile_and_empty_control() -> None:
    with tempfile.TemporaryDirectory() as raw:
        xml = Path(raw) / "results.xml"
        xml.write_text("<testsuite><testcase name='ok'/><testcase name='skip'><skipped/></testcase><testcase name='bad'><failure/></testcase></testsuite>")
        assert driver.junit_counts(xml) == {"total": 3, "passed": 1, "failed": 1, "errors": 0, "skipped": 1}
        empty = Path(raw) / "empty.xml"
        empty.write_text("<testsuite/>")
        assert driver.junit_counts(empty)["total"] == 0


def test_cpu_lock_derivation_ignores_non_cpu_torch_variants() -> None:
    lock = {
        "package": [
            {"name": "pytest", "version": "9.0.3"},
            {"name": "typeguard", "version": "4.5.2"},
            {"name": "torch", "version": "2.13.0+rocm7.1", "source": {"registry": "https://download.pytorch.org/whl/rocm7.1"}},
            {"name": "torch", "version": "2.12.0", "source": {"registry": "https://download.pytorch.org/whl/cpu"}},
            {"name": "torch", "version": "2.12.0+cpu", "source": {"registry": "https://download.pytorch.org/whl/cpu"}},
        ]
    }
    assert job.expected_versions(lock) == {"torch": "2.12.0+cpu", "pytest": "9.0.3", "typeguard": "4.5.2"}


def test_cpu_lock_derivation_control_rejects_missing_cpu_variant() -> None:
    try:
        job.expected_versions({"package": [{"name": "torch", "version": "2.13.0+rocm7.1", "source": {"registry": "https://download.pytorch.org/whl/rocm7.1"}}]})
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a lock without CPU torch resolutions")


def test_canary_is_not_normal_collection_name() -> None:
    assert not (ROOT / "fixtures" / "canary.py").name.startswith("test_")
    assert "assert False" in (ROOT / "fixtures" / "canary.py").read_text()


def test_normal_collection_excludes_canary() -> None:
    repo = ROOT.parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "test_deliberate_failure" not in result.stdout


def test_stored_job_has_bash_and_manifest_boundary() -> None:
    job = (ROOT / "job.pbs").read_text()
    assert job.startswith("#!/usr/bin/env bash")
    assert "RUN_ENV_PATH" in job
    assert "source \"$RUN_ENV_PATH\"" in job
    assert "qsub" not in job
