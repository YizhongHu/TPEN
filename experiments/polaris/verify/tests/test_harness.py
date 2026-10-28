"""CPU-only tests for the versioned Polaris harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import re
import tomllib


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
    harness_files = [path for path in ROOT.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    source = "\n".join(path.read_text() for path in harness_files)
    assert not re.search(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", source)
    run_root_marker = "/" + "eagle/"
    assert run_root_marker not in source
    planted = source + "\nrun_root=" + run_root_marker + "fake\n" + "a" * 40
    assert re.search(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", planted)
    assert run_root_marker in planted


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
            {"name": "tpen", "source": {"editable": "."}, "optional-dependencies": {"cpu": [{"name": "torch"}]}},
            {"name": "pytest", "version": "9.0.3"},
            {"name": "typeguard", "version": "4.5.2"},
            {"name": "torch", "version": "2.13.0+rocm7.1", "source": {"registry": "https://download.pytorch.org/whl/rocm7.1"}},
            {"name": "torch", "version": "2.12.0", "source": {"registry": "https://download.pytorch.org/whl/cpu"}, "marker": "sys_platform == 'darwin'"},
            {"name": "torch", "version": "2.12.0+cpu", "source": {"registry": "https://download.pytorch.org/whl/cpu"}, "marker": "sys_platform != 'darwin'"},
        ]
    }
    assert job.expected_versions(lock) == {"torch": "2.12.0+cpu", "pytest": "9.0.3", "typeguard": "4.5.2"}


def test_cpu_lock_derivation_control_rejects_missing_cpu_variant() -> None:
    try:
        job.expected_versions({"package": [{"name": "tpen", "source": {"editable": "."}, "optional-dependencies": {"cpu": [{"name": "torch"}]}}, {"name": "torch", "version": "2.13.0+rocm7.1", "source": {"registry": "https://download.pytorch.org/whl/rocm7.1"}}]})
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a lock without CPU torch resolutions")


def test_cpu_lock_derivation_matches_repository_lock() -> None:
    expected = job.expected_versions(tomllib.loads((ROOT.parents[2] / "uv.lock").read_text()))
    assert expected == {"torch": "2.12.0+cpu", "pytest": "9.0.3", "typeguard": "4.5.2"}


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


def test_explicit_canary_is_collected() -> None:
    repo = ROOT.parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "fixtures" / "canary.py")],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "test_deliberate_failure" in result.stdout


def test_stored_job_has_bash_and_manifest_boundary() -> None:
    job = (ROOT / "job.pbs").read_text()
    assert job.startswith("#!/usr/bin/env bash")
    assert "#PBS -S /bin/bash" in job
    assert "RUN_ENV_PATH" in job
    assert "source \"$RUN_ENV_PATH\"" in job
    assert 'exec "$RUN_PYTHON"' in job
    assert "qsub" not in job


def test_static_defect_scanners_have_positive_and_failing_controls() -> None:
    files = [ROOT / name for name in ("README.md", "facility_profile.json", "stage.py", "job.py", "driver.py", "job.pbs")]
    source = "\n".join(path.read_text() for path in files)
    assert "#!/bin/sh" not in source
    assert "LD_LIBRARY_PATH" not in source
    assert "<<" not in source
    planted = source + "\n#!/bin/sh\nLD_LIBRARY_PATH=bad\ncat <<EOF\nscript\nEOF\n"
    assert "#!/bin/sh" in planted
    assert "LD_LIBRARY_PATH" in planted
    assert "<<" in planted


def test_host_and_receipt_controls() -> None:
    assert job.host_matches("nid001", ["nid001.polaris.alcf.anl.gov"])
    assert not job.host_matches("nid001", ["nid002.polaris.alcf.anl.gov"])
    manifest = {"target_sha": "a" * 40, "cache_dir": "/cache"}
    assert job.receipt_matches(manifest, dict(manifest))
    assert not job.receipt_matches(manifest, {"target_sha": "b" * 40, "cache_dir": "/cache"})
    with tempfile.TemporaryDirectory() as raw:
        receipt = Path(raw) / "receipt.json"
        assert job.receipt_code(manifest, receipt) == 60
        receipt.write_text(json.dumps({"target_sha": "b" * 40, "cache_dir": "/cache"}))
        assert job.receipt_code(manifest, receipt) == 61
        receipt.write_text(json.dumps(dict(manifest)))
        assert job.receipt_code(manifest, receipt) == 0


def test_head_mismatch_control(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Harness Test"], check=True)
    (repo / "file").write_text("one")
    subprocess.run(["git", "-C", str(repo), "add", "file"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "one"], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert job.head_code(repo, head) == 0
    assert job.head_code(repo, "b" * 40) == 72


def test_required_ancestor_control(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Harness Test"], check=True)
    (repo / "file").write_text("one")
    subprocess.run(["git", "-C", str(repo), "add", "file"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "one"], check=True)
    ancestor = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    (repo / "file").write_text("two")
    subprocess.run(["git", "-C", str(repo), "commit", "-qam", "two"], check=True)
    target = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert stage._is_ancestor(repo, ancestor, target)
    assert not stage._is_ancestor(repo, target, ancestor)


def test_driver_exit_and_evidence_boundaries(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        repo = Path(raw) / "repo"
        evidence = Path(raw) / "evidence"
        repo.mkdir()
        (repo / "test_arms.py").write_text(
            "import pytest\n"
            "def test_pass(): pass\n"
            "def test_fail(): assert False\n"
            "@pytest.mark.skip\n"
            "def test_skip(): pass\n"
        )
        manifest = Path(raw) / "manifest.json"

        def invoke(arms, name):
            manifest.write_text(json.dumps({"target_sha": "a" * 40, "target_repo": str(repo), "arms": arms}))
            out = evidence / name
            monkeypatch.setattr(sys, "argv", ["driver.py", "--manifest", str(manifest), "--evidence", str(out), "--job-python", sys.executable])
            return driver.main(), out

        rc, out = invoke(["test_arms.py::test_pass"], "green")
        assert rc == 0
        assert all(path.parent == out for path in out.iterdir())
        rc, out = invoke(["test_arms.py"], "red")
        assert rc == 1
        rows = json.loads((out / "results.json").read_text())["arms"][0]
        assert rows["counts"]["failed"] == 1 and rows["counts"]["passed"] == 1
        rc, _ = invoke(["test_arms.py::test_skip"], "skipped")
        assert rc == 1

        class Completed:
            returncode = 1

        monkeypatch.setattr(driver.subprocess, "run", lambda *args, **kwargs: Completed())
        missing = evidence / "missing-junit"
        missing.mkdir()
        result = driver.run_arm("missing", repo, "test_arms.py::test_pass", missing)
        assert result["status"] == "FAIL" and result["reason"] == "missing JUnit"
