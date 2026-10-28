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
import copy


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
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
        assert stage.validate_arm_syntax("tests/test_one.py::test_one[n=3]") == "tests/test_one.py::test_one[n=3]"
        for bad in ("runner.trainer.max_steps=1", "../outside.py", "-k test_one"):
            try:
                stage.validate_arm(bad, repo)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted invalid arm: {bad}")


def test_literal_env_round_trip_and_single_manifest_source() -> None:
    with tempfile.TemporaryDirectory() as raw:
        manifest = Path(raw) / "manifest with 'quote' $X.json"
        env = Path(raw) / "run.env"
        manifest.write_text(json.dumps({"literal": "$X `quoted`"}))
        stage._write_literal_env(env, manifest, "/facility/python", Path(raw) / "job's helper.py")
        assert job._read_env(env) == {"RUN_MANIFEST": str(manifest), "RUN_PYTHON": "/facility/python", "RUN_JOB_PY": str(Path(raw) / "job's helper.py")}
        assert "EXPECTED_HEAD" not in (ROOT / "job.pbs").read_text()


def test_harness_has_no_embedded_commit_or_run_path() -> None:
    harness_files = [path for path in ROOT.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    assert not scan_identity_literals(harness_files)
    run_root_marker = "/" + "eagle/"
    with tempfile.TemporaryDirectory() as raw:
        planted = Path(raw) / "planted.py"
        planted.write_text("run_root=" + run_root_marker + "fake\n" + "a" * 40)
        assert scan_identity_literals([planted])


def scan_identity_literals(paths) -> bool:
    """Return whether any harness file contains a forbidden run identity literal."""
    pattern = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")
    run_root_marker = "/" + "eagle/"
    return any(pattern.search(path.read_text()) or run_root_marker in path.read_text() for path in paths)


def scan_shell_defects(paths) -> set[str]:
    """Scan stored production files for shell defects."""
    source = "\n".join(path.read_text() for path in paths)
    return {needle for needle in ("#!/bin/sh", "LD_LIBRARY_PATH", "<<") if needle in source}


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
            {"name": "tpen", "source": {"editable": "."}, "optional-dependencies": {"cpu": [{"name": "torch", "version": "2.12.0", "source": {"registry": "https://download.pytorch.org/whl/cpu"}, "marker": "sys_platform == 'darwin'"}, {"name": "torch", "version": "2.12.0+cpu", "source": {"registry": "https://download.pytorch.org/whl/cpu"}, "marker": "sys_platform != 'darwin'"}]}},
            {"name": "pytest", "version": "9.0.3"},
            {"name": "typeguard", "version": "4.5.2"},
            {"name": "torch", "version": "2.13.0+rocm7.1", "source": {"registry": "https://download.pytorch.org/whl/rocm7.1"}},
        ]
    }
    assert job.expected_versions(lock) == {"torch": "2.12.0+cpu", "pytest": "9.0.3", "typeguard": "4.5.2"}


def test_cpu_lock_derivation_control_rejects_missing_cpu_variant() -> None:
    try:
        job.expected_versions({"package": [{"name": "tpen", "source": {"editable": "."}, "optional-dependencies": {"cpu": []}}, {"name": "torch", "version": "2.13.0+rocm7.1", "source": {"registry": "https://download.pytorch.org/whl/rocm7.1"}}]})
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a lock without CPU torch resolutions")


def test_cpu_lock_derivation_matches_repository_lock() -> None:
    real_lock = tomllib.loads((ROOT.parents[2] / "uv.lock").read_text())
    expected = job.expected_versions(real_lock)
    assert expected == {"torch": "2.12.0+cpu", "pytest": "9.0.3", "typeguard": "4.5.2"}
    missing_linux = copy.deepcopy(real_lock)
    project = next(package for package in missing_linux["package"] if package.get("source", {}).get("editable") == ".")
    project["optional-dependencies"]["cpu"] = [entry for entry in project["optional-dependencies"]["cpu"] if "sys_platform != 'darwin'" not in entry.get("marker", "")]
    try:
        job.expected_versions(missing_linux)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted real lock with Linux CPU entry removed")


def test_environment_builder_and_profile_control() -> None:
    profile = json.loads((ROOT / "facility_profile.json").read_text())
    env = stage.build_env(profile, Path("/cache"), Path("/env"), base={})
    assert env["UV_PYTHON"] == profile["facility_python"]
    assert env["UV_CACHE_DIR"] == "/cache"
    assert env["OMP_NUM_THREADS"] == "1" and env["TZ"] == "UTC"
    altered = copy.deepcopy(profile)
    altered["thread_limits"]["OMP_NUM_THREADS"] = "2"
    assert stage.build_env(altered, Path("/cache"), Path("/env"), base={})["OMP_NUM_THREADS"] == "2"


def test_layout_is_one_source_for_sync_paths() -> None:
    paths = stage.layout(Path("/run-root"))
    env = stage.build_env(json.loads((ROOT / "facility_profile.json").read_text()), paths["cache_dir"], paths["prime_env"], base={})
    assert env["UV_CACHE_DIR"] == str(paths["cache_dir"])


def test_canary_is_not_normal_collection_name() -> None:
    assert not (ROOT / "fixtures" / "canary.py").name.startswith("test_")
    assert "assert False" in (ROOT / "fixtures" / "canary.py").read_text()


def test_normal_collection_excludes_canary() -> None:
    repo = ROOT.parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT)],
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
    assert '"$RUN_JOB_PY"' in job
    assert "$0" not in job and "dirname" not in job and "BASH_SOURCE" not in job
    assert "qsub" not in job


def test_static_defect_scanners_have_positive_and_failing_controls() -> None:
    files = [ROOT / name for name in ("README.md", "facility_profile.json", "stage.py", "job.py", "driver.py", "common.py", "job.pbs")]
    assert scan_shell_defects(files) == set()
    with tempfile.TemporaryDirectory() as raw:
        planted = Path(raw) / "planted.pbs"
        planted.write_text("#!/bin/sh\nLD_LIBRARY_PATH=bad\ncat <<EOF\nscript\nEOF\n")
        assert scan_shell_defects([planted]) == {"#!/bin/sh", "LD_LIBRARY_PATH", "<<"}
    job_script = (ROOT / "job.pbs").read_text()
    planted_job = job_script + '\nexec "$RUN_PYTHON" "$(dirname "$0")/job.py"\n'
    assert "$0" in planted_job and "dirname" in planted_job


def test_cell_id_is_one_safe_path_component() -> None:
    assert stage.validate_cell_id("verify-20261028T190602Z")
    for bad in ("../x", "a/b", ".", ".."):
        try:
            stage.validate_cell_id(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe cell id: {bad}")


def test_evidence_root_pbs_safety_control() -> None:
    assert stage.validate_evidence_root("/evidence/root")
    for bad in ("/evidence/a,b", "/evidence/a b", "/evidence/a'quote"):
        try:
            stage.validate_evidence_root(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe evidence root: {bad}")


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


def test_injected_preflight_codes_and_controls() -> None:
    base = dict(login_host=False, node_match=True, apple_double_count=0, python_exists=True, interpreter_ok=True, versions_ok=True, uv_ok=True, lock_ok=True, sync_ok=True, copied_ok=True)
    assert job.preflight_decision(**base) == 0
    for field, code in (("login_host", 70), ("node_match", 71), ("apple_double_count", 73), ("python_exists", 74), ("interpreter_ok", 76), ("versions_ok", 75), ("uv_ok", 77), ("lock_ok", 78), ("sync_ok", 80), ("copied_ok", 81)):
        altered = dict(base)
        altered[field] = 1 if field in {"login_host", "apple_double_count"} else False
        assert job.preflight_decision(**altered) == code


def test_job_main_uses_one_injected_observation_path(monkeypatch, tmp_path) -> None:
    """The verify/ directory is inserted at sys.path[0], so common imports
    resolve to the copied harness module rather than an unrelated package.
    """
    root = tmp_path
    evidence = root / "evidence"
    repo = root / "repo"
    repo.mkdir()
    manifest = root / "manifest.json"
    env = root / "run.env"
    profile = {"compute_host_guard": {"refuse_prefix": "login"}, "job_scratch_root": str(root / "scratch"), "facility_python": sys.executable, "uv_version": "0.8.23", "uv_flags": [], "thread_limits": {}, "timezone": "UTC"}
    copied = {"common.py": "expected"}
    manifest.write_text(json.dumps({"evidence_dir": str(evidence), "target_repo": str(repo), "target_sha": "a" * 40, "required_ancestors": [], "profile": profile, "cache_dir": str(root / "cache"), "staging_receipt": str(root / "receipt.json"), "job_helper": str(root / "harness" / "job.py"), "copied_file_sha256": copied, "driver": str(ROOT / "driver.py")}))
    env.write_text(f"RUN_MANIFEST={manifest}\n")
    (root / "receipt.json").write_text(json.dumps({"target_sha": "a" * 40, "cache_dir": str(root / "cache")}))
    nodefile = root / "nodefile"
    nodefile.write_text("nid001.fqdn\n")
    monkeypatch.setenv("RUN_ENV_PATH", str(env))
    monkeypatch.setenv("PBS_JOBID", "123")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))

    def configure() -> None:
        manifest_data = json.loads(manifest.read_text())
        manifest_data["required_ancestors"] = []
        manifest.write_text(json.dumps(manifest_data))
        monkeypatch.setattr(job, "copied_hashes", lambda paths: copied)
        monkeypatch.setattr(job, "_hostname", lambda: "nid001")
        monkeypatch.setattr(job, "_nodefile_entries", lambda: ["nid001.fqdn"])
        monkeypatch.setattr(job, "_git_head", lambda path: "a" * 40)
        monkeypatch.setattr(job, "_appledoubles", lambda path: [])
        monkeypatch.setattr(job, "_lock_text", lambda path: "")
        monkeypatch.setattr(job, "expected_versions", lambda lock: {"torch": "cpu", "pytest": "p", "typeguard": "t"})
        monkeypatch.setattr(job, "_uv_version", lambda uv: "uv 0.8.23")
        monkeypatch.setattr(job, "_offline_sync", lambda uv, path, runtime, log: 0)
        monkeypatch.setattr(job, "_venv_exists", lambda path: True)
        monkeypatch.setattr(job, "_venv_identity", lambda path: {"prefix": str(root / "scratch" / "123" / "tpen-verify-env"), "base_executable": sys.executable, "base_prefix": str(Path(sys.executable).resolve().parent.parent)})
        monkeypatch.setattr(job, "_package_versions", lambda path: {"torch": "cpu", "pytest": "p", "typeguard": "t"})
        monkeypatch.setattr(job, "_run_driver", lambda *args: 7)
        monkeypatch.setattr(job, "receipt_code", lambda manifest, path: 0)

    monkeypatch.setattr(sys, "argv", ["job.py", str(env)])
    for name, code in (("receipt_code", 60), ("receipt_code", 61), ("_hostname", 70), ("_nodefile_entries", 71), ("ancestor", 79), ("_git_head", 72), ("_appledoubles", 73), ("expected_versions", 78), ("_uv_version", 77), ("_offline_sync", 80), ("python", 74), ("identity", 76), ("_package_versions", 75), ("copied", 81)):
        # Reset setup before each arm; this also undoes monkeypatches made before the loop.
        monkeypatch.undo()
        monkeypatch.setenv("RUN_ENV_PATH", str(env)); monkeypatch.setenv("PBS_JOBID", "123"); monkeypatch.setenv("PBS_NODEFILE", str(nodefile)); monkeypatch.setattr(sys, "argv", ["job.py", str(env)])
        configure()
        if name == "receipt_code": monkeypatch.setattr(job, "receipt_code", lambda manifest, path, value=code: value)
        elif name == "_hostname": monkeypatch.setattr(job, "_hostname", lambda: "login-node")
        elif name == "_nodefile_entries": monkeypatch.setattr(job, "_nodefile_entries", lambda: ["other.fqdn"])
        elif name == "_git_head": monkeypatch.setattr(job, "_git_head", lambda path: "b" * 40)
        elif name == "_appledoubles": monkeypatch.setattr(job, "_appledoubles", lambda path: ["._bad"])
        elif name == "python": monkeypatch.setattr(job, "_venv_exists", lambda path: False)
        elif name == "_package_versions": monkeypatch.setattr(job, "_package_versions", lambda path: {"torch": "wrong"})
        elif name == "identity": monkeypatch.setattr(job, "_venv_identity", lambda path: {"prefix": "/wrong", "base_executable": "/wrong", "base_prefix": "/wrong"})
        elif name == "_uv_version": monkeypatch.setattr(job, "_uv_version", lambda uv: "uv 9.9.9")
        elif name == "expected_versions": monkeypatch.setattr(job, "expected_versions", lambda lock: (_ for _ in ()).throw(ValueError("bad lock")))
        elif name == "_offline_sync": monkeypatch.setattr(job, "_offline_sync", lambda uv, path, runtime, log: 1)
        elif name == "ancestor":
            manifest_data = json.loads(manifest.read_text())
            manifest_data["required_ancestors"] = ["c" * 40]
            manifest.write_text(json.dumps(manifest_data))
            monkeypatch.setattr(job, "is_ancestor", lambda *args: False)
        elif name == "copied": monkeypatch.setattr(job, "copied_hashes", lambda paths: {"common.py": "wrong"})
        assert job.main() == code
    monkeypatch.undo()
    monkeypatch.setenv("RUN_ENV_PATH", str(env)); monkeypatch.setenv("PBS_JOBID", "123"); monkeypatch.setenv("PBS_NODEFILE", str(nodefile)); monkeypatch.setattr(sys, "argv", ["job.py", str(env)])
    configure()
    manifest_data = json.loads(manifest.read_text())
    manifest_data["required_ancestors"] = []
    manifest.write_text(json.dumps(manifest_data))
    assert job.main() == 7


def test_copied_hash_positive_and_tamper_control(tmp_path) -> None:
    copied = tmp_path / "job.py"
    copied.write_text("original")
    expected = job.copied_hashes([copied])
    assert job.copied_hashes([copied]) == expected
    copied.write_text("tampered")
    assert job.copied_hashes([copied]) != expected


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
    assert stage.is_ancestor(repo, ancestor, target)
    assert not stage.is_ancestor(repo, target, ancestor)


def test_stage_end_to_end_fakes_and_ordering_controls(monkeypatch, tmp_path) -> None:
    """Exercise stage() through clone, both syncs, receipts, and qsub."""
    real_run = subprocess.run

    def git_repo(path: Path, filename: str = "test_target.py") -> str:
        path.mkdir()
        real_run(["git", "init", "-q", str(path)], check=True)
        real_run(["git", "-C", str(path), "config", "user.email", "test@example.invalid"], check=True)
        real_run(["git", "-C", str(path), "config", "user.name", "Harness Test"], check=True)
        (path / filename).write_text("def test_target(): pass\n")
        real_run(["git", "-C", str(path), "add", filename], check=True)
        real_run(["git", "-C", str(path), "commit", "-q", "-m", "target"], check=True)
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

    target_source = tmp_path / "target-source"
    target_sha = git_repo(target_source)
    bare = tmp_path / "remote.git"
    real_run(["git", "clone", "-q", "--bare", str(target_source), str(bare)], check=True)
    harness_repo = tmp_path / "harness-repo"
    git_repo(harness_repo, "placeholder")
    monkeypatch.setattr(stage, "_harness_repo", lambda: harness_repo)
    monkeypatch.setattr(stage.getpass, "getuser", lambda: "rhu")

    qsub_calls = []
    uv_envs = []
    qstat_calls = 0
    queue_mode = "debug-other"
    original_read_profile = stage.read_profile

    def full_job(job_id: str, user: str, queue: str, state: str) -> str:
        return (f"Job Id: {job_id}\n    Job_Owner = {user}@polaris-login\n"
                f"    job_state = {state}\n    queue = {queue}\n")

    def queue_text(queue: str) -> str:
        if queue == "debug-scaling":
            text = ("Queue: debug-scaling\n"
                    "max_queued = [u:PBS_GENERIC=1]\n"
                    "queued_jobs_threshold = [u:PBS_GENERIC=1]\n"
                    "resources_min.walltime = 00:05:00\n")
        else:
            text = ("Queue: debug\n"
                    "max_run = [u:PBS_GENERIC=1]\n"
                    "queued_jobs_threshold = [u:PBS_GENERIC=1]\n"
                    "max_run_res.nodect = [o:PBS_ALL=24]\n"
                    "resources_min.walltime = 00:05:00\n")
        if queue_mode == "wrapped-limit":
            text = ("Queue: debug\nmax_queued = [u:PBS_GENERIC=1],\n"
                    "\t[u:rhu=2]\nresources_min.walltime = 00:05:00\n")
        if queue_mode == "unterminated-limit":
            text = "Queue: debug\nmax_queued = [u:rhu=1\n"
        if queue_mode == "unknown-user-limit":
            text += "mystery_limit = [u:PBS_GENERIC=1]\n"
        return text

    def jobs_text(queue: str) -> str:
        if queue_mode in {"debug-other", "debug-running"} and queue == "debug":
            records = [full_job("7795000.polaris", "rhu", "debug", "R")]
            if queue_mode == "debug-other":
                records += [full_job("7795001.polaris", "other", "debug", "Q"),
                            full_job("7795002.polaris", "other", "debug", "R")]
            return "".join(records)
        if queue_mode == "debug-queued" and queue == "debug":
            return full_job("7795003.polaris", "rhu", "debug", "Q")
        if queue_mode == "debug-truncated" and queue == "debug":
            return full_job("7795004.polaris", "rhu", "debug-s*", "Q")
        if queue_mode == "scaling-running" and queue == "debug-scaling":
            return full_job("7795005.polaris", "rhu", "debug-scaling", "R")
        if queue_mode == "scaling-flip" and queue == "debug-scaling" and qstat_calls >= 2:
            return full_job("7795006.polaris", "rhu", "debug-scaling", "Q")
        if queue_mode == "debug-other-only" and queue == "debug":
            return full_job("7795007.polaris", "other", "debug", "R")
        return ""

    def table_text(queue: str) -> str:
        records = []
        if queue_mode in {"debug-other", "debug-running", "debug-queued", "debug-truncated"} and queue == "debug":
            records.append("7795000.polaris rhu 00:00:00 R debug\n")
        if queue_mode == "debug-queued":
            records = ["7795003.polaris rhu 00:00:00 Q debug\n"]
        if queue_mode == "debug-truncated":
            records = ["7795004.polaris rhu 00:00:00 Q debug-s*\n"]
        if queue_mode == "debug-other-only":
            records = []
        if queue_mode == "missing-full":
            records = ["7795008.polaris rhu 00:00:00 Q debug\n"]
        if queue_mode == "alternate-only":
            records = ["7795009.polaris rhu 00:00:00 Q debug\n"]
        if queue_mode == "scaling-running" and queue == "debug-scaling":
            records = ["7795005.polaris rhu 00:00:00 R debug-scaling\n"]
        if queue_mode == "scaling-flip" and queue == "debug-scaling" and qstat_calls >= 2:
            records = ["7795006.polaris rhu 00:00:00 Q debug-scaling\n"]
        return "Job id User Time Use S Queue\n" + "".join(records)

    def fake_subprocess(args, **kwargs):
        nonlocal qstat_calls
        if args[:2] == ["qstat", "-Qf"]:
            queue = args[2]
            if queue_mode == "unparseable":
                return subprocess.CompletedProcess(args, 0, "not qstat output\n", "")
            return subprocess.CompletedProcess(args, 0, queue_text(queue), "")
        if args == ["qstat", "-f"]:
            qstat_calls += 1
            queue = stage.read_profile(ROOT / "facility_profile.json")["pbs"]["queue"]
            if queue_mode == "alternate-only":
                return subprocess.CompletedProcess(args, 0, table_text(queue), "")
            return subprocess.CompletedProcess(args, 0, jobs_text(queue), "")
        if args[:3] == ["qstat", "-f", "-u"]:
            queue = stage.read_profile(ROOT / "facility_profile.json")["pbs"]["queue"]
            return subprocess.CompletedProcess(args, 0, table_text(queue), "")
        if args[:2] == ["qstat", "-u"]:
            queue = stage.read_profile(ROOT / "facility_profile.json")["pbs"]["queue"]
            return subprocess.CompletedProcess(args, 0, table_text(queue), "")
        if "sync" in args:
            uv_envs.append(kwargs["env"])
            Path(kwargs["env"]["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            Path(kwargs["env"]["UV_PROJECT_ENVIRONMENT"]).mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(args, 0, "", "")
        return real_run(args, **kwargs)

    original_stage_run = stage._run

    def fake_run(args, cwd=None, *, capture=False, env=None):
        if "--version" in args and "uv" in args:
            return "uv 0.8.23"
        if args and args[0] == "qsub":
            qsub_calls.append(args)
            return "123.polaris-pbs-01"
        if "sync" in args:
            uv_envs.append(env)
            Path(env["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            Path(env["UV_PROJECT_ENVIRONMENT"]).mkdir(parents=True, exist_ok=True)
            return ""
        return original_stage_run(args, cwd=cwd, capture=capture, env=env)

    monkeypatch.setattr(stage.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(stage, "_run", fake_run)
    args = type("Args", (), {"target_sha": target_sha, "arm": ["test_target.py"],
                              "evidence_root": str(tmp_path / "cells"), "cell_id": "cell-ok",
                              "required_ancestor": [], "remote": str(bare)})()
    assert stage.stage(args) == 0
    run_root = Path(args.evidence_root) / args.cell_id
    manifest = json.loads((run_root / "manifest.json").read_text())
    assert (run_root / "run.env").is_file() and (run_root / "staging-receipt.json").is_file()
    assert (run_root / "submission.txt").is_file()
    assert all((Path(args.evidence_root) / name).is_file() for name in ("cell-ok.pre-submit-qstat-Qf.txt", "cell-ok.pre-submit-qstat-f.txt.gz", "cell-ok.pre-submit-qstat-u.txt", "cell-ok.pre-submit-qstat-decision.json"))
    assert all((run_root / name).is_file() for name in ("pre-submit-qstat-Qf.txt", "pre-submit-qstat-f.txt.gz", "pre-submit-qstat-u.txt", "pre-submit-qstat-decision.json"))
    assert json.loads((run_root / "pre-submit-qstat-decision.json").read_text())["allowed"] is True
    assert len(uv_envs) == 2
    assert all(env["UV_CACHE_DIR"] == manifest["cache_dir"] for env in uv_envs)
    profile = manifest["profile"]
    expected_qsub = ["qsub", "-A", profile["pbs"]["account"], "-q", profile["pbs"]["queue"],
                     "-l", f"select={profile['pbs']['select']}", "-l", f"walltime={profile['pbs']['walltime']}",
                     "-l", f"filesystems={profile['pbs']['filesystems']}", "-o", manifest["pbs_dir"] + "/",
                     "-e", manifest["pbs_dir"] + "/", "-v", f"RUN_ENV_PATH={run_root.resolve() / 'run.env'}",
                     manifest["job_script"]]
    assert qsub_calls == [expected_qsub]

    def run_control(name, mode):
        nonlocal qstat_calls, queue_mode
        qstat_calls = 0
        queue_mode = mode
        monkeypatch.setattr(stage, "read_profile", original_read_profile)
        if mode.startswith("scaling"):
            control_profile = json.loads((ROOT / "facility_profile.json").read_text())
            control_profile["pbs"]["queue"] = "debug-scaling"
            monkeypatch.setattr(stage, "read_profile", lambda path, value=control_profile: value)
        cell = tmp_path / "cells" / name
        control_args = type("Args", (), {"target_sha": target_sha, "arm": ["test_target.py"],
                                          "evidence_root": str(tmp_path / "cells"), "cell_id": name,
                                          "required_ancestor": [], "remote": str(bare)})()
        def control_run(call_args, cwd=None, *, capture=False, env=None):
            if "--version" in call_args and "uv" in call_args:
                return "uv 0.8.23"
            if "sync" in call_args and mode == "prime-fail":
                raise RuntimeError("prime failed")
            if call_args and call_args[0] == "qsub":
                if mode in {"scaling-empty", "debug-truncated", "debug-other-only", "wrapped-limit"}:
                    qsub_calls.append(call_args)
                    return "123.polaris-pbs-01"
                raise AssertionError("qsub reached control")
            return original_stage_run(call_args, cwd=cwd, capture=capture, env=env)
        def control_subprocess(call_args, **kwargs):
            nonlocal qstat_calls
            if call_args[:2] == ["qstat", "-Qf"] or call_args == ["qstat", "-f"] or call_args[:2] == ["qstat", "-u"]:
                return fake_subprocess(call_args, **kwargs)
            if "--offline" in call_args and mode == "dry-fail":
                Path(kwargs["env"]["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
                Path(kwargs["env"]["UV_PROJECT_ENVIRONMENT"]).mkdir(parents=True, exist_ok=True)
                if kwargs.get("check", False):
                    raise subprocess.CalledProcessError(1, call_args)
                return subprocess.CompletedProcess(call_args, 1, "", "")
            return fake_subprocess(call_args, **kwargs)
        monkeypatch.setattr(stage, "_run", control_run)
        monkeypatch.setattr(stage.subprocess, "run", control_subprocess)
        try:
            stage.stage(control_args)
        except (RuntimeError, subprocess.CalledProcessError):
            pass
        if mode in {"scaling-empty", "debug-truncated", "debug-other-only", "wrapped-limit"}:
            assert (cell / "submission.txt").exists()
        else:
            assert not (cell / "submission.txt").exists()
            if mode not in {"scaling-flip"}:
                assert not (cell / "staging-receipt.json").exists()

    run_control("cell-debug-queued", "debug-queued")
    run_control("cell-debug-truncated", "debug-truncated")
    run_control("cell-debug-other-only", "debug-other-only")
    run_control("cell-alternate-only", "alternate-only")
    run_control("cell-missing-full", "missing-full")
    run_control("cell-wrapped-limit", "wrapped-limit")
    run_control("cell-unterminated-limit", "unterminated-limit")
    run_control("cell-scaling-running", "scaling-running")
    run_control("cell-unknown-user-limit", "unknown-user-limit")
    run_control("cell-scaling-empty", "scaling-empty")
    run_control("cell-unparseable", "unparseable")
    run_control("cell-scaling-flip", "scaling-flip")


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

        def invoke(arms, name, job_env=sys.prefix):
            manifest.write_text(json.dumps({"target_sha": "a" * 40, "target_repo": str(repo), "arms": arms}))
            out = evidence / name
            monkeypatch.setattr(sys, "argv", ["driver.py", "--manifest", str(manifest), "--evidence", str(out), "--job-python", sys.executable, "--job-env", job_env])
            return driver.main(), out

        rc, out = invoke(["test_arms.py::test_pass"], "green")
        assert rc == 0
        assert all(path.parent == out for path in out.iterdir())
        rc, _ = invoke(["test_arms.py::test_pass"], "wrong-interpreter", str(Path(raw) / "wrong-env"))
        assert rc == 76
        rc, out = invoke(["test_arms.py"], "red")
        assert rc == 1
        rows = json.loads((out / "results.json").read_text())["arms"][0]
        assert rows["counts"]["failed"] == 1 and rows["counts"]["passed"] == 1
        rc, _ = invoke(["test_arms.py::test_skip"], "skipped")
        assert rc == 1

        rc, _ = invoke([], "empty")
        assert rc == 1

        class Completed:
            returncode = 1

        monkeypatch.setattr(driver.subprocess, "run", lambda *args, **kwargs: Completed())
        missing = evidence / "missing-junit"
        missing.mkdir()
        result = driver.run_arm("missing", repo, "test_arms.py::test_pass", missing)
        assert result["status"] == "FAIL" and result["reason"] == "missing JUnit"

        def malformed(*args, **kwargs):
            Path(args[0][-1]).write_text("<testsuite")
            return Completed()

        monkeypatch.setattr(driver.subprocess, "run", malformed)
        malformed_dir = evidence / "malformed-junit"
        malformed_dir.mkdir()
        result = driver.run_arm("malformed", repo, "test_arms.py::test_pass", malformed_dir)
        assert result["status"] == "FAIL" and "unparseable JUnit" in result["reason"]
