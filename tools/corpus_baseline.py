#!/usr/bin/env python3
"""Run and reconcile a complete pytest corpus in one allocation.

The script is intentionally standard-library-only so it can be copied into an
allocation before the project environment is provisioned.  It never supplies
a pytest path: each arm starts at its checkout root and pytest self-selects.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import xml.etree.ElementTree as ET


PLUGIN = r'''
import json
import os
from pathlib import Path

_OBS = None

def _out(config):
    return Path(config.getoption("--corpus-observations"))

def pytest_addoption(parser):
    parser.addoption("--corpus-observations", required=True)

def pytest_configure(config):
    global _OBS
    _OBS = _out(config)
    _OBS.write_text("", encoding="utf-8")
    phase = os.environ.get("CORPUS_PHASE", "run")
    _OBS.with_name(phase + "-collection-skipped.txt").write_text("", encoding="utf-8")

def pytest_collection_modifyitems(config, items):
    phase = os.environ.get("CORPUS_PHASE", "run")
    p = _out(config).with_name(phase + "-collected.txt")
    p.write_text("\n".join(sorted(item.nodeid for item in items)) + "\n", encoding="utf-8")

def pytest_collectreport(report):
    if report.failed:
        with _OBS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"nodeid": report.nodeid, "when": "collection", "outcome": "error"}) + "\n")
        return
    if report.skipped:
        phase = os.environ.get("CORPUS_PHASE", "run")
        p = _OBS.with_name(phase + "-collection-skipped.txt")
        with p.open("a", encoding="utf-8") as stream:
            stream.write(report.nodeid + "\n")

def pytest_runtest_logreport(report):
    p = _OBS
    label = "passed"
    if report.failed:
        label = "failed" if report.when == "call" else "error"
    elif report.skipped:
        label = "skipped"
    with p.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"nodeid": report.nodeid, "when": report.when, "outcome": label}) + "\n")
'''


def run(cmd, *, cwd, stdout, env, label, quota_root=None, quota_evidence=None, quota_commands=(), cache_dir=None):
    """Run one step with a durable merged stream and an unambiguous result."""
    stdout.parent.mkdir(parents=True, exist_ok=True)
    if quota_root is not None and quota_evidence is not None:
        safe_quota_snapshot(quota_root, quota_evidence, label + "-before", quota_commands, cache_dir or env.get("UV_CACHE_DIR", quota_root))
    capture_error = None
    rc = None
    try:
        with stdout.open("wb") as stream:
            begin = f"BEGIN {label} COMMAND={json.dumps(cmd)}\n".encode()
            stream.write(begin); stream.flush()
            try:
                child = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT)
            except OSError as exc:
                capture_error = f"LAUNCH_ERROR {exc!r}"
            else:
                assert child.stdout is not None
                for chunk in iter(child.stdout.readline, b""):
                    try:
                        stream.write(chunk); stream.flush()
                    except OSError as exc:
                        capture_error = f"LOG_WRITE_ERROR {exc!r}"
                        try: sys.stderr.buffer.write(chunk); sys.stderr.flush()
                        except OSError: capture_error += " EMERGENCY_STREAM_FAILED"
                rc = child.wait()
            if capture_error:
                try: stream.write((capture_error + "\n").encode()); stream.flush()
                except OSError:
                    try: sys.stderr.write(capture_error + "\n"); sys.stderr.flush()
                    except OSError: pass
            end = f"END {label} RC={rc if rc is not None else 'LAUNCH_ERROR'} PEAK_RSS_CUMULATIVE_CHILDREN_LINUX_KIB={resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}\n".encode()
            try: stream.write(end); stream.flush()
            except OSError:
                try: sys.stderr.buffer.write(end); sys.stderr.flush()
                except OSError: pass
    except OSError as exc:
        capture_error = capture_error or f"LOG_OPEN_ERROR {exc!r}"
        try: sys.stderr.write(capture_error + "\n"); sys.stderr.flush()
        except OSError: pass
    if rc is None:
        rc = 125
    if quota_root is not None and quota_evidence is not None:
        safe_quota_snapshot(quota_root, quota_evidence, label + "-after", quota_commands, cache_dir or env.get("UV_CACHE_DIR", quota_root))
    return rc


def quota_snapshot(root, evidence, label, quota_commands, cache_dir):
    """Persist supported quota calls and disk headroom without inventing fields."""
    read_at = datetime.now(timezone.utc).isoformat()
    payload = {"label": label, "read_at_utc": read_at, "observed_at_utc": read_at,
               "uv_cache_dir": str(cache_dir), "disk": {}, "quota": [],
               "low_cardinality_agreement": {"status": "UNKNOWN",
                 "qualification": "Displayed/quantized fields are not treated as byte-exact agreement."}}
    for path in (root, Path(cache_dir), Path.home() / ".local" / "bin" / "uv"):
        target = path if path.exists() else path.parent
        usage = shutil.disk_usage(target)
        payload["disk"][str(path)] = {"free": usage.free, "total": usage.total, "used": usage.used}
    for command in quota_commands:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        def cat_a(value):
            return value.replace("\\", "\\\\").replace("\t", "^I").replace("\r", "^M").replace("\n", "$\\n")
        payload["quota"].append({"command": command, "rc": result.returncode,
                                 "stdout": result.stdout, "stderr": result.stderr,
                                 "stdout_cat_a": cat_a(result.stdout), "stderr_cat_a": cat_a(result.stderr),
                                 "read_at_utc": read_at,
                                 "space_headroom": "UNKNOWN", "file_headroom": "UNKNOWN",
                                 "headroom_units": "UNKNOWN", "table_freshness": "UNKNOWN"})
    (evidence / f"quota-{label}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def safe_quota_snapshot(root, evidence, label, quota_commands, cache_dir):
    try:
        quota_snapshot(root, evidence, label, quota_commands, cache_dir)
    except BaseException as exc:
        with (evidence / "capture-errors.log").open("a", encoding="utf-8") as stream:
            stream.write(f"QUOTA_SNAPSHOT_ERROR label={label} error={exc!r}\n")
            stream.flush()


def xml_pairs(path):
    pairs = []
    for case in ET.parse(path).getroot().iter("testcase"):
        pairs.append((case.attrib.get("classname", ""), case.attrib.get("name", "")))
    return pairs


def xml_outcomes(path):
    outcomes = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        pair = (case.attrib.get("classname", ""), case.attrib.get("name", ""))
        children = {child.tag for child in case}
        outcomes[pair] = "error" if "error" in children else "failed" if "failure" in children else "skipped" if "skipped" in children else "passed"
    return outcomes


def pair_for_nodeid(nodeid):
    if "::" not in nodeid:
        return ("", nodeid.replace("/", ".").removesuffix(".py"))
    module, *parts = nodeid.split("::")
    dotted = module.replace("/", ".").removesuffix(".py")
    return (".".join([dotted, *parts[:-1]]) if len(parts) > 1 else dotted, parts[-1])


def identity(checkout, uv, env, evidence, quota_commands):
    code = (
        "import os, pathlib, shutil, sys; import torch; "
        "print('SYS_EXECUTABLE='+sys.executable, flush=True); "
        "print('PYTHON_VERSION='+sys.version, flush=True); "
        "print('TORCH_VERSION='+torch.__version__, flush=True); "
        "print('TORCH_FILE='+str(pathlib.Path(torch.__file__).absolute()), flush=True); "
        "print('PYTHON_WHICH='+str(shutil.which('python')), flush=True); "
        "print('PYTHONPATH='+os.environ.get('PYTHONPATH',''), flush=True)"
    )
    output = evidence / "identity-command.log"
    rc = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-c", code],
             cwd=checkout, stdout=output, env=env, label="identity",
             quota_root=evidence.parents[1], quota_evidence=evidence, quota_commands=quota_commands)
    if rc != 0:
        raise RuntimeError(f"identity command rc={rc}; see {output}")
    return "\n".join(line for line in output.read_text(encoding="utf-8").splitlines()
                         if not line.startswith(("BEGIN ", "END "))) + "\n"


def arm(name, revision, root, uv, deliberate_red, quota_commands):
    checkout = root / name
    evidence = root / "evidence" / name
    evidence.mkdir(parents=True, exist_ok=True)
    envdir = root / ("venv-" + name)
    cachedir = root / ("uv-cache-" + name)
    env = os.environ.copy()
    env.update({"UV_PROJECT_ENVIRONMENT": str(envdir), "UV_CACHE_DIR": str(cachedir), "PYTHONDONTWRITEBYTECODE": "1"})
    env["PATH"] = str(envdir / "bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(checkout) + os.pathsep + env.get("PYTHONPATH", "")
    if os.environ.get("SLURM_JOB_ID", "") == "":
        raise RuntimeError("SLURM_JOB_ID is required before any test command")
    safe_quota_snapshot(root, evidence, "arm-before", quota_commands, cachedir)
    if not checkout.exists():
        if run(["git", "clone", "--no-single-branch", os.environ["CORPUS_SOURCE"], str(checkout)], cwd=root,
               stdout=evidence / "git-clone.log", env=env, label="git-clone",
               quota_root=root, quota_evidence=evidence, quota_commands=quota_commands) != 0:
            raise RuntimeError("git clone failed")
    if run(["git", "fetch", "--tags", "--all"], cwd=checkout, stdout=evidence / "git-fetch.log", env=env, label="git-fetch", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands) != 0:
        raise RuntimeError("git fetch failed")
    if run(["git", "checkout", "--detach", revision], cwd=checkout, stdout=evidence / "git-checkout.log", env=env, label="git-checkout", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands) != 0:
        raise RuntimeError("git checkout failed")
    measured = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, text=True, capture_output=True, check=True).stdout.strip()
    if measured != revision:
        raise RuntimeError(f"{name}: measured {measured}, expected {revision}")
    tag_count = subprocess.run(["git", "tag", "--list"], cwd=checkout, text=True, capture_output=True, check=True).stdout.splitlines()
    shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=checkout, text=True, capture_output=True, check=True).stdout.strip()
    sync_rc = run([str(uv), "sync", "--extra", "cpu", "--locked"], cwd=checkout, stdout=evidence / "uv-sync.log", env=env, label="uv-sync", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
    if sync_rc != 0:
        raise RuntimeError(f"{name}: uv sync rc={sync_rc}")
    (evidence / "identity.txt").write_text(identity(checkout, uv, env, evidence, quota_commands) + f"REVISION={measured}\nTAG_COUNT={len(tag_count)}\nSHALLOW={shallow}\n", encoding="utf-8")
    plugin = checkout / "corpus_plugin.py"
    plugin.write_text(PLUGIN, encoding="utf-8")
    (evidence / "command-provenance.txt").write_text("uv run --extra cpu --locked python -m pytest -q --junitxml=corpus.xml -p corpus_plugin --corpus-observations=observations.jsonl\n", encoding="utf-8")
    env["CORPUS_PHASE"] = "collect"
    collect_rc = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-m", "pytest", "-q", "--collect-only", "-p", "corpus_plugin", "--corpus-observations=collect-observations.jsonl"], cwd=checkout, stdout=evidence / "collect.log", env=env, label="pytest-collect", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
    for filename in ("collect-collected.txt", "collect-collection-skipped.txt", "collect-observations.jsonl"):
        source = checkout / filename
        if source.exists():
            shutil.copy2(source, evidence / filename)
    env["CORPUS_PHASE"] = "run"
    pytest_rc = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-m", "pytest", "-q", "--junitxml=corpus.xml", "-p", "corpus_plugin", "--corpus-observations=observations.jsonl"], cwd=checkout, stdout=evidence / "pytest.log", env=env, label="pytest-run", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
    if (checkout / "corpus.xml").exists():
        shutil.copy2(checkout / "corpus.xml", evidence / "corpus.xml")
    for filename in ("run-collected.txt", "run-collection-skipped.txt", "observations.jsonl"):
        source = checkout / filename
        if source.exists():
            shutil.copy2(source, evidence / filename)
    (evidence / "statuses.json").write_text(json.dumps({"collect_rc": collect_rc, "pytest_rc": pytest_rc}, indent=2) + "\n", encoding="utf-8")
    if collect_rc != 0:
        raise RuntimeError(f"{name}: collection instrument rc={collect_rc}")
    if deliberate_red:
        control = checkout / "test_corpus_deliberate_red.py"
        control.write_text("def test_corpus_deliberate_red():\n    assert False, 'intentional corpus extraction control'\n", encoding="utf-8")
        env["CORPUS_PHASE"] = "control"
        control_rc = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-m", "pytest", "-q", "-k", "test_corpus_deliberate_red", "--junitxml=deliberate-red.xml", "-p", "corpus_plugin", "--corpus-observations=deliberate-red-observations.jsonl"], cwd=checkout, stdout=evidence / "deliberate-red.log", env=env, label="deliberate-red", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
        shutil.copy2(control, evidence / control.name)
        shutil.copy2(checkout / "deliberate-red.xml", evidence / "deliberate-red.xml")
        shutil.copy2(checkout / "deliberate-red-observations.jsonl", evidence / "deliberate-red-observations.jsonl")
        (evidence / "deliberate-red-status.txt").write_text(f"INNER_RC={control_rc}\n", encoding="utf-8")
        if control_rc != 1:
            raise RuntimeError(f"{name}: deliberate red rc={control_rc}, expected 1")
        control_pairs = xml_pairs(evidence / "deliberate-red.xml")
        control_xml = xml_outcomes(evidence / "deliberate-red.xml")
        control_pair = next((pair for pair in control_pairs if pair[1] == "test_corpus_deliberate_red"), None)
        if control_pair is None or control_xml.get(control_pair) != "failed":
            raise RuntimeError(f"{name}: deliberate red node is not a JUnit failure")
        control_observations = evidence / "deliberate-red-observations.jsonl"
        if not control_observations.exists() or not any(
            json.loads(line).get("nodeid", "").endswith("test_corpus_deliberate_red") and
            json.loads(line).get("outcome") == "failed"
            for line in control_observations.read_text(encoding="utf-8").splitlines()
        ):
            raise RuntimeError(f"{name}: deliberate red failure was not extracted")
    return {
        "revision": measured, "tag_count": len(tag_count), "shallow": shallow,
        "collect_rc": collect_rc, "pytest_rc": pytest_rc,
        "collected": (evidence / "collect-collected.txt").read_text(encoding="utf-8").splitlines(),
        "collection_skipped": (evidence / "collect-collection-skipped.txt").read_text(encoding="utf-8").splitlines() if (evidence / "collect-collection-skipped.txt").exists() else [],
        "run_collected": (evidence / "run-collected.txt").read_text(encoding="utf-8").splitlines() if (evidence / "run-collected.txt").exists() else [],
        "run_collection_skipped": (evidence / "run-collection-skipped.txt").read_text(encoding="utf-8").splitlines() if (evidence / "run-collection-skipped.txt").exists() else [],
        "xml_pairs": xml_pairs(evidence / "corpus.xml"),
        "xml_outcomes": [{"classname": pair[0], "name": pair[1], "outcome": outcome}
                         for pair, outcome in xml_outcomes(evidence / "corpus.xml").items()],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--quota-spec", required=True, help="JSON list of supported quota argv lists")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("must run inside a Slurm allocation")
    args.run_root.mkdir(parents=True, exist_ok=True)
    os.environ["CORPUS_SOURCE"] = args.source
    results = {}
    try:
        quota_commands = json.loads(args.quota_spec)
        if len(quota_commands) != 3 or not all(isinstance(command, list) for command in quota_commands):
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"--quota-spec must be JSON list of three argv lists: {exc}")
    results["trunk"] = arm("trunk", args.revision, args.run_root, args.uv, True, quota_commands)
    for result in results.values():
        result["xml_outcomes"] = {(row["classname"], row["name"]): row["outcome"] for row in result["xml_outcomes"]}
        result["failed"] = []
        result["errors"] = []
        result["skipped"] = []
        for filename in ("collect-observations.jsonl", "observations.jsonl"):
            observation_file = args.run_root / "evidence" / "trunk" / filename
            if not observation_file.exists():
                raise RuntimeError(f"missing required observation artifact: {observation_file}")
            for line in observation_file.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                target = {"failed": "failed", "error": "errors", "skipped": "skipped"}.get(row["outcome"])
                if target:
                    result[target].append(row["nodeid"])
        if set(result["collected"]) != set(result["run_collected"]):
            raise RuntimeError("collect-only and run collected sets differ")
        if set(result["collection_skipped"]) != set(result["run_collection_skipped"]):
            raise RuntimeError("collect-only and run collection-skip sets differ")
        for key in ("failed", "errors", "skipped"):
            result[key] = sorted(set(result[key]))
        result["skipped"] = sorted(set(result["skipped"]) | set(result["collection_skipped"]))
        floor = len(set(result["collected"]) | set(result["collection_skipped"]))
        result["floor"] = floor
        expected_pairs = {pair_for_nodeid(node) for node in result["collected"] + result["collection_skipped"]}
        if set(result["xml_pairs"]) - expected_pairs:
            raise RuntimeError("JUnit outcome contains an identity absent from collect-only set")
        if len(result["xml_pairs"]) != len(set(result["xml_pairs"])):
            raise RuntimeError("JUnit contains duplicate (classname,name) identities")
        if len(result["xml_pairs"]) != floor:
            raise RuntimeError(f"JUnit rows {len(result['xml_pairs'])} do not equal in-job floor {floor}")
        observed = {}
        for filename in ("collect-observations.jsonl", "observations.jsonl"):
            path = args.run_root / "evidence" / "trunk" / filename
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    if row["nodeid"] in result["collected"]:
                        pair = pair_for_nodeid(row["nodeid"])
                        if pair in observed and observed[pair] != row["outcome"]:
                            raise RuntimeError(f"multiple outcomes for collected node {row['nodeid']}")
                        observed[pair] = row["outcome"]
        if set(observed) != set(result["xml_outcomes"]):
            raise RuntimeError("JUnit identities do not exactly reconcile with observed collected nodes")
        for outcome in ("failed", "error", "skipped"):
            result_key = "errors" if outcome == "error" else outcome
            observed_set = {node for node in result["collected"] if observed.get(pair_for_nodeid(node)) == outcome}
            expected_set = set(result[result_key]) - (set(result["collection_skipped"]) if outcome == "skipped" else set())
            if observed_set != expected_set:
                raise RuntimeError(f"observed {outcome} set is not stable")
            xml_set = {node for node in result["collected"] if result["xml_outcomes"].get(pair_for_nodeid(node)) == outcome}
            if xml_set != observed_set:
                raise RuntimeError(f"JUnit {outcome} set does not reconcile")
    for result in results.values():
        result["xml_outcomes"] = [{"classname": pair[0], "name": pair[1], "outcome": outcome}
                                  for pair, outcome in result["xml_outcomes"].items()]
    (args.run_root / "corpus-result.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"revision": args.revision, "collected": len(results["trunk"]["collected"]), "failed": results["trunk"]["failed"], "errors": results["trunk"]["errors"], "skipped": results["trunk"]["skipped"]}, sort_keys=True), flush=True)
    return 0 if all(result["pytest_rc"] == 0 for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
