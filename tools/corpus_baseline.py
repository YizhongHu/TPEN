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
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


PLUGIN = r'''
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
    _OBS.with_name("collection_skipped.txt").write_text("", encoding="utf-8")

def pytest_collection_modifyitems(config, items):
    p = _out(config).with_name("collected.txt")
    p.write_text("\n".join(sorted(item.nodeid for item in items)) + "\n", encoding="utf-8")

def pytest_collectreport(report):
    if report.failed:
        return
    if report.skipped:
        p = _OBS.with_name("collection_skipped.txt")
        with p.open("a", encoding="utf-8") as stream:
            stream.write(report.nodeid + "\n")

def pytest_runtest_logreport(report):
    if report.when != "call":
        return
    p = _OBS
    label = "passed"
    if report.failed:
        label = "error" if report.when != "call" else "failed"
    elif report.skipped:
        label = "skipped"
    with p.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"nodeid": report.nodeid, "outcome": label}) + "\n")
'''


def run(cmd, *, cwd, stdout, env):
    with stdout.open("w", encoding="utf-8") as stream:
        return subprocess.run(cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode


def xml_pairs(path):
    pairs = []
    for case in ET.parse(path).getroot().iter("testcase"):
        pairs.append((case.attrib.get("classname", ""), case.attrib.get("name", "")))
    return pairs


def pair_for_nodeid(nodeid):
    if "::" not in nodeid:
        return ("", nodeid.replace("/", ".").removesuffix(".py"))
    module, *parts = nodeid.split("::")
    dotted = module.replace("/", ".").removesuffix(".py")
    return (".".join([dotted, *parts[:-1]]) if len(parts) > 1 else dotted, parts[-1])


def identity(checkout, uv, env):
    code = (
        "import os, pathlib, shutil, sys; import torch; "
        "print('SYS_EXECUTABLE='+sys.executable, flush=True); "
        "print('PYTHON_VERSION='+sys.version, flush=True); "
        "print('TORCH_VERSION='+torch.__version__, flush=True); "
        "print('TORCH_FILE='+str(pathlib.Path(torch.__file__).absolute()), flush=True); "
        "print('PYTHON_WHICH='+str(shutil.which('python')), flush=True); "
        "print('PYTHONPATH='+os.environ.get('PYTHONPATH',''), flush=True)"
    )
    return subprocess.run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-c", code], cwd=checkout, env=env, text=True, capture_output=True, check=True).stdout


def arm(name, revision, root, uv, deliberate_red):
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
    if not checkout.exists():
        subprocess.run(["git", "clone", "--no-single-branch", os.environ["CORPUS_SOURCE"], str(checkout)], check=True)
    subprocess.run(["git", "fetch", "--tags", "--all"], cwd=checkout, check=True)
    subprocess.run(["git", "checkout", "--detach", revision], cwd=checkout, check=True)
    measured = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, text=True, capture_output=True, check=True).stdout.strip()
    if measured != revision:
        raise RuntimeError(f"{name}: measured {measured}, expected {revision}")
    tag_count = subprocess.run(["git", "tag", "--list"], cwd=checkout, text=True, capture_output=True, check=True).stdout.splitlines()
    shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=checkout, text=True, capture_output=True, check=True).stdout.strip()
    sync_rc = run([str(uv), "sync", "--extra", "cpu", "--locked"], cwd=checkout, stdout=evidence / "uv-sync.log", env=env)
    if sync_rc != 0:
        raise RuntimeError(f"{name}: uv sync rc={sync_rc}")
    (evidence / "identity.txt").write_text(identity(checkout, uv, env) + f"REVISION={measured}\nTAG_COUNT={len(tag_count)}\nSHALLOW={shallow}\n", encoding="utf-8")
    plugin = checkout / ".corpus_plugin.py"
    plugin.write_text(PLUGIN, encoding="utf-8")
    (evidence / "command-provenance.txt").write_text("uv run --extra cpu --locked python -m pytest -q --junitxml=corpus.xml -p corpus_plugin --corpus-observations=observations.jsonl\n", encoding="utf-8")
    collect_rc = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-m", "pytest", "-q", "--collect-only", "-p", "corpus_plugin", "--corpus-observations=collect-observations.jsonl"], cwd=checkout, stdout=evidence / "collect.log", env=env)
    pytest_rc = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-m", "pytest", "-q", "--junitxml=corpus.xml", "-p", "corpus_plugin", "--corpus-observations=observations.jsonl"], cwd=checkout, stdout=evidence / "pytest.log", env=env)
    for filename in ("collected.txt", "collection_skipped.txt", "observations.jsonl"):
        source = checkout / filename
        if source.exists():
            shutil.copy2(source, evidence / filename)
    (evidence / "statuses.json").write_text(json.dumps({"collect_rc": collect_rc, "pytest_rc": pytest_rc}, indent=2) + "\n", encoding="utf-8")
    if collect_rc != 0:
        raise RuntimeError(f"{name}: collection instrument rc={collect_rc}")
    if deliberate_red:
        control = checkout / "test_corpus_deliberate_red.py"
        control.write_text("def test_corpus_deliberate_red():\n    assert False, 'intentional corpus extraction control'\n", encoding="utf-8")
        control_rc = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-m", "pytest", "-q", "-k", "test_corpus_deliberate_red", "--junitxml=deliberate-red.xml"], cwd=checkout, stdout=evidence / "deliberate-red.log", env=env)
        shutil.copy2(control, evidence / control.name)
        shutil.copy2(checkout / "deliberate-red.xml", evidence / "deliberate-red.xml")
        (evidence / "deliberate-red-status.txt").write_text(f"INNER_RC={control_rc}\n", encoding="utf-8")
        if control_rc != 1:
            raise RuntimeError(f"{name}: deliberate red rc={control_rc}, expected 1")
        control_pairs = xml_pairs(evidence / "deliberate-red.xml")
        if not any(name == "test_corpus_deliberate_red" for _, name in control_pairs):
            raise RuntimeError(f"{name}: deliberate red node absent from JUnit")
        control.unlink()
    return {
        "revision": measured, "tag_count": len(tag_count), "shallow": shallow,
        "collect_rc": collect_rc, "pytest_rc": pytest_rc,
        "collected": (evidence / "collected.txt").read_text(encoding="utf-8").splitlines(),
        "collection_skipped": (evidence / "collection_skipped.txt").read_text(encoding="utf-8").splitlines() if (evidence / "collection_skipped.txt").exists() else [],
        "xml_pairs": xml_pairs(evidence / "corpus.xml"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("must run inside a Slurm allocation")
    args.run_root.mkdir(parents=True, exist_ok=True)
    os.environ["CORPUS_SOURCE"] = args.source
    results = {}
    results["trunk"] = arm("trunk", args.revision, args.run_root, args.uv, True)
    for result in results.values():
        result["failed"] = []
        result["errors"] = []
        result["skipped"] = result["collection_skipped"]
        for line in (args.run_root / "evidence" / "trunk" / "observations.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            result[{"failed": "failed", "error": "errors", "skipped": "skipped"}.get(row["outcome"], "ignored")].append(row["nodeid"]) if row["outcome"] in {"failed", "error", "skipped"} else None
        floor = len(set(result["collected"]) | set(result["collection_skipped"]))
        result["floor"] = floor
        expected_pairs = {pair_for_nodeid(node) for node in result["collected"] + result["collection_skipped"]}
        if set(result["xml_pairs"]) - expected_pairs:
            raise RuntimeError("JUnit outcome contains an identity absent from collect-only set")
        if len(result["xml_pairs"]) != len(set(result["xml_pairs"])):
            raise RuntimeError("JUnit contains duplicate (classname,name) identities")
        if len(result["xml_pairs"]) != floor:
            raise RuntimeError(f"JUnit rows {len(result['xml_pairs'])} do not equal in-job floor {floor}")
    (args.run_root / "corpus-result.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"revision": args.revision, "collected": len(results["trunk"]["collected"]), "failed": results["trunk"]["failed"], "errors": results["trunk"]["errors"], "skipped": results["trunk"]["skipped"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
