#!/usr/bin/env python3
"""Run and reconcile a complete pytest corpus in one allocation.

The script is intentionally standard-library-only so it can be copied into an
allocation before the project environment is provisioned.  It never supplies
a pytest path: each arm starts at its checkout root and pytest self-selects.
"""

from __future__ import annotations

import argparse
import ast
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
_SELECTED = []

def _out(config):
    return Path(config.getoption("--corpus-observations"))

def pytest_addoption(parser):
    parser.addoption("--corpus-observations", required=True)

def pytest_configure(config):
    global _OBS, _SELECTED
    _OBS = _out(config)
    _SELECTED = []
    _OBS.write_text("", encoding="utf-8")
    phase = os.environ.get("CORPUS_PHASE", "run")
    _OBS.with_name(phase + "-collection-skipped.txt").write_text("", encoding="utf-8")

def pytest_collection_modifyitems(config, items):
    phase = os.environ.get("CORPUS_PHASE", "run")
    order = os.environ.get("CORPUS_COLLECTION_ORDER", "UNSET")
    if order == "reverse":
        items.reverse()
    elif order != "natural":
        raise RuntimeError("CORPUS_COLLECTION_ORDER must be named natural or reverse")
    _SELECTED[:] = [item.nodeid for item in items]
    p = _out(config).with_name(phase + "-collected.txt")
    p.write_text("\n".join(_SELECTED) + "\n", encoding="utf-8")

def pytest_collection_finish(session):
    phase = os.environ.get("CORPUS_PHASE", "run")
    metadata = {
        "phase": phase,
        "order": os.environ.get("CORPUS_COLLECTION_ORDER", "UNSET"),
        "selected_sequence": list(_SELECTED),
        "argv": list(session.config.invocation_params.args),
        "cwd": os.getcwd(),
        "rootdir": str(session.config.rootpath),
        "pytest_version": __import__("pytest").__version__,
        "plugin_version": "corpus-order-v1",
        "config_identity": str(session.config.inifile) if session.config.inifile else "UNSET",
        "ordering_seed": os.environ.get("PYTEST_RANDOMLY_SEED", "UNSET"),
    }
    _out(config=session.config).with_name(phase + "-collection-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

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


def check_plugin_literal():
    """Parse the emitted plugin payload and prove the gate rejects a broken one."""
    source = Path(__file__).read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(Path(__file__)))
    plugin_node = next(
        node for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PLUGIN" for target in node.targets)
    )
    plugin = ast.literal_eval(plugin_node.value)
    ast.parse(plugin, filename="corpus_plugin.py")
    broken = plugin + "\ndef deliberately_broken(:\n"
    try:
        ast.parse(broken, filename="broken-corpus_plugin.py")
    except SyntaxError:
        print("PLUGIN_PAYLOAD_PARSE=PASS")
        print("PLUGIN_BROKEN_CONTROL=PASS")
        return 0
    raise AssertionError("deliberately broken plugin control was accepted")


_OUTCOME_PRIORITY = {"passed": 0, "skipped": 1, "failed": 2, "error": 3}


def reduce_node_outcome(outcomes):
    """Collapse all pytest phase/subtest reports for one node by severity."""
    if not outcomes:
        raise ValueError("cannot reduce an empty outcome list")
    return max(outcomes, key=lambda outcome: _OUTCOME_PRIORITY[outcome])


def check_outcome_reducer_controls():
    """Prove skip, teardown failure, and repeated subtest reports reduce safely."""
    controls = {
        "skip": ["passed", "skipped", "passed"],
        "teardown-failure": ["passed", "passed", "error"],
        "subtests": ["passed", "passed", "passed"],
    }
    expected = {"skip": "skipped", "teardown-failure": "error", "subtests": "passed"}
    reduced = {name: reduce_node_outcome(outcomes) for name, outcomes in controls.items()}
    if reduced != expected:
        raise AssertionError(f"outcome reducer controls failed: {reduced!r}")
    print("OUTCOME_REDUCER_CONTROLS=PASS")
    return 0


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


def write_status(evidence, **values):
    """Update the durable per-step receipt before any step can raise."""
    path = evidence / "statuses.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(values)
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_complete_xml(path):
    """Reject a short, well-formed XML prefix by checking its tail explicitly."""
    tail = path.read_bytes()[-256:]
    if b"</testsuites>" not in tail and b"</testsuite>" not in tail:
        raise RuntimeError(f"JUnit closing tag missing from tail: {path}")


def write_error_sweep(evidence):
    """Exercise the sweep and exclude the report from its own searched set."""
    predicate = "WRITE_ERROR_SWEEP_TRIGGER"
    control = evidence / "write-error-positive-control.log"
    report = evidence / "write-error-sweep-report.txt"
    control.write_text(predicate + "\n", encoding="utf-8")
    matches = []
    for path in sorted(evidence.rglob("*")):
        if not path.is_file() or path == report:
            continue
        if predicate in path.read_text(encoding="utf-8", errors="replace"):
            matches.append(str(path.relative_to(evidence)))
    report.write_text(f"PREDICATE={predicate}\nMATCHES={json.dumps(matches)}\n", encoding="utf-8")
    if not matches:
        raise RuntimeError("write-error sweep did not find positive control")
    return {"predicate": predicate, "matches": matches, "report_excluded": str(report)}


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


def arm(name, order, revision, root, uv, deliberate_red, quota_commands):
    checkout = root / name
    evidence = root / "evidence" / name
    evidence.mkdir(parents=True, exist_ok=True)
    envdir = root / ("venv-" + name)
    cachedir = root / ("uv-cache-" + name)
    env = os.environ.copy()
    env.update({"UV_PROJECT_ENVIRONMENT": str(envdir), "UV_CACHE_DIR": str(cachedir), "PYTHONDONTWRITEBYTECODE": "1"})
    env["CORPUS_COLLECTION_ORDER"] = order
    env["PATH"] = str(envdir / "bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(checkout) + os.pathsep + env.get("PYTHONPATH", "")
    if os.environ.get("SLURM_JOB_ID", "") == "":
        raise RuntimeError("SLURM_JOB_ID is required before any test command")
    write_status(evidence, arm=name, collection_order=order, sync_rc=None,
                 collect_rc=None, pytest_rc=None, control_rc=None)
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
    cache_log = evidence / "uv-cache-dir.log"
    cache_rc = run([str(uv), "cache", "dir"], cwd=checkout, stdout=cache_log, env=env,
                   label="uv-cache-dir", quota_root=root, quota_evidence=evidence,
                   quota_commands=quota_commands, cache_dir=cachedir)
    cache_lines = [line for line in cache_log.read_text(encoding="utf-8").splitlines()
                   if line and not line.startswith(("BEGIN ", "END "))]
    resolved_cache = cache_lines[-1] if cache_lines else "UNSET"
    write_status(evidence, uv_cache_dir_bound=str(cachedir), uv_cache_dir_resolved=resolved_cache,
                 uv_cache_dir_rc=cache_rc)
    if cache_rc != 0:
        raise RuntimeError(f"{name}: uv cache dir rc={cache_rc}")
    sync_rc = run([str(uv), "sync", "--extra", "cpu", "--locked"], cwd=checkout, stdout=evidence / "uv-sync.log", env=env, label="uv-sync", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
    write_status(evidence, sync_rc=sync_rc)
    if sync_rc != 0:
        raise RuntimeError(f"{name}: uv sync rc={sync_rc}")
    (evidence / "identity.txt").write_text(identity(checkout, uv, env, evidence, quota_commands) + f"REVISION={measured}\nTAG_COUNT={len(tag_count)}\nSHALLOW={shallow}\n", encoding="utf-8")
    plugin = checkout / "corpus_plugin.py"
    plugin.write_text(PLUGIN, encoding="utf-8")
    (evidence / "command-provenance.txt").write_text("uv run --extra cpu --locked python -m pytest -q --junitxml=corpus.xml -p corpus_plugin --corpus-observations=observations.jsonl\n", encoding="utf-8")
    env["CORPUS_PHASE"] = "collect"
    collect_rc = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-m", "pytest", "-q", "--collect-only", "-p", "corpus_plugin", "--corpus-observations=collect-observations.jsonl"], cwd=checkout, stdout=evidence / "collect.log", env=env, label="pytest-collect", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
    write_status(evidence, collect_rc=collect_rc)
    for filename in ("collect-collected.txt", "collect-collection-skipped.txt", "collect-collection-metadata.json", "collect-observations.jsonl"):
        source = checkout / filename
        if source.exists():
            shutil.copy2(source, evidence / filename)
    env["CORPUS_PHASE"] = "run"
    pytest_rc = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-m", "pytest", "-q", "--junitxml=corpus.xml", "-p", "corpus_plugin", "--corpus-observations=observations.jsonl"], cwd=checkout, stdout=evidence / "pytest.log", env=env, label="pytest-run", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
    write_status(evidence, pytest_rc=pytest_rc)
    if (checkout / "corpus.xml").exists():
        shutil.copy2(checkout / "corpus.xml", evidence / "corpus.xml")
    for filename in ("run-collected.txt", "run-collection-skipped.txt", "run-collection-metadata.json", "observations.jsonl"):
        source = checkout / filename
        if source.exists():
            shutil.copy2(source, evidence / filename)
    if collect_rc != 0:
        raise RuntimeError(f"{name}: collection instrument rc={collect_rc}")
    if pytest_rc != 0:
        raise RuntimeError(f"{name}: pytest rc={pytest_rc}")
    assert_complete_xml(evidence / "corpus.xml")
    truncated_xml = evidence / "corpus-truncated.xml"
    xml_bytes = (evidence / "corpus.xml").read_bytes()
    truncated_xml.write_bytes(xml_bytes[:-20])
    try:
        assert_complete_xml(truncated_xml)
    except RuntimeError:
        truncation_rejected = True
    else:
        raise RuntimeError(f"truncated JUnit was accepted: {truncated_xml}")
    sweep = write_error_sweep(evidence)
    if deliberate_red:
        control = checkout / "test_corpus_deliberate_red.py"
        control.write_text("def test_corpus_deliberate_red():\n    assert False, 'intentional corpus extraction control'\n", encoding="utf-8")
        env["CORPUS_PHASE"] = "control"
        control_rc = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-m", "pytest", "-q", "-k", "test_corpus_deliberate_red", "--junitxml=deliberate-red.xml", "-p", "corpus_plugin", "--corpus-observations=deliberate-red-observations.jsonl"], cwd=checkout, stdout=evidence / "deliberate-red.log", env=env, label="deliberate-red", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
        shutil.copy2(control, evidence / control.name)
        shutil.copy2(checkout / "deliberate-red.xml", evidence / "deliberate-red.xml")
        shutil.copy2(checkout / "deliberate-red-observations.jsonl", evidence / "deliberate-red-observations.jsonl")
        (evidence / "deliberate-red-status.txt").write_text(f"INNER_RC={control_rc}\n", encoding="utf-8")
        write_status(evidence, control_rc=control_rc)
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
    write_status(evidence, run_sweep=sweep, run_complete=False)
    (evidence / "RUN_COMPLETE.marker").write_text(
        f"RUN_COMPLETE arm={name} order={order} revision={measured}\n", encoding="utf-8")
    return {
        "revision": measured, "tag_count": len(tag_count), "shallow": shallow,
        "collection_order": order, "collect_rc": collect_rc, "pytest_rc": pytest_rc,
        "sync_rc": sync_rc, "uv_cache_dir_bound": str(cachedir),
        "uv_cache_dir_resolved": resolved_cache, "run_complete": True,
        "truncation_rejected": truncation_rejected,
        "run_sweep": sweep,
        "split_report": {"arm": name, "collect_only": True, "junit": True},
        "collect_sequence": (evidence / "collect-collected.txt").read_text(encoding="utf-8").splitlines(),
        "run_sequence": (evidence / "run-collected.txt").read_text(encoding="utf-8").splitlines(),
        "collected": (evidence / "collect-collected.txt").read_text(encoding="utf-8").splitlines(),
        "collection_skipped": (evidence / "collect-collection-skipped.txt").read_text(encoding="utf-8").splitlines() if (evidence / "collect-collection-skipped.txt").exists() else [],
        "run_collected": (evidence / "run-collected.txt").read_text(encoding="utf-8").splitlines() if (evidence / "run-collected.txt").exists() else [],
        "run_collection_skipped": (evidence / "run-collection-skipped.txt").read_text(encoding="utf-8").splitlines() if (evidence / "run-collection-skipped.txt").exists() else [],
        "xml_pairs": xml_pairs(evidence / "corpus.xml"),
        "xml_outcomes": [{"classname": pair[0], "name": pair[1], "outcome": outcome}
                         for pair, outcome in xml_outcomes(evidence / "corpus.xml").items()],
    }


def main():
    if "--check-plugin" in sys.argv:
        return check_plugin_literal()
    if "--check-outcome-reducer" in sys.argv:
        return check_outcome_reducer_controls()
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
    results["natural"] = arm("natural", "natural", args.revision, args.run_root, args.uv, True, quota_commands)
    results["reverse"] = arm("reverse", "reverse", args.revision, args.run_root, args.uv, False, quota_commands)
    for result in results.values():
        result["xml_outcomes"] = {(row["classname"], row["name"]): row["outcome"] for row in result["xml_outcomes"]}
        result["failed"] = []
        result["errors"] = []
        result["skipped"] = []
        evidence = args.run_root / "evidence" / result["collection_order"]
        if not (evidence / "RUN_COMPLETE.marker").exists():
            raise RuntimeError(f"missing run-level completion marker for {result['collection_order']}")
        if result["collect_sequence"] != result["run_sequence"]:
            raise RuntimeError(f"ORDER_MISMATCH for {result['collection_order']}")
        for filename in ("collect-collection-metadata.json", "run-collection-metadata.json"):
            metadata = json.loads((evidence / filename).read_text(encoding="utf-8"))
            if metadata["order"] != result["collection_order"]:
                raise RuntimeError(f"collection metadata order mismatch for {result['collection_order']}")
        for filename in ("collect-observations.jsonl", "observations.jsonl"):
            observation_file = evidence / filename
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
        observed_reports = {}
        for filename in ("collect-observations.jsonl", "observations.jsonl"):
            path = evidence / filename
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    if row["nodeid"] in result["collected"]:
                        observed_reports.setdefault(row["nodeid"], []).append(row["outcome"])
        observed = {pair_for_nodeid(nodeid): reduce_node_outcome(outcomes)
                    for nodeid, outcomes in observed_reports.items()}
        for nodeid in result["collection_skipped"]:
            observed[pair_for_nodeid(nodeid)] = "skipped"
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
    if set(results["natural"]["collected"]) != set(results["reverse"]["collected"]):
        raise RuntimeError("ORDER_MISMATCH: two named arms collected different node-ID sets")
    if results["natural"]["collect_sequence"] == results["reverse"]["collect_sequence"]:
        raise RuntimeError("ORDER_MISMATCH: named natural and reverse arms did not differ")
    for result in results.values():
        result["xml_outcomes"] = [{"classname": pair[0], "name": pair[1], "outcome": outcome}
                                  for pair, outcome in result["xml_outcomes"].items()]
    (args.run_root / "corpus-result.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"revision": args.revision, "arms": {
        name: {"order": result["collection_order"], "collected": len(result["collected"]),
               "failed": result["failed"], "errors": result["errors"], "skipped": result["skipped"]}
        for name, result in results.items()}}, sort_keys=True), flush=True)
    return 0 if all(result["pytest_rc"] == 0 for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
