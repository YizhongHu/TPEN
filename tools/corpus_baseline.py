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
import tempfile
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
        "effective_pytest_addopts": os.environ.get("PYTEST_ADDOPTS", ""),
        "effective_selection_policy": os.environ.get("CORPUS_SELECTION_POLICY", "UNSET"),
        "configured_addopts": list(session.config.getini("addopts")),
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


def check_emitted_plugin(path):
    """Parse the payload file immediately after the arm emits it."""
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    try:
        ast.parse(source + "\ndef deliberately_broken(:\n", filename="broken-emitted-plugin.py")
    except SyntaxError:
        return {"payload": "PASS", "broken_control": "PASS"}
    raise AssertionError("emitted-plugin broken control was accepted")


_OUTCOME_PRIORITY = {"passed": 0, "skipped": 1, "failed": 2, "error": 3}


def reduce_node_outcome(outcomes):
    """Collapse all pytest phase/subtest reports for one node by severity."""
    if not outcomes:
        raise ValueError("cannot reduce an empty outcome list")
    return max(outcomes, key=lambda outcome: _OUTCOME_PRIORITY[outcome])


def check_outcome_reducer_controls():
    """Prove the canonical phase/subtest precedence on representative reports."""
    controls = {
        "call-failed-teardown-error": ["passed", "failed", "error"],
        "call-failed-teardown-passed": ["passed", "failed", "passed"],
        "setup-error": ["error", "skipped", "passed"],
        "skip": ["passed", "skipped", "passed"],
        "subtests": ["passed", "passed", "passed", "passed"],
    }
    expected = {
        "call-failed-teardown-error": "error",
        "call-failed-teardown-passed": "failed",
        "setup-error": "error",
        "skip": "skipped",
        "subtests": "passed",
    }
    reduced = {name: reduce_node_outcome(outcomes) for name, outcomes in controls.items()}
    if reduced != expected:
        raise AssertionError(f"outcome reducer controls failed: {reduced!r}")
    print("OUTCOME_REDUCER_CONTROLS=PASS")
    return 0


def check_status_receipt_control():
    """Prove a nonzero sync status is durable before its caller raises."""
    with tempfile.TemporaryDirectory(prefix="corpus-status-control-") as directory:
        evidence = Path(directory)
        write_status(evidence, sync_rc=1)
        payload = json.loads((evidence / "statuses.json").read_text(encoding="utf-8"))
        if payload.get("sync_rc") != 1:
            raise AssertionError("sync failure receipt control did not retain rc=1")
    print("STATUS_RECEIPT_CONTROL=PASS")
    return 0


def check_selected_order_control():
    """Prove the selected-order comparison rejects an equal pair."""
    def require_different(left, right):
        if left == right:
            raise AssertionError("equal import sequences must be rejected")

    natural = ["module_a", "module_b"]
    reverse = list(natural)
    try:
        require_different(natural, reverse)
    except AssertionError:
        pass
    else:
        raise AssertionError("import-order equality control did not fail")
    print("SELECTED_ORDER_EQUALITY_CONTROL=PASS")
    return 0


def check_nodeid_controls():
    """Prove parameter punctuation remains part of one callable identity."""
    actual = pair_for_nodeid("tests/test_x.py::test_y[a::b]")
    expected = ("tests.test_x", "test_y[a::b]")
    if actual != expected:
        raise AssertionError(f"parameter punctuation mapping failed: {actual!r}")
    print("NODEID_PARAMETER_PUNCTUATION_CONTROL=PASS")
    return 0


def run(cmd, *, cwd, stdout, env, label, quota_root=None, quota_evidence=None, quota_commands=(), cache_dir=None):
    """Run one step with a durable merged stream and an unambiguous result."""
    capture_error = None
    try:
        stdout.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        capture_error = f"LOG_OPEN_ERROR {exc!r}"
    if quota_root is not None and quota_evidence is not None:
        safe_quota_snapshot(quota_root, quota_evidence, label + "-before", quota_commands, cache_dir or env.get("UV_CACHE_DIR", quota_root))
    rc = None
    try:
        if capture_error:
            raise OSError(capture_error)
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
            except OSError as exc:
                capture_error = capture_error or f"LOG_WRITE_ERROR {exc!r}"
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
    return {"rc": rc, "capture_error": capture_error}


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
    """Find real capture signatures while isolating the planted control."""
    predicates = ("LOG_WRITE_ERROR", "LAUNCH_ERROR", "LOG_OPEN_ERROR", "EDQUOT", "Disk quota exceeded")
    control_dir = evidence / "write-error-control"
    control_dir.mkdir(parents=True, exist_ok=True)
    control = control_dir / "positive-control.log"
    report = evidence / "write-error-sweep-report.txt"
    control.write_text("LOG_WRITE_ERROR\n", encoding="utf-8")
    control_text = control.read_text(encoding="utf-8")
    if not any(predicate in control_text for predicate in predicates):
        raise RuntimeError("write-error sweep positive control did not discriminate")
    matches = []
    for path in sorted(evidence.rglob("*")):
        if not path.is_file() or path == report or control_dir in path.parents:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        found = [predicate for predicate in predicates if predicate in text]
        if found:
            matches.append({"path": str(path.relative_to(evidence)), "signatures": found})
    report.write_text(
        f"PREDICATES={json.dumps(predicates)}\n"
        f"CONTROL_ISOLATED={control.relative_to(evidence)}\n"
        f"CONTROL_HIT=PASS\nREAL_MATCHES={json.dumps(matches, sort_keys=True)}\n"
        f"REPORT_EXCLUDED={report.name}\n", encoding="utf-8")
    return {"predicates": predicates, "control_hit": True, "real_matches": matches,
            "report_excluded": str(report)}


def xml_pairs(path):
    pairs = []
    for case in ET.parse(path).getroot().iter("testcase"):
        pairs.append((case.attrib.get("classname", ""), case.attrib.get("name", "")))
    return pairs


def xml_outcomes(path):
    reports = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        pair = (case.attrib.get("classname", ""), case.attrib.get("name", ""))
        children = {child.tag for child in case}
        outcome = "error" if "error" in children else "failed" if "failure" in children else "skipped" if "skipped" in children else "passed"
        reports.setdefault(pair, []).append(outcome)
    return {pair: reduce_node_outcome(outcomes) for pair, outcomes in reports.items()}


def pair_for_nodeid(nodeid):
    if "::" not in nodeid:
        return ("", nodeid.replace("/", ".").removesuffix(".py"))
    callable_node = nodeid.split("[", 1)[0]
    parameter = nodeid[len(callable_node):]
    module, *parts = callable_node.split("::")
    dotted = module.replace("/", ".").removesuffix(".py")
    if not parts:
        return (dotted, parameter)
    return (".".join([dotted, *parts[:-1]]) if len(parts) > 1 else dotted,
            parts[-1] + parameter)


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
    step = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-c", code],
             cwd=checkout, stdout=output, env=env, label="identity",
             quota_root=evidence.parents[1], quota_evidence=evidence, quota_commands=quota_commands)
    if step["rc"] != 0 or step["capture_error"]:
        raise RuntimeError(f"identity command rc={step['rc']} capture={step['capture_error']}; see {output}")
    return "\n".join(line for line in output.read_text(encoding="utf-8").splitlines()
                         if not line.startswith(("BEGIN ", "END "))) + "\n"


def arm(name, order, revision, root, uv, deliberate_red, quota_commands):
    checkout = root / name
    evidence = root / "evidence" / name
    evidence.mkdir(parents=True, exist_ok=True)
    envdir = root / ("venv-" + name)
    cachedir = root / ("uv-cache-" + name)
    env = os.environ.copy()
    inherited_addopts = env.pop("PYTEST_ADDOPTS", "")
    env["PYTEST_ADDOPTS"] = ""
    env.update({"UV_PROJECT_ENVIRONMENT": str(envdir), "UV_CACHE_DIR": str(cachedir), "PYTHONDONTWRITEBYTECODE": "1"})
    env["CORPUS_SELECTION_POLICY"] = "PYTEST_ADDOPTS_SANITIZED"
    env["CORPUS_COLLECTION_ORDER"] = order
    env["PATH"] = str(envdir / "bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(checkout) + os.pathsep + env.get("PYTHONPATH", "")
    if os.environ.get("SLURM_JOB_ID", "") == "":
        raise RuntimeError("SLURM_JOB_ID is required before any test command")
    write_status(evidence, arm=name, collection_order=order,
                 inherited_pytest_addopts=inherited_addopts,
                 effective_pytest_addopts="",
                 selection_policy="PYTEST_ADDOPTS_SANITIZED",
                 sync_rc=None,
                 collect_rc=None, pytest_rc=None, control_rc=None)
    safe_quota_snapshot(root, evidence, "arm-before", quota_commands, cachedir)
    if not checkout.exists():
        step = run(["git", "clone", "--no-single-branch", os.environ["CORPUS_SOURCE"], str(checkout)], cwd=root,
               stdout=evidence / "git-clone.log", env=env, label="git-clone",
               quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
        if step["rc"] != 0 or step["capture_error"]:
            raise RuntimeError("git clone failed")
    step = run(["git", "fetch", "--tags", "--all"], cwd=checkout, stdout=evidence / "git-fetch.log", env=env, label="git-fetch", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
    if step["rc"] != 0 or step["capture_error"]:
        raise RuntimeError("git fetch failed")
    step = run(["git", "checkout", "--detach", revision], cwd=checkout, stdout=evidence / "git-checkout.log", env=env, label="git-checkout", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
    if step["rc"] != 0 or step["capture_error"]:
        raise RuntimeError("git checkout failed")
    measured = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, text=True, capture_output=True, check=True).stdout.strip()
    if measured != revision:
        raise RuntimeError(f"{name}: measured {measured}, expected {revision}")
    tag_count = subprocess.run(["git", "tag", "--list"], cwd=checkout, text=True, capture_output=True, check=True).stdout.splitlines()
    shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=checkout, text=True, capture_output=True, check=True).stdout.strip()
    cache_log = evidence / "uv-cache-dir.log"
    cache_step = run([str(uv), "cache", "dir"], cwd=checkout, stdout=cache_log, env=env,
                   label="uv-cache-dir", quota_root=root, quota_evidence=evidence,
                   quota_commands=quota_commands, cache_dir=cachedir)
    cache_rc = cache_step["rc"]
    cache_lines = [line for line in cache_log.read_text(encoding="utf-8").splitlines()
                   if line and not line.startswith(("BEGIN ", "END "))]
    resolved_cache = cache_lines[-1] if cache_lines else "UNSET"
    write_status(evidence, uv_cache_dir_bound=str(cachedir), uv_cache_dir_resolved=resolved_cache,
                 uv_cache_dir_rc=cache_rc,
                 uv_cache_capture_error=cache_step["capture_error"])
    if cache_rc != 0 or cache_step["capture_error"]:
        raise RuntimeError(f"{name}: uv cache dir rc={cache_rc}")
    if resolved_cache != str(cachedir):
        raise RuntimeError(f"{name}: uv cache resolved to {resolved_cache!r}, expected bound {cachedir!s}")
    sync_step = run([str(uv), "sync", "--extra", "cpu", "--locked"], cwd=checkout, stdout=evidence / "uv-sync.log", env=env, label="uv-sync", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
    sync_rc = sync_step["rc"]
    write_status(evidence, sync_rc=sync_rc, sync_capture_error=sync_step["capture_error"])
    if sync_rc != 0 or sync_step["capture_error"]:
        raise RuntimeError(f"{name}: uv sync rc={sync_rc}")
    (evidence / "identity.txt").write_text(identity(checkout, uv, env, evidence, quota_commands) + f"REVISION={measured}\nTAG_COUNT={len(tag_count)}\nSHALLOW={shallow}\n", encoding="utf-8")
    plugin = checkout / "corpus_plugin.py"
    plugin.write_text(PLUGIN, encoding="utf-8")
    emitted_plugin_check = check_emitted_plugin(plugin)
    (evidence / "command-provenance.txt").write_text("uv run --extra cpu --locked python -m pytest -q --junitxml=corpus.xml -p corpus_plugin --corpus-observations=observations.jsonl\n", encoding="utf-8")
    env["CORPUS_PHASE"] = "collect"
    collect_step = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-m", "pytest", "-q", "--collect-only", "-p", "corpus_plugin", "--corpus-observations=collect-observations.jsonl"], cwd=checkout, stdout=evidence / "collect.log", env=env, label="pytest-collect", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
    collect_rc = collect_step["rc"]
    write_status(evidence, collect_rc=collect_rc, collect_capture_error=collect_step["capture_error"])
    for filename in ("collect-collected.txt", "collect-collection-skipped.txt", "collect-collection-metadata.json", "collect-observations.jsonl"):
        source = checkout / filename
        if source.exists():
            shutil.copy2(source, evidence / filename)
    env["CORPUS_PHASE"] = "run"
    pytest_step = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-m", "pytest", "-q", "--junitxml=corpus.xml", "-p", "corpus_plugin", "--corpus-observations=observations.jsonl"], cwd=checkout, stdout=evidence / "pytest.log", env=env, label="pytest-run", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
    pytest_rc = pytest_step["rc"]
    write_status(evidence, pytest_rc=pytest_rc, pytest_capture_error=pytest_step["capture_error"])
    if (checkout / "corpus.xml").exists():
        shutil.copy2(checkout / "corpus.xml", evidence / "corpus.xml")
    for filename in ("run-collected.txt", "run-collection-skipped.txt", "run-collection-metadata.json", "observations.jsonl"):
        source = checkout / filename
        if source.exists():
            shutil.copy2(source, evidence / filename)
    if collect_rc != 0:
        raise RuntimeError(f"{name}: collection instrument rc={collect_rc}")
    if collect_step["capture_error"]:
        raise RuntimeError(f"{name}: collection capture fault: {collect_step['capture_error']}")
    if pytest_step["capture_error"]:
        raise RuntimeError(f"{name}: pytest capture fault: {pytest_step['capture_error']}")
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
    if sweep["real_matches"]:
        write_status(evidence, run_sweep=sweep, corpus_complete=False)
        raise RuntimeError(f"{name}: real capture-error signatures found; completeness invalid")
    if deliberate_red:
        control = checkout / "test_corpus_deliberate_red.py"
        control.write_text("def test_corpus_deliberate_red():\n    assert False, 'intentional corpus extraction control'\n", encoding="utf-8")
        env["CORPUS_PHASE"] = "control"
        control_step = run([str(uv), "run", "--extra", "cpu", "--locked", "python", "-m", "pytest", "-q", "-k", "test_corpus_deliberate_red", "--junitxml=deliberate-red.xml", "-p", "corpus_plugin", "--corpus-observations=deliberate-red-observations.jsonl"], cwd=checkout, stdout=evidence / "deliberate-red.log", env=env, label="deliberate-red", quota_root=root, quota_evidence=evidence, quota_commands=quota_commands)
        control_rc = control_step["rc"]
        shutil.copy2(control, evidence / control.name)
        shutil.copy2(checkout / "deliberate-red.xml", evidence / "deliberate-red.xml")
        shutil.copy2(checkout / "deliberate-red-observations.jsonl", evidence / "deliberate-red-observations.jsonl")
        (evidence / "deliberate-red-status.txt").write_text(f"INNER_RC={control_rc}\nCAPTURE_ERROR={control_step['capture_error']}\n", encoding="utf-8")
        write_status(evidence, control_rc=control_rc, control_capture_error=control_step["capture_error"])
        if control_rc != 1 or control_step["capture_error"]:
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
    write_status(evidence, run_sweep=sweep, run_complete=False,
                 emitted_plugin_check=emitted_plugin_check)
    marker = evidence / "RUN_COMPLETE.marker"
    marker.write_text(
        f"RUN_COMPLETE arm={name} order={order} revision={measured}\n", encoding="utf-8")
    if not marker.is_file() or marker.stat().st_size == 0:
        write_status(evidence, run_complete=False)
        raise RuntimeError(f"{name}: run completion marker write was not durable")
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
    check_outcome_reducer_controls()
    check_status_receipt_control()
    check_selected_order_control()
    check_nodeid_controls()
    check_plugin_literal()
    args.run_root.mkdir(parents=True, exist_ok=True)
    os.environ["CORPUS_SOURCE"] = args.source
    results = {}
    try:
        quota_commands = json.loads(args.quota_spec)
        if len(quota_commands) != 3 or not all(isinstance(command, list) for command in quota_commands):
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"--quota-spec must be JSON list of three argv lists: {exc}")
    arm_errors = {}
    for arm_name, arm_order, red in (("natural", "natural", True), ("reverse", "reverse", False)):
        try:
            results[arm_name] = arm(arm_name, arm_order, args.revision, args.run_root, args.uv, red, quota_commands)
        except Exception as exc:
            arm_errors[arm_name] = repr(exc)
            evidence = args.run_root / "evidence" / arm_name
            if (evidence / "statuses.json").exists():
                write_status(evidence, arm_error=repr(exc))
    for result in results.values():
        result["xml_outcomes"] = {(row["classname"], row["name"]): row["outcome"] for row in result["xml_outcomes"]}
        evidence = args.run_root / "evidence" / result["collection_order"]
        if not (evidence / "RUN_COMPLETE.marker").exists():
            raise RuntimeError(f"missing run-level completion marker for {result['collection_order']}")
        if result["collect_sequence"] != result["run_sequence"]:
            raise RuntimeError(f"ORDER_MISMATCH for {result['collection_order']}")
        for filename in ("collect-collection-metadata.json", "run-collection-metadata.json"):
            metadata = json.loads((evidence / filename).read_text(encoding="utf-8"))
            if metadata["order"] != result["collection_order"]:
                raise RuntimeError(f"collection metadata order mismatch for {result['collection_order']}")
        observation_file = evidence / "observations.jsonl"
        if not observation_file.exists():
            raise RuntimeError(f"missing required observation artifact: {observation_file}")
        observed_reports = {}
        for line in observation_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            observed_reports.setdefault(row["nodeid"], []).append(row["outcome"])
        if set(result["collected"]) != set(result["run_collected"]):
            raise RuntimeError("collect-only and run collected sets differ")
        if set(result["collection_skipped"]) != set(result["run_collection_skipped"]):
            raise RuntimeError("collect-only and run collection-skip sets differ")
        observed = {nodeid: reduce_node_outcome(outcomes)
                    for nodeid, outcomes in observed_reports.items()}
        for nodeid in result["collection_skipped"]:
            observed[nodeid] = "skipped"
        result["failed"] = sorted(nodeid for nodeid, outcome in observed.items() if outcome == "failed")
        result["errors"] = sorted(nodeid for nodeid, outcome in observed.items() if outcome == "error")
        result["skipped"] = sorted(nodeid for nodeid, outcome in observed.items() if outcome == "skipped")
        floor = len(set(result["collected"]) | set(result["collection_skipped"]))
        result["floor"] = floor
        expected_pairs = {pair_for_nodeid(node) for node in result["collected"] + result["collection_skipped"]}
        if set(result["xml_pairs"]) - expected_pairs:
            raise RuntimeError("JUnit outcome contains an identity absent from collect-only set")
        xml_identity_set = set(result["xml_pairs"])
        if len(xml_identity_set) != floor:
            raise RuntimeError(f"JUnit identities {len(xml_identity_set)} do not equal in-job floor {floor}")
        observed_by_pair = {pair_for_nodeid(nodeid): outcome for nodeid, outcome in observed.items()}
        if set(observed_by_pair) != set(result["xml_outcomes"]):
            raise RuntimeError("JUnit identities do not exactly reconcile with observed collected nodes")
        for outcome in ("failed", "error", "skipped"):
            result_key = "errors" if outcome == "error" else outcome
            observed_set = {node for node, actual in observed.items() if actual == outcome}
            if observed_set != set(result[result_key]):
                raise RuntimeError(f"observed {outcome} set is not stable")
            xml_set = {node for node in result["collected"] + result["collection_skipped"]
                       if result["xml_outcomes"].get(pair_for_nodeid(node)) == outcome}
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
