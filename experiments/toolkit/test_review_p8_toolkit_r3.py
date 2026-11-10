"""L2 consumer proof: real executor placement output crosses into P8.

Mutant controls
---------------
M7
    Old: accept only ``science-placement-receipt-v1`` and discard v2 fields.
    New: decode the producer's closed L1a v2 receipt and manifest-v3 verdict.
    Killer: the real executor producer-to-reader contract test in the root
    suite, using the real executor dispatch/store producer.
M8
    Old: freeze placement evidence after core inspection.
    New: validate the closed L1a JSON evidence while reading the manifest.
    Killer: ``test_p8_reader_refuses_v2_wire_mutations_before_inspection``.
M9
    Dropped: ``final_verdict2.py`` is not present in this repository.  There
    is no source mutant or root-suite killer for it.

L1b (S2b-2 sampler layer)
    The worker that owns each declared-GPU v2 child samples it for its whole
    lifetime, so the real ``matched`` mode yields a v3 ``qualified`` receipt.
    The fake ``nvidia-smi`` runs as a child of the worker, so it reports the
    science child's PID from the child placement sidecar (and only while that
    PID is alive) instead of ``$PPID``.  The L1a v2 backward read is proven on
    untouched L1a producer bytes in ``test_run_results_l1b.py``.
    ``test_l1b_real_producer_outcomes_cross_p8_reader`` drives every v3
    outcome through the real CLI dispatch/store with deterministic fake
    ``nvidia-smi`` modes, and ``test_l1b_executor_and_p8_refuse_identical_tampers``
    feeds the same tampered bytes to both decoders.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import textwrap

from omegaconf import OmegaConf
import pytest

from experiments.toolkit.config_identity import expected_resolved_config_identity
from experiments.toolkit.run_results import (
    PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE,
    PLACEMENT_UNQUALIFIED_MESSAGE,
    ManifestRefusal,
    read,
)


ROOT = Path(__file__).resolve().parents[2]
# The consumer config is the single source for both the producer's config file
# and the identity the stub core stamps, mirroring the real core boundary.
CONSUMER_CONFIG = "mode: consumer-test\n"


MUTANT_TABLE = (
    {
        "id": "M7",
        "target": "experiments/toolkit/run_results.py",
        "old": "    if schema == SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2:\n",
        "new": "    if False:\n",
        "killer": "test_l2_real_executor_producer_to_p8_reader_contract",
        "suite": "root",
    },
    {
        "id": "M8",
        "target": "experiments/toolkit/run_results.py",
        "old": "    _validate_json(value)\n    schema = value.get(\"schema\") if isinstance(value, dict) else None\n",
        "new": "    schema = value.get(\"schema\") if isinstance(value, dict) else None\n",
        "killer": "test_p8_reader_rejects_nonfinite_before_inspection",
        "suite": "root",
    },
    {
        "id": "M10",
        "target": "experiments/toolkit/run_results.py",
        "old": "    if declared[\"devices\"]:\n",
        "new": "    if True:\n",
        "killer": "test_p8_reader_accepts_cpu_only_v2_receipts",
        "suite": "root",
    },
)


def test_l2_mutant_table_has_exact_unique_source_anchors() -> None:
    """Mutation packets must name literal production replacements."""

    assert {row["id"] for row in MUTANT_TABLE} == {"M7", "M8", "M10"}
    for row in MUTANT_TABLE:
        source = (ROOT / row["target"]).read_text(encoding="utf-8")
        assert source.count(row["old"]) == 1, row
        assert row["old"] != row["new"]
        assert row["killer"] and row["suite"] in {"root", "executor"}


def _executor_python() -> str:
    value = os.environ.get("P8_EXECUTOR_PYTHON") or os.environ.get("P9C_EXECUTOR_PYTHON")
    if not value:
        pytest.skip("named skip: P8_EXECUTOR_PYTHON/P9C_EXECUTOR_PYTHON is unset")
    return value


def _consumer_identity() -> dict[str, str]:
    """Compute the consumer config identity with core's public function."""

    return expected_resolved_config_identity(
        OmegaConf.to_container(OmegaConf.create(CONSUMER_CONFIG), resolve=False)
    )


def _producer_code() -> str:
    source = textwrap.dedent(
        """
        import json, os, socket, subprocess, sys
        from pathlib import Path
        from tpen_executor._codec import digest
        from tpen_executor.allocations import ConfiguredAllocation
        from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
        from tpen_executor.placement import AllocationSlot
        from tpen_executor.tasks import ConfiguredRun, ConfiguredRunTable

        root = Path(sys.argv[1])
        mode = sys.argv[2]
        fake_bin = Path(sys.argv[3])
        config_text = sys.argv[4]
        identity = sys.argv[5]
        stub = root / "stub" / "tpen"
        stub.mkdir(parents=True)
        (stub / "__init__.py").write_text("", encoding="utf-8")
        (stub / "run.py").write_text(
            "import json, os, sys\\n"
            "from pathlib import Path\\n"
            "if Path(__file__).resolve().parent != Path(os.environ['L2_STUB_ORIGIN']).resolve():\\n"
            "    raise RuntimeError('consumer-test imported an unexpected tpen.run')\\n"
            "def main(argv):\\n"
            "    import time; time.sleep(float(os.environ.get('L1B_STUB_SECONDS', '0')))\\n"
            "    values = dict(item.split('=', 1) for item in argv if '=' in item)\\n"
            "    run_dir = Path(values['run.root']) / values.get('run.run_id', 'core')\\n"
            "    checkpoint = run_dir / 'checkpoint' / 'step_000001'\\n"
            "    checkpoint.mkdir(parents=True, exist_ok=True)\\n"
            "    (run_dir / 'evidence.json').write_text(json.dumps({\\n"
            "        'schema': 'tpen.run-evidence/v2', 'run_dir': str(run_dir),\\n"
            "        'verdict': 'completed', 'run_status': 'completed',\\n"
            "        'checkpoint': {'checkpoint_dir': str(checkpoint), 'schema_version': 1,\\n"
            "                       'next_iteration': 1, 'completed_updates': 1},\\n"
            "        'refusal': None,\\n"
            "        'resolved_config_identity': json.loads(os.environ['L2_RESOLVED_CONFIG_IDENTITY']),\\n"
            "    }), encoding='utf-8')\\n"
            "    return 3 if os.environ.get('L2_REFUSED') == '1' else 0\\n"
            "if __name__ == '__main__': raise SystemExit(main(sys.argv[1:]))\\n",
            encoding="utf-8",
        )
        (stub / "inspection.py").write_text('''
import json, sys
from pathlib import Path
print(Path(sys.argv[1], "evidence.json").read_text(encoding="utf-8"))
''', encoding="utf-8")

        config = root / "config.yaml"
        config.write_text(config_text, encoding="utf-8")
        group = ReadyGroup("l2-consumer", (TaskSpec("task-a", "task-a", ProgramSpecRef({}, digest({})), digest("task-a"), 1, "steps"),), digest("l2-consumer"))
        cpu = min(getattr(os, "sched_getaffinity", lambda _pid: {0})(0))
        devices = {"cpu-not-sampled": (), "gpu-unresolved": ("GPU-12345678-1234-1234-1234-123456789ABC", "GPU-12345678-1234-1234-1234-123456789ABE")}.get(mode, ("GPU-12345678-1234-1234-1234-123456789ABC",))
        slot = AllocationSlot("logical-slot", "logical-label", devices, (cpu,), (), ()).to_dict()
        allocation = ConfiguredAllocation(
            AllocationId("allocation-a"), BackendJobId("job-a"),
            {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot]},
             "scheduler": ({"PBS_NODEFILE": str(root / "missing-nodefile")} if mode == "unavailable" else {"PBS_NODEFILE": ("other-node" if mode == "mismatch" else socket.gethostname()) + "\\n"})},
        )
        paths = {name: root / name for name in ("input.json", "allocation.json", "runs.json", "worker.yaml", "store")}
        paths["input.json"].write_bytes(group.encode())
        paths["allocation.json"].write_bytes(allocation.encode())
        paths["runs.json"].write_bytes(ConfiguredRunTable(group.identity, (ConfiguredRun("task-a", str(config)),)).encode())
        child_env = {
            "PYTHONPATH": str(stub.parent),
            "L2_STUB_ORIGIN": str(stub.resolve()),
            "L2_RESOLVED_CONFIG_IDENTITY": identity,
            "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
            # The worker samples the child's lifetime; keep it alive for
            # several sampling cadences after its sidecar exists.
            "L1B_STUB_SECONDS": "1.0",
            "TPEN_PLACEMENT_GPU_SAMPLER_CADENCE": "0.05",
        }
        if mode == "refused":
            child_env["TPEN_PLACEMENT_REFUSAL"] = json.dumps({"code": "child-placement-refused", "reason": "consumer refusal control"})
            child_env["L2_REFUSED"] = "1"
        worker = {
            "worker": {"_target_": "tpen_executor.programs.WorkerProgram", "_recursive_": False,
                "environment": {"_target_": "tpen_executor.environments.PreparedPythonEnvironment", "_recursive_": False,
                    "python": sys.executable, "env": child_env},
                "process": {"_target_": "tpen_executor.workers.SubprocessWorker", "_recursive_": False},
                "runner": {"_target_": "tpen_executor.runners.TPENCommand", "_recursive_": False, "config": str(config)},
                "evidence": {"_target_": "tpen_executor.runners.TPENEvidence", "_recursive_": False}}}
        # This is intentionally a tiny YAML emitter for scalar/map values used by
        # the executor CLI; it keeps the producer entirely inside the executor env.
        lines = ["worker:", "  _target_: tpen_executor.programs.WorkerProgram", "  _recursive_: false", "  environment:", "    _target_: tpen_executor.environments.PreparedPythonEnvironment", "    _recursive_: false", "    python: " + json.dumps(sys.executable), "    env:"]
        for key, value in child_env.items(): lines.append("      " + key + ": " + json.dumps(value))
        lines += ["  process:", "    _target_: tpen_executor.workers.SubprocessWorker", "    _recursive_: false", "  runner:", "    _target_: tpen_executor.runners.TPENCommand", "    _recursive_: false", "    config: " + json.dumps(str(config)), "  evidence:", "    _target_: tpen_executor.runners.TPENEvidence", "    _recursive_: false"]
        paths["worker.yaml"].write_text("\\n".join(lines) + "\\n", encoding="utf-8")
        command = [sys.executable, "-m", "tpen_executor", "dispatch", "--input", str(paths["input.json"]), "--store", str(paths["store"]), "--allocation", str(paths["allocation.json"]), "--worker-config", str(paths["worker.yaml"]), "--runs", str(paths["runs.json"])]
        result = subprocess.run(command, cwd=root, env=dict(os.environ, PYTHONPATH=os.environ.get("PYTHONPATH", "")), capture_output=True, text=True)
        expected_rc = 3 if mode in {"unavailable", "refused"} else 0
        if result.returncode != expected_rc:
            print(result.stdout, file=sys.stderr); print(result.stderr, file=sys.stderr); raise SystemExit(result.returncode)
        print("MANIFEST=" + str(paths["store"] / "run-manifest.json"))
        """
    )
    return "\n".join(line[8:] if line.startswith("        ") else line for line in source.splitlines()) + "\n"


def _run_real_producer(tmp_path: Path, mode: str) -> Path:
    executor_python = _executor_python()
    fake_bin = tmp_path / mode / "bin"
    fake_bin.mkdir(parents=True)
    fake = fake_bin / "nvidia-smi"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "def child_pid():\n"
        "    # The worker runs this fake; the science child PID is in its sidecar.\n"
        "    try:\n"
        "        pid = json.loads(Path(os.environ['TPEN_PLACEMENT_OUTPUT']).read_text())['pid']\n"
        "        os.kill(pid, 0)\n"
        "        return pid\n"
        "    except Exception:\n"
        "        return None\n"
        "marker = os.environ.get('L2_GPU_PARTIAL_MARKER')\n"
        "if marker:\n"
        "    path = Path(marker)\n"
        "    count = int(path.read_text()) + 1 if path.exists() else 1\n"
        "    path.write_text(str(count))\n"
        "    if count > 3: raise SystemExit(1)\n"
        "if '--version' in sys.argv: print('NVIDIA-SMI consumer-test')\n"
        "elif '--query-gpu=uuid,pci.bus_id' in sys.argv:\n"
        "    print('GPU-12345678-1234-1234-1234-123456789ABC, 0000:00:01.0')\n"
        "    if os.environ.get('L2_GPU_MISMATCH') == '1': print('GPU-12345678-1234-1234-1234-123456789ABD, 0000:00:02.0')\n"
        "elif '--query-compute-apps=pid,gpu_uuid' in sys.argv and child_pid() is not None:\n"
        "    print(f\"{child_pid()},GPU-12345678-1234-1234-1234-123456789ABD\" if os.environ.get('L2_GPU_MISMATCH') == '1' else f\"{child_pid()},GPU-12345678-1234-1234-1234-123456789ABC\")\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    root = tmp_path / mode
    env = dict(os.environ, PYTHONPATH=str(ROOT / "executor" / "src"))
    if mode in {"unavailable", "gpu-telemetry-failed"}:
        fake.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        fake.chmod(0o755)
    elif mode in L1B_FAKE_ROWS:
        # Deterministic shell fake run by the worker: the science child PID is
        # read from its placement sidecar and reported only while it is alive.
        fake.write_text(
            "#!/bin/sh\n"
            "pid=$(sed -n 's/.*\"pid\": \\([0-9][0-9]*\\).*/\\1/p' \"$TPEN_PLACEMENT_OUTPUT\" 2>/dev/null)\n"
            "if [ -z \"$pid\" ] || ! kill -0 \"$pid\" 2>/dev/null; then pid=''; fi\n"
            "case \"$1\" in\n"
            "  --version) echo 'NVIDIA-SMI consumer-test' ;;\n"
            f"  --query-gpu=*) printf '%s, 0000:00:01.0\\n%s, 0000:00:02.0\\n' {GPU_A} {GPU_B} ;;\n"
            f"  *) {L1B_FAKE_ROWS[mode]} ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
    # Identity is computed here, in the root core environment, because the
    # stub ``tpen`` package shadows core inside the executor-side child.
    result = subprocess.run(
        [executor_python, "-c", _producer_code(), str(root), mode, str(fake_bin), CONSUMER_CONFIG, json.dumps(_consumer_identity())],
        cwd=ROOT / "executor", env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    marker = next(line for line in result.stdout.splitlines() if line.startswith("MANIFEST="))
    return Path(marker.removeprefix("MANIFEST="))


GPU_A = "GPU-12345678-1234-1234-1234-123456789ABC"
GPU_B = "GPU-12345678-1234-1234-1234-123456789ABD"
# Process-row command of each L1b fake mode; ``$pid`` is the live science child.
# Every command exits 0 when the child has no row yet.
L1B_FAKE_ROWS = {
    "gpu-matched": f'if [ -n "$pid" ]; then echo "$pid, {GPU_A}"; fi',
    "gpu-out-of-entitlement": f'if [ -n "$pid" ]; then echo "$pid, {GPU_B}"; fi',
    "gpu-no-row": f'echo "999999, {GPU_A}"',
    "gpu-unresolved": f'if [ -n "$pid" ]; then echo "$pid, {GPU_A}"; fi',
    "cpu-not-sampled": f'if [ -n "$pid" ]; then echo "$pid, {GPU_A}"; fi',
}


def _reader_inspector(manifest: dict[str, object]):
    checkpoint = manifest["segments"][0]["checkpoint_dir"]
    identity = _consumer_identity()
    return lambda _: {"verdict": "completed", "checkpoint": {"checkpoint_dir": checkpoint, "schema_version": 1, "next_iteration": 1, "completed_updates": 1}, "resolved_config_identity": identity}


def test_l2_real_executor_producer_to_p8_reader_contract(tmp_path: Path) -> None:
    """M7-M8: real dispatch output crosses P8 for host-map and refusal classes.

    With L1b the declared-GPU ``matched`` child is sampled, so it is a v3
    qualified receipt; refusals keep the unchanged v2 refusal receipt.
    """

    matched_path = _run_real_producer(tmp_path, "matched")
    matched_manifest = json.loads(matched_path.read_text(encoding="utf-8"))
    matched = read(matched_path, _reader_inspector(matched_manifest))
    assert matched.chains[0].segments[0].resolved_config_identity == _consumer_identity()
    assert matched.chains[0].segments[0].placement_receipt["schema"] == "tpen-executor/science-placement-receipt-v3"
    assert matched.chains[0].segments[0].placement_receipt["verdict"]["qualification"] == "qualified"
    assert matched.chains[0].segments[0].placement_receipt["verdict"]["reason_class"] == "qualified"
    assert not matched.placement_unqualified

    for mode, reason_class in (("mismatch", "mismatch"), ("unavailable", "refused"), ("refused", "refused")):
        path = _run_real_producer(tmp_path, mode)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        result = read(path, _reader_inspector(manifest) if manifest["segments"] else None)
        attempts = result.chains[0].segments if result.chains else result.preserved_attempts
        placement = attempts[0].placement_receipt
        assert placement["verdict"]["reason_class"] == reason_class
        if mode in {"unavailable", "refused"}:
            assert result.chains == ()
            assert result.preserved_attempts
            assert result.preserved_attempts[0].outcome == "launch_failed"
            assert result.placement_unqualified
            assert "GPU PLACEMENT UNQUALIFIED" in result.placement_verdict["message"]

    foreign = copy.deepcopy(matched_manifest)
    foreign["attempts"][0]["placement_receipt"]["attempt_id"] = "foreign-attempt"
    foreign_path = matched_path.with_name("foreign.json")
    foreign_path.write_text(json.dumps(foreign), encoding="utf-8")
    with pytest.raises(ManifestRefusal, match="attempt disagrees"):
        read(foreign_path, lambda _: pytest.fail("cross-attempt receipt must refuse before inspection"))

    v1 = ROOT / "experiments" / "toolkit" / "fixtures" / "run-manifest-v1.json"
    backward = read(v1, lambda _: pytest.fail("v1 backward read must not inspect an empty chain"))
    assert backward.chains == ()


L1B_OUTCOMES = {
    # mode: (reason_class, gpu.reason, science_acceptance, manifest message or None)
    "gpu-matched": ("qualified", "matched", "accepted", None),
    "gpu-out-of-entitlement": ("mismatch", "out-of-entitlement", "not-accepted", PLACEMENT_UNQUALIFIED_MESSAGE),
    "gpu-telemetry-failed": ("telemetry-failed", "sampler-failed", "not-accepted", PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
    "gpu-no-row": ("telemetry-failed", "no-exact-pid-row", "not-accepted", PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
    "gpu-unresolved": ("telemetry-failed", "entitlement-unresolved", "not-accepted", PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
    "cpu-not-sampled": ("not-sampled", "cpu-only", "accepted", None),
}


def test_l1b_real_producer_outcomes_cross_p8_reader(tmp_path: Path) -> None:
    """F7: real CLI dispatch/store output for every v3 outcome crosses P8.

    Every L1b outcome is post-launch evidence: the producer asserts dispatch
    rc 0, and P8 must see the accepted segment, the typed verdict, the
    preserved samples, and the exact manifest message.
    """

    for mode, (reason_class, code, acceptance, message) in L1B_OUTCOMES.items():
        path = _run_real_producer(tmp_path, mode)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        result = read(path, _reader_inspector(manifest))
        assert result.chains, mode
        receipt = result.chains[0].segments[0].placement_receipt
        sampler = receipt["gpu_sampler"]
        assert receipt["schema"] == "tpen-executor/science-placement-receipt-v3", mode
        assert receipt["verdict"]["reason_class"] == reason_class, (mode, receipt["verdict"], sampler["failure"])
        assert receipt["gpu"]["reason"] == code, mode
        assert receipt["verdict"]["science_acceptance"] == acceptance, mode
        if mode == "cpu-not-sampled":
            assert sampler["attempted"] is False and sampler["count"] == 0, mode
        elif mode == "gpu-telemetry-failed":
            # Every command fails: the second consecutive gap exceeds the
            # recorded budget (manager ruling Q1).
            assert sampler["attempted"] is True and sampler["failure"]["code"] == "gap-budget-exceeded", mode
            assert sampler["gaps"]["count"] == 2 and sampler["gaps"]["max_consecutive"] == 2, mode
        else:
            assert sampler["reason"] == "observed" and sampler["quiescent"] is True, mode
            # Decoded P8 records are frozen: JSON arrays read back as tuples.
            assert sampler["survivors"] == () and sampler["count"] >= 1, mode
            # The worker resolved the fake through the prepared PATH.
            assert sampler["executable"].endswith("/bin/nvidia-smi") and sampler["version"], mode
            # The worker sampled the exact science child for its lifetime.
            assert sampler["target_pid"] == receipt["child"]["pid"], mode
            assert sampler["aggregate"]["samples_taken"] >= sampler["count"], mode
            child_rows = {row["pid"] for sample in sampler["samples"] for row in sample["pid_rows"]}
            assert child_rows == ({999999} if mode == "gpu-no-row" else {receipt["child"]["pid"]}), mode
        if mode == "gpu-out-of-entitlement":
            assert receipt["gpu"]["active_uuids"] == (GPU_B,), mode
        if message is None:
            assert result.placement_verdict is None and not result.placement_unqualified, mode
        else:
            assert result.placement_unqualified, mode
            assert result.placement_verdict["message"] == message, mode


def test_l1b_executor_and_p8_refuse_identical_tampers(tmp_path: Path) -> None:
    """F5: the executor decoder and P8 agree on every sampler contradiction.

    The same tampered manifest bytes are decoded by the real executor
    ``RunManifest`` (in the executor interpreter) and by P8 here.
    """

    executor_python = _executor_python()
    from experiments.toolkit.test_run_results_l1b import P8_CONTRADICTIONS, P8_OTHER_CONTRADICTIONS

    fixtures = json.loads((ROOT / "experiments" / "toolkit" / "fixtures" / "run-manifests-v3-sampler.json").read_text(encoding="utf-8"))
    corpus = {name: ("matched", mutate) for name, mutate in P8_CONTRADICTIONS.items()}
    corpus.update(P8_OTHER_CONTRADICTIONS)
    cases = {f"baseline-{name}": fixtures[name] for name in sorted({baseline for baseline, _ in corpus.values()})}
    for name, (baseline, mutate) in corpus.items():
        tampered = copy.deepcopy(fixtures[baseline])
        mutate(tampered["attempts"][0]["placement_receipt"])
        cases[name] = tampered
    paths = {}
    for name, manifest in cases.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(manifest), encoding="utf-8")
    probe = (
        "import json, sys\n"
        "from tpen_executor.results import RunManifest\n"
        "verdicts = {}\n"
        "for path in sys.argv[1:]:\n"
        "    try:\n"
        "        RunManifest.decode(open(path, encoding='utf-8').read()); verdicts[path] = 'accept'\n"
        "    except (TypeError, ValueError):\n"
        "        verdicts[path] = 'reject'\n"
        "print(json.dumps(verdicts))\n"
    )
    result = subprocess.run(
        [executor_python, "-c", probe, *map(str, paths.values())], cwd=ROOT / "executor",
        env=dict(os.environ, PYTHONPATH=str(ROOT / "executor" / "src")), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    executor_verdicts = json.loads(result.stdout)
    for name, path in paths.items():
        try:
            read(path, lambda _: pytest.fail("attempt-only manifest must not inspect core"))
            p8 = "accept"
        except ManifestRefusal:
            p8 = "reject"
        expected = "accept" if name.startswith("baseline-") else "reject"
        assert executor_verdicts[str(path)] == p8 == expected, name
