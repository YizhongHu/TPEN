"""PR 641 T2 round-1 proposals, based on 31cb4d5f over baa1bd67.

Authoring only: NOT executed by the reviewer. Run on Cannon CPU with the
approved interpreter harness. No xfails: the report names predicted REDs
separately from passing controls. Every test documents its target mutant.
Fixtures below exercise study APIs; the last test uses only the public
run_from_config subprocess boundary for real checkpoint/receipt evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRAIN_SHA = "a" * 40
EVAL_SHA = "b" * 40


def _module(name):
    key = f"h_v2_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # tests/unit/experiments/test_study_module_identity.py: publish under
        # spec.name itself (the unique key), never a bare Name subscript --
        # the module-identity rule's structural check matches only the
        # attribute-access form.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


job = _module("run_job")
ev = job.eval_job
protocol = ev.protocol
launch = _module("launch")


@pytest.fixture(autouse=True)
def _shared_study_modules(monkeypatch):
    # Other test files reload the h_v2_* keys during collection. Exercise
    # one consistent module graph without depending on collection order.
    monkeypatch.setattr(job, "protocol", protocol)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _checkpoint(root, update):
    path = root / "checkpoints" / f"step_{update + 1:06d}"
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(f"synthetic model {update}".encode())
    (path / "COMPLETE").write_text("complete\n")
    _json(path / "manifest.json", {"schema_version": 2, "completed_updates": update,
                                  "next_iteration": update + 1, "files": {"model": "model.pt"}})
    return {"update": update, "path": str(path),
            "files": {name: _sha(path / name) for name in ("model.pt", "manifest.json", "COMPLETE")}}


def _train(root, payload="capacity-pilot", updates=range(0, 21000, 1000)):
    config, identity = protocol.pilot_training_config(payload)
    materialized = root / "train.yaml"
    materialized.parent.mkdir(parents=True, exist_ok=True)
    materialized.write_text(yaml.safe_dump(config))
    identity["materialized_config_sha256"] = _sha(materialized)
    receipt = {"schema": job.attached.H_V2_ACCEPTED_RECEIPT_SCHEMA, "payload": payload,
               "science_config_identity": identity,
               "segments": [{"source_commit": TRAIN_SHA, "resolved_config_sha256": _sha(materialized),
                             "artifacts": {"resolved_config": {"path": str(materialized),
                                                                "sha256": _sha(materialized)}}}],
               "checkpoint_inventory": [_checkpoint(root, u) for u in updates]}
    path = root / "attached" / "receipts" / f"h-v2-train-{payload}.h-v2.json"
    _json(path, receipt)
    return receipt, path


def _output(config):
    """Fake compute only; create real producer-shaped output bytes and hashes."""
    run_dir = Path(config["run"]["root"]) / config["run"]["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "metrics.jsonl").write_text('{"event":"completed"}\n')
    _json(run_dir / "metadata.json", {"git_commit": EVAL_SHA})
    _json(run_dir / "status.json", {"status": "completed"})
    for task in config["evaluator"]["tasks"]:
        directory = run_dir / task["output_dir"].removeprefix("${run.dir}/")
        directory.mkdir(parents=True, exist_ok=True)
        csv = directory / "sampled_eval_table.csv"
        csv.write_text("local_energy\n-1\n0\n1\n")
        _json(directory / "sampled_eval_table.metadata.json", {"csv_sha256": _sha(csv)})
        if task["generator"]["_target_"] == protocol.READOUT_TRAJECTORY_GENERATOR_TARGET:
            raw = directory / "raw_readout.csv"
            raw.write_text("sample_index,raw_channel_sum,floor_active\n0,1,False\n")
            _json(directory / "raw_readout.json", {"csv_sha256": _sha(raw)})
    return run_dir


def _fake_compute(monkeypatch):
    def run(argv, **kwargs):
        config = yaml.safe_load(Path(argv[-1]).read_text())
        _output(config)
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(ev.subprocess, "run", run)
    monkeypatch.setattr(ev, "probe_readout_trajectory_generator", lambda *a: True)


def test_k1_extra_present_file_is_never_opened(tmp_path, monkeypatch):
    """K1 control: kills reading an uninventoried sibling, with NO missing file."""
    entry = _checkpoint(tmp_path, 1000)
    extra = Path(entry["path"]) / "extra.bin"
    extra.write_bytes(b"not inventoried")
    opened = []
    original_open = Path.open

    def spy(path, *args, **kwargs):
        opened.append(path.resolve())
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    ev.verify_checkpoint_files(entry)
    assert extra.resolve() not in opened
    assert (Path(entry["path"]) / "model.pt").resolve() in opened


@pytest.mark.parametrize("omitted", ["model.pt", "manifest.json", "COMPLETE", "all"])
def test_checkpoint_load_closure_must_be_inventoried(tmp_path, omitted):
    """F2: kill allowing a present load-required file absent from the inventory."""
    entry = _checkpoint(tmp_path, 1000)
    ev.resolve_checkpoint({"checkpoint_inventory": [entry]}, update=1000)
    if omitted == "all":
        entry["files"] = {}
    else:
        del entry["files"][omitted]
    with pytest.raises(ev.EvalJobRefusal):
        ev.resolve_checkpoint({"checkpoint_inventory": [entry]}, update=1000)


@pytest.mark.parametrize("field,value", [
    ("seed_namespace", "pipeline_gpu_smoke_g"),
    ("contract_inputs_sha256", None),
    ("payload_overlay_sha256", None),
    ("materialized_config_sha256", None),
])
def test_eval_refuses_wrong_train_identity(tmp_path, field, value):
    """F3/D4 (review r2 delta): kill accepting correct lineage/commit but a
    wrong seed namespace, or a DROPPED identity field. D4 stopped re-hashing
    this checkout's own current overlay/contract-inputs files against the
    receipt's recorded hashes (a real EVAL-only submission against an
    earlier-commit TRAIN receipt legitimately disagrees with "this
    checkout"'s current frozen inputs) -- these three fields are instead
    required PRESENT; a dropped field is refused, never a silently-missing
    identity block.
    """
    receipt, _ = _train(tmp_path, updates=(1000,))
    ev.verify_train_receipt_identity(receipt, expected_commit=TRAIN_SHA,
                                     expected_lineage="P", expected_payload="capacity-pilot")
    receipt["science_config_identity"][field] = value
    with pytest.raises(ev.EvalJobRefusal):
        ev.verify_train_receipt_identity(receipt, expected_commit=TRAIN_SHA,
                                         expected_lineage="P", expected_payload="capacity-pilot")


def test_k2_completed_generation_survives_next_failure(tmp_path, monkeypatch):
    """K2 control: kill end-only publication; real job/receipt reader, fake compute."""
    _train(tmp_path, updates=(0, 1000))
    receipt_path = job._eval_receipt_path(tmp_path, "h-v2-eval-capacity-pilot-stage1")
    observations = []
    original = ev.run_generation_direct
    _fake_compute(monkeypatch)

    def run(**kwargs):
        observations.append(ev.read_eval_receipt(receipt_path) if receipt_path.exists() else None)
        if len(observations) == 2:
            raise ev.EvalJobRefusal("injected-failure", "second generation failed")
        return original(**kwargs)

    monkeypatch.setattr(ev, "run_generation_direct", run)
    monkeypatch.setattr(job, "check_identity", lambda *a, **k: None)
    for name, value in {"RUN": str(tmp_path), "SHA": EVAL_SHA, "SCIENCE_PYTHON": sys.executable}.items():
        monkeypatch.setenv(name, value)
    rc = job.main(["eval-fanout", "capacity-pilot", "--pilot-stage", "1",
                   "--expected-train-commit", TRAIN_SHA])
    assert rc != 0
    assert len(observations) == 2 and observations[0] is None
    assert observations[1] is not None, "first completion was not durable before the second started"
    receipt = ev.read_eval_receipt(receipt_path)
    assert receipt == observations[1]
    assert [(g["update"], g["stream"]) for g in receipt["generations"]] == [(0, 0)]
    for artifact in receipt["generations"][0]["artifacts"].values():
        assert _sha(artifact["path"]) == artifact["sha256"]


def _b1(root, *, train_receipt_sha256="c" * 64):
    generations = []
    for ordinal in range(21):
        width = 1 if ordinal in (9, 10, 11) else 10
        path = root / f"b1-{ordinal}.csv"
        path.write_text(f"local_energy\n{-width}\n0\n{width}\n")
        generations.append({"ordinal": ordinal, "update": ordinal * 1000, "stream": 0,
                            "seed": 996000 + 10 * ordinal,
                            "artifacts": {"trajectory_csv": {"path": str(path), "sha256": _sha(path)}}})
    path = job._eval_receipt_path(root, "h-v2-eval-capacity-pilot-stage1")
    receipt = ev.write_eval_receipt(path, eval_commit=EVAL_SHA, train_commit=TRAIN_SHA,
                                   train_receipt_path=root / "train.json", train_receipt_sha256=train_receipt_sha256,
                                   payload="capacity-pilot", generations=generations,
                                   readout_generator_enabled=True)
    return path, receipt


def test_sel_uses_the_bound_b1_bytes(tmp_path):
    """F4: kill hash-blind SEL after a CSV is replaced without changing its receipt."""
    _, receipt = _b1(tmp_path)
    assert ev.compute_pilot_selection(receipt)["c1"] == 10  # positive control
    for ordinal in (9, 10, 11):
        Path(receipt["generations"][ordinal]["artifacts"]["trajectory_csv"]["path"]).write_text(
            "local_energy\n-100\n0\n100\n")
    with pytest.raises((ev.EvalJobRefusal, ValueError)):
        ev.compute_pilot_selection(receipt)


def test_sel_requires_all_21_b1_generations(tmp_path):
    """F4: kill permitting an incomplete B1 receipt because 0/1 are outside triples."""
    _, receipt = _b1(tmp_path)
    assert ev.compute_pilot_selection(receipt)["c1"] == 10
    receipt["generations"] = receipt["generations"][2:]
    with pytest.raises((ev.EvalJobRefusal, ValueError)):
        ev.compute_pilot_selection(receipt)


@pytest.mark.parametrize("payload", ["capacity-pilot", "gpu-route-g", "science"])
def test_eval_launch_refuses_outside_run_root_before_submit(tmp_path, monkeypatch, payload):
    """F5: kill eval paths bypassing the run-root amendment; no real sbatch."""
    submitted = []
    monkeypatch.setattr(launch, "_default_runner", lambda argv: (
        submitted.append(argv) or subprocess.CompletedProcess(argv, 0, "123\n", "")))
    args = ["--run-root", str(tmp_path / "outside"), "--run-root-prefix", str(tmp_path / "runs"),
            "--commit", EVAL_SHA, "--payload", payload, "--stage", "eval",
            "--time", "00:30:00", "--expected-train-commit", TRAIN_SHA,
            "--science-python", sys.executable, "--executor-python", sys.executable]
    if payload == "capacity-pilot":
        args += ["--pilot-stage", "1"]
    with pytest.raises(launch.LaunchError):
        launch.main(args)
    assert submitted == []


@pytest.mark.parametrize("stage", ["train", "eval"])
def test_science_single_stage_refuses_more_than_whole_lineage_budget(tmp_path, monkeypatch, stage):
    """F5: kill bypassing C3 through single-stage paths (33h exceeds total 32h)."""
    monkeypatch.setenv("T_WALL", "33:00:00")
    args = ["--run-root", str(tmp_path / "runs" / "h-v2" / "science" / "L1"),
            "--run-root-prefix", str(tmp_path / "runs"), "--commit", EVAL_SHA,
            "--payload", "science", "--stage", stage, "--science-python", sys.executable,
            "--executor-python", sys.executable, "--dry-run"]
    if stage == "eval":
        args += ["--time", "33:00:00", "--expected-train-commit", TRAIN_SHA]
    with pytest.raises(launch.LaunchError, match="(?i)32|budget|wall"):
        launch.main(args)


def test_receipt_hashes_all_real_output_files(tmp_path, monkeypatch):
    """F6: kill dropping status/config/metrics/readout sidecar hashes."""
    _fake_compute(monkeypatch)
    config = protocol.science_evaluation_config(tmp_path / "checkpoint", seed=999004)
    config["run"].update(root=str(tmp_path), run_id="eval", layout="flat")
    result = ev.run_generation_direct(science_python=sys.executable, config=config,
                                      config_path=tmp_path / "request.yaml", expected_source_commit=EVAL_SHA)
    paths = {Path(a["path"]) for a in result["artifacts"].values()}
    # Enumerating TEST outputs is an independent coverage oracle, not checkpoint discovery.
    written = {p for p in (tmp_path / "eval").rglob("*") if p.is_file()}
    assert written <= paths, f"unreceipted outputs: {written - paths}"


def test_stage2_receipt_retains_selection_and_checkpoint_digests(tmp_path, monkeypatch):
    """F6: kill dropping the chosen selection record or input-checkpoint hashes."""
    train, _ = _train(tmp_path)
    _b1(tmp_path)
    job.run_pilot_sel(tmp_path)
    selection_path = tmp_path / "attached" / "receipts" / "h-v2-eval-capacity-pilot-selection.json"
    _fake_compute(monkeypatch)
    job.run_eval_fanout(run_root=tmp_path, expected_commit=EVAL_SHA,
                        science_python=sys.executable, payload="capacity-pilot", lineage="P",
                        expected_train_commit=TRAIN_SHA, pilot_stage="2")
    receipt = ev.read_eval_receipt(job._eval_receipt_path(tmp_path, "h-v2-eval-capacity-pilot-stage2"))
    encoded = json.dumps(receipt)
    assert _sha(selection_path) in encoded, "stage 2 forgot the selection it executed"
    for generation in receipt["generations"]:
        entry = next(e for e in train["checkpoint_inventory"] if e["update"] == generation["update"])
        assert all(digest in encoded for digest in entry["files"].values())


def test_final_basin_earliest_tie_and_neighbor_controls():
    """F7 control: kill c*=c1 or latest-tie selection, independent numerical oracle."""
    widths = {o: 10.0 for o in range(8, 13)}
    widths.update({8: 1.0, 9: 1.0, 10: 1.0})
    assert protocol.select_final_basin_ordinal(10, widths) == 9
    assert protocol.select_final_basin_ordinal(10, {o: 1.0 for o in range(8, 13)}) == 9


def test_w_is_nearest_rank_not_interpolated():
    """Passing control: kill linear interpolation missed by the old symmetric sample."""
    assert protocol.w_statistic([0.0] * 999 + [1000.0]) == 0.0


@pytest.mark.parametrize("c1", [3, 10, 18, 19])
def test_full_b2_run_set_has_independent_boundary_oracle(c1):
    """Passing control: kill drifting the window used by both old test and producer."""
    windows = {3: {1, 2, 3, 4, 5}, 10: {8, 9, 10, 11, 12},
               18: {16, 17, 18, 19, 20}, 19: {16, 17, 18, 19, 20}}
    expected = {(o, 1) for o in windows[c1]} | {(c1, 2), (c1, 3)}
    expected |= {(o, s) for o in (0, 20) for s in (1, 2, 3)}
    actual = ev.pilot_fanout("2", c1=c1)
    assert {(g["ordinal"], g["stream"]) for g in actual} == expected
    assert len(actual) == len(expected)
    assert all(g["seed"] == 996000 + 10 * g["ordinal"] + g["stream"] for g in actual)


def test_final_sel_is_a_separate_cpu_command_after_b2(tmp_path, monkeypatch):
    """F7 (implemented): the final-basin selector has an explicit CPU caller.

    Per the orchestrator's fix (run_job.py final-sel capacity-pilot /
    job.run_pilot_final_sel), B2 itself never computes c* -- it is a
    separate, explicit completion step run after B2, reading the verified
    B1/B2 eval receipts and the SEL selection record. Replaces the review's
    original assertion (B2 calls the selector inline), which predated that
    design choice and is noted in the implementation receipt.
    """
    _, train_path = _train(tmp_path)
    # D2 (review r2 delta): B1 and B2's train_receipt_sha256 must agree (the
    # real producer binds both to the SAME TRAIN receipt's actual bytes).
    _b1(tmp_path, train_receipt_sha256=_sha(train_path))
    job.run_pilot_sel(tmp_path)
    calls = []
    original = protocol.select_final_basin_ordinal

    def select(c1, widths):
        calls.append((c1, dict(widths)))
        return original(c1, widths)

    monkeypatch.setattr(protocol, "select_final_basin_ordinal", select)
    _fake_compute(monkeypatch)
    job.run_eval_fanout(run_root=tmp_path, expected_commit=EVAL_SHA,
                        science_python=sys.executable, payload="capacity-pilot", lineage="P",
                        expected_train_commit=TRAIN_SHA, pilot_stage="2")
    assert not calls, "B2 must not compute c* itself -- that is final-sel's own job"

    job.run_pilot_final_sel(tmp_path)
    assert calls, "run_pilot_final_sel never called the final-basin selector"
    assert calls[0][0] == 10
    final_path = tmp_path / "attached" / "receipts" / "h-v2-eval-capacity-pilot-final-selection.json"
    assert final_path.is_file()
    record = json.loads(final_path.read_text(encoding="utf-8"))
    assert record["c1"] == 10
    assert "c_star" in record


@pytest.mark.parametrize("rc,stdout", [(1, ""), (0, "unparseable output\n")])
def test_readout_probe_errors_are_not_optional_absence(monkeypatch, rc, stdout):
    """F8: kill silently converting a crashed/malformed probe into disabled readout."""
    monkeypatch.setattr(ev.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(
        argv, rc, stdout, "injected probe failure"))
    with pytest.raises(ev.EvalJobRefusal):
        ev.probe_readout_trajectory_generator(sys.executable)


def test_science_train_leg_is_byte_identical_dry_run_stays_six_lines(tmp_path, monkeypatch, capsys):
    """F9 (orchestrator decision, partial accept, comment 6774677893): E6's
    'byte-identical' covers the sbatch wrapper and sidecar handling, not the
    eval leg's arguments -- replacing the eval refusal is this item's
    purpose. So the frozen contract is: (a) the TRAIN leg's own sbatch
    command never changes shape when built through the eval-fanout-aware
    call site, and (b) the whole science dry run stays exactly 6 lines.
    Replaces the review's original full-command comparison, which would
    have frozen the eval leg's arguments too.
    """
    common = dict(science_python=sys.executable, executor_python=sys.executable,
                  run_root=tmp_path, checkout_root=tmp_path / "checkout", time_value="08:00:00",
                  dependency=None, commit=EVAL_SHA)
    train_via_direct_call = launch.build_sbatch_command("train", "L1", payload="science", **common)
    train_via_submit_lineage = []
    launch.submit_lineage(
        "L1", payload="science", science_python=sys.executable, executor_python=sys.executable,
        run_root=tmp_path, checkout_root=tmp_path / "checkout", t_wall="08:00:00", e_wall="24:00:00",
        commit=EVAL_SHA,
        runner=lambda command: (train_via_submit_lineage.append(list(command))
                                 or __import__("subprocess").CompletedProcess(command, 0, "1;c\n", "")),
    )
    assert train_via_submit_lineage[0] == train_via_direct_call

    monkeypatch.setenv("T_WALL", "08:00:00")
    monkeypatch.setenv("E_WALL", "24:00:00")
    # D6 (review r2 delta): the run-root guard is now required on every path.
    code = launch.main(["--run-root", str(tmp_path / "runs" / "h-v2" / "science"),
                        "--run-root-prefix", str(tmp_path / "runs"),
                        "--commit", EVAL_SHA, "--payload", "science",
                        "--science-python", sys.executable, "--executor-python", sys.executable, "--dry-run"])
    assert code == 0
    assert capsys.readouterr().out.count("sbatch --parsable") == 6


@pytest.mark.parametrize("c1,expected_core_count", [(18, 10), (19, 10)])
def test_b2_core_late_basin_obeys_amendment_r5(c1, expected_core_count):
    """F10 (orchestrator decision r5, comment 6774677893): 'stage 2-core: the
    same as stage 2 without ordinal 20' means without the endpoint ANCHOR
    runs at ordinal 20 (streams 1/2/3 added only as anchors) -- a run at
    ordinal 20 that is already in the c1 window (stream 1 at lo..lo+4) or
    at c1 itself is KEPT, so c*=19 stays reachable. For c1 in {18, 19},
    lo=16 so the window already spans 16..20: core drops only (20, 2) and
    (20, 3) (the anchor-only pairs), never (20, 1).
    """
    full = {(g["ordinal"], g["stream"]) for g in ev.pilot_fanout("2", c1=c1)}
    core = {(g["ordinal"], g["stream"]) for g in ev.pilot_fanout("2-core", c1=c1)}
    assert (20, 1) in core, "ordinal 20 stream 1 is inside the c1 window and must stay in core"
    assert full - core == {(20, 2), (20, 3)}
    assert len(core) == expected_core_count


def test_science_eval_seed_formula_matches_ci(tmp_path):
    """F12 (orchestrator decision, accepted from scope notes): science eval
    seeds must be read-checked against CI's own frozen formula field
    (contract_inputs.json measurement.seed_roster.evaluation_stream_seed_formula),
    never a hard-coded literal with no CI tie, the way pilot_eval_seed/
    gpu_route_g_eval_seed already are against their own roster fields.
    """
    payload = json.loads(protocol.CONTRACT_INPUTS_PATH.read_text(encoding="utf-8"))
    formula = payload["measurement"]["seed_roster"]["evaluation_stream_seed_formula"]
    base, lineage_coeff, ordinal_coeff = (
        int(n)
        for n in __import__("re").findall(
            r"(\d+)\s*\+\s*(\d+)\*lineage_index\s*\+\s*(\d+)\*evaluated_checkpoint_ordinal\s*\+\s*stream_index",
            formula,
        )[0]
    )
    for lineage_index in range(3):
        for ordinal in range(27):
            for stream in range(4):
                expected = base + lineage_coeff * lineage_index + ordinal_coeff * ordinal + stream
                assert protocol.science_eval_seed(lineage_index, ordinal, stream) == expected


@pytest.mark.parametrize("explicit_eps", [False, True], ids=["p-base", "e-current-control"])
def test_real_eval_restores_p_and_e_checkpoints(tmp_path, monkeypatch, explicit_eps):
    """F1 + consumer: kill the P->E model-hash drift; real CPU core subprocesses.

    Preserve P's model mapping (no explicit default eps) and reduce only CPU
    sampler/update counts. An old TRAIN source checkout is unnecessary for
    this defect: model_config hashes the mapping, not constructor defaults.
    """
    science_python = os.environ.get("H_V2_SCIENCE_PYTHON")
    executor_python = os.environ.get("H_V2_EXECUTOR_PYTHON")
    assert science_python and executor_python, "approved Cannon CPU harness interpreters required"
    train, identity = protocol.pilot_training_config("gpu-route-g")
    if not explicit_eps:
        train["model"]["readout"].pop("eps", None)
    train["runtime"]["device"] = "cpu"
    train["sampler"].update(n_walkers=4, burn_in=1, n_steps=1)
    train["trainer"]["max_steps"] = 1
    # Mirror the production pilot TRAIN assembly: candidate callbacks are
    # admitted against the exact science interpreter before the final plan is
    # activated.  Building directly from the pre-candidate pilot config would
    # exercise the completeness gate with an intentionally incomplete payload.
    job.attached.probe_admitted_a1_modules(
        train,
        protocol.candidate_modules_to_append(),
        science_python=science_python,
        probe_config_path=tmp_path / "handoff" / "h-v2-train-gpu-route-g.probe.yaml",
        cwd=REPO,
    )
    plan, overrides, train = job.attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="gpu-route-g",
        config=train, science_config_identity=identity, payload_label="gpu-route-g")
    prepared = job.attached.prepare_h_v2_run(plan, overrides, train, output_dir=tmp_path / "handoff")
    allocation, worker = job.write_local_allocation_and_worker(
        tmp_path / "allocation", science_python=science_python,
        placeholder_config=prepared.effective_config, backend_job_id="t2-review-cpu-only")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    train_path = tmp_path / "train.h-v2.json"
    job.attached.activate_h_v2_run(
        prepared, science_python=science_python, executor_python=executor_python,
        allocation=allocation, worker_config=worker, store=tmp_path / "store",
        activation_receipt_path=tmp_path / "activation.json", h_v2_receipt_path=train_path,
        expected_source_commit=commit, dispatcher="local", timeout=180, cwd=REPO)
    receipt, train_sha = ev.load_train_receipt(train_path)
    checkpoint = ev.resolve_checkpoint(receipt, update=1)
    if not explicit_eps:
        resolved_config = yaml.safe_load(
            Path(receipt["segments"][0]["artifacts"]["resolved_config"]["path"]).read_text()
        )
        assert "eps" not in resolved_config["model"]["readout"], (
            "P historical model mapping must omit model.readout.eps"
        )
        model_state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=False)
        assert not any("model.readout.eps" in key for key in model_state), (
            "P checkpoint model state must not contain model.readout.eps"
        )
    evaluation = protocol.science_evaluation_config(checkpoint, seed=999004)
    evaluation["runtime"]["device"] = "cpu"
    evaluation["sampler"].update(n_walkers=4, burn_in=1, n_steps=1)
    evaluation["run"].update(root=str(tmp_path), run_id="evaluation", layout="flat")
    for task in evaluation["evaluator"]["tasks"]:
        task["generator"]["n_draws"] = 2
        task["generator"]["chunk_size"] = 4
        task["summaries"][0]["max_samples"] = 8
    # env -i equivalent for the REAL EVAL entry: no fake core or executor dispatch.
    original_run = subprocess.run
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONPATH": str(REPO),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

    def isolated_run(argv, **kwargs):
        return original_run(argv, **dict(kwargs, env=env))

    monkeypatch.setattr(ev.subprocess, "run", isolated_run)
    result = ev.run_generation_direct(science_python=science_python, config=evaluation,
                                      config_path=tmp_path / "eval.yaml", expected_source_commit=commit,
                                      cwd=REPO, timeout=180)
    generation = ev.generation_record(ordinal=1, update=1, stream=0, seed=999004,
                                       task_id="cpu-review-eval", run_result=result)
    receipt_path = tmp_path / "eval-receipt.json"
    ev.write_eval_receipt(receipt_path, eval_commit=commit, train_commit=commit,
                          train_receipt_path=train_path, train_receipt_sha256=train_sha,
                          payload="gpu-route-g", generations=[generation], readout_generator_enabled=True)
    read_back = ev.read_eval_receipt(receipt_path)
    assert len(read_back["generations"]) == 1
    for artifact in read_back["generations"][0]["artifacts"].values():
        assert _sha(artifact["path"]) == artifact["sha256"]


def test_k3_production_wiring_loads_from_the_verified_snapshot(tmp_path, monkeypatch):
    """K3 (final-verify finding, mutant M1 at run_job.py:386-388): kill dropping
    the ``snapshot_dir`` kwarg from the PRODUCTION ``resolve_checkpoint`` call
    site. Drives the real eval fan-out path (``job.run_eval_fanout``, not the
    unit-level ``resolve_checkpoint`` helper test in test_eval_job.py), spies
    on the checkpoint directory core actually receives, and then corrupts the
    original checkpoint AFTER the run to prove core never re-reads it.
    """
    train, _ = _train(tmp_path, payload="gpu-route-g", updates=(1000,))
    entry = train["checkpoint_inventory"][0]
    original_checkpoint_dir = Path(entry["path"])
    _fake_compute(monkeypatch)

    captured_dirs = []
    original_science_evaluation_config = protocol.science_evaluation_config

    def spy(checkpoint_dir, **kwargs):
        captured_dirs.append(Path(checkpoint_dir))
        return original_science_evaluation_config(checkpoint_dir, **kwargs)

    monkeypatch.setattr(protocol, "science_evaluation_config", spy)

    job.run_eval_fanout(run_root=tmp_path, expected_commit=EVAL_SHA,
                        science_python=sys.executable, payload="gpu-route-g", lineage="G",
                        expected_train_commit=TRAIN_SHA)

    assert len(captured_dirs) == 1
    loaded_dir = captured_dirs[0]
    assert loaded_dir != original_checkpoint_dir, (
        "core was pointed at the original checkpoint directory, not a job-local snapshot"
    )
    assert str(loaded_dir).startswith(str(tmp_path / "attached" / "checkpoint-snapshots")), (
        f"loaded checkpoint dir {loaded_dir} is not under this job's own snapshot root"
    )
    for relative, expected_sha256 in entry["files"].items():
        assert _sha(loaded_dir / relative) == expected_sha256

    # Corrupt the ORIGINAL directory after the run; the snapshot core actually
    # loaded from is a private copy and must be unaffected -- the production
    # call site, not just the unit-level helper, must close the check/use
    # window.
    (original_checkpoint_dir / "model.pt").write_bytes(b"corrupted after verification")
    for relative, expected_sha256 in entry["files"].items():
        assert _sha(loaded_dir / relative) == expected_sha256


@pytest.mark.parametrize("center,expected_interior", [(3, False), (10, True), (19, False)])
def test_k4_interior_minimum_flag_on_sel_and_final_sel_records(tmp_path, center, expected_interior):
    """K4 (operator-rulings-pilot-20261110-1900, item 3): kill a mutant that
    always reports ``interior_minimum=True``. A basin center at the window
    edge (``c1`` of 3 or 19) never had a full neighboring triple on both
    sides and must be flagged ``False`` in both the SEL selection record and
    the final-SEL ``c*`` record; ``c1=10`` is the ordinary interior case and
    must stay ``True``. Stage 2 still runs in every case (the window is
    defined even at the edge).
    """
    narrow = {center - 1, center, center + 1}
    generations = []
    for ordinal in range(21):
        width = 1 if ordinal in narrow else 10
        path = tmp_path / f"b1-{center}-{ordinal}.csv"
        path.write_text(f"local_energy\n{-width}\n0\n{width}\n")
        generations.append({"ordinal": ordinal, "update": ordinal * 1000, "stream": 0,
                            "seed": 996000 + 10 * ordinal,
                            "artifacts": {"trajectory_csv": {"path": str(path), "sha256": _sha(path)}}})
    stage1_path = job._eval_receipt_path(tmp_path, "h-v2-eval-capacity-pilot-stage1")
    ev.write_eval_receipt(stage1_path, eval_commit=EVAL_SHA, train_commit=TRAIN_SHA,
                         train_receipt_path=tmp_path / "train.json", train_receipt_sha256="c" * 64,
                         payload="capacity-pilot", generations=generations, readout_generator_enabled=True)

    job.run_pilot_sel(tmp_path)
    selection_path = tmp_path / "attached" / "receipts" / "h-v2-eval-capacity-pilot-selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["c1"] == center, "the narrow-triple fixture must pick the intended center"
    assert selection["interior_minimum"] is expected_interior

    candidates = sorted({c for c in (center - 1, center, center + 1) if 3 <= c <= 19} | {center})
    needed = sorted({o for c in candidates for o in (c - 1, c, c + 1)})
    stage2_generations = []
    for ordinal in needed:
        path = tmp_path / f"b2-{center}-{ordinal}.csv"
        path.write_text("local_energy\n-1\n0\n1\n")
        stage2_generations.append({"ordinal": ordinal, "update": ordinal * 1000, "stream": 1, "seed": 1,
                                   "artifacts": {"trajectory_csv": {"path": str(path), "sha256": _sha(path)}}})
    # D2 (review r2 delta): B2's own receipt must carry the SEL selection it
    # actually ran against, bound by the selection file's own sha256 -- the
    # same shape ``run_job.run_eval_fanout`` writes.
    stage2_selection = {**selection, "selection_record_sha256": _sha(selection_path)}
    stage2_path = job._eval_receipt_path(tmp_path, "h-v2-eval-capacity-pilot-stage2")
    ev.write_eval_receipt(stage2_path, eval_commit=EVAL_SHA, train_commit=TRAIN_SHA,
                         train_receipt_path=tmp_path / "train.json", train_receipt_sha256="c" * 64,
                         payload="capacity-pilot", generations=stage2_generations, readout_generator_enabled=True,
                         selection=stage2_selection)

    job.run_pilot_final_sel(tmp_path)
    final_path = tmp_path / "attached" / "receipts" / "h-v2-eval-capacity-pilot-final-selection.json"
    final_record = json.loads(final_path.read_text(encoding="utf-8"))
    assert final_record["c1"] == center
    assert final_record["interior_minimum"] is expected_interior
