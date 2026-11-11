#!/usr/bin/env python3
"""H-v2 in-job attached route (design target sections A4/B3, item 9d3aa0a3).

Invoked by the launch script's generated wrapper (``launch.py``, item
67f2eb57, F1/G2): the exact, validated ``--science-python`` interpreter runs
``<checkout>/.../run_job.py train L1`` or ``... run_job.py eval L1``, always
with an absolute path to this file inside the explicit checkout (never the
Slurm-spooled copy of this file alone, which has no siblings to import).
Runs entirely inside its own allocation; the launch script only submits it.

Required environment (G2: set explicitly by the wrapper from the same
literal, validated interpreter paths the launch script was given --
never left to whatever the submitting shell happened to export):

``RUN``
    Durable run root; ``$RUN/checkout`` is the checkout this job runs from.
``SHA``
    Expected checkout commit, asserted against ``git -C "$RUN/checkout"
    rev-parse HEAD`` before any payload.
``SCIENCE_PYTHON``
    Absolute science environment Python (CUDA asserted against it).
``EXECUTOR_PY``
    Absolute executor environment Python (``cd executor && uv sync --locked``,
    separate from the science environment).

NOT YET COMPLETE. Section C2 schedules 27 checkpoints x 4 streams (108
evaluations) per lineage EVAL job, run one after another inside the same
allocation. This script invokes exactly one EVAL generation (stream 0
against the lineage's single recorded TRAIN checkpoint): ``attached.py``'s
``prepare_h_v2_run`` admits only one task per generation, and no
per-checkpoint/per-stream loop has been built or reviewed yet. That fan-out
is a follow-up, not invented here. EVAL also currently refuses
unconditionally before dispatch (``attached.EVAL_ROUTE_UNSUPPORTED_CODE``,
pending an operator decision on item 9d3aa0a3) -- this script reaches that
refusal rather than working around it, per the layer-9 acceptance contract.

This module imports no ``tpen``, no ``hydra``, and no ``tpen_executor``
(``experiments/README.md``); it talks to the executor only through
``attached.py``'s CLI subprocess route.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _h_v2_module(name: str):
    """Load an h-v2 sibling under its own ``h_v2_<name>`` :mod:`sys.modules` key.

    Mirrors ``attached.py``'s own private loader (see that module's
    docstring for why a bare ``import`` is unsafe here).
    """

    key = f"h_v2_{name}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


attached = _h_v2_module("attached")
launch_config = _h_v2_module("launch_config")
protocol = _h_v2_module("protocol")
eval_job = _h_v2_module("eval_job")

#: TO baff945c: TRAIN refuses with this code for any lineage outside
#: ``launch_config.LINEAGES`` (the approved L1-L3 roster), before any
#: dispatch -- never silently submitting ``protocol.training_config()``'s
#: CPU pipeline-check defaults relabeled with an unapproved lineage name.
SCIENCE_CONFIG_UNAVAILABLE_CODE = "h-v2-science-config-unavailable"

#: F3: the label an explicitly-selected non-science dispatch carries in its
#: task id, so its job name/logs/receipts say plainly that it is not a
#: science run.
PIPELINE_CHECK_LABEL = "pipeline-check"


class JobError(RuntimeError):
    """The in-job route cannot proceed (environment, identity, or usage)."""


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise JobError(f"required environment variable {name!r} is unset")
    return value


def check_identity(
    run_root: Path,
    expected_commit: str,
    science_python: str,
    *,
    expected_stratum: str = launch_config.APPROVED_GPU_STRATUM,
) -> None:
    """Section B3's in-job identity block: print, then assert, before any payload."""

    job_id = os.environ.get("SLURM_JOB_ID", "absent")
    partition = os.environ.get("SLURM_JOB_PARTITION", "absent")
    hostname = subprocess.run(["hostname"], check=False, capture_output=True, text=True).stdout.strip()
    print(f"[h-v2] job={job_id} partition={partition} host={hostname or 'absent'}")
    subprocess.run(["nvidia-smi", "-L"], check=False)

    checkout = run_root / "checkout"
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    if head.returncode != 0 or head.stdout.strip() != expected_commit:
        raise JobError(
            f"checkout HEAD {head.stdout.strip()!r} at {checkout} does not match expected commit "
            f"{expected_commit!r}"
        )

    torch_check = subprocess.run(
        [
            science_python,
            "-c",
            "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if torch_check.returncode != 0:
        raise JobError(f"CUDA torch check failed: {torch_check.stderr.strip()}")
    device_name = torch_check.stdout.strip()
    # F6: the delivered card is checked here, not merely requested at
    # submission (section B3: "must print NVIDIA A100-SXM4-80GB; anything
    # else fails the job loudly"). seas_gpu mixes A100-80GB and H200, and a
    # requeue can change which one a job lands on (policy
    # gpu-hardware-strata-and-constraint-rule-2026-08-15). TO 63690220: the
    # pilot/part-(g) payloads pass their own per-payload expected stratum
    # (``gpu-route-g`` serves a ``gpu_test`` MIG slice, a different hardware
    # stratum from the science/pipeline-check route's A100-80GB).
    if device_name != expected_stratum:
        raise JobError(
            f"delivered GPU {device_name!r} does not match the approved stratum "
            f"{expected_stratum!r} (design target section B3/C2); "
            "failing loudly rather than running on an unapproved card"
        )
    print(f"[h-v2] {device_name}")


def write_local_allocation_and_worker(
    alloc_dir: Path, *, science_python: str, placeholder_config: str, backend_job_id: str
) -> tuple[Path, Path]:
    """Write minimal real allocation/worker fixtures for ``--dispatcher local``.

    Mirrors ``test_attached.py``'s ``_real_cli_allocation_and_worker``: one
    GPU, one task, one node needs no multi-slot binding
    (``experiments/toolkit/attached_activation.py``'s
    ``bind_allocation_context`` exists for that case, not this one).
    ``runner.config`` is a placeholder: the executor overrides it per task
    from the configured-runs sidecar ``prepare_h_v2_run`` already wrote.
    """

    alloc_dir.mkdir(parents=True, exist_ok=True)
    allocation_path = alloc_dir / "allocation.json"
    allocation_path.write_text(
        json.dumps(
            {
                "schema": "tpen-executor/configured-allocation-v1",
                "allocation_id": f"h-v2-{backend_job_id}",
                "backend_job_id": backend_job_id,
                "facts": {},
            }
        ),
        encoding="utf-8",
    )
    worker_path = alloc_dir / "worker.yaml"
    worker_path.write_text(
        "worker:\n"
        "  _target_: tpen_executor.programs.WorkerProgram\n"
        "  _recursive_: false\n"
        "  environment:\n"
        "    _target_: tpen_executor.environments.PreparedPythonEnvironment\n"
        "    _recursive_: false\n"
        f"    python: {json.dumps(science_python)}\n"
        "    env: {}\n"
        "  process:\n"
        "    _target_: tpen_executor.workers.SubprocessWorker\n"
        "    _recursive_: false\n"
        "  runner:\n"
        "    _target_: tpen_executor.runners.TPENCommand\n"
        "    _recursive_: false\n"
        f"    config: {json.dumps(placeholder_config)}\n"
        "  evidence:\n"
        "    _target_: tpen_executor.runners.TPENEvidence\n"
        "    _recursive_: false\n",
        encoding="utf-8",
    )
    return allocation_path, worker_path


def _reserve_store_parent(store: Path) -> None:
    """Create *store*'s parent directory, never *store* itself.

    ``executor/src/tpen_executor/composition.py``'s ``_reserve_store`` does a
    bare, non-recursive ``mkdir()`` on the store root and must see an absent
    root to atomically reserve it; this only ensures the ancestry above it
    exists (TO 8d9a31b3: a fresh run root has no ``attached/store`` yet, so
    the bare ``mkdir()`` raised ``ENOENT`` before it ever reached the
    store-exists/store-create distinction).
    """

    store.parent.mkdir(parents=True, exist_ok=True)


def _train_task_id(lineage: str, *, pipeline_check: bool) -> str:
    """Return TRAIN's task id for *lineage* (K2/H1).

    :func:`run` uses this to build TRAIN's own task id, and
    :func:`checkpoint_dir_for_eval` uses it to locate TRAIN's accepted
    receipt from EVAL -- one function, so the two can never drift apart into
    two different names for the same lineage again (TRAIN wrote
    ``h-v2-train-pipeline-check-<lineage>`` under ``pipeline_check`` while
    EVAL read the unlabelled ``h-v2-train-<lineage>``, so EVAL always failed
    with "no TRAIN accepted receipt" before reaching its own documented
    ``h-v2-eval-route-unsupported`` refusal).
    """

    build_lineage = f"{PIPELINE_CHECK_LABEL}-{lineage}" if pipeline_check else lineage
    return f"h-v2-train-{build_lineage}"


def checkpoint_dir_for_eval(run_root: Path, lineage: str, *, pipeline_check: bool = False) -> Path:
    """Read the lineage's TRAIN checkpoint dir from its accepted receipt.

    Parameters
    ----------
    pipeline_check : bool, default=False
        Must match the ``pipeline_check`` TRAIN actually ran under for this
        lineage (K2/H1): the receipt name is namespaced the same way.

    Raises
    ------
    JobError
        If the TRAIN accepted receipt is missing or carries no segments --
        this never invents a checkpoint path.
    """

    task_id = _train_task_id(lineage, pipeline_check=pipeline_check)
    receipt_path = run_root / "attached" / "receipts" / f"{task_id}.h-v2.json"
    if not receipt_path.is_file():
        raise JobError(f"no TRAIN accepted receipt at {receipt_path}; TRAIN must complete before EVAL")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    # TO baff945c review r1 (F1): require science identity whenever THIS CALL
    # asks for a science checkpoint (pipeline_check=False), not merely when
    # the receipt happens to carry a "science" payload string -- a receipt
    # producer bug or tamper that drops/malforms the payload field must not
    # silently bypass the identity check. The receipt's own payload is itself
    # validated here, not trusted blind.
    if not pipeline_check:
        if receipt.get("payload") != "science":
            raise JobError(
                f"TRAIN accepted receipt {receipt_path} does not carry payload='science' "
                f"(got {receipt.get('payload')!r}); refusing to trust it for a science checkpoint"
            )
        missing = attached.SCIENCE_CONFIG_IDENTITY_KEYS - set(receipt.get("science_config_identity") or {})
        if missing:
            raise JobError(
                f"TRAIN accepted receipt {receipt_path} claims payload=science but is missing "
                f"science config identity fields: {sorted(missing)!r}"
            )
    segments = receipt.get("segments") or []
    if not segments:
        raise JobError(f"TRAIN accepted receipt {receipt_path} carries no segments")
    return Path(segments[0]["checkpoint_dir"])


def _pilot_train_task_id(payload: str) -> str:
    """Return a pilot/part-(g) payload's fixed TRAIN task id (TO 63690220)."""

    return f"h-v2-train-{payload}"


#: TO 8a5e926f (item E): every eval-fanout payload's own TRAIN task id and
#: fixed lineage label (science's lineage is caller-supplied, checked
#: against ``launch_config.LINEAGES``).
_EVAL_FANOUT_PAYLOADS = ("science", "capacity-pilot", "gpu-route-g")


def _eval_receipt_path(run_root: Path, task_label: str) -> Path:
    return run_root / "attached" / "receipts" / f"{task_label}.h-v2-eval.json"


def run_eval_fanout(
    *,
    run_root: Path,
    expected_commit: str,
    science_python: str,
    payload: str,
    lineage: str,
    expected_train_commit: str,
    lineage_index: int | None = None,
    pilot_stage: str | None = None,
) -> None:
    """Dispatch one EVAL fan-out job (item E, acceptance-contract 8a5e926f).

    "A study-side job, NOT an executor dispatch": every scheduled generation
    runs directly through ``tpen.run.run_from_config``
    (:func:`eval_job.run_generation_direct`), never ``tpen.run.main`` or
    ``python -m tpen.run``, verified and consolidated into this job's own
    eval receipt -- never the
    TRAIN-only executor route ``attached.activate_h_v2_run`` refuses for
    EVAL (:data:`attached.EVAL_ROUTE_UNSUPPORTED_CODE`).

    Raises
    ------
    JobError
        Code :data:`SCIENCE_CONFIG_UNAVAILABLE_CODE`, for an unapproved
        science lineage; otherwise propagates :class:`eval_job.EvalJobRefusal`
        (checkpoint-hash mismatch, wrong-commit TRAIN receipt, a missing SEL
        selection record, a failed generation, ...) unchanged -- any of
        these is a non-zero exit via :func:`main`.
    """

    if payload not in _EVAL_FANOUT_PAYLOADS:
        raise JobError(f"eval fan-out payload must be one of {_EVAL_FANOUT_PAYLOADS!r}, got {payload!r}")

    selection_record: dict | None = None
    if payload == "science":
        if lineage not in launch_config.LINEAGES:
            raise JobError(
                f"{SCIENCE_CONFIG_UNAVAILABLE_CODE}: eval fan-out refuses lineage {lineage!r} outside "
                f"the approved roster {launch_config.LINEAGES!r}"
            )
        train_task_id = _train_task_id(lineage, pipeline_check=False)
        expected_lineage = lineage
        resolved_lineage_index = launch_config.LINEAGES.index(lineage) if lineage_index is None else lineage_index
        generations = eval_job.science_fanout(lineage_index=resolved_lineage_index)
        task_label = f"h-v2-eval-science-{lineage}"
    elif payload == "capacity-pilot":
        train_task_id = _pilot_train_task_id(payload)
        expected_lineage = launch_config.PILOT_PAYLOAD_LINEAGES[payload]
        if pilot_stage is None:
            raise JobError("capacity-pilot eval fan-out requires --pilot-stage")
        task_label = f"h-v2-eval-{payload}-stage{pilot_stage}"
        if pilot_stage == "1":
            generations = eval_job.pilot_fanout("1")
        else:
            selection_path = run_root / "attached" / "receipts" / "h-v2-eval-capacity-pilot-selection.json"
            if not selection_path.is_file():
                raise JobError(f"no SEL selection record at {selection_path}; run the SEL step first")
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            stage1_receipt_path = _eval_receipt_path(run_root, f"h-v2-eval-{payload}-stage1")
            _, stage1_sha256 = eval_job.load_train_receipt(stage1_receipt_path)
            c1 = eval_job.verify_selection_record(selection, b1_eval_receipt_sha256=stage1_sha256)
            generations = eval_job.pilot_fanout(pilot_stage, c1=c1)
            # F6: stage 2/2-core's receipt retains the SEL record it actually
            # executed, bound by the selection FILE's own sha256 -- never
            # left null for a consumer with no way to recover which
            # selection a given stage-2 run used.
            selection_record = {
                **selection,
                "selection_record_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            }
    else:  # "gpu-route-g"
        train_task_id = _pilot_train_task_id(payload)
        expected_lineage = launch_config.PILOT_PAYLOAD_LINEAGES[payload]
        task_label = f"h-v2-eval-{payload}"
        generations = eval_job.gpu_route_g_fanout()

    train_receipt_path = run_root / "attached" / "receipts" / f"{train_task_id}.h-v2.json"
    train_receipt, train_receipt_sha256 = eval_job.load_train_receipt(train_receipt_path)
    eval_job.verify_train_receipt_identity(
        train_receipt,
        expected_commit=expected_train_commit,
        expected_lineage=expected_lineage,
        expected_payload=payload,
    )

    readout_available = eval_job.probe_readout_trajectory_generator(science_python)
    results_root = run_root / "attached" / "results" / "attached"
    handoff_root = run_root / "attached" / "handoff"
    receipt_path = _eval_receipt_path(run_root, task_label)
    records: list[dict] = []
    checkpoint_snapshot_root = run_root / "attached" / "checkpoint-snapshots" / task_label
    for generation in generations:
        gen_task_id = f"{task_label}-ck{generation['update']:06d}-s{generation['stream']}"
        # F2: core loads from a job-local verified snapshot of the
        # inventoried checkpoint bytes, never the original (shared,
        # externally-writable) checkpoint directory -- closes the
        # verify/load check-use window.
        checkpoint_entry = eval_job.select_checkpoint(train_receipt, update=generation["update"])
        checkpoint_dir = eval_job.resolve_checkpoint(
            train_receipt, update=generation["update"], snapshot_dir=checkpoint_snapshot_root / gen_task_id
        )
        config = protocol.science_evaluation_config(
            checkpoint_dir, seed=generation["seed"], enable_readout_trajectory=readout_available
        )
        config["run"]["root"] = str(results_root)
        config["run"]["run_id"] = gen_task_id
        config["run"]["layout"] = "flat"
        run_result = eval_job.run_generation_direct(
            science_python=science_python,
            config=config,
            config_path=handoff_root / f"{gen_task_id}.yaml",
            expected_source_commit=expected_commit,
            cwd=run_root / "checkout",
        )
        records.append(
            eval_job.generation_record(
                ordinal=generation["ordinal"],
                update=generation["update"],
                stream=generation["stream"],
                seed=generation["seed"],
                task_id=gen_task_id,
                run_result=run_result,
                checkpoint_files=checkpoint_entry.get("files"),
            )
        )
        # E3: written on every generation's completion, so a wall-killed job
        # leaves every already-completed generation's output sha256s
        # verifiable even though the job still exits non-zero overall if it
        # does not finish the whole schedule.
        eval_job.write_eval_receipt(
            receipt_path,
            eval_commit=expected_commit,
            train_commit=expected_train_commit,
            train_receipt_path=train_receipt_path,
            train_receipt_sha256=train_receipt_sha256,
            payload=payload,
            generations=records,
            readout_generator_enabled=readout_available,
            selection=selection_record,
        )
    print(f"[h-v2] {task_label}: {len(records)} generation(s) accepted (readout_generator={readout_available})")


def run_pilot_sel(run_root: Path) -> None:
    """Compute and write the pilot SEL selection record (amendment r3/r4 E1).

    Reads the completed capacity-pilot stage-1 eval receipt, computes the
    stream-0 ``W`` per ordinal, and writes the selection record
    (:func:`eval_job.build_selection_record`) stage 2 verifies against.

    Raises
    ------
    JobError
        If the stage-1 eval receipt is missing or incomplete.
    """

    stage1_path = _eval_receipt_path(run_root, "h-v2-eval-capacity-pilot-stage1")
    if not stage1_path.is_file():
        raise JobError(f"no capacity-pilot stage-1 eval receipt at {stage1_path}; stage 1 must complete before SEL")
    # D5 (review r2 delta): hash the EXACT bytes parsed, not a later re-read
    # of the file (which an atomic replacement mid-computation could race).
    stage1_receipt, stage1_sha256 = eval_job.read_eval_receipt_with_hash(stage1_path)
    selection_core = eval_job.compute_pilot_selection(stage1_receipt)
    record = eval_job.build_selection_record(
        c1=selection_core["c1"],
        triple_means=selection_core["triple_means"],
        b1_eval_receipt_sha256=stage1_sha256,
    )
    selection_path = run_root / "attached" / "receipts" / "h-v2-eval-capacity-pilot-selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"[h-v2] SEL: c1={record['c1']}")


def run_pilot_final_sel(run_root: Path) -> None:
    """Compute and write the pilot final-basin ``c*`` record (F7, review r1).

    In scope per orchestrator decision 2026-11-10: the final-basin selector
    (:func:`protocol.select_final_basin_ordinal`) had no production caller --
    B2 finished without ever combining streams 0/1 or publishing ``c*``. This
    is that completion step: an explicit CPU command run after B2, reading
    the verified B1 (stage-1) and B2 (stage-2 or stage-2-core) eval receipts
    plus the SEL selection record, pooling each candidate ordinal's stream-0
    (B1) and stream-1 (B2) local energies, computing the instrument's ``W``
    per pooled ordinal, and writing the bound final-selection record.

    Raises
    ------
    JobError
        If the SEL selection record, the stage-1 receipt, or a stage-2/
        stage-2-core receipt is missing, pooled stream-0/1 data is missing
        for any candidate ordinal, or B2 is not bound to this same B1/SEL
        (D2, review r2 delta): wrong TRAIN receipt, a selection record
        belonging to a different B1, or a partial stage-2 run set.
    """

    selection_path = run_root / "attached" / "receipts" / "h-v2-eval-capacity-pilot-selection.json"
    if not selection_path.is_file():
        raise JobError(f"no SEL selection record at {selection_path}; run `sel capacity-pilot` first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    stage1_path = _eval_receipt_path(run_root, "h-v2-eval-capacity-pilot-stage1")
    if not stage1_path.is_file():
        raise JobError(f"no capacity-pilot stage-1 eval receipt at {stage1_path}")
    # D5: hash the exact bytes parsed, not a later re-read.
    stage1_receipt, stage1_sha256 = eval_job.read_eval_receipt_with_hash(stage1_path)
    c1 = eval_job.verify_selection_record(selection, b1_eval_receipt_sha256=stage1_sha256)

    stage2_path = _eval_receipt_path(run_root, "h-v2-eval-capacity-pilot-stage2")
    if not stage2_path.is_file():
        stage2_path = _eval_receipt_path(run_root, "h-v2-eval-capacity-pilot-stage2-core")
    if not stage2_path.is_file():
        raise JobError(
            f"no capacity-pilot stage-2/stage-2-core eval receipt under {run_root}; "
            "run `eval-fanout capacity-pilot --pilot-stage 2` (or 2-core) first"
        )
    stage2_receipt, stage2_sha256 = eval_job.read_eval_receipt_with_hash(stage2_path)

    # D2 (review r2 delta): B2 must be bound to THIS B1/TRAIN, not merely
    # schema-shaped. Refuse a stage-2 receipt dispatched against a different
    # TRAIN receipt, or against a different (or absent) SEL selection.
    if stage2_receipt.get("train_receipt_sha256") != stage1_receipt.get("train_receipt_sha256"):
        raise JobError(
            f"{stage2_path} train_receipt_sha256 {stage2_receipt.get('train_receipt_sha256')!r} does not "
            f"match the B1 receipt's {stage1_receipt.get('train_receipt_sha256')!r}"
        )
    stage2_selection = stage2_receipt.get("selection")
    if stage2_selection is None:
        raise JobError(f"{stage2_path} carries no selection record; it was not dispatched from SEL's own record")
    stage2_c1 = eval_job.verify_selection_record(stage2_selection, b1_eval_receipt_sha256=stage1_sha256)
    if stage2_c1 != c1 or stage2_selection.get("selection_record_sha256") != hashlib.sha256(
        selection_path.read_bytes()
    ).hexdigest():
        raise JobError(f"{stage2_path} selection record does not match the current SEL selection file")

    candidates = sorted({c for c in (c1 - 1, c1, c1 + 1) if 3 <= c <= 19} | {c1})
    needed_ordinals = sorted({o for c in candidates for o in (c - 1, c, c + 1)})
    stream0_energies: dict[int, list[float]] = {ordinal: [] for ordinal in needed_ordinals}
    stream1_energies: dict[int, list[float]] = {ordinal: [] for ordinal in needed_ordinals}
    for generation in stage1_receipt.get("generations") or []:
        if generation.get("stream") == 0 and generation["ordinal"] in stream0_energies:
            stream0_energies[generation["ordinal"]].extend(eval_job.read_verified_stream_local_energies(generation))
    for generation in stage2_receipt.get("generations") or []:
        if generation.get("stream") == 1 and generation["ordinal"] in stream1_energies:
            stream1_energies[generation["ordinal"]].extend(eval_job.read_verified_stream_local_energies(generation))
    # D2: the missing-data check must fail an ordinal missing EITHER stream
    # separately -- a complete B1 must never mask a dropped B2 stream-1 run.
    missing_stream0 = sorted(ordinal for ordinal, energies in stream0_energies.items() if not energies)
    missing_stream1 = sorted(ordinal for ordinal, energies in stream1_energies.items() if not energies)
    if missing_stream0 or missing_stream1:
        raise JobError(
            f"final-SEL is missing pooled data for ordinals: stream0={missing_stream0!r} "
            f"stream1={missing_stream1!r}"
        )

    # D1 (review r2 delta): each stream is centered about its OWN median
    # before its deviations are pooled -- a shift in one stream's location
    # (with its dispersion unchanged) must not move the pooled W.
    pooled01_w_by_ordinal = {
        ordinal: protocol.pooled_w_statistic([stream0_energies[ordinal], stream1_energies[ordinal]])
        for ordinal in needed_ordinals
    }
    c_star = protocol.select_final_basin_ordinal(c1, pooled01_w_by_ordinal)
    record = {
        "c1": c1,
        "c_star": c_star,
        "b1_eval_receipt_sha256": stage1_sha256,
        "stage2_eval_receipt_sha256": stage2_sha256,
        "pooled01_w_by_ordinal": {str(ordinal): w for ordinal, w in pooled01_w_by_ordinal.items()},
        # K4 (final-verify finding on operator-rulings-pilot-20261110-1900,
        # item 3): carried from the SEL record's own c1, not recomputed from
        # c_star -- the edge condition this ruling targets is "c1 had no full
        # neighboring triple", not a property of the final pick.
        "interior_minimum": protocol.is_interior_minimum(c1),
    }
    final_path = run_root / "attached" / "receipts" / "h-v2-eval-capacity-pilot-final-selection.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"[h-v2] final-SEL: c1={c1} c*={c_star}")


def run(
    stage: str,
    lineage: str,
    *,
    pipeline_check: bool = False,
    pilot_payload: str | None = None,
) -> None:
    """Build, prepare, and activate one H-v2 attached generation.

    Parameters
    ----------
    stage : {"train", "eval"}
        Job stage.
    lineage : str
        Opaque lineage identity (for example ``"L1"``), or the fixed
        per-payload lineage label (``"G"``/``"P"``) when *pilot_payload* is
        set.
    pipeline_check : bool, default=False
        Selects the non-science pipeline-check payload (its task id, job
        name and logs say so) instead of the science payload TRAIN builds
        by default for an approved lineage (TO baff945c,
        ``protocol.science_training_config``).
    pilot_payload : {"gpu-route-g", "capacity-pilot"}, optional
        TO 63690220: selects one of the two non-science, TRAIN-only pilot
        payloads instead. Mutually exclusive with *pipeline_check*; EVAL is
        refused unconditionally for a pilot payload (item E's scope).

    Raises
    ------
    JobError
        Code :data:`SCIENCE_CONFIG_UNAVAILABLE_CODE`, if ``stage`` is
        ``"train"``, neither ``pipeline_check`` nor *pilot_payload* is set,
        and ``lineage`` is not one of ``launch_config.LINEAGES`` (the
        approved L1-L3 roster). Also raised if *pilot_payload* is set and
        ``stage`` is not ``"train"``, or *lineage* does not match the
        payload's own fixed label.
    """

    run_root = Path(_require_env("RUN")).resolve()
    expected_commit = _require_env("SHA")
    science_python = _require_env("SCIENCE_PYTHON")
    executor_python = _require_env("EXECUTOR_PY")

    if pilot_payload is not None:
        if stage != "train":
            raise JobError(f"pilot payload {pilot_payload!r} only runs stage 'train' (item E owns eval), got {stage!r}")
        expected_lineage = launch_config.PILOT_PAYLOAD_LINEAGES[pilot_payload]
        if lineage != expected_lineage:
            raise JobError(
                f"pilot payload {pilot_payload!r} refuses lineage {lineage!r}; "
                f"it dispatches only its own fixed lineage {expected_lineage!r}"
            )
        check_identity(
            run_root,
            expected_commit,
            science_python,
            expected_stratum=launch_config.PILOT_PAYLOAD_GPU_STRATA[pilot_payload],
        )
        task_id = _pilot_train_task_id(pilot_payload)
        handoff_dir = run_root / "attached" / "handoff" / task_id
        results_root = run_root / "attached" / "results"
        store = run_root / "attached" / "store" / task_id
        _reserve_store_parent(store)
        receipts_dir = run_root / "attached" / "receipts"
        receipts_dir.mkdir(parents=True, exist_ok=True)

        config, identity = protocol.pilot_training_config(pilot_payload)
        probe_path = handoff_dir / f"{task_id}.probe.yaml"
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        enabled_modules, skipped_not_admitted = attached.probe_admitted_a1_modules(
            config,
            protocol.candidate_modules_to_append(),
            science_python=science_python,
            probe_config_path=probe_path,
            cwd=run_root / "checkout",
        )
        plan, overrides, config = attached.build_h_v2_train_plan(
            output_dir=handoff_dir,
            results_root=results_root,
            lineage=pilot_payload,
            config=config,
            science_config_identity=identity,
            payload_label=pilot_payload,
        )
        # F4: reconcile against the final materialized config, not merely the
        # probe's own bookkeeping -- this is the complete, disjoint partition
        # of the A1 candidate set that actually ran, including the base
        # DataIntegrity/GradientStats leaves the probe never appends/removes.
        enabled_modules = protocol.reconcile_enabled_modules(config)
        prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=handoff_dir)
        alloc_dir = run_root / "attached" / "alloc" / task_id
        allocation_path, worker_path = write_local_allocation_and_worker(
            alloc_dir,
            science_python=science_python,
            placeholder_config=prepared.effective_config,
            backend_job_id=os.environ.get("SLURM_JOB_ID", "unknown"),
        )
        attached.activate_h_v2_run(
            prepared,
            science_python=science_python,
            executor_python=executor_python,
            allocation=allocation_path,
            worker_config=worker_path,
            store=store,
            activation_receipt_path=receipts_dir / f"{task_id}.activation.json",
            h_v2_receipt_path=receipts_dir / f"{task_id}.h-v2.json",
            expected_source_commit=expected_commit,
            dispatcher="local",
            cwd=run_root / "checkout",
            enabled_modules=enabled_modules,
            skipped_not_admitted=skipped_not_admitted,
            skipped_unspecified=protocol.unspecified_modules(),
        )
        print(
            f"[h-v2] {task_id}: accepted (enabled={enabled_modules!r} "
            f"skipped={skipped_not_admitted!r} unspecified={protocol.unspecified_modules()!r})"
        )
        return

    check_identity(run_root, expected_commit, science_python)

    science_config = None
    science_config_identity = None
    enabled_modules: list[str] | None = None
    skipped_not_admitted: list[str] | None = None
    if stage == "train":
        if pipeline_check:
            build_lineage = f"{PIPELINE_CHECK_LABEL}-{lineage}"
        else:
            if lineage not in launch_config.LINEAGES:
                raise JobError(
                    f"{SCIENCE_CONFIG_UNAVAILABLE_CODE}: lineage {lineage!r} is not an approved "
                    f"science lineage ({launch_config.LINEAGES!r}); refusing before any dispatch "
                    "rather than silently submitting protocol.training_config()'s CPU "
                    "pipeline-check defaults under an unapproved lineage name. Pass "
                    "pipeline_check=True (CLI: --pipeline-check) to explicitly select "
                    "the non-science pipeline-check payload instead."
                )
            build_lineage = lineage
            lineage_index = launch_config.LINEAGES.index(lineage)
            science_config, science_config_identity = protocol.science_training_config(lineage, lineage_index)
            probe_path = run_root / "attached" / "handoff" / f"h-v2-train-{lineage}.probe.yaml"
            enabled_modules, skipped_not_admitted = attached.probe_admitted_a1_modules(
                science_config,
                protocol.candidate_modules_to_append(),
                science_python=science_python,
                probe_config_path=probe_path,
                cwd=run_root / "checkout",
            )
        task_id = _train_task_id(lineage, pipeline_check=pipeline_check)
    elif stage == "eval":
        build_lineage = lineage
        # K2/H1: labelled the same way TRAIN is, so EVAL's own task
        # id/logs/receipts say plainly which TRAIN receipt it read.
        label = f"{PIPELINE_CHECK_LABEL}-" if pipeline_check else ""
        task_id = f"h-v2-eval-{label}{lineage}-stream0"
    else:
        raise JobError(f"stage must be 'train' or 'eval', got {stage!r}")

    # F2: namespaced by task id so a concurrent sibling lineage/stage cannot
    # retire or overwrite the handoff sidecars this task is still bound to
    # (the fixed names executor_input.json/configured-runs.json were
    # previously shared by every lineage and stage under one directory).
    handoff_dir = run_root / "attached" / "handoff" / task_id
    results_root = run_root / "attached" / "results"
    store = run_root / "attached" / "store" / task_id
    _reserve_store_parent(store)
    receipts_dir = run_root / "attached" / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)

    if stage == "train":
        plan, overrides, config = attached.build_h_v2_train_plan(
            output_dir=handoff_dir,
            results_root=results_root,
            lineage=build_lineage,
            config=science_config,
            science_config_identity=science_config_identity,
        )
        enabled_modules = protocol.reconcile_enabled_modules(config)
    else:
        checkpoint_dir = checkpoint_dir_for_eval(run_root, lineage, pipeline_check=pipeline_check)
        plan, overrides, config = attached.build_h_v2_eval_plan(
            output_dir=handoff_dir,
            results_root=results_root,
            lineage=lineage,
            checkpoint=checkpoint_dir,
            stream=0,
        )

    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=handoff_dir)
    alloc_dir = run_root / "attached" / "alloc" / task_id
    allocation_path, worker_path = write_local_allocation_and_worker(
        alloc_dir,
        science_python=science_python,
        placeholder_config=prepared.effective_config,
        backend_job_id=os.environ.get("SLURM_JOB_ID", "unknown"),
    )
    attached.activate_h_v2_run(
        prepared,
        science_python=science_python,
        executor_python=executor_python,
        allocation=allocation_path,
        worker_config=worker_path,
        store=store,
        activation_receipt_path=receipts_dir / f"{task_id}.activation.json",
        h_v2_receipt_path=receipts_dir / f"{task_id}.h-v2.json",
        expected_source_commit=expected_commit,
        dispatcher="local",
        cwd=run_root / "checkout",
        enabled_modules=enabled_modules,
        skipped_not_admitted=skipped_not_admitted,
        skipped_unspecified=protocol.unspecified_modules(),
    )
    print(f"[h-v2] {task_id}: accepted")


def _main_eval_fanout(argv: list[str]) -> int:
    """Parse and dispatch ``run_job.py eval-fanout ...`` (item E)."""

    import argparse

    parser = argparse.ArgumentParser(prog="run_job.py eval-fanout")
    parser.add_argument("payload", choices=_EVAL_FANOUT_PAYLOADS)
    parser.add_argument("lineage", nargs="?", default=None, help="Required, science only.")
    parser.add_argument("--expected-train-commit", required=True)
    parser.add_argument("--pilot-stage", choices=("1", "2", "2-core"), default=None)
    args = parser.parse_args(argv)

    if args.payload == "science":
        if not args.lineage:
            print("usage: run_job.py eval-fanout science <lineage> --expected-train-commit <sha>", file=sys.stderr)
            return 2
        lineage = args.lineage
        expected_stratum = launch_config.APPROVED_GPU_STRATUM
    else:
        lineage = launch_config.PILOT_PAYLOAD_LINEAGES[args.payload]
        expected_stratum = launch_config.PILOT_PAYLOAD_GPU_STRATA[args.payload]
        if args.payload == "capacity-pilot" and args.pilot_stage is None:
            print(
                "usage: run_job.py eval-fanout capacity-pilot --pilot-stage {1,2,2-core} "
                "--expected-train-commit <sha>",
                file=sys.stderr,
            )
            return 2

    run_root = Path(_require_env("RUN")).resolve()
    expected_commit = _require_env("SHA")
    science_python = _require_env("SCIENCE_PYTHON")
    try:
        check_identity(run_root, expected_commit, science_python, expected_stratum=expected_stratum)
        run_eval_fanout(
            run_root=run_root,
            expected_commit=expected_commit,
            science_python=science_python,
            payload=args.payload,
            lineage=lineage,
            expected_train_commit=args.expected_train_commit,
            pilot_stage=args.pilot_stage,
        )
    except Exception as error:  # noqa: BLE001 - this is the job's own top-level boundary
        print(f"[h-v2] job failed: {error}", file=sys.stderr)
        return 1
    return 0


def _main_sel(argv: list[str]) -> int:
    """Parse and dispatch ``run_job.py sel capacity-pilot``."""

    if argv != ["capacity-pilot"]:
        print("usage: run_job.py sel capacity-pilot", file=sys.stderr)
        return 2
    run_root = Path(_require_env("RUN")).resolve()
    try:
        run_pilot_sel(run_root)
    except Exception as error:  # noqa: BLE001 - this is the job's own top-level boundary
        print(f"[h-v2] job failed: {error}", file=sys.stderr)
        return 1
    return 0


def _main_final_sel(argv: list[str]) -> int:
    """Parse and dispatch ``run_job.py final-sel capacity-pilot`` (F7)."""

    if argv != ["capacity-pilot"]:
        print("usage: run_job.py final-sel capacity-pilot", file=sys.stderr)
        return 2
    run_root = Path(_require_env("RUN")).resolve()
    try:
        run_pilot_final_sel(run_root)
    except Exception as error:  # noqa: BLE001 - this is the job's own top-level boundary
        print(f"[h-v2] job failed: {error}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse and dispatch the in-job CLI.

    Review r1 (F2): an explicit payload choice is required -- either
    ``--pipeline-check`` or ``--payload science`` -- for both ``train`` and
    ``eval``. A bare ``train L1``/``eval L1`` with neither refuses (exit 2)
    rather than defaulting to the science payload.

    TO 8a5e926f: ``eval-fanout``/``sel``/``final-sel`` are separate
    subcommands (item E's own direct-dispatch route), dispatched before the
    ``train``/``eval`` parsing below ever runs.
    """

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "eval-fanout":
        return _main_eval_fanout(argv[1:])
    if argv and argv[0] == "sel":
        return _main_sel(argv[1:])
    if argv and argv[0] == "final-sel":
        return _main_final_sel(argv[1:])
    pipeline_check = False
    explicit_science = False
    pilot_payload: str | None = None
    if argv and argv[-1] == "--pipeline-check":
        pipeline_check = True
        argv = argv[:-1]
    elif len(argv) >= 2 and argv[-2:] == ["--payload", "science"]:
        explicit_science = True
        argv = argv[:-2]
    elif len(argv) >= 2 and argv[-2] == "--payload" and argv[-1] in protocol.PILOT_PAYLOADS:
        pilot_payload = argv[-1]
        argv = argv[:-2]
    if len(argv) != 2 or not (pipeline_check or explicit_science or pilot_payload):
        print(
            "usage: run_job.py {train|eval} <lineage> "
            "{--payload science|--payload gpu-route-g|--payload capacity-pilot|--pipeline-check}",
            file=sys.stderr,
        )
        return 2
    stage, lineage = argv
    try:
        run(stage, lineage, pipeline_check=pipeline_check, pilot_payload=pilot_payload)
    except Exception as error:  # noqa: BLE001 - this is the job's own top-level boundary
        # afterok sees only this exit code: any refusal here must be non-zero,
        # never a bare 0 (executor CLI exit 0 alone is not science complete).
        print(f"[h-v2] job failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
