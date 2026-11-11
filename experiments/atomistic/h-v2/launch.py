"""H-v2 study-side launch script (item 67f2eb57, design target section A4).

Per the manager's Q1 ruling (note ``manager-ruling-q1-submission-20261109``
on item 9d3aa0a3): the executor does not submit jobs, it only dispatches
inside an allocation it is already attached to. This script is the minimal
study-side submitter that closes that gap: per lineage, it runs ``sbatch``
for the training job, then ``sbatch --dependency=afterok:<train job id>``
for the evaluation job, and nothing else -- no retry, no polling, no
fallback. Each job runs ``run_job.py`` (the executor's attached route from
item 9d3aa0a3) inside its own allocation.

Round 2 (PR 619 decisions F1/F3): ``--payload`` is required, with no
default. ``pipeline-check`` explicitly selects the non-science payload,
labelled as such in job names and in-job argv. ``--science-python``/
``--executor-python`` are also required, absolute, existing-file paths,
embedded literally into the job wrapper rather than left to whatever the
submitting shell happened to export.

TO baff945c: ``science`` dispatches the approved per-lineage science
training config (``protocol.science_training_config``) for lineages L1-L3
(``launch_config.LINEAGES``); any other lineage still refuses before any
``sbatch`` call. ``run_job.py``'s own in-job gate for the same refusal is
kept as defence in depth.

Round 3 (K4). Invocation: run this file with the SCIENCE environment's
interpreter (the one later passed as ``--science-python``), not a bare
system Python -- even ``--dry-run`` and a bad ``--payload`` eagerly load
``run_job.py`` (and therefore ``attached.py``: ``yaml``, ``omegaconf``,
``experiments/toolkit``) to read shared constants, so those packages must
already be importable::

    "$TPEN_UV" run --project "$TPEN_CHECKOUT" --locked --extra cpu python \\
        "$TPEN_CHECKOUT/experiments/atomistic/h-v2/launch.py" \\
        --run-root "$RUN_ROOT" --commit "$SHA" --payload pipeline-check \\
        --science-python "$SCIENCE_PYTHON" --executor-python "$EXECUTOR_PY" \\
        --dry-run

No ``PYTHONPATH`` setup is required for that invocation: this module puts
its own checkout root on ``sys.path`` (K1; the same convention as
``experiments/hooke/pair_stability_v3/launch.py``'s own ``REPO_ROOT``
insertion), since ``experiments`` is a namespace package with no
``__init__.py`` and resolves only when the repository root is on
``sys.path`` (``pyproject.toml``'s ``[tool.pytest.ini_options]`` comment).
The in-job wrapper this script generates sets its own ``PYTHONPATH`` the
same way, for the same reason, in its own child process (:func:`_wrap_command`).
``T_WALL``/``E_WALL`` environment variables are required for every
invocation, dry-run included (K3): there is no placeholder fallback.

This script never cancels a scheduler job, and never invokes the ``tpen.run``
module directly as a script entrypoint (no ``__main__`` guard, operator
policy 2026-11-06); use the installed ``tpen`` console command instead. It
imports no ``tpen``, no ``tpen_executor``, and no ``hydra``
(``experiments/README.md``).
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent

# K1: ``experiments`` is a namespace package with no ``__init__.py`` and
# resolves only when the repository root is on ``sys.path`` (pytest supplies
# this for free via ``pythonpath = ["."]``; a plain script invocation does
# not). ``run_job.py``'s module-level load of ``attached.py`` needs it the
# moment ``_run_job_module()`` is first called below, including under
# ``--dry-run`` and even for an invalid ``--payload``. Mirrors
# ``experiments/hooke/pair_stability_v3/launch.py``'s own ``REPO_ROOT``
# insertion.
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _h_v2_module(name: str):
    """Load an h-v2 sibling under its own ``h_v2_<name>`` :mod:`sys.modules` key.

    ``experiments/`` has several same-named files across studies (notably
    ``launch.py`` itself, see ``he-v1``/``he-importance``), so a bare
    ``import`` would silently resolve to whichever study's copy loaded
    first. Mirrors ``attached.py``'s own private loader.
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


launch_config = _h_v2_module("launch_config")


def _run_job_module():
    """Lazily load ``run_job.py`` (G1: reuse its constants, not a duplicate literal).

    Loaded on demand rather than at module import time, so a plain
    ``--dry-run`` or argument-parsing failure never pays for (or depends on)
    ``run_job.py``'s own module-level load of ``attached.py``.
    """

    return _h_v2_module("run_job")


class LaunchError(RuntimeError):
    """A lineage's submission aborted before completing its two jobs."""


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), check=False, capture_output=True, text=True)


#: TO 63690220: the two non-science, TRAIN-only pilot/part-(g) payloads
#: (acceptance-contract). Each dispatches exactly one lineage, refuses any
#: other lineage or stage, and never chains an EVAL job (item E's scope).
PILOT_PAYLOADS = ("gpu-route-g", "capacity-pilot")


def _validate_payload(payload: str) -> None:
    """Refuse any ``--payload`` value outside the declared set.

    TO baff945c lifted the former blanket refusal of ``payload == "science"``
    for the approved L1-L3 lineage roster; :func:`_validate_science_lineage`
    now carries the per-lineage refusal that used to live here, reusing
    ``run_job.py``'s own refusal code
    (:data:`run_job.SCIENCE_CONFIG_UNAVAILABLE_CODE`) so the two refusal
    sites -- there, before any ``sbatch`` call, and ``run_job.py``'s in-job
    gate, kept as defence in depth -- never drift apart into two different
    codes for the same condition. TO 63690220 adds the two pilot payloads
    (:data:`PILOT_PAYLOADS`), dispatched by :func:`submit_pilot_payload`
    rather than :func:`submit_lineage`/:func:`submit_all`.

    Raises
    ------
    LaunchError
        If ``payload`` is not ``"science"``, ``"pipeline-check"``, or one of
        :data:`PILOT_PAYLOADS`.
    """

    if payload not in ("science", "pipeline-check") + PILOT_PAYLOADS:
        raise LaunchError(
            f"--payload must be 'science', 'pipeline-check', or one of {PILOT_PAYLOADS!r}, got {payload!r}"
        )


def _validate_science_lineage(payload: str, lineage: str) -> None:
    """TO baff945c: refuse a science payload for an unapproved lineage.

    EVAL stays refused unconditionally regardless of lineage
    (``attached.EVAL_ROUTE_UNSUPPORTED_CODE``, BLOCKER-EVAL); this only
    gates which lineage a ``science`` TRAIN/EVAL job may be submitted for.

    Raises
    ------
    LaunchError
        Code :data:`run_job.SCIENCE_CONFIG_UNAVAILABLE_CODE`, if
        ``payload == "science"`` and ``lineage`` is not one of
        ``launch_config.LINEAGES``.
    """

    if payload == "science" and lineage not in launch_config.LINEAGES:
        code = _run_job_module().SCIENCE_CONFIG_UNAVAILABLE_CODE
        raise LaunchError(
            f"{code}: lineage {lineage!r} is not an approved science lineage "
            f"({launch_config.LINEAGES!r}); refusing before any sbatch submission"
        )


def _validate_interpreter(path_value: str, *, label: str) -> str:
    """G2 (decision F1): require an absolute, existing interpreter path.

    Symlinks are deliberately left unresolved: a venv's ``bin/python`` is
    commonly a symlink, and resolving it would embed a path outside the
    venv it was selected for.

    Raises
    ------
    LaunchError
        If ``path_value`` is not absolute, or does not exist as a file.
    """

    if not os.path.isabs(path_value):
        raise LaunchError(f"{label} must be an absolute path, got {path_value!r}")
    if not Path(path_value).is_file():
        raise LaunchError(f"{label} does not exist: {path_value!r}")
    return path_value


#: F2: a self-contained GPU-utilization sidecar for the pilot/part-(g)
#: payloads, run via ``<science_python> - <log_path>`` so no sibling file
#: dependency exists (mirrors :func:`_wrap_command`'s own rationale). Stdlib
#: only (``os``/``signal``/``subprocess``/``sys``/``time``): no new
#: dependency. Samples ``memory.used``/``utilization.gpu`` roughly every
#: five seconds with a bounded per-query timeout; a failing or hung
#: ``nvidia-smi`` invocation is isolated (never raised to the caller) and
#: never blocks the next sample. On SIGTERM (the wrapper's own cleanup,
#: after the payload exits) it kills its own in-flight ``nvidia-smi`` process
#: group, if any, and exits -- it is never left running or orphaned past the
#: payload's own lifetime.
_GPU_SIDECAR_SOURCE = """\
import os
import signal
import subprocess
import sys
import time

LOG_PATH = sys.argv[1]
SAMPLE_INTERVAL_SECONDS = 5.0
QUERY_TIMEOUT_SECONDS = 3.0
_current = {"proc": None}


def _cleanup(signum, frame):
    proc = _current["proc"]
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # Reap immediately: an unreaped killed child stays a zombie (its pid
        # stays valid for a pid-0 liveness probe) until this process exits
        # AND the kernel reparents/collects it -- a race a caller polling
        # os.kill(pid, 0) right after this process exits can lose.
        try:
            proc.wait()
        except Exception:
            pass
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _cleanup)

while True:
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        _current["proc"] = proc
        try:
            output, _ = proc.communicate(timeout=QUERY_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = proc.communicate()
        _current["proc"] = None
        if output:
            with open(LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(output if output.endswith("\\n") else output + "\\n")
    except Exception:
        _current["proc"] = None
    elapsed = time.monotonic() - start
    time.sleep(max(0.0, SAMPLE_INTERVAL_SECONDS - elapsed))
"""

#: F2 (fixed, round 2): the dry-run printer (:func:`submit_pilot_payload`) and
#: every test built around it assume ``print(shlex.join(command))`` emits
#: exactly ONE printed line per job -- a ``--wrap`` value that itself
#: contains a literal newline (the original heredoc-based sidecar launch)
#: breaks that invariant even though it is still a single shell *token*
#: (``shlex.quote`` escapes it, but escaping does not stop a raw newline
#: byte from starting a new terminal line when printed). Launching the
#: sidecar via a base64-encoded ``-c`` payload avoids a heredoc entirely, so
#: the whole pilot wrapper body can be ONE line, joined by ``;``/``&``
#: instead of real newlines.
_GPU_SIDECAR_LAUNCHER = "import base64,sys;exec(base64.b64decode(sys.argv.pop(1)).decode())"
_GPU_SIDECAR_SOURCE_B64 = base64.b64encode(_GPU_SIDECAR_SOURCE.encode("utf-8")).decode("ascii")


def _gpu_sidecar_wrap_lines(
    *, checkout_quoted: str, science_python: str, executor_python: str, payload_command: str
) -> list[str]:
    """Return the pilot-only wrapper statements: sidecar, payload, bounded cleanup.

    F2: unlike the plain ``exec``-based science/pipeline-check wrapper
    (:func:`_wrap_command`), this never ``exec``s the payload -- its exit
    status is captured (``RC=$?``) so the sidecar can be stopped and reaped
    (``kill``/``wait``) before the wrapper itself exits with that SAME
    status, preserved unchanged. Every statement here is joined on one line
    (see :data:`_GPU_SIDECAR_LAUNCHER`); none may contain a literal newline.
    """

    science_quoted = shlex.quote(science_python)
    return [
        "set -e",
        f"cd {checkout_quoted}",
        f"CHECKOUT={checkout_quoted}",
        'export PYTHONPATH="$CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"',
        f"export SCIENCE_PYTHON={shlex.quote(science_python)}",
        f"export EXECUTOR_PY={shlex.quote(executor_python)}",
        'mkdir -p "$RUN/logs"',
        f"{science_quoted} -c {shlex.quote(_GPU_SIDECAR_LAUNCHER)} {shlex.quote(_GPU_SIDECAR_SOURCE_B64)} "
        '"$RUN/logs/gpu-sidecar.log" &',
        "MONITOR_PID=$!",
        # The payload's own exit status is captured with errexit disabled
        # around it, never allowed to abort the script before it is read.
        "set +e",
        payload_command,
        "RC=$?",
        "set -e",
        'kill "$MONITOR_PID" >/dev/null 2>&1 || true',
        'wait "$MONITOR_PID" >/dev/null 2>&1 || true',
        "exit $RC",
    ]


def _wrap_command(
    stage: str,
    lineage: str,
    *,
    checkout_root: Path,
    science_python: str,
    executor_python: str,
    payload: str,
) -> str:
    """Return the ``sh`` body for a self-contained, location-independent wrapper.

    F1: a ``sbatch`` submission only spools the SUBMITTED script's own bytes;
    it does not copy the checkout alongside it. Submitting ``run_job.py``
    directly therefore fails to find its siblings (``attached.py`` etc.) once
    Slurm runs the spooled copy. The fix is to submit this wrapper instead:
    it carries no sibling dependency of its own (every path is embedded as a
    literal, ``shlex``-quoted argument), ``cd``s into the explicit checkout,
    and ``exec``s the explicit science interpreter on the checkout's own
    ``run_job.py``.

    G2: the interpreter is the caller's explicit, validated
    ``--science-python``/``--executor-python`` value, embedded here as a
    literal ``shlex``-quoted argument -- never ``$SCIENCE_PYTHON`` read back
    from whatever the submitting shell happened to export (``sbatch``
    forwards the submitter's environment via ``--export=ALL``, which is not
    the same guarantee as an explicit, validated path). ``SCIENCE_PYTHON``
    and ``EXECUTOR_PY`` are still set in the job's own environment, from the
    same literal values, because ``run_job.py``'s in-job route reads them by
    name; setting them here from a literal rather than inheriting them
    closes the same gap for both.

    G1: when ``payload == "pipeline-check"``, ``--pipeline-check`` is
    appended to the in-job argv, selecting the explicit non-science payload
    in ``run_job.py``. For ``payload == "science"`` (an approved L1-L3
    lineage; :func:`_validate_science_lineage` refuses any other lineage
    before this function is reached), ``--payload science`` is appended
    instead (review r1 F2): ``run_job.py``'s in-job parser requires one of
    these two explicit tokens and refuses a bare ``train``/``eval`` argv
    rather than defaulting to science. TO 63690220: for a pilot payload
    (:data:`PILOT_PAYLOADS`), ``--payload <payload>`` is appended with the
    exact pilot payload name, so ``run_job.py`` dispatches
    ``protocol.pilot_training_config`` instead of either of the above.

    K1: Slurm spools only the submitted ``--wrap`` body itself, as its own
    fresh process -- ``sys.path`` there starts at the checkout's
    ``experiments/atomistic/h-v2`` directory (``run_job.py``'s own
    directory), never the repository root, and nothing sets ``PYTHONPATH``
    for it. ``run_job.py``'s module-level load of ``attached.py`` (``from
    experiments.toolkit.attached_activation import ...``) then fails with
    ``ModuleNotFoundError: No module named 'experiments'`` before any
    payload runs, every time, for every stage. The wrapper sets
    ``PYTHONPATH`` to the explicit checkout root itself, via a real shell
    variable rather than inline string interpolation (so a run root
    containing shell metacharacters, as :func:`shlex.quote` already handles
    for every other embedded path here, can never be reinterpreted),
    appending any already-inherited ``PYTHONPATH`` rather than replacing it
    -- the same convention as ``experiments/atomistic/he-cutover``'s sbatch
    templates.
    """

    job_script = checkout_root / launch_config.JOB_SCRIPT_RELATIVE
    if payload == "pipeline-check":
        extra_arg = f" {shlex.quote('--pipeline-check')}"
    elif payload in PILOT_PAYLOADS:
        extra_arg = f" {shlex.quote('--payload')} {shlex.quote(payload)}"
    else:
        # F2 (review r1): an explicit science token, never an implicit default
        # -- a dropped flag or a bare "train L1"/"eval L1" invocation must
        # refuse in run_job.py's main(), not silently dispatch science.
        extra_arg = f" {shlex.quote('--payload')} {shlex.quote('science')}"
    checkout_quoted = shlex.quote(str(checkout_root))
    payload_command = (
        f"{shlex.quote(science_python)} {shlex.quote(str(job_script))} "
        f"{shlex.quote(stage)} {shlex.quote(lineage)}{extra_arg}"
    )
    if payload in PILOT_PAYLOADS:
        # F2: these two payloads run under a bounded GPU-utilization sidecar
        # (:data:`_GPU_SIDECAR_SOURCE`); the science/pipeline-check wrapper
        # below stays byte-identical to #629 (no sidecar). Joined onto ONE
        # line (never a literal newline): a `--wrap` value containing one
        # breaks the "print(shlex.join(command)) is exactly one printed
        # line" convention every dry-run caller/test relies on, even though
        # it is still a single shell token. A statement already ending in
        # `&` (the backgrounded sidecar launch) needs no extra separator.
        statements = _gpu_sidecar_wrap_lines(
            checkout_quoted=checkout_quoted,
            science_python=science_python,
            executor_python=executor_python,
            payload_command=payload_command,
        )
        body = statements[0]
        for statement in statements[1:]:
            body += statement if body.endswith("&") else f"; {statement}"
        return body
    return (
        f"cd {checkout_quoted} && "
        f"CHECKOUT={checkout_quoted} && "
        'PYTHONPATH="$CHECKOUT${PYTHONPATH:+:$PYTHONPATH}" '
        f"SCIENCE_PYTHON={shlex.quote(science_python)} "
        f"EXECUTOR_PY={shlex.quote(executor_python)} "
        f"exec {payload_command}"
    )


def build_sbatch_command(
    stage: str,
    lineage: str,
    *,
    payload: str,
    science_python: str,
    executor_python: str,
    run_root: Path,
    checkout_root: Path,
    time_value: str,
    dependency: str | None,
    commit: str | None = None,
    resources: launch_config.JobResources = launch_config.RESOURCES,
) -> list[str]:
    """Return the exact ``sbatch`` argv for one job (design target section A4).

    Parameters
    ----------
    stage : {"train", "eval"}
        Job stage.
    lineage : str
        Opaque lineage identity (for example ``"L1"``).
    payload : {"science", "pipeline-check"}
        G1/TO baff945c: which payload this job dispatches. ``"science"`` is
        refused by :func:`_validate_science_lineage` for any lineage outside
        ``launch_config.LINEAGES``; ``"pipeline-check"`` labels the job name
        and appends ``--pipeline-check`` to the in-job argv
        (:func:`_wrap_command`).
    science_python, executor_python : str
        G2: explicit, pre-validated absolute interpreter paths, embedded
        literally into the wrapper (:func:`_wrap_command`).
    run_root, checkout_root : pathlib.Path
        Durable run root (logs live under ``run_root/logs``) and the
        checkout the job script is read from.
    time_value : str
        Exact Slurm ``--time`` value (``T_WALL`` for train, ``E_WALL`` for
        eval); this module performs no wall-time arithmetic.
    dependency : str or None
        Exact ``--dependency`` value (for example ``"afterok:12345"``), or
        ``None`` for a job with no dependency.
    commit : str, optional
        Expected checkout commit (F5). When given, bound into the job's
        environment as ``--export=...,SHA=<commit>`` so the in-job identity
        check cannot be satisfied by a stale inherited ``SHA`` instead of the
        commit this script was actually told to launch.
    resources : JobResources
        Section C2 geometry; defaults to the single approved
        :data:`launch_config.RESOURCES`.

    Returns
    -------
    list of str
        The full ``sbatch`` argv: flags, then ``--wrap=<wrapper body>`` as
        the final argument (the wrapper runs the job script; see
        :func:`_wrap_command`).
    """

    if stage not in ("train", "eval"):
        raise LaunchError(f"stage must be 'train' or 'eval', got {stage!r}")
    _validate_payload(payload)
    _validate_science_lineage(payload, lineage)
    run_root = Path(run_root)
    checkout_root = Path(checkout_root)
    label = _run_job_module().PIPELINE_CHECK_LABEL if payload == "pipeline-check" else payload
    job_name = f"h-v2-{label}-{lineage}-{stage}"
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
    ]
    if resources.constraint:
        command.append(f"--constraint={resources.constraint}")
    # K2: an optional typed GRES model renders as gpu:<model>:<count>; every
    # other resource profile (science, pipeline-check, capacity-pilot) keeps
    # the generic gpu:<count> it always has.
    gres = f"gpu:{resources.gpu_model}:{resources.gpus}" if resources.gpu_model else f"gpu:{resources.gpus}"
    command += [
        f"--gres={gres}",
        f"--nodes={resources.nodes}",
        f"--ntasks={resources.ntasks}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.mem}",
        f"--time={time_value}",
    ]
    if resources.no_requeue:
        command.append("--no-requeue")
    # F5/G2: RUN and (when known) SHA are bound explicitly here, from this
    # script's own parsed arguments, rather than left to whatever the
    # submitting shell happened to have exported. SCIENCE_PYTHON and
    # EXECUTOR_PY no longer rely on ALL forwarding the submitting shell's
    # environment either -- the wrapper (_wrap_command) sets them itself
    # from the same literal, validated paths used to exec the interpreter.
    # ALL is kept for anything else the environment may still need (sbatch
    # does not export by default -- facility note
    # sbatch-env-var-not-exported-by-default-20261104).
    export_parts = ["ALL", f"RUN={run_root}"]
    if commit is not None:
        export_parts.append(f"SHA={commit}")
    command.append("--export=" + ",".join(export_parts))
    command += [
        f"--output={run_root / 'logs' / '%x-%j.out'}",
        f"--error={run_root / 'logs' / '%x-%j.err'}",
    ]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(
        f"--wrap={_wrap_command(stage, lineage, checkout_root=checkout_root, science_python=science_python, executor_python=executor_python, payload=payload)}"
    )
    return command


def _wrap_eval_fanout_command(
    payload: str,
    *,
    lineage: str | None,
    pilot_stage: str | None,
    expected_train_commit: str,
    checkout_root: Path,
    science_python: str,
    executor_python: str,
) -> str:
    """Return the wrapper body for ``run_job.py eval-fanout ...`` (item E).

    TO 8a5e926f amendment r4 E6: pilot/g eval-fanout jobs reuse the pilot
    TRAIN jobs' GPU sidecar (:func:`_gpu_sidecar_wrap_lines`); science's
    wrapper stays the plain ``exec`` form (no sidecar), matching #629's own
    wrapper.
    """

    job_script = checkout_root / launch_config.JOB_SCRIPT_RELATIVE
    argv = [science_python, str(job_script), "eval-fanout", payload]
    if payload == "science":
        assert lineage is not None
        argv.append(lineage)
    argv += ["--expected-train-commit", expected_train_commit]
    if pilot_stage is not None:
        argv += ["--pilot-stage", pilot_stage]
    checkout_quoted = shlex.quote(str(checkout_root))
    payload_command = shlex.join(argv)
    if payload in PILOT_PAYLOADS:
        statements = _gpu_sidecar_wrap_lines(
            checkout_quoted=checkout_quoted,
            science_python=science_python,
            executor_python=executor_python,
            payload_command=payload_command,
        )
        body = statements[0]
        for statement in statements[1:]:
            body += statement if body.endswith("&") else f"; {statement}"
        return body
    return (
        f"cd {checkout_quoted} && "
        f"CHECKOUT={checkout_quoted} && "
        'PYTHONPATH="$CHECKOUT${PYTHONPATH:+:$PYTHONPATH}" '
        f"SCIENCE_PYTHON={shlex.quote(science_python)} "
        f"EXECUTOR_PY={shlex.quote(executor_python)} "
        f"exec {payload_command}"
    )


def build_eval_fanout_sbatch_command(
    payload: str,
    *,
    lineage: str | None = None,
    pilot_stage: str | None = None,
    expected_train_commit: str,
    science_python: str,
    executor_python: str,
    run_root: Path,
    checkout_root: Path,
    time_value: str,
    dependency: str | None,
    commit: str,
    resources: launch_config.JobResources,
) -> list[str]:
    """Return the exact ``sbatch`` argv for one EVAL fan-out job (item E).

    Unlike :func:`build_sbatch_command` (the single-generation TRAIN/EVAL
    route), this submits ``run_job.py eval-fanout <payload> ...`` -- the
    checkpoint-verified, direct-dispatch fan-out
    (:mod:`experiments.atomistic.h-v2.eval_job`), never the TRAIN-only
    executor route that refuses EVAL unconditionally.

    Parameters
    ----------
    payload : {"science", "pipeline-check", "gpu-route-g", "capacity-pilot"}
    lineage : str, optional
        Required for ``payload == "science"``; ignored (the payload's own
        fixed label is used) for every other payload.
    pilot_stage : {"1", "2", "2-core"}, optional
        Required for ``payload == "capacity-pilot"``; refused for any other
        payload by the caller before this function is reached.
    expected_train_commit : str
        The TRAIN commit this EVAL job's receipt must have been written at
        (bound into the in-job argv, never guessed).
    commit : str
        This job's OWN expected checkout commit (bound as ``SHA``, the same
        way :func:`build_sbatch_command` does) -- may differ from
        *expected_train_commit* for an EVAL-only submission against an
        earlier TRAIN commit.

    Returns
    -------
    list of str
        The full ``sbatch`` argv, ``--wrap=...`` last.

    Raises
    ------
    LaunchError
        If *payload* is not recognized, or *payload == "science"* with an
        unapproved *lineage*.
    """

    _validate_payload(payload)
    if payload in PILOT_PAYLOADS:
        job_lineage = launch_config.PILOT_PAYLOAD_LINEAGES[payload]
    else:
        _validate_science_lineage(payload, lineage or "")
        job_lineage = lineage
    run_root = Path(run_root)
    checkout_root = Path(checkout_root)
    label = _run_job_module().PIPELINE_CHECK_LABEL if payload == "pipeline-check" else payload
    stage_label = f"eval-{pilot_stage}" if pilot_stage else "eval"
    job_name = f"h-v2-{label}-{job_lineage}-{stage_label}"
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
    ]
    if resources.constraint:
        command.append(f"--constraint={resources.constraint}")
    gres = f"gpu:{resources.gpu_model}:{resources.gpus}" if resources.gpu_model else f"gpu:{resources.gpus}"
    command += [
        f"--gres={gres}",
        f"--nodes={resources.nodes}",
        f"--ntasks={resources.ntasks}",
        f"--cpus-per-task={resources.cpus_per_task}",
        f"--mem={resources.mem}",
        f"--time={time_value}",
    ]
    if resources.no_requeue:
        command.append("--no-requeue")
    command.append("--export=" + ",".join(["ALL", f"RUN={run_root}", f"SHA={commit}"]))
    command += [
        f"--output={run_root / 'logs' / '%x-%j.out'}",
        f"--error={run_root / 'logs' / '%x-%j.err'}",
    ]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(
        "--wrap="
        + _wrap_eval_fanout_command(
            payload,
            lineage=lineage if payload == "science" else None,
            pilot_stage=pilot_stage,
            expected_train_commit=expected_train_commit,
            checkout_root=checkout_root,
            science_python=science_python,
            executor_python=executor_python,
        )
    )
    return command


def _validate_pilot_stage2_ready(run_root: Path, *, pilot_stage: str) -> None:
    """Refuse pilot ``--pilot-stage 2``/``2-core`` before any ``sbatch`` call.

    TO 8a5e926f (orchestrator round-1 correction): reads the SEL selection
    record item E's ``sel`` subcommand already wrote, and defensively
    re-validates ``c1`` (never trusts a possibly-stale or tampered file
    blind) -- the real gate is still ``run_job.py``'s own
    ``eval_job.verify_selection_record``, run in-job; this is an early,
    cheap client-side refusal before spending a GPU allocation on a doomed
    submission.

    Raises
    ------
    LaunchError
        If the selection record is missing, unreadable, or carries a
        ``c1`` outside ``3..19``.
    """

    selection_path = run_root / "attached" / "receipts" / "h-v2-eval-capacity-pilot-selection.json"
    if not selection_path.is_file():
        raise LaunchError(
            f"no SEL selection record at {selection_path}; run `run_job.py sel capacity-pilot` "
            f"before --pilot-stage {pilot_stage}"
        )
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LaunchError(f"selection record {selection_path} is not valid JSON: {error}") from error
    c1 = selection.get("c1")
    if not isinstance(c1, int) or isinstance(c1, bool) or not 3 <= c1 <= 19:
        raise LaunchError(
            f"selection record {selection_path} carries c1={c1!r}, not an int in 3..19; "
            "refusing before any sbatch call"
        )


def _submit(command: Sequence[str], *, runner: Runner) -> str:
    """Run one ``sbatch`` command and return its parsed job id.

    Raises
    ------
    LaunchError
        If ``sbatch`` exits non-zero, or ``--parsable`` yields no job id.
        Either is a hard stop: the caller must not submit the dependent job.
    """

    result = runner(command)
    if result.returncode != 0:
        raise LaunchError(
            f"sbatch exited {result.returncode} for {shlex.join(command)}: {result.stderr.strip()}"
        )
    job_id = result.stdout.strip().split(";")[0]
    if not job_id:
        raise LaunchError(f"sbatch returned no job id for {shlex.join(command)}")
    return job_id


def _parse_wall_seconds(value: str) -> int:
    """Parse a Slurm ``--time`` value (``[D-]HH:MM:SS``/``HH:MM``/``MM``) into seconds.

    Raises
    ------
    LaunchError
        If ``value`` is malformed, or resolves to a non-positive duration.
    """

    remainder = value
    days = 0
    if "-" in value:
        days_part, remainder = value.split("-", 1)
        try:
            days = int(days_part)
        except ValueError:
            raise LaunchError(f"malformed wall-time value {value!r}") from None
    fields = remainder.split(":")
    try:
        parts = [int(field) for field in fields]
    except ValueError:
        raise LaunchError(f"malformed wall-time value {value!r}") from None
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = parts[0], parts[1], 0
    elif len(parts) == 1:
        hours, minutes, seconds = 0, 0, parts[0]
    else:
        raise LaunchError(f"malformed wall-time value {value!r}")
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise LaunchError(f"wall-time value {value!r} must be positive")
    return total


def _validate_wall_budget(t_wall: str, e_wall: str) -> None:
    """F4: refuse an over-budget T_WALL/E_WALL pair before any submission.

    Section C3's rule is FROZEN by D1: each of T_WALL and E_WALL is at most
    the ``seas_gpu`` ceiling (:data:`launch_config.MAX_JOB_WALL_SECONDS`),
    and their sum is at most the per-lineage cap
    (:data:`launch_config.MAX_LINEAGE_WALL_SECONDS`). Slurm itself then
    enforces the 32/96 GPU-hour budget once jobs are submitted, but nothing
    stops an over-budget pair from being accepted at submit time -- this is
    the check that refuses it before the first ``sbatch`` call.
    """

    t_seconds = _parse_wall_seconds(t_wall)
    e_seconds = _parse_wall_seconds(e_wall)
    if t_seconds > launch_config.MAX_JOB_WALL_SECONDS or e_seconds > launch_config.MAX_JOB_WALL_SECONDS:
        raise LaunchError(
            f"T_WALL and E_WALL must each be at most {launch_config.MAX_JOB_WALL_SECONDS}s "
            f"(2 days, the seas_gpu ceiling); got T_WALL={t_wall!r} ({t_seconds}s), "
            f"E_WALL={e_wall!r} ({e_seconds}s)"
        )
    total = t_seconds + e_seconds
    if total > launch_config.MAX_LINEAGE_WALL_SECONDS:
        raise LaunchError(
            f"T_WALL + E_WALL must be at most {launch_config.MAX_LINEAGE_WALL_SECONDS}s "
            f"(32 hours, design target section C3); got T_WALL={t_wall!r} + E_WALL={e_wall!r} "
            f"= {total}s"
        )


def _validate_single_stage_wall_seconds(time_value: str, *, label: str) -> None:
    """F5: refuse a single-stage (TRAIN-only or EVAL-only) submission's own time budget.

    :func:`_validate_wall_budget` only runs for the chained ``--stage
    both``/omitted science route, where both ``T_WALL`` and ``E_WALL`` are
    known together. A single-stage submission supplies only one side, but it
    still must not alone exceed the per-job ceiling or the whole per-lineage
    budget (review r1 F5: a standalone 33-hour request already exceeds the
    32-hour total lineage allowance and was accepted).

    Raises
    ------
    LaunchError
        If *time_value* exceeds :data:`launch_config.MAX_JOB_WALL_SECONDS` or
        :data:`launch_config.MAX_LINEAGE_WALL_SECONDS`.
    """

    seconds = _parse_wall_seconds(time_value)
    if seconds > launch_config.MAX_JOB_WALL_SECONDS:
        raise LaunchError(
            f"{label} must be at most {launch_config.MAX_JOB_WALL_SECONDS}s (2 days, the seas_gpu "
            f"ceiling); got {time_value!r} ({seconds}s)"
        )
    if seconds > launch_config.MAX_LINEAGE_WALL_SECONDS:
        raise LaunchError(
            f"{label} must be at most the per-lineage wall budget "
            f"{launch_config.MAX_LINEAGE_WALL_SECONDS}s (32 hours, design target section C3) even alone; "
            f"got {time_value!r} ({seconds}s)"
        )


def submit_lineage(
    lineage: str,
    *,
    payload: str,
    science_python: str,
    executor_python: str,
    run_root: Path,
    checkout_root: Path,
    t_wall: str,
    e_wall: str,
    commit: str | None = None,
    runner: Runner = _default_runner,
    dry_run: bool = False,
) -> tuple[str | None, str | None]:
    """Submit one lineage's train job, then its eval job chained on it.

    Returns
    -------
    tuple of (str or None, str or None)
        The train and eval job ids, or ``(None, None)`` when ``dry_run`` is
        set: nothing is submitted, so no id exists yet, and this prints the
        exact commands (the eval command's ``--dependency`` carries a
        placeholder, since the real train job id is not known without
        submitting).

    Raises
    ------
    LaunchError
        If ``payload`` is not ``"science"``/``"pipeline-check"``, or
        ``payload == "science"`` for a lineage outside
        ``launch_config.LINEAGES``, before any submission, dry-run included.
        If the train submission fails or yields no job id, the eval job is
        never submitted in that case.
    """

    _validate_payload(payload)
    train_command = build_sbatch_command(
        "train",
        lineage,
        payload=payload,
        science_python=science_python,
        executor_python=executor_python,
        run_root=run_root,
        checkout_root=checkout_root,
        time_value=t_wall,
        dependency=None,
        commit=commit,
    )

    def _eval_command(*, dependency: str | None, expected_commit: str) -> list[str]:
        # TO 8a5e926f (item E): 'science' chains the checkpoint-verified
        # direct-dispatch fan-out (never the TRAIN-only executor route,
        # which refuses EVAL unconditionally for it). 'pipeline-check'
        # keeps the pre-existing single-generation route BYTE-IDENTICAL --
        # item E's fan-out is science-scale only (eval_job.py has no
        # pipeline-check grid); that route's own EVAL refusal is unchanged,
        # pre-existing, and out of this item's scope.
        if payload == "science":
            return build_eval_fanout_sbatch_command(
                payload,
                lineage=lineage,
                expected_train_commit=expected_commit,
                science_python=science_python,
                executor_python=executor_python,
                run_root=run_root,
                checkout_root=checkout_root,
                time_value=e_wall,
                dependency=dependency,
                commit=expected_commit,
                resources=launch_config.RESOURCES,
            )
        return build_sbatch_command(
            "eval",
            lineage,
            payload=payload,
            science_python=science_python,
            executor_python=executor_python,
            run_root=run_root,
            checkout_root=checkout_root,
            time_value=e_wall,
            dependency=dependency,
            commit=commit,
        )

    if dry_run:
        print(shlex.join(train_command))
        placeholder_eval = _eval_command(dependency="afterok:<train job id>", expected_commit=commit or "<commit>")
        print(shlex.join(placeholder_eval))
        return None, None

    train_id = _submit(train_command, runner=runner)
    # Submitted ONLY after the train job id exists, chained on it via
    # afterok -- never auto-submitted ahead of or without that dependency.
    eval_command = _eval_command(dependency=f"afterok:{train_id}", expected_commit=commit or "")
    eval_id = _submit(eval_command, runner=runner)
    return train_id, eval_id


#: F5: the fixed study segment every pilot/part-(g) run root lives under,
#: between the durable prefix and the payload/lineage pair
#: (``<prefix>/h-v2/<payload>/<lineage>``, harness r3 addendum3).
_PILOT_RUN_ROOT_STUDY_SEGMENT = "h-v2"


def _validate_run_root_prefix(
    run_root: Path, run_root_prefix: str, *, payload: str, lineage: str | None = None
) -> None:
    """Refuse a run root outside its payload's own durable location.

    Run-root amendment (harness r3 addendum3): every H-v2 run root -- pilot,
    part-(g), and science -- must live under a persistent, lifecycle-governed
    prefix, in the shape ``<prefix>/h-v2/<payload>`` (or, with *lineage*
    given, the narrower ``<prefix>/h-v2/<payload>/<lineage>``), or a child of
    it (for example an attempt subdirectory) -- not merely anywhere under the
    broad prefix (review r1 F5: a caller-supplied prefix ending in ``runs``
    otherwise accepted any sibling payload's or an unrelated root). The exact
    facility path is never hardcoded here
    (``netscratch-never-referenced-from-repo``); the caller supplies the
    prefix (:data:`launch_config.PILOT_RUN_ROOT_PREFIX_ENV`).

    Raises
    ------
    LaunchError
        If ``run_root_prefix`` is not absolute, or ``run_root`` is not the
        expected directory or a descendant of it.
    """

    prefix = Path(run_root_prefix)
    if not prefix.is_absolute():
        raise LaunchError(f"{launch_config.PILOT_RUN_ROOT_PREFIX_ENV} must be an absolute path, got {run_root_prefix!r}")
    segments = (_PILOT_RUN_ROOT_STUDY_SEGMENT, payload) + ((lineage,) if lineage is not None else ())
    expected_root = prefix.joinpath(*segments).resolve()
    try:
        run_root.resolve().relative_to(expected_root)
    except ValueError:
        raise LaunchError(
            f"run root {run_root} must be {expected_root} or a descendant of it "
            f"({launch_config.PILOT_RUN_ROOT_PREFIX_ENV}={run_root_prefix!r}, payload={payload!r}, "
            f"lineage={lineage!r}); refusing before any sbatch call"
        ) from None


def submit_pilot_payload(
    payload: str,
    *,
    science_python: str,
    executor_python: str,
    run_root: Path,
    checkout_root: Path,
    run_root_prefix: str,
    t_wall: str | None = None,
    commit: str | None = None,
    runner: Runner = _default_runner,
    dry_run: bool = False,
) -> str | None:
    """Submit one pilot/part-(g) payload's single TRAIN job (TO 63690220).

    Unlike :func:`submit_lineage`, this never chains an EVAL job (item E's
    scope) and never accepts a caller-supplied lineage: each pilot payload
    carries exactly one fixed lineage label
    (:data:`launch_config.PILOT_PAYLOAD_LINEAGES`).

    Parameters
    ----------
    payload : {"gpu-route-g", "capacity-pilot"}
    run_root_prefix : str
        The durable, lifecycle-governed prefix *run_root* must lie under
        (:func:`_validate_run_root_prefix`), checked before any ``sbatch``
        call, dry-run included.
    t_wall : str, optional
        Exact Slurm ``--time`` value. Defaults to the payload's own cap
        (:data:`launch_config.PILOT_PAYLOAD_MAX_WALL_SECONDS`) formatted as
        ``HH:MM:SS``. A caller-supplied value above the cap refuses.

    Returns
    -------
    str or None
        The submitted TRAIN job id, or ``None`` under ``--dry-run``.

    Raises
    ------
    LaunchError
        If *payload* is not one of :data:`PILOT_PAYLOADS`, *run_root* is
        outside *run_root_prefix*, or *t_wall* exceeds the payload's cap.
    """

    if payload not in PILOT_PAYLOADS:
        raise LaunchError(f"--payload must be one of {PILOT_PAYLOADS!r}, got {payload!r}")
    run_root = Path(run_root)
    lineage = launch_config.PILOT_PAYLOAD_LINEAGES[payload]
    _validate_run_root_prefix(run_root, run_root_prefix, payload=payload, lineage=lineage)
    cap_seconds = launch_config.PILOT_PAYLOAD_MAX_WALL_SECONDS[payload]
    if t_wall is None:
        time_value = _format_wall_seconds(cap_seconds)
    else:
        time_value = t_wall
        requested_seconds = _parse_wall_seconds(time_value)
        if requested_seconds > cap_seconds:
            raise LaunchError(
                f"--time for payload {payload!r} must be at most {cap_seconds}s "
                f"({_format_wall_seconds(cap_seconds)}); got {time_value!r} ({requested_seconds}s)"
            )
    command = build_sbatch_command(
        "train",
        lineage,
        payload=payload,
        science_python=science_python,
        executor_python=executor_python,
        run_root=run_root,
        checkout_root=checkout_root,
        time_value=time_value,
        dependency=None,
        commit=commit,
        resources=launch_config.PILOT_PAYLOAD_RESOURCES[payload],
    )
    if dry_run:
        print(shlex.join(command))
        return None
    return _submit(command, runner=runner)


def _format_wall_seconds(total_seconds: int) -> str:
    """Return a Slurm ``[D-]HH:MM:SS`` value for *total_seconds* (inverse of :func:`_parse_wall_seconds`)."""

    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    body = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}-{body}" if days else body


def submit_all(
    *,
    payload: str,
    science_python: str,
    executor_python: str,
    run_root: Path,
    checkout_root: Path,
    t_wall: str,
    e_wall: str,
    commit: str | None = None,
    lineages: Sequence[str] = launch_config.LINEAGES,
    runner: Runner = _default_runner,
    dry_run: bool = False,
) -> dict[str, tuple[str | None, str | None]]:
    """Submit every lineage's two jobs in order; abort on the first failure.

    G1: validates ``payload`` first, before anything else, so an
    unrecognized value raises before the T_WALL/E_WALL budget is even
    checked, with zero runner calls, dry-run included (the default
    ``lineages`` roster, ``launch_config.LINEAGES``, is always approved for
    ``payload == "science"``). Review r1 (F3): the ENTIRE requested
    ``lineages`` roster is then validated before any submission or dry-run
    print -- a caller-supplied roster mixing an approved and an unapproved
    lineage (for example ``("L1", "L9")``) refuses before ``L1`` is ever
    submitted, not partway through. :func:`submit_lineage`/
    :func:`build_sbatch_command` repeat the same per-lineage check as
    defence in depth. Then validates the T_WALL/E_WALL budget (F4) before
    any lineage is attempted. A lineage whose train submission fails raises
    before any later lineage is attempted: this function does no
    partial-progress bookkeeping beyond what each already-successful
    :func:`submit_lineage` call returned to the caller before the failure.
    """

    _validate_payload(payload)
    for lineage in lineages:
        _validate_science_lineage(payload, lineage)
    _validate_wall_budget(t_wall, e_wall)
    results: dict[str, tuple[str | None, str | None]] = {}
    for lineage in lineages:
        results[lineage] = submit_lineage(
            lineage,
            payload=payload,
            science_python=science_python,
            executor_python=executor_python,
            run_root=run_root,
            checkout_root=checkout_root,
            t_wall=t_wall,
            e_wall=e_wall,
            commit=commit,
            runner=runner,
            dry_run=dry_run,
        )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse launch command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Durable run root (logs, checkout live under it).")
    parser.add_argument(
        "--payload",
        required=True,
        choices=("science", "pipeline-check") + PILOT_PAYLOADS,
        help=(
            "Which payload every lineage dispatches (G1/decision F3; TO baff945c). "
            "'science' dispatches the approved per-lineage science training config for "
            "L1-L3; any other lineage still refuses before any sbatch submission. "
            "'pipeline-check' explicitly selects the non-science pipeline-check payload, "
            "labelled as such in job names and in-job argv. 'gpu-route-g'/'capacity-pilot' "
            "(TO 63690220) each submit exactly one TRAIN-only job; --stage train is "
            "required for these, and --run-root-prefix. No default: the caller must "
            "choose."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("train", "eval", "both"),
        default=None,
        help=(
            "'train'/'eval' submits only that stage (TO 8a5e926f); 'eval' requires "
            "--expected-train-commit and --time, and (for payload=capacity-pilot) "
            "--pilot-stage. Omitted or 'both' (equivalent) submits science/"
            "pipeline-check's train then eval, chained by afterok -- unchanged shape "
            "(exactly one sbatch line per stage per lineage). Required, and must be "
            "'train' or 'eval', for a pilot payload (TO 63690220); never 'both' for "
            "those (a pilot payload never chains train and eval in one submission)."
        ),
    )
    parser.add_argument(
        "--pilot-stage",
        choices=("1", "2", "2-core"),
        default=None,
        help=(
            "Required with --payload capacity-pilot --stage eval (TO 8a5e926f amendment "
            "r3/r4 E1): each stage is a SEPARATE sbatch submission, never auto-chained. "
            "'2'/'2-core' refuse before any sbatch call unless the SEL selection record "
            "(run_job.py sel capacity-pilot) already exists with c1 in 3..19."
        ),
    )
    parser.add_argument(
        "--expected-train-commit",
        default=None,
        help=(
            "Required with --stage eval (TO 8a5e926f): the TRAIN commit this EVAL job's "
            "receipt must have been written at. May differ from --commit (this job's own "
            "checkout commit) for an EVAL-only submission against an earlier TRAIN commit."
        ),
    )
    parser.add_argument(
        "--time",
        default=None,
        help=(
            "Required with --stage eval (any payload): the exact Slurm --time value. No "
            "default -- T_WALL/E_WALL govern only the --stage train/both science route."
        ),
    )
    parser.add_argument(
        "--run-root-prefix",
        default=None,
        help=(
            f"Required for a pilot payload: the durable, lifecycle-governed run-root prefix "
            f"(falls back to the {launch_config.PILOT_RUN_ROOT_PREFIX_ENV} environment variable)."
        ),
    )
    parser.add_argument(
        "--science-python",
        required=True,
        help=(
            "Absolute path to the science environment's python, embedded literally "
            "into the job wrapper. Not resolved: a venv bin/python symlink is kept "
            "as-is, since resolving it would escape the venv."
        ),
    )
    parser.add_argument(
        "--executor-python",
        required=True,
        help="Absolute path to the executor environment's python; validated the same way.",
    )
    parser.add_argument(
        "--commit",
        required=True,
        help=(
            "Expected checkout commit. Recorded, not enforced here: each job's own "
            "identity block (section B3) asserts it inside the allocation."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact sbatch commands; submit nothing.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Submit (or print, under ``--dry-run``) every lineage's two jobs."""

    args = parse_args(argv)
    run_root = Path(args.run_root).resolve()
    # F5: no custom-checkout option. run_job.py's in-job identity check always
    # asserts the checkout at RUN/checkout (section B3), so a launch-time
    # override here would submit a job whose wrapper and whose own identity
    # check disagree about where the checkout lives.
    checkout_root = run_root / "checkout"
    science_python = _validate_interpreter(args.science_python, label="--science-python")
    executor_python = _validate_interpreter(args.executor_python, label="--executor-python")
    if args.payload in PILOT_PAYLOADS:
        if args.stage not in ("train", "eval"):
            raise LaunchError(
                f"--stage train or --stage eval is required for payload {args.payload!r} "
                "(TO 63690220/8a5e926f); a pilot payload never submits 'both'"
            )
        if args.stage == "train":
            if args.pilot_stage is not None:
                raise LaunchError("--pilot-stage is only valid with --stage eval")
            run_root_prefix = args.run_root_prefix or os.environ.get(launch_config.PILOT_RUN_ROOT_PREFIX_ENV)
            if not run_root_prefix:
                raise LaunchError(
                    f"--run-root-prefix or {launch_config.PILOT_RUN_ROOT_PREFIX_ENV} is required for "
                    f"payload {args.payload!r}"
                )
            # F6: an explicit, smaller TRAIN wall time is honored (via
            # :func:`submit_pilot_payload`'s own cap check), not silently
            # replaced by the payload's maximum.
            train_id = submit_pilot_payload(
                args.payload,
                science_python=science_python,
                executor_python=executor_python,
                run_root=run_root,
                checkout_root=checkout_root,
                run_root_prefix=run_root_prefix,
                t_wall=os.environ.get("T_WALL"),
                commit=args.commit,
                dry_run=args.dry_run,
            )
            print(f"[h-v2] {launch_config.PILOT_PAYLOAD_LINEAGES[args.payload]}: train={train_id}")
            return 0
        # --stage eval (TO 8a5e926f): standalone EVAL-fanout submission, one
        # sbatch call, never auto-chained after another stage.
        if not args.expected_train_commit:
            raise LaunchError("--expected-train-commit is required for --stage eval")
        if not args.time:
            raise LaunchError("--time is required for --stage eval")
        # F5/D6 (review r2 delta): the pilot/part-(g) EVAL route must pass
        # the same run-root containment guard --stage train already
        # enforces, on EVERY submission path, before any sbatch call -- an
        # omitted prefix is refused rather than silently admitting a
        # disposable root (the guard is never optional).
        pilot_run_root_prefix = args.run_root_prefix or os.environ.get(launch_config.PILOT_RUN_ROOT_PREFIX_ENV)
        if not pilot_run_root_prefix:
            raise LaunchError(
                f"--run-root-prefix or {launch_config.PILOT_RUN_ROOT_PREFIX_ENV} is required for "
                f"payload {args.payload!r} --stage eval"
            )
        _validate_run_root_prefix(
            run_root, pilot_run_root_prefix, payload=args.payload,
            lineage=launch_config.PILOT_PAYLOAD_LINEAGES[args.payload],
        )
        if args.payload == "capacity-pilot":
            if args.pilot_stage is None:
                raise LaunchError("--pilot-stage {1,2,2-core} is required for capacity-pilot --stage eval")
            if args.pilot_stage in ("2", "2-core"):
                _validate_pilot_stage2_ready(run_root, pilot_stage=args.pilot_stage)
        elif args.pilot_stage is not None:
            raise LaunchError(f"--pilot-stage is not supported for payload {args.payload!r}")
        eval_command = build_eval_fanout_sbatch_command(
            args.payload,
            pilot_stage=args.pilot_stage,
            expected_train_commit=args.expected_train_commit,
            science_python=science_python,
            executor_python=executor_python,
            run_root=run_root,
            checkout_root=checkout_root,
            time_value=args.time,
            dependency=None,
            commit=args.commit,
            resources=launch_config.PILOT_PAYLOAD_RESOURCES[args.payload],
        )
        if args.dry_run:
            print(shlex.join(eval_command))
            return 0
        eval_id = _submit(eval_command, runner=_default_runner)
        print(f"[h-v2] {launch_config.PILOT_PAYLOAD_LINEAGES[args.payload]}: eval={eval_id}")
        return 0

    # science / pipeline-check
    if args.stage == "train":
        if args.pilot_stage is not None:
            raise LaunchError(f"--pilot-stage is not supported for payload {args.payload!r}")
        t_wall = os.environ.get("T_WALL")
        if not t_wall:
            raise LaunchError("T_WALL environment variable is required")
        # F5: a single-stage TRAIN-only submission bypassed the chained
        # route's wall-budget check; it alone must still fit the per-lineage
        # budget.
        _validate_single_stage_wall_seconds(t_wall, label="T_WALL")
        # D6 (review r2 delta): the guard is REQUIRED on every submission
        # path, never optional just because this is a single-stage route.
        science_run_root_prefix = args.run_root_prefix or os.environ.get(launch_config.PILOT_RUN_ROOT_PREFIX_ENV)
        if not science_run_root_prefix:
            raise LaunchError(
                f"--run-root-prefix or {launch_config.PILOT_RUN_ROOT_PREFIX_ENV} is required for "
                f"payload {args.payload!r} --stage train"
            )
        _validate_run_root_prefix(run_root, science_run_root_prefix, payload=args.payload)
        for lineage in launch_config.LINEAGES:
            _validate_science_lineage(args.payload, lineage)
        for lineage in launch_config.LINEAGES:
            command = build_sbatch_command(
                "train", lineage, payload=args.payload, science_python=science_python,
                executor_python=executor_python, run_root=run_root, checkout_root=checkout_root,
                time_value=t_wall, dependency=None, commit=args.commit,
            )
            if args.dry_run:
                print(shlex.join(command))
                continue
            job_id = _submit(command, runner=_default_runner)
            print(f"[h-v2] {lineage}: train={job_id}")
        return 0
    if args.stage == "eval":
        if args.pilot_stage is not None:
            raise LaunchError(f"--pilot-stage is not supported for payload {args.payload!r}")
        if not args.expected_train_commit:
            raise LaunchError("--expected-train-commit is required for --stage eval")
        if not args.time:
            raise LaunchError("--time is required for --stage eval")
        # F5/D6: same single-stage wall-budget and MANDATORY run-root
        # containment checks as --stage train, applied to the standalone
        # EVAL route.
        _validate_single_stage_wall_seconds(args.time, label="--time")
        science_run_root_prefix = args.run_root_prefix or os.environ.get(launch_config.PILOT_RUN_ROOT_PREFIX_ENV)
        if not science_run_root_prefix:
            raise LaunchError(
                f"--run-root-prefix or {launch_config.PILOT_RUN_ROOT_PREFIX_ENV} is required for "
                f"payload {args.payload!r} --stage eval"
            )
        _validate_run_root_prefix(run_root, science_run_root_prefix, payload=args.payload)
        for lineage in launch_config.LINEAGES:
            _validate_science_lineage(args.payload, lineage)
        for lineage in launch_config.LINEAGES:
            command = build_eval_fanout_sbatch_command(
                args.payload, lineage=lineage, expected_train_commit=args.expected_train_commit,
                science_python=science_python, executor_python=executor_python, run_root=run_root,
                checkout_root=checkout_root, time_value=args.time, dependency=None, commit=args.commit,
                resources=launch_config.RESOURCES,
            )
            if args.dry_run:
                print(shlex.join(command))
                continue
            job_id = _submit(command, runner=_default_runner)
            print(f"[h-v2] {lineage}: eval={job_id}")
        return 0

    # --stage omitted or "both": UNCHANGED shape -- exactly one train line and
    # one chained-afterok eval line per lineage (B3: the science dry run
    # stays exactly 6 lines for 3 lineages).
    if args.pilot_stage is not None:
        raise LaunchError(f"--pilot-stage is not supported for payload {args.payload!r}")
    # K3: required unconditionally, dry-run included -- _validate_wall_budget
    # cannot parse the prior placeholder values ("<T_WALL>"/"<E_WALL>"), so a
    # dry run without both set always raised "malformed wall-time value"
    # anyway; that dead placeholder path is removed rather than fixed.
    t_wall = os.environ.get("T_WALL")
    e_wall = os.environ.get("E_WALL")
    if not t_wall or not e_wall:
        raise LaunchError("T_WALL and E_WALL environment variables are required")
    # D6 (review r2 delta): F5's run-root guard reached every OTHER
    # submission path but never this chained one -- the worst gap, since it
    # is the default route (``--stage`` omitted). Required and enforced
    # here too, before any sbatch call.
    chained_run_root_prefix = args.run_root_prefix or os.environ.get(launch_config.PILOT_RUN_ROOT_PREFIX_ENV)
    if not chained_run_root_prefix:
        raise LaunchError(
            f"--run-root-prefix or {launch_config.PILOT_RUN_ROOT_PREFIX_ENV} is required for "
            f"payload {args.payload!r}"
        )
    _validate_run_root_prefix(run_root, chained_run_root_prefix, payload=args.payload)
    results = submit_all(
        payload=args.payload,
        science_python=science_python,
        executor_python=executor_python,
        run_root=run_root,
        checkout_root=checkout_root,
        t_wall=t_wall,
        e_wall=e_wall,
        commit=args.commit,
        dry_run=args.dry_run,
    )
    for lineage, (train_id, eval_id) in results.items():
        print(f"[h-v2] {lineage}: train={train_id} eval={eval_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
