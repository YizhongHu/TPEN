"""Sequential local dispatch for one already-attached allocation."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from typing import Any

from .._codec import digest
from ..contracts import ReadyGroup
from ..lifecycle.attached import AttachedAllocation
from ..programs import WorkerProgram
from ..results import RunManifest
from ..runners import CoreRunEvidence, EvidenceUnavailable
from ..tasks import ConfiguredRun


_ACTIVE_DISPATCH: ContextVar[bool] = ContextVar("tpen_executor_local_dispatch", default=False)


def _lexical(path: str | os.PathLike[str]) -> str:
    """Return an absolute normalized path without resolving symlinks."""

    return os.path.abspath(os.path.normpath(os.fspath(path)))


def _inside(root: str, path: str) -> bool:
    """Return whether ``path`` is lexically below ``root``."""

    return path == root or path.startswith(root + os.sep)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for lifecycle receipts."""

    return datetime.now(timezone.utc).isoformat()


def _evidence_payload(evidence: CoreRunEvidence) -> dict[str, Any]:
    """Return the canonical JSON-shaped form of one decoded evidence record."""

    if not isinstance(evidence, CoreRunEvidence):
        raise TypeError("evidence reader must return CoreRunEvidence")
    return {
        "schema": evidence.schema,
        "run_dir": evidence.run_dir,
        "verdict": evidence.verdict,
        "run_status": evidence.run_status,
        "checkpoint": evidence.checkpoint,
        "refusal": evidence.refusal,
    }


def _member_paths(fence: Any) -> tuple[str, str, str, str, str, str]:
    """Validate and derive all dispatcher-owned paths for one attempt."""

    root = _lexical(fence.root)
    output_dir = _lexical(fence.output_dir)
    if not _inside(root, output_dir) or output_dir == root:
        raise RuntimeError("attempt output directory escapes the allocation store root")
    core_dir = _lexical(os.path.join(output_dir, "core"))
    if not _inside(output_dir, core_dir):
        raise RuntimeError("core run directory escapes the attempt fence")
    stdout_path = _lexical(os.path.join(output_dir, "stdout.log"))
    stderr_path = _lexical(os.path.join(output_dir, "stderr.log"))
    evidence_stdout = _lexical(os.path.join(output_dir, "evidence.stdout.log"))
    evidence_stderr = _lexical(os.path.join(output_dir, "evidence.stderr.log"))
    for path in (stdout_path, stderr_path, evidence_stdout, evidence_stderr):
        if not _inside(output_dir, path):
            raise RuntimeError("dispatcher log path escapes the attempt fence")
    return output_dir, core_dir, stdout_path, stderr_path, evidence_stdout, evidence_stderr


def _status(process_outcome: Any) -> str:
    """Map one reaped child outcome to the closed operational vocabulary."""

    if type(getattr(process_outcome, "timed_out", None)) is not bool:
        raise TypeError("process outcome has an invalid timed_out flag")
    return "timed_out" if process_outcome.timed_out else (
        "completed" if process_outcome.returncode == 0 else "failed"
    )


@dataclass(frozen=True, slots=True)
class LocalDispatcher:
    """Run independent group members sequentially in one attached allocation.

    Parameters
    ----------
    worker : WorkerProgram
        Unresolved worker handle. It is materialized in the owner process for
        each member using the environment supplied to :meth:`run`.

    Notes
    -----
    This baseline intentionally has no concurrency, retry, recovery, scheduler,
    Parsl, or Balsam behavior. The public core run-directory overrides are
    ``run.root=<attempt output>``, ``run.layout=flat``, and ``run.run_id=core``.
    They keep core's derived run directory below the store-owned attempt fence.
    A concrete store root containing the literal ``${`` is refused before the
    first ``begin`` call so unresolved OmegaConf syntax cannot redirect core.
    A timeout, when supplied, must be finite and positive. Cancellation makes
    one best-effort ``finish`` publication for the begun member, preserves a
    known ``timed_out`` status (otherwise using ``failed``), and propagates the
    original cancellation after worker cleanup.
    """

    worker: WorkerProgram

    def __post_init__(self) -> None:
        if not callable(getattr(self.worker, "materialize", None)):
            raise TypeError("worker must be a materializable WorkerProgram handle")

    def run(
        self,
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> RunManifest:
        """Dispatch every admitted member and return the durable group manifest.

        Parameters
        ----------
        group : ReadyGroup
            Ordered group whose members are launched in order.
        runs : mapping[str, ConfiguredRun]
            Exactly one configured run for every group member.
        lifecycle : AttachedAllocation
            Attached allocation that owns the fences and receipt store.
        env : mapping, optional
            Owner-process materialization environment for the worker handle.
        timeout : float or None, optional
            Positive child and evidence timeout in seconds.

        Returns
        -------
        RunManifest
            Manifest rebuilt from the lifecycle store after all members finish.

        Raises
        ------
        TypeError, ValueError
            If the group, run mapping, lifecycle, environment, or timeout is
            malformed. Admission happens before the first store write. A
            literal ``${`` in the concrete store root is also rejected before
            ``begin``; the timeout must be finite and positive when present.
        RuntimeError
            If dispatch is re-entered or a fenced path is not lexically safe.
        Exception
            The first member-local ``begin`` exception is deferred until all
            other members have been visited. If a later ``finish`` exception
            also occurs, it is attached as a note to that same original begin
            exception rather than replacing it.
        """

        if _ACTIVE_DISPATCH.get():
            raise RuntimeError("LocalDispatcher.run is not reentrant")
        token = _ACTIVE_DISPATCH.set(True)
        try:
            self._validate_inputs(group, runs, lifecycle, env, timeout)
            begin_error: Exception | None = None
            finish_error: Exception | None = None
            for member in group.members:
                configured = runs[member.task_id]
                try:
                    fence = lifecycle.begin(group, member.task_id, _utc_now())
                except Exception as error:
                    if begin_error is None:
                        begin_error = error
                    continue
                result_digest: str | None = None
                status = "launch_failed"
                try:
                    worker = self.worker.materialize(env)
                    environment = worker["environment"]
                    process = worker["process"]
                    worker["runner"]
                    evidence = worker["evidence"]
                    if not callable(getattr(process, "run", None)):
                        raise TypeError("materialized worker process lacks run()")
                    if not callable(getattr(evidence, "read", None)):
                        raise TypeError("materialized worker evidence reader lacks read()")
                    output_dir, core_dir, stdout_path, stderr_path, evidence_stdout, evidence_stderr = _member_paths(fence)
                    args = (*configured.command().args, f"run.root={output_dir}", "run.layout=flat", "run.run_id=core")
                    outcome = process.run(
                        environment,
                        args,
                        cwd=output_dir,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        timeout=timeout,
                    )
                    status = _status(outcome)
                    try:
                        decoded = evidence.read(
                            process,
                            environment,
                            core_dir,
                            cwd=output_dir,
                            stdout_path=evidence_stdout,
                            stderr_path=evidence_stderr,
                            timeout=timeout,
                        )
                    except EvidenceUnavailable:
                        result_digest = None
                    else:
                        result_digest = digest(_evidence_payload(decoded))
                except EvidenceUnavailable:
                    # Evidence absence never changes the process outcome or
                    # fabricates a digest.
                    result_digest = None
                except Exception:
                    status = "launch_failed"
                    result_digest = None
                except BaseException:
                    # Cancellation and interpreter exits still receive the
                    # one finish attempt before propagating to the caller.
                    if status != "timed_out":
                        status = "failed"
                    result_digest = None
                    try:
                        lifecycle.finish(fence, status, _utc_now(), result_digest)
                    except BaseException:
                        pass
                    raise
                try:
                    lifecycle.finish(fence, status, _utc_now(), result_digest)
                except Exception as error:
                    if finish_error is None:
                        finish_error = error
            if begin_error is not None:
                if finish_error is not None:
                    begin_error.add_note(f"finish error after deferred begin failure: {finish_error!r}")
                raise begin_error
            if finish_error is not None:
                raise finish_error
            return lifecycle.store.manifest(group)
        finally:
            _ACTIVE_DISPATCH.reset(token)

    @staticmethod
    def _validate_inputs(
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None,
        timeout: float | None,
    ) -> None:
        """Validate all admission inputs before touching the lifecycle store."""

        if not isinstance(group, ReadyGroup):
            raise TypeError("group must be ReadyGroup")
        if isinstance(runs, (str, bytes, bytearray)) or not isinstance(runs, Mapping):
            raise ValueError("runs must be a mapping of task IDs to ConfiguredRun")
        if not callable(getattr(lifecycle, "begin", None)) or not callable(getattr(lifecycle, "finish", None)):
            raise TypeError("lifecycle must provide begin() and finish()")
        if not hasattr(lifecycle, "store") or not callable(getattr(lifecycle.store, "manifest", None)):
            raise TypeError("lifecycle must expose a manifest-capable store")
        if env is not None and (isinstance(env, (str, bytes, bytearray)) or not isinstance(env, Mapping)):
            raise TypeError("env must be a mapping or None")
        if timeout is not None and (
            type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive or None")
        store_root = getattr(lifecycle.store, "root", None)
        if isinstance(store_root, (str, os.PathLike)) and "${" in os.fspath(store_root):
            raise ValueError("store root must not contain OmegaConf interpolation syntax '${'")
        expected = {member.task_id for member in group.members}
        actual = set(runs)
        if actual != expected:
            raise ValueError("runs keys must match group member task IDs exactly")
        for task_id, configured in runs.items():
            if not isinstance(configured, ConfiguredRun):
                raise TypeError("runs values must be ConfiguredRun records")
            if configured.id != task_id:
                raise ValueError("ConfiguredRun.id must match its mapping task ID")


__all__ = ["LocalDispatcher"]
