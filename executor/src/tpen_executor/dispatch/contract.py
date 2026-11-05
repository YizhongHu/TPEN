"""Shared lifecycle contract for attached executor dispatchers."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from datetime import datetime, timezone
import math
import os
from typing import Any

from .._codec import digest
from ..contracts import ReadyGroup
from ..lifecycle.attached import AttachedAllocation
from ..results import RunManifest
from ..runners import CoreRunEvidence, EvidenceUnavailable
from ..tasks import ConfiguredRun


_ACTIVE_DISPATCH: ContextVar[bool] = ContextVar("tpen_executor_dispatch", default=False)


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


class DispatcherContract:
    """Implement admission, fencing, outcome, evidence, and segment semantics.

    Concrete dispatchers only provide :meth:`_execute_process`.  This keeps
    backend framework state out of the durable executor contract and ensures
    that Local and Parsl dispatchers publish identical records.
    """

    worker: Any

    def _worker_environment(self, materialized: Any) -> Any:
        """Return the prepared environment used by the child and inspector."""

        return materialized["environment"]

    def _execute_process(
        self,
        materialized: Any,
        environment: Any,
        args: tuple[str, ...],
        *,
        cwd: str,
        stdout_path: str,
        stderr_path: str,
        timeout: float | None,
    ) -> Any:
        """Run one child in the backend represented by ``self``."""

        process = materialized["process"]
        return process.run(
            environment,
            args,
            cwd=cwd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=timeout,
        )

    def validate(
        self,
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> None:
        """Perform common admission without touching backend or store state."""

        self._validate_inputs(group, runs, lifecycle, env, timeout)

    def _begin_dispatch(
        self,
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None,
        timeout: float | None,
    ) -> Any:
        """Validate inputs and reserve the process-local dispatch slot."""

        if _ACTIVE_DISPATCH.get():
            raise RuntimeError("dispatcher.run is not reentrant")
        self.validate(group, runs, lifecycle, env, timeout)
        return _ACTIVE_DISPATCH.set(True)

    @staticmethod
    def _end_dispatch(token: Any) -> None:
        """Release a dispatch slot acquired by :meth:`_begin_dispatch`."""

        _ACTIVE_DISPATCH.reset(token)

    def run(
        self,
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        *,
        _admitted: bool = False,
    ) -> RunManifest:
        """Dispatch every admitted member and return its durable group manifest.

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
            malformed. Admission happens before the first store write.
        RuntimeError
            If dispatch is re-entered or a fenced path is not lexically safe.

        Notes
        -----
        Every begun fence receives one best-effort finish on cancellation.
        Child return code and timeout determine operational status; backend
        future success is never used as a completion signal.

        Admission is completed before the first lifecycle write. Ordinary
        member failures are recorded and processing continues to later members.
        """

        if _admitted:
            if not _ACTIVE_DISPATCH.get():
                raise RuntimeError("admitted dispatch requires an active dispatch slot")
            return self._run_admitted(group, runs, lifecycle, env, timeout)
        token = self._begin_dispatch(group, runs, lifecycle, env, timeout)
        try:
            return self._run_admitted(group, runs, lifecycle, env, timeout)
        finally:
            self._end_dispatch(token)

    def _run_admitted(
        self,
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None,
        timeout: float | None,
    ) -> RunManifest:
        """Run the shared lifecycle after admission and guard setup."""

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
            segment_evidence: CoreRunEvidence | None = None
            status = "launch_failed"
            try:
                materialized = self.worker.materialize(env)
                environment = self._worker_environment(materialized)
                process = materialized["process"]
                materialized["runner"]
                evidence = materialized["evidence"]
                if not callable(getattr(process, "run", None)):
                    raise TypeError("materialized worker process lacks run()")
                if not callable(getattr(evidence, "read", None)):
                    raise TypeError("materialized worker evidence reader lacks read()")
                output_dir, core_dir, stdout_path, stderr_path, evidence_stdout, evidence_stderr = _member_paths(fence)
                args = (*configured.command().args, f"run.root={output_dir}", "run.layout=flat", "run.run_id=core")
                outcome = self._execute_process(
                    materialized,
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
                    segment_evidence = decoded
            except Exception:
                status = "launch_failed"
                result_digest = None
            except BaseException as error:
                # A begun fence is terminalized exactly once before any
                # cancellation escapes. Preserve a known timeout because it
                # is more specific than the later evidence interruption.
                if status != "timed_out":
                    status = "failed"
                try:
                    lifecycle.finish(fence, status, _utc_now(), result_digest)
                except BaseException as finish_exc:
                    try:
                        error.add_note(f"finish error during cancellation: {finish_exc!r}")
                    except BaseException:
                        pass
                raise
            try:
                lifecycle.finish(fence, status, _utc_now(), result_digest)
            except Exception as error:
                if finish_error is None:
                    finish_error = error
            if (
                status == "completed"
                and segment_evidence is not None
                and segment_evidence.verdict in {"completed", "incomplete"}
                and segment_evidence.checkpoint is not None
            ):
                existing_segments = lifecycle.store.manifest(group).segments
                if not any(segment.task_id == fence.task_id for segment in existing_segments):
                    try:
                        lifecycle.store.accept_segment(fence, segment_evidence, None)
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


__all__ = ["DispatcherContract"]
