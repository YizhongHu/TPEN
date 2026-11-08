"""Public command and evidence clients for configured TPEN child processes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from ._codec import canonical_bytes, decode_json
from .environments import PreparedPythonEnvironment
from ._records import frozen_record
from .workers import SubprocessWorker

RUN_EVIDENCE_SCHEMA = "tpen.run-evidence/v1"
CHILD_CONTEXT_KEYS_ENV = "TPEN_EXECUTOR_CONTEXT_ENV_KEYS"
CONTEXT_ENV_KEY_PATTERN = r"^(CUDA_VISIBLE_DEVICES|TPEN_CTX_[A-Z0-9_]{1,64})$"
RUN_SHIM = (
    "import json, os, re, socket, sys\n"
    "from pathlib import Path\n"
    "from datetime import datetime, timezone\n"
    "try:\n"
    f"    context_keys = json.loads(os.environ.get('{CHILD_CONTEXT_KEYS_ENV}', '[]'))\n"
    "except (TypeError, ValueError):\n"
    "    context_keys = []\n"
    f"context_keys = [key for key in context_keys if type(key) is str and re.fullmatch({CONTEXT_ENV_KEY_PATTERN!r}, key)] if isinstance(context_keys, list) else []\n"
    "observation_keys = sorted(set(context_keys) | {'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'})\n"
    "observed_environment = {key: os.environ.get(key) for key in observation_keys}\n"
    "run_roots = [arg.split('=', 1)[1] for arg in sys.argv[1:] if arg.startswith('run.root=')]\n"
    "placement_path = os.environ.get('TPEN_PLACEMENT_OUTPUT')\n"
    "placement_pid = os.getpid()\n"
    "placement_host = socket.gethostname()\n"
    "if run_roots and placement_path:\n"
    "    placement_refusal = None\n"
    "    placement_ticks = 0\n"
    "    placement_affinity = []\n"
    "    placement_stamp = datetime.now(timezone.utc).isoformat()\n"
    "    try:\n"
    "        raw_cpus = os.environ.get('TPEN_PLACEMENT_DECLARED_CPUS')\n"
    "        if raw_cpus is not None:\n"
    "            if not raw_cpus:\n"
    "                raise ValueError('declared CPU set is empty')\n"
    "            declared_cpus = tuple(sorted({int(part) for part in raw_cpus.split(',')}))\n"
    "            if not declared_cpus or any(cpu < 0 for cpu in declared_cpus):\n"
    "                raise ValueError('declared CPU set is invalid')\n"
    "            os.sched_setaffinity(0, declared_cpus)\n"
    "        placement_affinity = sorted(os.sched_getaffinity(0))\n"
    "        stat_text = open(f'/proc/{placement_pid}/stat', encoding='utf-8').read()\n"
    "        stat_end = stat_text.rfind(')')\n"
    "        stat_fields = stat_text[stat_end + 1:].strip().split() if stat_end >= 0 else []\n"
    "        if len(stat_fields) <= 19:\n"
    "            raise ValueError('process stat lacks start ticks')\n"
    "        placement_ticks = int(stat_fields[19])\n"
    "        declared_host = os.environ.get('TPEN_PLACEMENT_DECLARED_HOST')\n"
    "        if declared_host and placement_host != declared_host:\n"
    "            raise RuntimeError(f'child host {placement_host!r} differs from declared host {declared_host!r}')\n"
    "    except Exception as error:\n"
    "        placement_refusal = {'code': 'child-placement-refused', 'reason': str(error)}\n"
    "    observation = {\n"
    "        'schema': 'tpen-executor/child-placement-v1',\n"
    "        'task_id': os.environ.get('TPEN_PLACEMENT_TASK_ID', ''),\n"
    "        'attempt_id': os.environ.get('TPEN_PLACEMENT_ATTEMPT_ID', ''),\n"
    "        'group_digest': os.environ.get('TPEN_PLACEMENT_GROUP_DIGEST', ''),\n"
    "        'allocation_id': os.environ.get('TPEN_PLACEMENT_ALLOCATION_ID', ''),\n"
    "        'backend_job_id': os.environ.get('TPEN_PLACEMENT_BACKEND_JOB_ID', ''),\n"
    "        'slot_id': os.environ.get('TPEN_PLACEMENT_SLOT_ID', ''),\n"
    "        'payload_digest': os.environ.get('TPEN_PLACEMENT_PAYLOAD_DIGEST', ''),\n"
    "        'output_dir': os.environ.get('TPEN_PLACEMENT_OUTPUT_DIR', run_roots[-1]),\n"
    "        'host': placement_host,\n"
    "        'pid': placement_pid,\n"
    "        'proc_start_ticks': placement_ticks,\n"
    "        'affinity': placement_affinity,\n"
    "        'thread_settings': {name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},\n"
    "        'cuda_visible_devices': observed_environment.get('CUDA_VISIBLE_DEVICES'),\n"
    "        'observed_at_utc': placement_stamp,\n"
    "        'refusal': placement_refusal,\n"
    "    }\n"
    "    if placement_refusal is not None:\n"
    "        print(f\"child-placement-refused: {placement_refusal['reason']}\", file=sys.stderr)\n"
    "    try:\n"
    "        Path(placement_path).write_text(json.dumps(observation, sort_keys=True) + '\\n', encoding='utf-8')\n"
    "    except OSError as error:\n"
    "        print(f\"child-placement-refused: could not write sidecar: {error}\", file=sys.stderr)\n"
    "        raise SystemExit(2)\n"
    "    Path(run_roots[-1], 'executor-child-observation.json').write_text(json.dumps({'schema': 'tpen-executor/child-observation-v1', 'cuda_visible_devices': observed_environment.get('CUDA_VISIBLE_DEVICES'), 'environment': observed_environment, 'thread_settings': {name: observed_environment.get(name) for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}, 'hostname': placement_host, 'pid': placement_pid}, sort_keys=True) + '\\n', encoding='utf-8')\n"
    "    if placement_refusal is not None:\n"
    "        raise SystemExit(2)\n"
    "from tpen.run import main\n"
    "raise SystemExit(main(sys.argv[1:]))"
)
REFUSAL_CODES = frozenset(
    {
        "missing-run-dir", "missing-status", "multiple-status", "malformed-status",
        "status-io-error", "incomplete-checkpoint", "corrupt-checkpoint",
        "checkpoint-io-error", "corrupt-publication-catalog", "torn-publication-catalog",
        "missing-checkpoint", "no-train-resume-checkpoint",
    }
)


class EvidenceUnavailable(RuntimeError):
    """Raised when a child cannot provide trustworthy public evidence."""


@frozen_record
class CoreRunEvidence:
    """Strict, immutable executor-owned mirror of core run evidence.

    Parameters
    ----------
    schema, run_dir, verdict : str
        Validated public evidence fields.
    run_status : str or None
        Core-reported run status.
    checkpoint, refusal : mapping or None
        Validated nested evidence, copied into read-only mappings.
    """

    schema: str
    run_dir: str
    verdict: str
    run_status: str | None
    checkpoint: Mapping[str, Any] | None
    refusal: Mapping[str, str] | None

    SCHEMA: ClassVar[str] = RUN_EVIDENCE_SCHEMA
    VERDICTS: ClassVar[frozenset[str]] = frozenset({"completed", "incomplete", "failed", "refused"})
    REFUSAL_CODES: ClassVar[frozenset[str]] = REFUSAL_CODES

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", self._checkpoint(self.checkpoint))
        object.__setattr__(self, "refusal", self._refusal(self.refusal))

    @classmethod
    def from_mapping(cls, value: Any) -> "CoreRunEvidence":
        """Decode one strict public evidence mapping.

        Parameters
        ----------
        value : mapping
            JSON-shaped public evidence.

        Returns
        -------
        CoreRunEvidence
            A validated record with frozen nested mappings.

        Raises
        ------
        EvidenceUnavailable
            If the schema, keys, types, or verdict invariants are invalid.
        """

        if type(value) is not dict:
            raise EvidenceUnavailable("evidence is not a JSON object")
        expected = {"schema", "run_dir", "verdict", "run_status", "checkpoint", "refusal"}
        if set(value) != expected:
            raise EvidenceUnavailable("evidence has unexpected keys")
        if value["schema"] != cls.SCHEMA:
            raise EvidenceUnavailable("unknown evidence schema")
        if type(value["run_dir"]) is not str or not value["run_dir"]:
            raise EvidenceUnavailable("run_dir must be a non-empty string")
        verdict = value["verdict"]
        if type(verdict) is not str or verdict not in cls.VERDICTS:
            raise EvidenceUnavailable("unknown evidence verdict")
        run_status = value["run_status"]
        if run_status is not None and type(run_status) is not str:
            raise EvidenceUnavailable("run_status must be a string or null")
        checkpoint = value["checkpoint"]
        refusal = value["refusal"]
        if verdict == "refused" and refusal is None:
            raise EvidenceUnavailable("refused evidence requires refusal")
        if verdict == "refused" and checkpoint is not None:
            raise EvidenceUnavailable("refused evidence cannot contain checkpoint")
        if verdict == "completed" and checkpoint is None:
            raise EvidenceUnavailable("completed evidence requires checkpoint")
        if verdict != "refused" and refusal is not None:
            raise EvidenceUnavailable("non-refused evidence cannot contain refusal")
        return cls(value["schema"], value["run_dir"], verdict, run_status, checkpoint, refusal)

    def __hash__(self) -> int:
        """Hash the canonical JSON representation used by equality fields."""

        return hash(canonical_bytes({
            "schema": self.schema,
            "run_dir": self.run_dir,
            "verdict": self.verdict,
            "run_status": self.run_status,
            "checkpoint": self.checkpoint,
            "refusal": self.refusal,
        }))

    @staticmethod
    def _checkpoint(value: Any) -> MappingProxyType | None:
        if value is None:
            return None
        keys = {"checkpoint_dir", "schema_version", "next_iteration", "completed_updates"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise EvidenceUnavailable("checkpoint has invalid keys")
        if type(value["checkpoint_dir"]) is not str:
            raise EvidenceUnavailable("checkpoint text fields have invalid types")
        if type(value["schema_version"]) is not int:
            raise EvidenceUnavailable("checkpoint schema_version must be an integer")
        if type(value["next_iteration"]) is not int or type(value["completed_updates"]) is not int:
            raise EvidenceUnavailable("checkpoint counters have invalid types")
        if value["next_iteration"] < 0 or value["completed_updates"] < 0:
            raise EvidenceUnavailable("checkpoint counters must be nonnegative")
        return MappingProxyType(dict(value))

    @classmethod
    def _refusal(cls, value: Any) -> MappingProxyType | None:
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != {"code", "reason"}:
            raise EvidenceUnavailable("refusal has invalid keys")
        if type(value["code"]) is not str or value["code"] not in cls.REFUSAL_CODES:
            raise EvidenceUnavailable("unknown refusal code")
        if type(value["reason"]) is not str:
            raise EvidenceUnavailable("refusal reason must be a string")
        return MappingProxyType({"code": value["code"], "reason": value["reason"]})


@frozen_record
class TPENCommand:
    """Command specification for the public configured-run entrypoint.

    Parameters
    ----------
    config : str
        Absolute path to the Hydra configuration.
    overrides : sequence of str, optional
        Positional dot-list overrides passed unchanged to core.

    Raises
    ------
    TypeError
        If overrides is a scalar string/bytes value or contains non-strings.
    ValueError
        If config is not an absolute path.
    """

    config: str
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.config) is not str or not os.path.isabs(self.config):
            raise ValueError("config must be an absolute path")
        if isinstance(self.overrides, (str, bytes, bytearray)):
            raise TypeError("overrides must be a sequence of strings")
        if any(type(item) is not str for item in self.overrides):
            raise TypeError("overrides must be strings")
        object.__setattr__(self, "overrides", tuple(self.overrides))

    @property
    def args(self) -> tuple[str, ...]:
        """Return arguments following the selected Python interpreter."""

        return ("-c", RUN_SHIM, "--config", self.config, *self.overrides)


@frozen_record
class TPENEvidence:
    """Read public run evidence through a child inspection process.

    Notes
    -----
    Inspection output is obtained only from the named stdout log.  The
    decoder rejects duplicate JSON members at every depth and never opens
    ``run_dir`` directly.
    """

    def read(
        self,
        worker: SubprocessWorker,
        environment: PreparedPythonEnvironment,
        run_dir: str | os.PathLike[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        stdout_path: str | os.PathLike[str],
        stderr_path: str | os.PathLike[str],
        timeout: float | None = None,
    ) -> CoreRunEvidence:
        """Launch inspection and decode exactly one JSON object.

        Parameters
        ----------
        worker : SubprocessWorker
            Worker used to launch the selected science interpreter.
        environment : PreparedPythonEnvironment
            Prepared environment consumed by the worker.
        run_dir : str or os.PathLike
            Run directory passed to public inspection; it is not read here.
        cwd : str or os.PathLike or None, optional
            Child working directory.
        stdout_path, stderr_path : str or os.PathLike
            Required named logs.
        timeout : float or None, optional
            Timeout in seconds.

        Returns
        -------
        CoreRunEvidence
            Frozen, validated public evidence.

        Raises
        ------
        EvidenceUnavailable
            If transport, process, decoding, or evidence validation fails.

        Notes
        -----
        The worker applies the owner environment, then the descriptor overlay,
        then the setup script; setup is the last writer. The timeout covers the
        worker's complete process group.
        """

        try:
            outcome = worker.run(
                environment,
                ("-m", "tpen.inspection", os.fspath(run_dir)),
                cwd=cwd,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout=timeout,
            )
        except OSError as exc:
            raise EvidenceUnavailable("inspection process could not be launched") from exc
        if outcome.timed_out or outcome.returncode != 0:
            raise EvidenceUnavailable("inspection process did not complete successfully")
        try:
            payload = Path(outcome.stdout_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise EvidenceUnavailable("inspection stdout could not be read") from exc
        if not payload.strip():
            raise EvidenceUnavailable("inspection stdout is empty")
        try:
            decoded = decode_json(payload)
        except (TypeError, ValueError) as exc:
            raise EvidenceUnavailable("inspection stdout is not one JSON object") from exc
        return CoreRunEvidence.from_mapping(decoded)


__all__ = ["CHILD_CONTEXT_KEYS_ENV", "CONTEXT_ENV_KEY_PATTERN", "CoreRunEvidence", "EvidenceUnavailable", "REFUSAL_CODES", "RUN_EVIDENCE_SCHEMA", "TPENCommand", "TPENEvidence"]
