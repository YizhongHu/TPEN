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
from .workers import LaunchPurpose, SubprocessWorker

RUN_EVIDENCE_SCHEMA = "tpen.run-evidence/v1"
RUN_EVIDENCE_SCHEMA_V2 = "tpen.run-evidence/v2"
RUN_EVIDENCE_SCHEMA_V3 = "tpen.run-evidence/v3"
RUN_EVIDENCE_OUTCOMES = frozenset({"completed", "planned_yield", "failed", "unknown", "refused"})
_CHECKPOINT_REF_KEYS = frozenset(
    {
        "schema", "checkpoint_dir", "schema_version", "kind", "next_iteration",
        "completed_updates", "manifest_sha256", "model_sha256", "provenance",
        "content_id", "profile",
    }
)
CHILD_CONTEXT_KEYS_ENV = "TPEN_EXECUTOR_CONTEXT_ENV_KEYS"
CHILD_OBSERVATION_FILENAME = "executor-child-observation.json"
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
    "placement_abort = False\n"
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
    "    except Exception as error:\n"
    "        placement_refusal = {'code': 'child-placement-refused', 'reason': str(error)}\n"
    "        placement_abort = True\n"
    "    declared_host = os.environ.get('TPEN_PLACEMENT_DECLARED_HOST')\n"
    "    host_key = lambda value: value.strip().lower().rstrip('.').split('.', 1)[0]\n"
    "    if declared_host and host_key(placement_host) != host_key(declared_host) and placement_refusal is None:\n"
    "        placement_refusal = {'code': 'child-placement-mismatch', 'reason': f'child host {placement_host!r} differs from declared host {declared_host!r}'}\n"
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
    f"Path(run_roots[-1], {CHILD_OBSERVATION_FILENAME!r}).write_text(json.dumps({{'schema': 'tpen-executor/child-observation-v1', 'cuda_visible_devices': observed_environment.get('CUDA_VISIBLE_DEVICES'), 'environment': observed_environment, 'thread_settings': {{name: observed_environment.get(name) for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}, 'hostname': placement_host, 'pid': placement_pid}}, sort_keys=True) + '\\n', encoding='utf-8') if run_roots else None\n"
    "if placement_abort:\n"
    "    raise SystemExit(2)\n"
    "from tpen.run import main\n"
    "raise SystemExit(main(sys.argv[1:]))"
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
    resolved_config_identity: Mapping[str, str] | None = None
    outcome: str | None = None
    checkpoint_ref: Mapping[str, Any] | None = None

    SCHEMA: ClassVar[str] = RUN_EVIDENCE_SCHEMA
    SCHEMA_V2: ClassVar[str] = RUN_EVIDENCE_SCHEMA_V2
    SCHEMA_V3: ClassVar[str] = RUN_EVIDENCE_SCHEMA_V3
    VERDICTS: ClassVar[frozenset[str]] = frozenset({"completed", "incomplete", "failed", "refused"})
    OUTCOMES: ClassVar[frozenset[str]] = RUN_EVIDENCE_OUTCOMES
    def __post_init__(self) -> None:
        if self.schema not in {RUN_EVIDENCE_SCHEMA, RUN_EVIDENCE_SCHEMA_V2, RUN_EVIDENCE_SCHEMA_V3}:
            raise EvidenceUnavailable(f"unsupported run-evidence schema {self.schema!r}")
        if type(self.run_dir) is not str:
            raise EvidenceUnavailable("run_dir must be a string")
        if self.run_status is not None and not isinstance(self.run_status, str):
            raise EvidenceUnavailable("run_status must be a string or null")
        if self.verdict not in self.VERDICTS:
            raise EvidenceUnavailable(f"unsupported run-evidence verdict {self.verdict!r}")
        object.__setattr__(self, "checkpoint", self._checkpoint(self.checkpoint))
        object.__setattr__(self, "refusal", self._refusal(self.refusal))
        object.__setattr__(self, "resolved_config_identity", self._identity(self.resolved_config_identity))
        object.__setattr__(self, "checkpoint_ref", self._checkpoint_ref(self.checkpoint_ref))
        if self.verdict == "completed":
            if self.checkpoint is None or self.refusal is not None:
                raise EvidenceUnavailable("completed evidence requires checkpoint and no refusal")
        elif self.verdict == "refused":
            if self.refusal is None or self.checkpoint is not None:
                raise EvidenceUnavailable("refused evidence requires refusal and no checkpoint")
        elif self.refusal is not None:
            raise EvidenceUnavailable(f"{self.verdict} evidence cannot contain a refusal")
        if self.schema == RUN_EVIDENCE_SCHEMA_V3:
            if self.resolved_config_identity is None:
                raise EvidenceUnavailable("v3 evidence requires a non-null resolved_config_identity")
            if self.outcome not in self.OUTCOMES:
                raise EvidenceUnavailable("unknown v3 typed outcome")
            if self.verdict != "refused" and self.run_status != self.outcome:
                raise EvidenceUnavailable("v3 run_status must equal outcome")
            if self.outcome == "planned_yield" and self.verdict != "refused":
                if self.checkpoint is None or self.checkpoint_ref is None:
                    raise EvidenceUnavailable("planned_yield evidence requires a checkpoint_ref")
                if self.checkpoint_ref["profile"] != "train_resume":
                    raise EvidenceUnavailable("planned_yield evidence requires a train_resume checkpoint_ref")
                for key in ("checkpoint_dir", "next_iteration", "completed_updates"):
                    if self.checkpoint_ref[key] != self.checkpoint[key]:
                        raise EvidenceUnavailable("planned_yield checkpoint_ref does not match checkpoint summary")
        elif self.outcome is not None or self.checkpoint_ref is not None:
            raise EvidenceUnavailable("typed outcome fields are only valid in v3 evidence")

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

        if not isinstance(value, Mapping):
            raise EvidenceUnavailable("evidence is not a mapping")
        schema = value.get("schema")
        expected = {"schema", "run_dir", "verdict", "run_status", "checkpoint", "refusal"}
        if schema == RUN_EVIDENCE_SCHEMA_V2:
            expected.add("resolved_config_identity")
        elif schema == RUN_EVIDENCE_SCHEMA_V3:
            expected.update({"resolved_config_identity", "outcome", "checkpoint_ref"})
        if schema not in {RUN_EVIDENCE_SCHEMA, RUN_EVIDENCE_SCHEMA_V2, RUN_EVIDENCE_SCHEMA_V3} or set(value) != expected:
            raise EvidenceUnavailable("evidence has unexpected keys")
        if schema in {RUN_EVIDENCE_SCHEMA_V2, RUN_EVIDENCE_SCHEMA_V3} and value.get("resolved_config_identity") is None:
            raise EvidenceUnavailable(f"{schema.rsplit('/', 1)[-1]} evidence requires a non-null resolved_config_identity")
        if value["schema"] not in {RUN_EVIDENCE_SCHEMA, RUN_EVIDENCE_SCHEMA_V2, RUN_EVIDENCE_SCHEMA_V3}:
            raise EvidenceUnavailable("unknown evidence schema")
        if type(value["run_dir"]) is not str:
            raise EvidenceUnavailable("run_dir must be a string")
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
        outcome = value.get("outcome")
        checkpoint_ref = value.get("checkpoint_ref")
        if schema == RUN_EVIDENCE_SCHEMA_V3:
            if type(outcome) is not str or outcome not in cls.OUTCOMES:
                raise EvidenceUnavailable("unknown v3 typed outcome")
            if checkpoint_ref is not None and not isinstance(checkpoint_ref, Mapping):
                raise EvidenceUnavailable("v3 checkpoint_ref must be a mapping or null")
        return cls(
            value["schema"], value["run_dir"], verdict, run_status, checkpoint, refusal,
            value.get("resolved_config_identity"), outcome, checkpoint_ref,
        )

    def __hash__(self) -> int:
        """Hash the canonical JSON representation used by equality fields."""

        return hash(canonical_bytes({
            "schema": self.schema,
            "run_dir": self.run_dir,
            "verdict": self.verdict,
            "run_status": self.run_status,
            "checkpoint": self.checkpoint,
            "refusal": self.refusal,
            "resolved_config_identity": self.resolved_config_identity,
            "outcome": self.outcome,
            "checkpoint_ref": self.checkpoint_ref,
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
        if value["schema_version"] < 0:
            raise EvidenceUnavailable("checkpoint schema_version must be nonnegative")
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
        return MappingProxyType({str(key): str(item) for key, item in value.items()})

    @staticmethod
    def _identity(value: Any) -> MappingProxyType | None:
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != {"schema", "sha256"}:
            raise EvidenceUnavailable("resolved_config_identity has invalid keys")
        if value["schema"] != "tpen-config-identity-v1":
            raise EvidenceUnavailable("unknown resolved_config_identity schema")
        digest = value["sha256"]
        if type(digest) is not str or len(digest) != 64 or digest != digest.lower() or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise EvidenceUnavailable("resolved_config_identity sha256 is invalid")
        return MappingProxyType({"schema": value["schema"], "sha256": digest})

    @staticmethod
    def _checkpoint_ref(value: Any) -> MappingProxyType | None:
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != _CHECKPOINT_REF_KEYS:
            raise EvidenceUnavailable("checkpoint_ref has invalid keys")
        if value["schema"] != "tpen.checkpoint-ref/v1":
            raise EvidenceUnavailable("checkpoint_ref has an unsupported schema")
        if type(value["checkpoint_dir"]) is not str or type(value["kind"]) is not str:
            raise EvidenceUnavailable("checkpoint_ref paths and kind must be strings")
        for key in ("schema_version", "next_iteration", "completed_updates"):
            if type(value[key]) is not int or value[key] < 0:
                raise EvidenceUnavailable(f"checkpoint_ref {key} must be a non-negative integer")
        for key in ("manifest_sha256", "model_sha256", "content_id"):
            digest = value[key]
            if type(digest) is not str or len(digest) != 64 or digest != digest.lower() or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise EvidenceUnavailable(f"checkpoint_ref {key} must be a 64-hex SHA-256 value")
        if not isinstance(value["provenance"], Mapping):
            raise EvidenceUnavailable("checkpoint_ref provenance must be a mapping")
        if value["profile"] not in {"train_resume", "model_only"}:
            raise EvidenceUnavailable("checkpoint_ref has an invalid payload profile")
        return MappingProxyType(dict(value))

    @property
    def continuation_eligible(self) -> bool:
        """Whether this v3 record is an explicitly valid continuation leaf."""

        return (
            self.schema == RUN_EVIDENCE_SCHEMA_V3
            and self.verdict != "refused"
            and self.refusal is None
            and self.outcome == "planned_yield"
            and self.checkpoint is not None
            and self.checkpoint_ref is not None
            and self.checkpoint_ref.get("profile") == "train_resume"
            and self.checkpoint_ref.get("checkpoint_dir") == self.checkpoint.get("checkpoint_dir")
            and self.checkpoint_ref.get("next_iteration") == self.checkpoint.get("next_iteration")
            and self.checkpoint_ref.get("completed_updates") == self.checkpoint.get("completed_updates")
        )


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
        cancel_path: str | os.PathLike[str] | None = None,
        cancel_identity: Mapping[str, str] | None = None,
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

        This is the only trusted inspection call site: it launches with
        ``purpose=LaunchPurpose.INSPECTION``, which exempts the read-only
        inspection of a member's own output from the slot quarantine and
        from GPU sampling.  The argument is a correctness guard against
        accidental slot reuse, not a security boundary (review delta-4 D4-1).
        """

        try:
            outcome = worker.run(
                environment,
                ("-m", "tpen.inspection", os.fspath(run_dir)),
                cwd=cwd,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout=timeout,
                purpose=LaunchPurpose.INSPECTION,
                cancel_path=cancel_path,
                cancel_identity=cancel_identity,
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


__all__ = ["CHILD_CONTEXT_KEYS_ENV", "CHILD_OBSERVATION_FILENAME", "CONTEXT_ENV_KEY_PATTERN", "CoreRunEvidence", "EvidenceUnavailable", "RUN_EVIDENCE_SCHEMA", "RUN_EVIDENCE_SCHEMA_V2", "RUN_EVIDENCE_SCHEMA_V3", "TPENCommand", "TPENEvidence"]
