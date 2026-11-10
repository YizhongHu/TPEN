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
    "import json, os, re, socket, subprocess, sys, threading, time\n"
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
    "placement_gpu_sampler = {'status': 'unavailable', 'reason': 'sampler-not-started', 'method': 'nvidia-smi-pid-uuid-v1', 'version': None, 'cadence_seconds': 0.0, 'count': 0, 'monotonic_start': None, 'monotonic_end': None, 'inventory': [], 'pid_rows': [], 'samples': [], 'command_pids': [], 'survivors': [], 'elapsed_seconds': 0.0, 'join_timeout_seconds': 1.0, 'quiescent': True}\n"
    "placement_sampler_stop = threading.Event()\n"
    "placement_sampler_thread = None\n"
    "GPU_UUID_RE = re.compile(r'^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$')\n"
    "GPU_COMMAND_TIMEOUT = 1.0\n"
    "def _gpu_command(args, command_pids):\n"
    "    # Inherit the science child's process group so worker cancellation or\n"
    "    # a SIGKILL of the parent owns the sampler command too.\n"
    "    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)\n"
    "    command_pids.append(process.pid)\n"
    "    try:\n"
    "        stdout, stderr = process.communicate(timeout=GPU_COMMAND_TIMEOUT)\n"
    "    except subprocess.TimeoutExpired:\n"
    "        try:\n"
    "            process.kill()\n"
    "        except ProcessLookupError:\n"
    "            pass\n"
    "        try:\n"
    "            stdout, stderr = process.communicate(timeout=GPU_COMMAND_TIMEOUT)\n"
    "        except subprocess.TimeoutExpired:\n"
    "            # Do not wait without a bound after killing a command: a\n"
    "            # descendant retaining a pipe must not hold the sampler open.\n"
    "            process.kill()\n"
    "            for stream in (process.stdout, process.stderr):\n"
    "                if stream is not None:\n"
    "                    stream.close()\n"
    "            try:\n"
    "                process.wait(timeout=GPU_COMMAND_TIMEOUT)\n"
    "            except subprocess.TimeoutExpired:\n"
    "                pass\n"
    "            stdout, stderr = '', ''\n"
    "        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr), 'nvidia-smi-timeout'\n"
    "    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr), ''\n"
    "def _gpu_rows(text, width):\n"
    "    rows = []\n"
    "    for line in text.splitlines():\n"
    "        parts = [part.strip() for part in line.split(',')]\n"
        "        if len(parts) != width or not all(parts):\n"
        "            raise ValueError('nvidia-smi output has malformed columns')\n"
        "        rows.append(parts)\n"
    "    return rows\n"
    "def _gpu_sample(command_pids):\n"
    "    try:\n"
    "        version, error = _gpu_command(['nvidia-smi', '--version'], command_pids)\n"
    "        if error or version.returncode != 0:\n"
    "            return {'ok': False, 'reason': error or 'nvidia-smi-command-failed', 'version': version.stdout.strip() or None, 'inventory': [], 'pid_rows': []}\n"
    "        inventory, error = _gpu_command(['nvidia-smi', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits'], command_pids)\n"
    "        if error or inventory.returncode != 0:\n"
    "            return {'ok': False, 'reason': error or 'nvidia-smi-command-failed', 'version': version.stdout.strip() or None, 'inventory': [], 'pid_rows': []}\n"
    "        processes, error = _gpu_command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], command_pids)\n"
    "        if error or processes.returncode != 0:\n"
    "            return {'ok': False, 'reason': error or 'nvidia-smi-command-failed', 'version': version.stdout.strip() or None, 'inventory': [], 'pid_rows': []}\n"
    "        inv = []\n"
    "        for row in _gpu_rows(inventory.stdout, 2):\n"
    "            if not GPU_UUID_RE.fullmatch(row[0]):\n"
    "                raise ValueError('nvidia-smi inventory UUID is not a physical UUID')\n"
    "            inv.append({'uuid': row[0].upper(), 'pci.bus_id': row[1]})\n"
    "        if not inv:\n"
    "            raise ValueError('nvidia-smi inventory is empty')\n"
    "        pids = []\n"
    "        for row in _gpu_rows(processes.stdout, 2):\n"
    "            if not row[0].isdigit() or not GPU_UUID_RE.fullmatch(row[1]):\n"
    "                raise ValueError('nvidia-smi process row is malformed')\n"
    "            pids.append({'pid': int(row[0]), 'gpu_uuid': row[1].upper()})\n"
    "        return {'ok': True, 'reason': '', 'version': version.stdout.strip() or None, 'inventory': inv, 'pid_rows': pids}\n"
    "    except (OSError, subprocess.SubprocessError, ValueError) as error:\n"
    "        return {'ok': False, 'reason': 'nvidia-smi-unavailable:' + str(error), 'version': None, 'inventory': [], 'pid_rows': []}\n"
    "def _gpu_sampler(cadence, limit):\n"
    "    started = time.monotonic()\n"
    "    samples = []\n"
    "    inventory = []\n"
    "    pid_rows = []\n"
    "    command_pids = []\n"
    "    reason = ''\n"
    "    version = None\n"
    "    while len(samples) < limit and not placement_sampler_stop.is_set():\n"
    "        tick = time.monotonic()\n"
    "        result = _gpu_sample(command_pids)\n"
    "        sample_end = time.monotonic()\n"
    "        version = result['version'] or version\n"
    "        reason = result['reason']\n"
    "        if not result['ok']:\n"
    "            break\n"
    "        inventory = result['inventory']\n"
    "        pid_rows = result['pid_rows']\n"
    "        samples.append({'monotonic_start': tick, 'monotonic_end': sample_end, 'inventory': inventory, 'pid_rows': pid_rows})\n"
    "        if placement_sampler_stop.wait(cadence):\n"
    "            break\n"
    "    ended = time.monotonic()\n"
    "    unavailable = bool(reason) or not samples\n"
    "    placement_gpu_sampler.update({'status': 'unavailable' if unavailable else 'observed', 'reason': reason or 'observed', 'version': version, 'cadence_seconds': cadence, 'count': len(samples), 'monotonic_start': started if samples else None, 'monotonic_end': ended if samples else None, 'inventory': inventory if samples else [], 'pid_rows': pid_rows if samples else [], 'samples': samples, 'command_pids': command_pids, 'survivors': [], 'elapsed_seconds': ended - started, 'join_timeout_seconds': GPU_COMMAND_TIMEOUT * 3 + 1.0, 'quiescent': True})\n"
    "def _write_placement(path, observation):\n"
    "    Path(path).write_text(json.dumps(observation, sort_keys=True) + '\\n', encoding='utf-8')\n"
    "placement_enabled = bool(run_roots and placement_path)\n"
    "placement_code = 0\n"
    "if placement_enabled:\n"
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
    "        'schema': 'tpen-executor/child-placement-v2',\n"
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
    "        'gpu_sampler': placement_gpu_sampler,\n"
    "    }\n"
    "    try:\n"
    "        sampler_count = min(16, max(1, int(os.environ.get('TPEN_PLACEMENT_GPU_SAMPLER_COUNT', '3'))))\n"
    "        sampler_cadence = min(1.0, max(0.01, float(os.environ.get('TPEN_PLACEMENT_GPU_SAMPLER_CADENCE', '0.1'))))\n"
    "    except (TypeError, ValueError):\n"
    "        sampler_count, sampler_cadence = 3, 0.1\n"
    "    if placement_refusal is not None:\n"
    "        print(f\"child-placement-refused: {placement_refusal['reason']}\", file=sys.stderr)\n"
    "    if not placement_abort:\n"
    "        placement_sampler_thread = threading.Thread(target=_gpu_sampler, args=(sampler_cadence, sampler_count), daemon=False, name='tpen-gpu-sampler')\n"
    "        placement_sampler_thread.start()\n"
    "    placement_code = 2 if placement_abort else 0\n"
    "    try:\n"
    "        if not placement_abort:\n"
    "            Path(run_roots[-1], 'executor-child-observation.json').write_text(json.dumps({'schema': 'tpen-executor/child-observation-v1', 'cuda_visible_devices': observed_environment.get('CUDA_VISIBLE_DEVICES'), 'environment': observed_environment, 'thread_settings': {name: observed_environment.get(name) for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}, 'hostname': placement_host, 'pid': placement_pid}, sort_keys=True) + '\\n', encoding='utf-8')\n"
    "            from tpen.run import main\n"
    "            placement_code = main(sys.argv[1:])\n"
    "    finally:\n"
    "        if placement_sampler_thread is not None:\n"
    "            placement_sampler_stop.set()\n"
    "            placement_sampler_thread.join(timeout=GPU_COMMAND_TIMEOUT * 3 + 1.0)\n"
    "            if placement_sampler_thread.is_alive():\n"
    "                placement_gpu_sampler.update({'status': 'unavailable', 'reason': 'sampler-thread-survived-join-bound', 'survivors': ['tpen-gpu-sampler'], 'quiescent': False})\n"
    "        if placement_path and run_roots:\n"
    "            observation['gpu_sampler'] = placement_gpu_sampler\n"
    "            observation['refusal'] = placement_refusal\n"
    "            _write_placement(placement_path, observation)\n"
    "else:\n"
    "    # Placement is optional for legacy/ordinary TPENCommand dispatch.\n"
    "    # Preserve the child observation file when run.root is supplied, then\n"
    "    # invoke core unconditionally as the pre-S2b-2 shim did.\n"
    "    if run_roots:\n"
    "        Path(run_roots[-1], 'executor-child-observation.json').write_text(json.dumps({'schema': 'tpen-executor/child-observation-v1', 'cuda_visible_devices': observed_environment.get('CUDA_VISIBLE_DEVICES'), 'environment': observed_environment, 'thread_settings': {name: observed_environment.get(name) for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}, 'hostname': placement_host, 'pid': placement_pid}, sort_keys=True) + '\\n', encoding='utf-8')\n"
    "    from tpen.run import main\n"
    "    placement_code = main(sys.argv[1:])\n"
    "if placement_abort:\n"
    "    raise SystemExit(placement_code)\n"
    "raise SystemExit(placement_code)"
)
REFUSAL_CODES = frozenset(
    {
        "missing-run-dir", "missing-status", "multiple-status", "malformed-status",
        "status-io-error", "incomplete-checkpoint", "corrupt-checkpoint",
        "checkpoint-io-error", "corrupt-publication-catalog", "torn-publication-catalog",
        "missing-checkpoint", "no-train-resume-checkpoint",
        "unsupported-callback-cadence",
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
