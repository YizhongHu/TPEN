"""In-allocation execution shared by the train and eval drivers.

What happens here, in order, before any physics runs:

1. the job refuses to run outside a Slurm allocation, because the login-node
   boundary is not advisory and a login-node "run" produces a receipt that
   cites no node;
2. the delivered GPU is read from inside the allocation and asserted against
   the stratum the row was constrained for, and a mismatch FAILS the row; and
3. an allocation receipt is written before the run starts, so a row that dies
   later still says which card it held.

Only then is the configured run started through ``tpen.run.run_from_config``,
the single ``tpen`` symbol ``experiments/README.md`` permits here. The config
itself is loaded with OmegaConf directly rather than through ``tpen.run``'s
loader, because that loader is not part of the sanctioned exception.
"""

from __future__ import annotations

import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

# Siblings are loaded study-scoped, not by bare import: experiments/ has several
# same-named modules and the first study loaded would otherwise own the bare name
# for every study after it. See experiments/toolkit/study_imports.py.
#
# The loader is reached BY PATH rather than by putting the repository root on
# sys.path. A study directory that mutates sys.path is the mechanism behind the
# very defect this import exists to fix, and he-cutover's gateway test forbids it
# outright -- so the fix must not reintroduce it in order to install itself.
import importlib.util as _tpen_importlib  # noqa: E402
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

if "_tpen_study_imports" not in _tpen_sys.modules:
    _tpen_spec = _tpen_importlib.spec_from_file_location(
        "_tpen_study_imports",
        _TpenPath(__file__).resolve().parents[3] / "experiments" / "toolkit" / "study_imports.py",
    )
    _tpen_module = _tpen_importlib.module_from_spec(_tpen_spec)
    _tpen_sys.modules["_tpen_study_imports"] = _tpen_module
    _tpen_spec.loader.exec_module(_tpen_module)
sibling = _tpen_sys.modules["_tpen_study_imports"].sibling

layout = sibling(__file__, 'layout')

plan_stage = sibling(__file__, 'plan')
strata = sibling(__file__, 'strata')

ALLOCATION_RECEIPT = "allocation_receipt.json"

DeviceReader = Callable[[], "str | None"]
ConfigRunner = Callable[..., int]
ConfigTransform = Callable[[Any], Any | None]


class DriverError(RuntimeError):
    """The row cannot run as planned inside this allocation."""


def require_scheduler(environ: Mapping[str, str] | None = None) -> str:
    """Return the Slurm job id, refusing to run outside an allocation.

    Raises
    ------
    DriverError
        If ``SLURM_JOB_ID`` is unset or empty. Tests, training, smokes and
        production all run inside a submitted job; a login-node execution is a
        policy violation, not a fallback.
    """

    environ = os.environ if environ is None else environ
    job_id = str(environ.get("SLURM_JOB_ID") or "").strip()
    if not job_id:
        raise DriverError(
            "SLURM_JOB_ID is empty: He-v1 rows run only inside a Slurm allocation"
        )
    return job_id


def torch_device_name() -> str | None:
    """Return the delivered CUDA device name, or ``None`` when none is visible.

    ``torch`` is imported lazily and inside the allocation only: this module is
    imported by tests that must not require a GPU build.
    """

    try:
        import torch  # noqa: PLC0415 - deliberately lazy; the driver runs in-allocation
    except ImportError:
        return None
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        return None
    return str(torch.cuda.get_device_name(0))


def allocation_receipt(
    row: Mapping[str, Any],
    *,
    job_id: str,
    delivered_device: str | None,
    device_status: str,
    environ: Mapping[str, str],
    mismatch: str | None = None,
) -> dict[str, Any]:
    """Assemble the receipt describing what this allocation actually delivered."""

    resources = row["resources"]
    requested_stratum = str(resources["stratum"])
    return {
        "row_id": str(row["row_id"]),
        "kind": str(row["kind"]),
        "job_id": job_id,
        "hostname": socket.gethostname(),
        "partition": str(environ.get("SLURM_JOB_PARTITION") or resources["partition"]),
        "requested_partition": str(resources["partition"]),
        "requested_stratum": requested_stratum,
        "requested_constraint": strata.constraint_for(requested_stratum),
        "delivered_device": delivered_device,
        "delivered_device_status": device_status,
        "delivered_matches_requested": mismatch is None,
        "mismatch_reason": mismatch,
        "cuda_visible_devices": environ.get("CUDA_VISIBLE_DEVICES"),
        "python_executable": sys.executable,
        "recorded_at": datetime.now(ZoneInfo(plan_stage.STUDY_TIMEZONE)).isoformat(),
        "timezone": plan_stage.STUDY_TIMEZONE,
    }


def verify_delivered_device(
    row: Mapping[str, Any],
    *,
    receipt_dir: str | Path,
    job_id: str,
    device_reader: DeviceReader = torch_device_name,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assert the delivered card matches the row's constraint and record both.

    The receipt is written on both paths: a mismatched row must leave behind
    the evidence of what it was given, not just a traceback.

    Raises
    ------
    strata.DeliveredDeviceMismatch
        If the delivered device is missing or is not the constrained stratum.
    """

    environ = os.environ if environ is None else environ
    delivered = device_reader()
    mismatch: str | None = None
    try:
        strata.check_delivered_device(
            stratum_name=str(row["resources"]["stratum"]), delivered=delivered
        )
    except strata.DeliveredDeviceMismatch as exc:
        mismatch = str(exc)
    receipt = allocation_receipt(
        row,
        job_id=job_id,
        delivered_device=delivered,
        device_status="present" if delivered else "absent",
        environ=environ,
        mismatch=mismatch,
    )
    layout.write_json(Path(receipt_dir) / ALLOCATION_RECEIPT, receipt)
    if mismatch is not None:
        raise strata.DeliveredDeviceMismatch(mismatch)
    return receipt


def build_config(
    config_path: str | Path,
    overrides: Sequence[str],
    *,
    checked: Sequence[str] | None = None,
) -> Any:
    """Load one run config and apply dotlist overrides.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Base run config.
    overrides : sequence of str
        Dotlist overrides to apply.
    checked : sequence of str, optional
        Subset of ``overrides`` whose key paths must already exist in the base
        config. Defaults to all of them.

    Notes
    -----
    ``OmegaConf.merge`` with a dotlist SILENTLY CREATES unknown keys, so a
    mistyped override path no-ops on every row with no error anywhere. The
    scientific overrides are therefore checked structurally against the base
    config here. Run-plumbing overrides (``run.root``, ``run.run_id``,
    ``run.layout``) are deliberately exempt: they set launcher-owned keys that
    a study config need not spell out, and they are exercised end to end by the
    run directory a row actually writes.

    Presence is decided on the raw container rather than through
    ``OmegaConf.select``, because a key declared ``null`` or ``???`` -- both of
    which the He configs use -- is declared, not absent.
    """

    from omegaconf import OmegaConf  # noqa: PLC0415 - keeps import cost off the test path

    cfg = OmegaConf.load(str(config_path))
    container = OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)
    to_check = list(overrides if checked is None else checked)
    unknown = [
        override
        for override in to_check
        if not _key_exists(container, str(override).split("=", 1)[0])
    ]
    if unknown:
        raise DriverError(
            f"overrides target keys absent from {config_path}: {unknown}; "
            "OmegaConf would create them silently"
        )
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist([str(item) for item in overrides]))


def _key_exists(container: Any, dotted_key: str) -> bool:
    """Return whether one dotted key path is declared in a config container."""

    node = container
    for part in str(dotted_key).split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
    return True


def row_overrides(
    row: Mapping[str, Any],
    *,
    run_root: str | Path,
    extra: Sequence[str] = (),
) -> list[str]:
    """Return the full override list for one row.

    The run id is pinned to the row id and the layout to ``flat`` so the run
    directory is reproducible from the manifest alone, rather than from a
    generated timestamp.
    """

    return [
        *[str(item) for item in row["overrides"]],
        f"run.root={Path(run_root)}",
        f"run.run_id={row['row_id']}",
        "run.layout=flat",
        *[str(item) for item in extra],
    ]


def run_row(
    row: Mapping[str, Any],
    *,
    results_root: str | Path,
    plan_attempt_id: str,
    launch_attempt_id: str,
    extra_overrides: Sequence[str] = (),
    config_transform: ConfigTransform | None = None,
    device_reader: DeviceReader = torch_device_name,
    runner: ConfigRunner | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Verify the allocation, then run one row's configured run.

    Returns
    -------
    int
        The configured run's exit code: ``0`` on success, ``1`` on a handled
        failure. Nothing here converts a failure into a success.
    """

    environ = os.environ if environ is None else environ
    job_id = require_scheduler(environ)
    result_dir = layout.row_dir(results_root, str(row["stage"]), str(row["row_id"]), plan_attempt_id)
    result_dir.mkdir(parents=True, exist_ok=True)
    verify_delivered_device(
        row,
        receipt_dir=result_dir,
        job_id=job_id,
        device_reader=device_reader,
        environ=environ,
    )
    layout.write_json(
        result_dir / "row.json",
        {
            "row": dict(row),
            "plan_attempt_id": str(plan_attempt_id),
            "launch_attempt_id": str(launch_attempt_id),
            "job_id": job_id,
        },
    )
    overrides = row_overrides(row, run_root=result_dir, extra=extra_overrides)
    cfg = build_config(
        _config_path(row),
        overrides,
        checked=[*[str(item) for item in row["overrides"]], *[str(item) for item in extra_overrides]],
    )
    if config_transform is not None:
        transformed = config_transform(cfg)
        if transformed is not None:
            cfg = transformed
    run = runner if runner is not None else _run_from_config
    return int(
        run(
            cfg,
            config_path=str(_config_path(row)),
            command=f"he-v1 {row['kind']} {row['row_id']}",
        )
    )


def _config_path(row: Mapping[str, Any]) -> Path:
    """Return the run config of one row, resolved against the repo root."""

    configured = Path(str(row["config"]))
    if configured.is_absolute():
        return configured
    return STUDY_DIR.parents[2] / configured


def _run_from_config(cfg: Any, *, config_path: str, command: str) -> int:
    """Call the one sanctioned ``tpen`` entrypoint."""

    from tpen.run import run_from_config  # noqa: PLC0415 - sanctioned launcher exception

    return int(run_from_config(cfg, config_path=config_path, command=command))


def add_common_arguments(parser: Any) -> None:
    """Register the arguments shared by both drivers."""

    parser.add_argument("--results-root", required=True, help="Durable study results root.")
    parser.add_argument("--plan-attempt-id", required=True, help="Plan attempt this row belongs to.")
    parser.add_argument("--launch-attempt-id", required=True, help="Launch attempt that submitted it.")
    parser.add_argument("--row-id", required=True, help="Manifest row id to run.")


def load_row(results_root: str | Path, plan_attempt_id: str, row_id: str, *, kind: str) -> dict[str, Any]:
    """Return one manifest row, checking it is the kind this driver runs."""

    manifest = plan_stage.read_manifest(results_root, plan_attempt_id)
    row = plan_stage.row_by_id(manifest, row_id)
    if str(row["kind"]) != kind:
        raise DriverError(f"row {row_id!r} is a {row['kind']!r} row, not {kind!r}")
    return row


__all__ = [
    "ALLOCATION_RECEIPT",
    "DriverError",
    "add_common_arguments",
    "allocation_receipt",
    "build_config",
    "ConfigTransform",
    "load_row",
    "require_scheduler",
    "row_overrides",
    "run_row",
    "torch_device_name",
    "verify_delivered_device",
]
