"""Run one He-cutover training row inside Slurm or PBS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

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

cutover_strata = sibling(__file__, 'cutover_strata')


def require_allocation(environ: Mapping[str, str] | None = None) -> str:
    """Return the Slurm or PBS allocation id, refusing a login-node run."""

    environ = os.environ if environ is None else environ
    job_id = str(environ.get("SLURM_JOB_ID") or environ.get("PBS_JOBID") or "").strip()
    if not job_id:
        raise RuntimeError("SLURM_JOB_ID and PBS_JOBID are empty; allocation required")
    return job_id


def configure_smoke_training(cfg: Any, row: Mapping[str, Any]) -> Any:
    """Apply only the three declared smoke-scale mutations."""

    cfg.trainer.max_steps = int(row["max_steps"])
    cfg.sampler.n_walkers = int(row["n_walkers"])
    checkpoints = [callback for callback in cfg.callbacks if str(callback.get("_target_")) == "tpen.callback.Checkpoint"]
    if len(checkpoints) != 1:
        raise RuntimeError("training config must declare exactly one Checkpoint callback")
    checkpoints[0].schedule.every_n = int(row["max_steps"])
    return cfg


def configure_training(cfg: Any, row: Mapping[str, Any]) -> Any:
    """Apply smoke reductions or assert the frozen production scale."""

    if row.get("scale") == "smoke":
        return configure_smoke_training(cfg, row)
    if row.get("scale") != "production":
        raise RuntimeError("training row scale must be smoke or production")
    if int(cfg.trainer.max_steps) != int(row["max_steps"]) or int(cfg.sampler.n_walkers) != int(row["n_walkers"]):
        raise RuntimeError("production row disagrees with the frozen He-v1 train config")
    return cfg


def output_overrides(row: Mapping[str, Any]) -> list[str]:
    """Return overrides that realize the plan-owned row directory."""

    result_dir = Path(str(row["result_dir"]))
    if result_dir.name != str(row["row_id"]):
        raise ValueError("planned train result_dir must end with row_id")
    return [f"run.root={result_dir.parent}", f"run.run_id={row['row_id']}", "run.layout=flat"]


def run(row: Mapping[str, Any], *, plan_attempt_id: str, environ: Mapping[str, str] | None = None, device_reader=None, runner=None) -> int:
    """Validate allocation/device and invoke the sanctioned TPEN entrypoint."""

    environ = os.environ if environ is None else environ
    require_allocation(environ)
    import hev1

    device_reader = device_reader or hev1.driver.torch_device_name
    delivered = device_reader()
    cutover_strata.check_delivered_device(facility=str(row["facility"]), stratum=str(row["resources"]["stratum"]), delivered=delivered)
    config_path = Path(__file__).resolve().parents[3] / str(row["config"])
    overrides = [f"runtime.seed={row['seed']}", *output_overrides(row)]
    cfg = hev1.driver.build_config(config_path, overrides, checked=[overrides[0]])
    configure_training(cfg, row)
    if runner is None:
        from tpen.run import run_from_config
        runner = run_from_config
    return int(runner(cfg, config_path=str(config_path), command=f"he-cutover train {row['row_id']}"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-json", required=True)
    parser.add_argument("--plan-attempt-id", required=True)
    args = parser.parse_args(argv)
    return run(json.loads(args.row_json), plan_attempt_id=args.plan_attempt_id)


if __name__ == "__main__":
    raise SystemExit(main())
