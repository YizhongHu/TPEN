"""Render guarded Stage-1 sbatch scripts; submission is always explicit."""

from __future__ import annotations

import argparse
import base64
import json
import re
import shlex
from pathlib import Path
from typing import Any, Mapping

from experiments.atomistic.stage1.protocol import _require_sha, identity_expectations

ALLOWED_PARTITIONS = ("test", "gpu_test")
FACILITY_RESULTS_PREFIX = "/n/netscratch/kozinsky_lab/Lab/rhu/"
_SAFE_RENDERED_VALUE = re.compile(r"[A-Za-z0-9_./:+,@=-]+\Z")


def _require_safe_value(value: object, *, field: str) -> str:
    """Reject values that could alter the generated shell program."""

    if not isinstance(value, str) or not _SAFE_RENDERED_VALUE.fullmatch(value):
        raise ValueError(f"{field} contains shell-unsafe characters")
    return value


def _binding_token(row: Mapping[str, Any]) -> str:
    """Encode the complete row binding using shell-safe alphabet only."""

    payload = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def row_overrides(row: Mapping[str, Any]) -> list[str]:
    """Return explicit Hydra overrides for every launch row kind."""

    kind = row.get("kind")
    if kind not in {"smoke", "train", "eval"}:
        raise ValueError("manifest row has an invalid kind")
    overrides = [f"runtime.seed={row['seed']}"]
    if kind == "eval":
        overrides.extend([
            f"evaluation_sampler.n_walkers={row['walkers']}",
            f"evaluation_tasks.mcmc_energy.generator.n_draws={row['draws']}",
            f"load.path={row['checkpoint_path']}",
            "load.mode=model_only",
            "load.strict=true",
        ])
    else:
        overrides.extend([
            f"sampler.n_walkers={row['walkers']}",
            f"trainer.max_steps={row['updates']}",
        ])
    return overrides


def command_for_row(row: Mapping[str, Any], *, python: str, driver: str) -> str:
    """Build the only driver command permitted by the launcher."""

    if row.get("kind") not in {"smoke", "train", "eval"}:
        raise ValueError("manifest row has an invalid kind")
    _require_sha(row.get("source_sha"), length=40, field="source_sha")
    _require_sha(row.get("config_sha256"), length=64, field="config_sha256")
    values = {"python": python, "driver": driver, "config": row.get("config"),
              "row_id": row.get("row_id", "")}
    for field, value in values.items():
        _require_safe_value(value, field=field)
    args = [python, driver, "--config", str(row["config"]), "--kind", str(row["kind"])]
    for override in row_overrides(row):
        _require_safe_value(override, field="override")
        args.append(override)
    args.extend(["--binding", _binding_token(row)])
    if row.get("kind") == "eval":
        args.extend([
            "--expected-resolved-config-identity", _binding_token({
                "resolved_config_identity": row["resolved_config_identity"]}),
            "--expected-checkpoint-sha256", str(row["checkpoint_sha256"]),
        ])
    return shlex.join(args)


def render(*, row: Mapping[str, Any], driver: str, python: str, head_sha: str,
           results_root: str, partition: str = "test",
           scratch_base: str = "/scratch") -> str:
    """Render a fail-closed script for one identity-bound row."""

    if not Path(python).is_absolute():
        raise ValueError("python must be an absolute path")
    _require_safe_value(python, field="python")
    _require_safe_value(driver, field="driver")
    _require_safe_value(scratch_base, field="scratch_base")
    if not Path(scratch_base).is_absolute() or scratch_base == "/":
        raise ValueError("scratch_base must be absolute and not '/'")
    _require_sha(head_sha, length=40, field="head_sha")
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError(f"partition must be one of {ALLOWED_PARTITIONS}")
    if not results_root.startswith(FACILITY_RESULTS_PREFIX):
        raise ValueError("results_root must be below the approved Netscratch prefix")
    _require_safe_value(results_root, field="results_root")
    row_root = row.get("results_root")
    if not isinstance(row_root, str) or not row_root.startswith(results_root.rstrip("/") + "/"):
        raise ValueError("row results_root is not bound to the rendered root")
    _require_safe_value(row_root, field="row.results_root")
    if row.get("source_sha") != head_sha:
        raise ValueError("manifest source_sha does not match rendered head_sha")
    if row.get("kind") == "eval":
        identity_expectations(row)
    command = command_for_row(row, python=python, driver=driver)
    template_path = Path(__file__).with_name("templates") / "cannon.sbatch"
    template = template_path.read_text(encoding="utf-8")
    for marker, value in {
        "__PARTITION__": partition,
        "__RESULTS_ROOT__": results_root,
        "__HEAD_SHA__": head_sha,
        "__SCRATCH_BASE__": scratch_base.rstrip("/"),
        "__COMMAND__": command,
    }.items():
        template = template.replace(marker, value)
    return template


def main() -> int:
    """Write a script without submitting it."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", type=Path, required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--scratch-base", default="/scratch")
    args = parser.parse_args()
    row = json.loads(args.row.read_text(encoding="utf-8"))
    args.output.write_text(render(row=row, driver=args.driver, python=args.python,
                                  head_sha=args.head_sha, results_root=args.results_root,
                                  partition=args.partition, scratch_base=args.scratch_base), encoding="utf-8")
    args.output.chmod(0o750)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
