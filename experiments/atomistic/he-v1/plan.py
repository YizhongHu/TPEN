"""Expand the He-v1 production grid into an explicit ordered row manifest.

`production-grid-v0` asks for one frozen scientific arm evaluated over three
independent training seeds, a predeclared set of retained checkpoints, and four
independent fixed-model evaluation chains per checkpoint. This stage writes
that shape down -- completely, in order, before anything is submitted -- so the
grid a receipt refers to is a file rather than a memory of a command line.

Two properties are load-bearing:

no implicit defaults
    Every field of the grid config is required and every unknown field is
    rejected. A silently defaulted seed count or wall time would produce a
    green run answering a different question than the one asked.

stable row ids
    A row id is a pure function of the row's coordinates, so re-planning the
    same config produces the same ids, the same order, and the same plan hash.
    Row directories are addressed by that id, so a rerun lands beside its
    predecessor instead of on top of it.

This module imports no ``tpen`` (``experiments/README.md``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import yaml

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
strata = sibling(__file__, 'strata')

SCHEMA_VERSION = "he-v1-study/v1"
STUDY_TIMEZONE = "America/New_York"

#: Required top-level grid-config keys. Nothing is optional and nothing else is
#: accepted: an unrecognized key is a typo that would otherwise be ignored.
GRID_KEYS: frozenset[str] = frozenset(
    {
        "study",
        "train_config",
        "eval_config",
        "seeds",
        "checkpoint_steps",
        "eval_chains",
        "eval_chain_seed_base",
        "train_resources",
        "eval_resources",
        "gate_spec",
        "seed_stages",
        "convergence_assessment",
        "reporting_rules",
        "unemitted_requirements",
    }
)

#: Required keys of the `reporting_rules` block. A limit on what the study may
#: CONCLUDE, predeclared before the energy that would be judged against it.
REPORTING_KEYS: frozenset[str] = frozenset(
    {
        "chemical_accuracy_max_combined_uncertainty_mha",
        "combined_uncertainty_includes_seed_spread",
    }
)

#: Requirements of `production-grid-v0` this arm cannot discharge, each carrying
#: an explicit disposition. Recorded as data so the statement cannot be
#: separated from the artifact it qualifies.
UNEMITTED_REQUIREMENT_KEYS: frozenset[str] = frozenset(
    {"min_sampled_electron_nucleus_radius"}
)

#: Admissible dispositions. `not_emitted` says the quantity has no emitter at
#: all; a metric NAME would say which metric discharges it instead.
UNEMITTED_DISPOSITIONS: frozenset[str] = frozenset({"not_emitted"})

#: Required keys of the `convergence_assessment` block. Predeclared BEFORE any
#: production data exists, which is the only thing that makes the rule
#: legitimate rather than a post-hoc rescue.
CONVERGENCE_KEYS: frozenset[str] = frozenset(
    {
        "method",
        "n_windows",
        "n_trailing_windows",
        "on_inadequate",
        "may_reselect",
        "window_width_min_tau_multiple",
    }
)

#: Minimum trailing windows. The sign test's power is set by the NUMBER OF
#: DIFFERENCES, and under independent symmetric noise the probability that all
#: differences share a sign is ``2 * (1/2)**(n-1)``: 12.50% at 5 windows,
#: 6.25% at 6, 1.56% at 8. Five windows would trip this gate on a converged run
#: one time in eight, which is far too noisy for a criterion that declares a
#: 146 GPU-hour budget inadequate.
MIN_TRAILING_WINDOWS = 8

#: Required per-row resource keys.
RESOURCE_KEYS: frozenset[str] = frozenset(
    {"partition", "stratum", "timeout_min", "cpus", "mem_gb", "gpus"}
)

#: Override path fragments that would let a row resume instead of finishing.
#: `production-grid-v0` forbids relying on restart/resume until its semantics
#: are independently demonstrated, so a row that asks for one is a planning
#: error rather than something the launcher quietly honors.
_RESUME_TOKENS: tuple[str, ...] = ("resume", "restart", "requeue")


class PlanError(ValueError):
    """The grid config does not describe a submittable study."""


def load_grid_config(path: str | Path) -> dict[str, Any]:
    """Read and validate one grid config.

    Raises
    ------
    PlanError
        If a required key is missing, an unknown key is present, or a value is
        outside its contract.
    """

    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PlanError(f"grid config {path} is not a mapping")
    return validate_grid_config(payload)


def validate_grid_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a grid-config mapping and return it as a plain dict."""

    keys = set(payload)
    missing = sorted(GRID_KEYS - keys)
    unknown = sorted(keys - GRID_KEYS)
    if missing:
        raise PlanError(f"grid config is missing required keys: {missing}")
    if unknown:
        raise PlanError(f"grid config carries unknown keys: {unknown}")

    study = _require_text(payload["study"], "study")
    train_config = _require_text(payload["train_config"], "train_config")
    eval_config = _require_text(payload["eval_config"], "eval_config")
    seeds = _require_int_sequence(payload["seeds"], "seeds")
    checkpoint_steps = _require_int_sequence(payload["checkpoint_steps"], "checkpoint_steps")
    eval_chains = _require_positive_int(payload["eval_chains"], "eval_chains")
    seed_base = _require_int(payload["eval_chain_seed_base"], "eval_chain_seed_base")
    train_resources = _validate_resources(payload["train_resources"], "train_resources")
    eval_resources = _validate_resources(payload["eval_resources"], "eval_resources")
    gate_spec = payload["gate_spec"]
    if not isinstance(gate_spec, Mapping):
        raise PlanError(
            "grid config 'gate_spec' must be a mapping; declare it empty to state "
            "explicitly that no tolerance has been predeclared yet"
        )
    seed_stages = _validate_seed_stages(payload["seed_stages"], seeds)
    convergence = _validate_convergence_assessment(payload["convergence_assessment"])
    reporting = _validate_reporting_rules(payload["reporting_rules"])
    unemitted = _validate_unemitted_requirements(payload["unemitted_requirements"])

    if any(step <= 0 for step in checkpoint_steps):
        raise PlanError(f"checkpoint_steps must be positive: {checkpoint_steps}")
    if seed_base < 0:
        raise PlanError(f"eval_chain_seed_base must be non-negative: {seed_base}")

    # Chain seeds are derived, so a collision with a training seed would make a
    # "fresh" evaluation chain a replay of the sampler that trained the model.
    chain_seeds = {
        _chain_seed(seed_base, seed_index, checkpoint_index, chain)
        for seed_index in range(len(seeds))
        for checkpoint_index in range(len(checkpoint_steps))
        for chain in range(eval_chains)
    }
    collisions = sorted(chain_seeds & set(seeds))
    if collisions:
        raise PlanError(
            f"derived evaluation chain seeds collide with training seeds: {collisions}; "
            "raise eval_chain_seed_base"
        )

    return {
        "study": study,
        "train_config": train_config,
        "eval_config": eval_config,
        "seeds": list(seeds),
        "checkpoint_steps": list(checkpoint_steps),
        "eval_chains": eval_chains,
        "eval_chain_seed_base": seed_base,
        "train_resources": train_resources,
        "eval_resources": eval_resources,
        "gate_spec": dict(gate_spec),
        "seed_stages": seed_stages,
        "convergence_assessment": convergence,
        "reporting_rules": reporting,
        "unemitted_requirements": unemitted,
    }


def _validate_reporting_rules(payload: Any) -> dict[str, Any]:
    """Validate the predeclared limits on what this study may CONCLUDE.

    A conclusion rule is worthless if it can be written after the energy it
    constrains is known, so it is validated here as a required grid key rather
    than left to a report author's discretion.
    """

    if not isinstance(payload, Mapping):
        raise PlanError("reporting_rules must be a mapping")
    missing = sorted(REPORTING_KEYS - set(payload))
    unknown = sorted(set(payload) - REPORTING_KEYS)
    if missing:
        raise PlanError(f"reporting_rules is missing required keys: {missing}")
    if unknown:
        raise PlanError(f"reporting_rules carries unknown keys: {unknown}")

    threshold = payload["chemical_accuracy_max_combined_uncertainty_mha"]
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise PlanError(
            "reporting_rules.chemical_accuracy_max_combined_uncertainty_mha must be a real number"
        )
    if not (float(threshold) > 0.0):
        raise PlanError(
            f"chemical_accuracy_max_combined_uncertainty_mha must be positive, got {threshold}"
        )
    if payload["combined_uncertainty_includes_seed_spread"] is not True:
        # The MCSE of one chain is not the study's uncertainty. Three seeds
        # resolve to three means, and excluding their spread would understate
        # the bar by exactly the quantity replication exists to measure.
        raise PlanError(
            "reporting_rules.combined_uncertainty_includes_seed_spread must be true: a "
            "single chain's MCSE is not the combined uncertainty of a three-seed study"
        )
    return {
        "chemical_accuracy_max_combined_uncertainty_mha": float(threshold),
        "combined_uncertainty_includes_seed_spread": True,
    }


def _validate_unemitted_requirements(payload: Any) -> dict[str, str]:
    """Validate the explicit `absent` dispositions for requirements not met.

    A requirement nobody wrote down as unmet reads as met. Recording the
    disposition as DATA rather than prose is what stops it being separated from
    the artifact it qualifies.
    """

    if not isinstance(payload, Mapping):
        raise PlanError("unemitted_requirements must be a mapping")
    missing = sorted(UNEMITTED_REQUIREMENT_KEYS - set(payload))
    unknown = sorted(set(payload) - UNEMITTED_REQUIREMENT_KEYS)
    if missing:
        raise PlanError(
            f"unemitted_requirements is missing required keys: {missing}; a requirement "
            "this arm cannot discharge must be recorded, not omitted"
        )
    if unknown:
        raise PlanError(f"unemitted_requirements carries unknown keys: {unknown}")
    resolved: dict[str, str] = {}
    for key in sorted(UNEMITTED_REQUIREMENT_KEYS):
        value = _require_text(payload[key], f"unemitted_requirements.{key}")
        if value not in UNEMITTED_DISPOSITIONS:
            raise PlanError(
                f"unemitted_requirements.{key} must be one of {sorted(UNEMITTED_DISPOSITIONS)}, "
                f"got {value!r}"
            )
        resolved[key] = value
    return resolved


def _validate_seed_stages(payload: Any, seeds: Sequence[int]) -> list[list[int]]:
    """Validate the declared launch staging over the predeclared seeds.

    THIS KEY IS A PREDECLARED PROCEDURE, NOT AN EXECUTED DEPENDENCY. Nothing in
    this repository acts on it. `expand_rows` gives every training row
    ``depends_on: []``, and `launch.py` contains no reference to `seed_stages`
    at all, so ``launch.py --submit`` submits all three seeds simultaneously
    with no Slurm dependency between them. The staging is a commitment H-F3
    honours by hand: submit stage 1, wait for it to reach terminal, run
    `assess_convergence.py` on its loss trace, then submit stage 2.

    Validation here therefore establishes only that the declared stages are
    well-formed and cover exactly the predeclared seeds. It does NOT establish
    that anything sequences them, and the tests assert only the former.

    WHY STAGE AT ALL: nothing yet shows the loss is not still descending at
    300,000 updates. Staging costs about 98 h of wall instead of 49 and avoids
    committing 97 of 146 GPU-hours before a complete 300k trace for this arm
    exists.

    THE STAGING GATE MAY ONLY REPORT ON BUDGET ADEQUACY. It may never re-select
    an arm, a checkpoint, a budget, or a seed -- see `convergence_assessment`,
    whose `may_reselect` must be false.
    """

    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise PlanError("seed_stages must be a sequence of seed lists")
    stages = [_require_int_sequence(stage, f"seed_stages[{index}]") for index, stage in enumerate(payload)]
    if not stages:
        raise PlanError("seed_stages must be non-empty")
    flat = [seed for stage in stages for seed in stage]
    if len(set(flat)) != len(flat):
        raise PlanError(f"seed_stages repeats a seed: {flat}")
    # EVERY predeclared seed must appear exactly once. A staging that quietly
    # dropped a seed would turn a three-replicate study into a smaller one while
    # the `seeds` list still claimed three.
    if sorted(flat) != sorted(seeds):
        raise PlanError(
            f"seed_stages {flat} does not cover exactly the predeclared seeds {list(seeds)}"
        )
    return stages


def _validate_convergence_assessment(payload: Any) -> dict[str, Any]:
    """Validate the predeclared convergence-assessment rule.

    LIKE `seed_stages`, THIS BLOCK IS A PREDECLARED PROCEDURE AND NOT AN
    EXECUTED GATE. No stage of this pipeline computes a windowed mean or runs a
    sign test; `collect.py`, `report.py` and `driver.py` never read this key.
    The rule is carried into the plan manifest so it is frozen before any
    production data exists, and it is discharged by a human running
    `assess_convergence.py` on the stage-1 loss trace between stages.

    Validation here establishes that the rule is well-formed and that its
    parameters are the predeclared ones. The tests assert only that.
    """

    if not isinstance(payload, Mapping):
        raise PlanError("convergence_assessment must be a mapping")
    missing = sorted(CONVERGENCE_KEYS - set(payload))
    unknown = sorted(set(payload) - CONVERGENCE_KEYS)
    if missing:
        raise PlanError(f"convergence_assessment is missing required keys: {missing}")
    if unknown:
        raise PlanError(f"convergence_assessment carries unknown keys: {unknown}")
    method = _require_text(payload["method"], "convergence_assessment.method")
    if method != "windowed_means_sign_test":
        # Two rejected alternatives, both measured in a peer lane.
        # A TAIL AVERAGE hid a whole-budget-inside-the-transient failure, found
        # only after six runs had completed.
        # AN ERROR-BAR OVERLAP TEST fails toward FALSE REASSURANCE, which is the
        # dangerous direction: blocking takes the LARGEST standard error across
        # levels, so "flat within errors" gets EASIER to satisfy exactly when
        # autocorrelation is worst. Measured case: five windows drifting 37 uHa
        # monotonically against 10 uHa bars: adjacent steps are 0.93 bars so
        # EVERY ADJACENT PAIR OVERLAPS and an overlap test passes, while the
        # cumulative drift is 3.70 bars and the trend is real. The mirror case
        # scattered 5.8 bars and was pure noise because it was non-monotone.
        # THE SIGN PATTERN IS THE DISCRIMINATOR; bar magnitude is not.
        raise PlanError(
            f"convergence_assessment.method must be 'windowed_means_sign_test', got {method!r}; "
            "a tail average cannot see a persistent descent, and an error-bar overlap test "
            "passes one that drifts monotonically inside its own bars"
        )
    n_windows = _require_positive_int(payload["n_windows"], "convergence_assessment.n_windows")
    n_trailing = _require_positive_int(
        payload["n_trailing_windows"], "convergence_assessment.n_trailing_windows"
    )
    if n_trailing < MIN_TRAILING_WINDOWS:
        raise PlanError(
            f"convergence_assessment.n_trailing_windows {n_trailing} is below "
            f"{MIN_TRAILING_WINDOWS}; the sign test's false-alarm rate is "
            f"2*(1/2)**(n-1), so fewer windows declare a sound budget inadequate too often"
        )
    if n_trailing > n_windows:
        raise PlanError(
            f"convergence_assessment.n_trailing_windows {n_trailing} exceeds n_windows {n_windows}"
        )
    on_inadequate = _require_text(payload["on_inadequate"], "convergence_assessment.on_inadequate")
    if on_inadequate != "report_only":
        raise PlanError(
            f"convergence_assessment.on_inadequate must be 'report_only', got {on_inadequate!r}; "
            "no extension rule is predeclared, and inventing one after seeing the trace is "
            "exactly the violation production-grid-v0 forbids"
        )
    if payload["may_reselect"] is not False:
        raise PlanError(
            "convergence_assessment.may_reselect must be false: the rule may REPORT that the "
            "budget was inadequate and may never re-select an arm, checkpoint or budget"
        )
    tau_multiple = payload["window_width_min_tau_multiple"]
    if not isinstance(tau_multiple, (int, float)) or isinstance(tau_multiple, bool):
        raise PlanError(
            "convergence_assessment.window_width_min_tau_multiple must be a real number"
        )
    if float(tau_multiple) < 1.0:
        # THE SIGN TEST'S FALSE-ALARM RATE ASSUMES INDEPENDENT WINDOW MEANS, and
        # that assumption fails silently. If the window is narrower than the
        # LOSS SERIES autocorrelation time, consecutive means are positively
        # correlated, same-sign runs become far more likely under the null, and
        # the nominal 1.56% is optimistic by an unknown factor -- a gate whose
        # false-alarm rate nobody has bounded. The loss-series tau is NOT the
        # local-energy tau: the loss is a trajectory through parameter space
        # under an optimizer and its correlation time is a different, probably
        # much longer quantity, so H-C2's 1.15/5.56 must not be reused for it.
        raise PlanError(
            f"convergence_assessment.window_width_min_tau_multiple {tau_multiple} is below 1.0; "
            "windows narrower than the loss-series autocorrelation time make the sign test's "
            "false-alarm rate unbounded"
        )
    return {
        "method": method,
        "n_windows": n_windows,
        "n_trailing_windows": n_trailing,
        "on_inadequate": on_inadequate,
        "may_reselect": False,
        "window_width_min_tau_multiple": float(tau_multiple),
    }


def train_row_id(seed: int) -> str:
    """Return the stable row id of one training seed."""

    return f"train-seed{int(seed):04d}"


def eval_row_id(*, seed: int, checkpoint_step: int, chain: int) -> str:
    """Return the stable row id of one fixed-model evaluation chain."""

    return f"eval-seed{int(seed):04d}-step{int(checkpoint_step):09d}-chain{int(chain):02d}"


def checkpoint_dir_name(step: int) -> str:
    """Return the checkpoint directory name written by the Checkpoint callback.

    Mirrors ``tpen.checkpoint.checkpoint_step_dir_name``. It is spelled here
    rather than imported because ``experiments/`` may not import ``tpen``; the
    manifest test pins the format so a drift shows up as a failing test rather
    than as an eval row pointing at nothing.
    """

    if int(step) < 0:
        raise PlanError(f"checkpoint step must be non-negative, got {step}")
    return f"step_{int(step):06d}"


def expand_rows(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Expand a validated grid config into the ordered row manifest.

    Order is declaration order: every training seed first, then that seed's
    checkpoints in declared order, then that checkpoint's chains. The order is
    part of the contract -- a receipt cites row indexes.
    """

    seeds = list(config["seeds"])
    checkpoint_steps = list(config["checkpoint_steps"])
    n_chains = int(config["eval_chains"])
    seed_base = int(config["eval_chain_seed_base"])
    train_resources = dict(config["train_resources"])
    eval_resources = dict(config["eval_resources"])
    max_step = max(checkpoint_steps)

    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        row_id = train_row_id(seed)
        rows.append(
            {
                "row_id": row_id,
                "index": len(rows),
                "kind": "train",
                "stage": layout.STAGE_TRAIN,
                "seed": int(seed),
                "checkpoint_step": None,
                "chain": None,
                "chain_seed": None,
                "config": str(config["train_config"]),
                "overrides": [
                    f"runtime.seed={int(seed)}",
                    f"sampler.seed={int(seed)}",
                    f"trainer.max_steps={int(max_step)}",
                ],
                "retained_checkpoint_steps": [int(step) for step in checkpoint_steps],
                "depends_on": [],
                "resources": dict(train_resources),
            }
        )
        for checkpoint_index, step in enumerate(checkpoint_steps):
            for chain in range(n_chains):
                chain_seed = _chain_seed(seed_base, seed_index, checkpoint_index, chain)
                rows.append(
                    {
                        "row_id": eval_row_id(seed=seed, checkpoint_step=step, chain=chain),
                        "index": len(rows),
                        "kind": "eval",
                        "stage": layout.STAGE_EVAL,
                        "seed": int(seed),
                        "checkpoint_step": int(step),
                        "chain": int(chain),
                        "chain_seed": int(chain_seed),
                        "config": str(config["eval_config"]),
                        "overrides": [
                            f"runtime.seed={int(chain_seed)}",
                            f"evaluation.seed={int(chain_seed)}",
                        ],
                        "retained_checkpoint_steps": [],
                        "depends_on": [train_row_id(seed)],
                        "resources": dict(eval_resources),
                        "checkpoint_dir_name": checkpoint_dir_name(step),
                    }
                )

    _require_unique_row_ids(rows)
    for row in rows:
        reject_resume_overrides(row)
        _validate_row_placement(row)
    return tuple(rows)


def plan_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return the content hash of one expanded row list.

    Timestamps and attempt ids are excluded on purpose: two plans of the same
    grid must hash identically, or "the plan did not change" is unverifiable.
    """

    canonical = json.dumps(list(rows), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of one file's bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    *,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    attempt_id: str,
    results_root: str | Path,
    grid_config_path: str | Path | None,
    grid_config_sha256: str | None,
    created_at: str,
) -> dict[str, Any]:
    """Assemble the durable plan manifest."""

    return {
        "schema_version": SCHEMA_VERSION,
        "study": str(config["study"]),
        "attempt_id": str(attempt_id),
        "created_at": created_at,
        "timezone": STUDY_TIMEZONE,
        "results_root": str(results_root),
        "grid_config_path": None if grid_config_path is None else str(grid_config_path),
        "grid_config_sha256": grid_config_sha256,
        "grid_config": dict(config),
        "gate_spec": dict(config["gate_spec"]),
        "gate_spec_declared": bool(config["gate_spec"]),
        "seed_stages": [list(stage) for stage in config["seed_stages"]],
        "convergence_assessment": dict(config["convergence_assessment"]),
        "reporting_rules": dict(config["reporting_rules"]),
        "unemitted_requirements": dict(config["unemitted_requirements"]),
        "plan_hash": plan_hash(rows),
        "n_rows": len(rows),
        "n_train_rows": sum(1 for row in rows if row["kind"] == "train"),
        "n_eval_rows": sum(1 for row in rows if row["kind"] == "eval"),
        "resume_policy": "forbidden",
        "rows": [dict(row) for row in rows],
    }


def write_plan(manifest: Mapping[str, Any], *, results_root: str | Path) -> Path:
    """Write one plan attempt and return its directory."""

    attempt_id = str(manifest["attempt_id"])
    directory = layout.plan_attempt_dir(results_root, attempt_id)
    layout.write_json(directory / layout.MANIFEST_FILENAME, dict(manifest))
    _write_rows_csv(directory / layout.ROWS_FILENAME, manifest["rows"])
    layout.write_latest(layout.stage_dir(results_root, layout.STAGE_PLAN), attempt_id)
    return directory


def read_manifest(results_root: str | Path, attempt_id: str) -> dict[str, Any]:
    """Read one plan manifest, validating its schema version."""

    payload = layout.read_json(layout.manifest_path(results_root, attempt_id))
    if not isinstance(payload, dict):
        raise PlanError("plan manifest is not a mapping")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PlanError(
            f"plan manifest schema_version {payload.get('schema_version')!r} "
            f"does not match {SCHEMA_VERSION!r}"
        )
    return payload


def row_by_id(manifest: Mapping[str, Any], row_id: str) -> dict[str, Any]:
    """Return one manifest row by id."""

    for row in manifest.get("rows", []):
        if str(row.get("row_id")) == str(row_id):
            return dict(row)
    raise PlanError(f"row {row_id!r} is not in plan attempt {manifest.get('attempt_id')!r}")


def now_attempt_id(clock: datetime | None = None) -> str:
    """Return a timestamped attempt id in the study timezone."""

    moment = clock or datetime.now(ZoneInfo(STUDY_TIMEZONE))
    return moment.strftime("%Y%m%dT%H%M%S")


def _chain_seed(base: int, seed_index: int, checkpoint_index: int, chain: int) -> int:
    """Return the deterministic seed of one evaluation chain.

    The stride keeps chains of different checkpoints and different training
    seeds from sharing a sampler stream, which is what "independent chain"
    means here.
    """

    return int(base) + 10_000 * int(seed_index) + 100 * int(checkpoint_index) + int(chain)


def _validate_resources(payload: Any, field: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PlanError(f"{field} must be a mapping")
    keys = set(payload)
    missing = sorted(RESOURCE_KEYS - keys)
    unknown = sorted(keys - RESOURCE_KEYS)
    if missing:
        raise PlanError(f"{field} is missing required keys: {missing}")
    if unknown:
        raise PlanError(f"{field} carries unknown keys: {unknown}")
    return {
        "partition": _require_text(payload["partition"], f"{field}.partition"),
        "stratum": _require_text(payload["stratum"], f"{field}.stratum"),
        "timeout_min": _require_positive_int(payload["timeout_min"], f"{field}.timeout_min"),
        "cpus": _require_positive_int(payload["cpus"], f"{field}.cpus"),
        "mem_gb": _require_positive_int(payload["mem_gb"], f"{field}.mem_gb"),
        "gpus": _require_positive_int(payload["gpus"], f"{field}.gpus"),
    }


def _validate_row_placement(row: Mapping[str, Any]) -> None:
    """Validate one row's GPU placement, pin, and wall time."""

    resources = row["resources"]
    try:
        resolved = strata.validate_gpu_placement(
            partition=str(resources["partition"]),
            stratum_name=str(resources["stratum"]),
            timeout_min=int(resources["timeout_min"]),
        )
    except strata.StratumError as exc:
        raise PlanError(f"row {row['row_id']!r}: {exc}") from exc
    resources["constraint"] = resolved.constraint


def reject_resume_overrides(row: Mapping[str, Any]) -> None:
    for override in row["overrides"]:
        lowered = str(override).lower()
        if any(token in lowered for token in _RESUME_TOKENS):
            raise PlanError(
                f"row {row['row_id']!r} override {override!r} asks for restart/resume, "
                "which production-grid-v0 forbids until its semantics are demonstrated"
            )


def _require_unique_row_ids(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: dict[str, int] = {}
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in seen:
            raise PlanError(
                f"duplicate row id {row_id!r} at indexes {seen[row_id]} and {row['index']}; "
                "two rows would write into one directory"
            )
        seen[row_id] = int(row["index"])


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the human-readable row table.

    Structural fields that do not apply to a row kind are written as the
    literal ``absent`` rather than as an empty cell, for the same reason the
    collector does it: a blank cell parses to NaN and disappears.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "row_id",
        "kind",
        "seed",
        "checkpoint_step",
        "chain",
        "chain_seed",
        "partition",
        "stratum",
        "constraint",
        "timeout_min",
        "gpus",
        "depends_on",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            resources = row["resources"]
            writer.writerow(
                {
                    "index": row["index"],
                    "row_id": row["row_id"],
                    "kind": row["kind"],
                    "seed": row["seed"],
                    "checkpoint_step": _csv_cell(row["checkpoint_step"]),
                    "chain": _csv_cell(row["chain"]),
                    "chain_seed": _csv_cell(row["chain_seed"]),
                    "partition": resources["partition"],
                    "stratum": resources["stratum"],
                    "constraint": resources["constraint"],
                    "timeout_min": resources["timeout_min"],
                    "gpus": resources["gpus"],
                    "depends_on": " ".join(row["depends_on"]) or "none",
                }
            )


def _csv_cell(value: Any) -> str:
    return "absent" if value is None else str(value)


def _require_text(value: Any, field: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise PlanError(f"{field} must be a non-empty string")
    return text


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{field} must be an integer, got {value!r}")
    return int(value)


def _require_positive_int(value: Any, field: str) -> int:
    number = _require_int(value, field)
    if number <= 0:
        raise PlanError(f"{field} must be positive, got {number}")
    return number


def _require_int_sequence(value: Any, field: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PlanError(f"{field} must be a sequence of integers")
    numbers = [_require_int(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if not numbers:
        raise PlanError(f"{field} must be non-empty")
    if len(set(numbers)) != len(numbers):
        raise PlanError(f"{field} carries duplicates: {numbers}")
    return numbers


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse plan command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-config", required=True, help="Grid config YAML path.")
    parser.add_argument("--results-root", required=True, help="Durable study results root.")
    parser.add_argument(
        "--attempt-id",
        default=None,
        help="Explicit plan attempt id; defaults to a study-timezone timestamp.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Expand the grid and write one plan attempt."""

    args = parse_args(argv)
    grid_config_path = Path(args.grid_config).resolve()
    config = load_grid_config(grid_config_path)
    rows = expand_rows(config)
    attempt_id = args.attempt_id or now_attempt_id()
    manifest = build_manifest(
        config=config,
        rows=rows,
        attempt_id=attempt_id,
        results_root=str(Path(args.results_root).resolve()),
        grid_config_path=grid_config_path,
        grid_config_sha256=file_sha256(grid_config_path),
        created_at=datetime.now(ZoneInfo(STUDY_TIMEZONE)).isoformat(),
    )
    directory = write_plan(manifest, results_root=Path(args.results_root).resolve())
    print(
        f"[he-v1] planned {manifest['n_rows']} rows "
        f"({manifest['n_train_rows']} train, {manifest['n_eval_rows']} eval) "
        f"into {directory} plan_hash={manifest['plan_hash'][:12]}"
    )
    if not manifest["gate_spec_declared"]:
        print(
            "[he-v1] gate_spec is empty: tolerances are predeclared in H-F1, so every "
            "value gate will report 'absent' with its observed value retained"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
