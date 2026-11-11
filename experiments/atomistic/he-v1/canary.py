"""Plan and validate the minimal two-checkpoint He-v1 evaluation canary.

The tracked canary grid contains no facility checkpoint paths.  An external
source map supplies those paths and independently pins the ``model.pt``,
``manifest.json``, and ``COMPLETE`` bytes, both checkpoint progress counters,
and the training source SHA.  Every stage revalidates that typed contract; a
path alone is never checkpoint identity.

This module imports no ``tpen`` (``experiments/README.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
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

plan_stage = sibling(__file__, 'plan')
strata = sibling(__file__, 'strata')

GRID_SCHEMA = "he-v1-eval42-grid/v2"
LEGACY_GRID_SCHEMA = "he-v1-eval-canary-grid/v1"
SOURCE_SCHEMA = "he-v1-eval-canary-sources/v1"
CANARY_SCHEMA = "he-v1-eval-canary-plan/v1"
METRIC_NAMESPACE_SPEC_KEY = "metric_namespaces"
STUDY = "he-v1-eval-canary-v1"
FROZEN_STUDY = "he-v1-eval-42-v1"
# Compatibility default for the original two-row canary. Frozen v2 rows carry
# their own task_names and never consult this value.
DEFAULT_TASK_NAMES = ("mcmc_energy",)
CHECKPOINT_COORDINATES = (
    ("actual-step-025000", 25_000),
    ("actual-step-050000", 50_000),
)

_GRID_KEYS = frozenset(
    {"schema", "study", "eval_config", "task_names", "checkpoints", "scale", "resources"}
)
_GRID_V2_KEYS = frozenset(
    {"schema", "study", "eval_config", "checkpoints", "rows", "metric_namespaces"}
)
_GRID_CHECKPOINT_KEYS = frozenset({"source_id", "checkpoint_step", "evaluation_seed"})
_GRID_ROW_KEYS = frozenset(
    {"row_id", "checkpoint_step", "seed", "task_names", "scale", "resources", "factor_arm"}
)
_SCALE_KEYS = frozenset(
    {"n_walkers", "n_draws", "burn_in", "discard_draws", "stride", "chunk_size"}
)
_RESOURCE_KEYS = frozenset(
    {"partition", "stratum", "constraint", "timeout_min", "cpus", "mem_gb", "gpus"}
)
_SOURCE_KEYS = frozenset(
    {
        "checkpoint_dir",
        "next_iteration",
        "completed_updates",
        "model_sha256",
        "manifest_sha256",
        "complete_sha256",
        "training_source_sha",
    }
)


class CanaryError(ValueError):
    """The canary plan or one immutable checkpoint source is invalid."""


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Return a content id for one JSON-compatible value."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckpointSource:
    """External identity contract for one immutable real-format checkpoint."""

    source_id: str
    checkpoint_dir: Path
    next_iteration: int
    completed_updates: int
    model_sha256: str
    manifest_sha256: str
    complete_sha256: str
    training_source_sha: str

    @classmethod
    def from_mapping(cls, source_id: str, value: Any) -> "CheckpointSource":
        """Construct one source from the strict external-map representation."""

        if not isinstance(value, Mapping):
            raise CanaryError(f"checkpoint source {source_id!r} must be a mapping")
        _require_exact_keys(value, _SOURCE_KEYS, f"checkpoint source {source_id!r}")
        checkpoint_dir = Path(_require_text(value["checkpoint_dir"], "checkpoint_dir"))
        if not checkpoint_dir.is_absolute():
            raise CanaryError(
                f"checkpoint source {source_id!r} path must be absolute: {checkpoint_dir}"
            )
        return cls(
            source_id=_require_text(source_id, "source_id"),
            checkpoint_dir=checkpoint_dir,
            next_iteration=_require_positive_int(value["next_iteration"], "next_iteration"),
            completed_updates=_require_positive_int(
                value["completed_updates"], "completed_updates"
            ),
            model_sha256=_require_sha256(value["model_sha256"], "model_sha256"),
            manifest_sha256=_require_sha256(
                value["manifest_sha256"], "manifest_sha256"
            ),
            complete_sha256=_require_sha256(value["complete_sha256"], "complete_sha256"),
            training_source_sha=_require_git_sha(
                value["training_source_sha"], "training_source_sha"
            ),
        )

    def validate(self) -> dict[str, Any]:
        """Reconcile every externally pinned field against the live directory."""

        required = {
            "model": self.checkpoint_dir / "model.pt",
            "manifest": self.checkpoint_dir / "manifest.json",
            "complete": self.checkpoint_dir / "COMPLETE",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise CanaryError(
                f"checkpoint source {self.source_id!r} is incomplete; missing={missing}"
            )
        actual_hashes = {name: file_sha256(path) for name, path in required.items()}
        expected_hashes = {
            "model": self.model_sha256,
            "manifest": self.manifest_sha256,
            "complete": self.complete_sha256,
        }
        if actual_hashes != expected_hashes:
            raise CanaryError(
                f"checkpoint source {self.source_id!r} content mismatch: "
                f"expected={expected_hashes}, actual={actual_hashes}"
            )
        try:
            manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CanaryError(
                f"checkpoint source {self.source_id!r} manifest is unreadable: {exc}"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise CanaryError(f"checkpoint source {self.source_id!r} manifest is not a mapping")
        files = manifest.get("files")
        provenance = manifest.get("provenance")
        if manifest.get("schema_version") != 2 or manifest.get("kind") != "tpen.checkpoint":
            raise CanaryError(
                f"checkpoint source {self.source_id!r} is not a v2 tpen.checkpoint"
            )
        if manifest.get("next_iteration") != self.next_iteration:
            raise CanaryError(
                f"checkpoint source {self.source_id!r} next_iteration mismatch"
            )
        if manifest.get("completed_updates") != self.completed_updates:
            raise CanaryError(
                f"checkpoint source {self.source_id!r} completed_updates mismatch"
            )
        if not isinstance(files, Mapping) or files.get("model") != "model.pt":
            raise CanaryError(
                f"checkpoint source {self.source_id!r} manifest does not bind model.pt"
            )
        resolved_name = files.get("resolved_config")
        if not isinstance(resolved_name, str) or not (
            self.checkpoint_dir / resolved_name
        ).is_file():
            raise CanaryError(
                f"checkpoint source {self.source_id!r} lacks its real-format resolved config"
            )
        if not isinstance(provenance, Mapping) or provenance.get(
            "git_sha"
        ) != self.training_source_sha:
            raise CanaryError(
                f"checkpoint source {self.source_id!r} training source SHA mismatch"
            )
        source_tpen_version = provenance.get("tpen_version")
        if not isinstance(source_tpen_version, str) or not source_tpen_version.strip():
            raise CanaryError(
                f"checkpoint source {self.source_id!r} lacks source TPEN version"
            )
        binding = {
            "source_id": self.source_id,
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
            "model_sha256": self.model_sha256,
            "manifest_sha256": self.manifest_sha256,
            "complete_sha256": self.complete_sha256,
            "training_source_sha": self.training_source_sha,
            "source_tpen_version": source_tpen_version,
            "checkpoint_schema_version": 2,
            "checkpoint_kind": "tpen.checkpoint",
        }
        return {**binding, "content_id": canonical_sha256(binding)}

    def receipt(self) -> dict[str, Any]:
        """Return a runtime receipt with the external path plus validated identity."""

        return {
            "schema": "he-v1-eval-canary-checkpoint-binding/v1",
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_model_file": str(self.checkpoint_dir / "model.pt"),
            **self.validate(),
        }


def load_grid(path: str | Path) -> dict[str, Any]:
    """Load a tracked canary declaration, including the frozen multi-row plan."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CanaryError("canary grid must be a mapping")
    if payload.get("study") not in {STUDY, FROZEN_STUDY}:
        raise CanaryError("canary grid schema/study identity changed")
    if payload["eval_config"] != "experiments/atomistic/he-v1/configs/eval.yaml":
        raise CanaryError("canary must reuse the generic He-v1 evaluation config")
    if payload.get("schema") == LEGACY_GRID_SCHEMA:
        _require_exact_keys(payload, _GRID_KEYS, "canary grid")
    elif payload.get("schema") == GRID_SCHEMA:
        _require_exact_keys(payload, _GRID_V2_KEYS, "canary grid")
        return _load_grid_v2(payload)
    else:
        raise CanaryError("canary grid schema/study identity changed")
    if tuple(payload["task_names"]) != DEFAULT_TASK_NAMES:
        raise CanaryError("legacy canary task graph must contain only mcmc_energy")

    checkpoints = _mapping_sequence(payload["checkpoints"], "checkpoints")
    for checkpoint in checkpoints:
        _require_exact_keys(checkpoint, _GRID_CHECKPOINT_KEYS, "canary checkpoint")
        _require_positive_int(checkpoint["evaluation_seed"], "evaluation_seed")
    coordinates = tuple(
        (str(item["source_id"]), int(item["checkpoint_step"])) for item in checkpoints
    )
    if coordinates != CHECKPOINT_COORDINATES:
        raise CanaryError(
            f"canary checkpoints must be exactly {CHECKPOINT_COORDINATES!r}, got {coordinates!r}"
        )

    scale = payload["scale"]
    if not isinstance(scale, Mapping):
        raise CanaryError("canary scale must be a mapping")
    _require_exact_keys(scale, _SCALE_KEYS, "canary scale")
    resolved_scale = {
        key: _require_positive_int(scale[key], f"scale.{key}")
        for key in sorted(_SCALE_KEYS - {"discard_draws"})
    }
    resolved_scale["discard_draws"] = _require_nonnegative_int(
        scale["discard_draws"], "scale.discard_draws"
    )
    resources = payload["resources"]
    if not isinstance(resources, Mapping):
        raise CanaryError("canary resources must be a mapping")
    _require_exact_keys(resources, _RESOURCE_KEYS, "canary resources")
    resolved_resources = {
        "partition": _require_text(resources["partition"], "resources.partition"),
        "stratum": _require_text(resources["stratum"], "resources.stratum"),
        "constraint": resources["constraint"],
        **{
            key: _require_positive_int(resources[key], f"resources.{key}")
            for key in ("timeout_min", "cpus", "mem_gb", "gpus")
        },
    }
    try:
        resolved = strata.validate_canary_gpu_placement(
            partition=resolved_resources["partition"],
            stratum_name=resolved_resources["stratum"],
            timeout_min=resolved_resources["timeout_min"],
        )
    except strata.StratumError as exc:
        raise CanaryError(str(exc)) from exc
    if resolved_resources["constraint"] not in (None, "") or resolved.constraint:
        raise CanaryError("gpu_test A100-MIG canary must not invent a node constraint")
    if resolved_resources["gpus"] != 1:
        raise CanaryError("canary rows require exactly one gpu_test MIG allocation")
    resolved_resources["constraint"] = None

    return {
        "schema": LEGACY_GRID_SCHEMA,
        "study": STUDY,
        "eval_config": str(payload["eval_config"]),
        "task_names": list(DEFAULT_TASK_NAMES),
        "checkpoints": [dict(item) for item in checkpoints],
        "scale": resolved_scale,
        "resources": resolved_resources,
    }


def _load_grid_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the explicit per-row frozen evaluation grid."""

    checkpoints = _mapping_sequence(payload["checkpoints"], "checkpoints")
    raw_metric_namespaces = payload["metric_namespaces"]
    if not isinstance(raw_metric_namespaces, Mapping):
        raise CanaryError("canary metric_namespaces must be a mapping")
    metric_namespaces: dict[str, str] = {}
    for metric, namespace in raw_metric_namespaces.items():
        metric_namespaces[_require_text(metric, "metric_namespaces key")] = _require_text(
            namespace, f"metric_namespaces[{metric!r}]"
        ).strip("/")
    coordinates = tuple(
        (str(item["source_id"]), int(item["checkpoint_step"])) for item in checkpoints
    )
    if coordinates != CHECKPOINT_COORDINATES:
        raise CanaryError(
            f"canary checkpoints must be exactly {CHECKPOINT_COORDINATES!r}, got {coordinates!r}"
        )
    checkpoint_steps = {step for _, step in CHECKPOINT_COORDINATES}
    rows = _mapping_sequence(payload["rows"], "rows")
    if len(rows) != 42:
        raise CanaryError(f"frozen He-v1 evaluation grid must contain 42 rows, got {len(rows)}")
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        _require_exact_keys(raw, _GRID_ROW_KEYS, "canary row")
        row = dict(raw)
        step = _require_positive_int(row["checkpoint_step"], "row.checkpoint_step")
        if step not in checkpoint_steps:
            raise CanaryError(f"row checkpoint_step is not a retained checkpoint: {step}")
        row["row_id"] = _require_text(row["row_id"], "row.row_id")
        row["seed"] = _require_positive_int(row["seed"], "row.seed")
        tasks = row["task_names"]
        if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
            raise CanaryError(f"row {row['row_id']!r} task_names must be a sequence")
        row["task_names"] = [
            _require_text(name, "row.task_names entry") for name in tasks
        ]
        if not row["task_names"] or len(set(row["task_names"])) != len(row["task_names"]):
            raise CanaryError(f"row {row['row_id']!r} task_names must be unique and non-empty")
        scale = row["scale"]
        if not isinstance(scale, Mapping):
            raise CanaryError(f"row {row['row_id']!r} scale must be a mapping")
        _require_exact_keys(scale, _SCALE_KEYS, f"row {row['row_id']!r} scale")
        row["scale"] = {
            key: (_require_nonnegative_int(scale[key], f"row {row['row_id']!r} scale.{key}")
                  if key == "discard_draws" else
                  _require_positive_int(scale[key], f"row {row['row_id']!r} scale.{key}"))
            for key in _SCALE_KEYS
        }
        row["resources"] = _resolve_resources(row["resources"], f"row {row['row_id']!r}")
        if row["factor_arm"] is not None and not isinstance(row["factor_arm"], Mapping):
            raise CanaryError(f"row {row['row_id']!r} factor_arm must be a mapping or null")
        normalized.append(row)
    if len({row["row_id"] for row in normalized}) != len(normalized):
        raise CanaryError("frozen canary row ids must be unique")
    counts = {step: sum(row["checkpoint_step"] == step for row in normalized) for step in checkpoint_steps}
    if counts != {25_000: 21, 50_000: 21}:
        raise CanaryError(f"frozen canary grid must contain 21 rows per checkpoint, got {counts}")
    return {
        "schema": GRID_SCHEMA,
        "study": FROZEN_STUDY,
        "eval_config": str(payload["eval_config"]),
        "checkpoints": [dict(item) for item in checkpoints],
        "rows": normalized,
        "metric_namespaces": metric_namespaces,
    }


def _resolve_resources(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CanaryError(f"{field} resources must be a mapping")
    _require_exact_keys(value, _RESOURCE_KEYS, f"{field} resources")
    resolved_resources = {
        "partition": _require_text(value["partition"], f"{field}.resources.partition"),
        "stratum": _require_text(value["stratum"], f"{field}.resources.stratum"),
        "constraint": value["constraint"],
        **{key: _require_positive_int(value[key], f"{field}.resources.{key}")
           for key in ("timeout_min", "cpus", "mem_gb", "gpus")},
    }
    validator = resource_validator(
        resolved_resources["partition"], resolved_resources["stratum"]
    )
    try:
        resolved = validator(partition=resolved_resources["partition"],
                             stratum_name=resolved_resources["stratum"],
                             timeout_min=resolved_resources["timeout_min"])
    except strata.StratumError as exc:
        raise CanaryError(str(exc)) from exc
    if validator is strata.validate_canary_gpu_placement:
        if resolved_resources["constraint"] not in (None, "") or resolved.constraint:
            raise CanaryError("gpu_test A100-MIG canary must not invent a node constraint")
        resolved_resources["constraint"] = None
    elif resolved_resources["constraint"] != resolved.constraint:
        raise CanaryError(f"{field} resources constraint disagrees with its GPU stratum")
    return resolved_resources


def resource_validator(partition: str, stratum: str):
    """Select placement validation from the row's declared resource pair."""

    return (
        strata.validate_canary_gpu_placement
        if (str(partition), str(stratum)) == ("gpu_test", "a100_mig")
        else strata.validate_gpu_placement
    )


def load_source_map(path: str | Path) -> dict[str, CheckpointSource]:
    """Load the external map without accepting missing or extra source ids."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"schema", "sources"}:
        raise CanaryError("source map must contain exactly schema and sources")
    if payload["schema"] != SOURCE_SCHEMA or not isinstance(payload["sources"], Mapping):
        raise CanaryError(f"source map schema must be {SOURCE_SCHEMA!r}")
    return {
        str(source_id): CheckpointSource.from_mapping(str(source_id), value)
        for source_id, value in payload["sources"].items()
    }


def reconcile_grid_sources(
    grid: Mapping[str, Any], sources: Mapping[str, CheckpointSource]
) -> dict[str, dict[str, Any]]:
    """Validate both checkpoint directories transactionally before planning."""

    expected_ids = [str(item["source_id"]) for item in grid["checkpoints"]]
    if set(sources) != set(expected_ids):
        raise CanaryError(
            f"source map ids mismatch: expected={expected_ids}, actual={sorted(sources)}"
        )
    bindings: dict[str, dict[str, Any]] = {}
    for checkpoint in grid["checkpoints"]:
        source_id = str(checkpoint["source_id"])
        source = sources[source_id]
        expected_step = int(checkpoint["checkpoint_step"])
        if source.next_iteration != expected_step:
            raise CanaryError(
                f"checkpoint source {source_id!r} next_iteration must identify actual "
                f"checkpoint step {expected_step}; completed_updates remains independently "
                "bound to the real manifest value"
            )
        bindings[source_id] = source.validate()
    return bindings


def expand_rows(
    grid: Mapping[str, Any], bindings: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    """Expand the declared task and scale contract into immutable plan rows."""

    if "rows" in grid:
        bindings_by_step = {
            int(item["checkpoint_step"]): dict(bindings[str(item["source_id"])])
            for item in grid["checkpoints"]
        }
        rows: list[dict[str, Any]] = []
        for index, spec in enumerate(grid["rows"]):
            scale = dict(spec["scale"])
            n_walkers = int(scale["n_walkers"])
            n_draws = int(scale["n_draws"])
            row = {
                "row_id": str(spec["row_id"]), "index": index, "kind": "eval",
                "stage": layout.STAGE_EVAL, "seed": int(spec["seed"]),
                "checkpoint_step": int(spec["checkpoint_step"]), "chain": 0,
                "chain_seed": int(spec["seed"]), "config": str(grid["eval_config"]),
                "overrides": [f"runtime.seed={spec['seed']}", f"evaluation.seed={spec['seed']}"],
                "retained_checkpoint_steps": [], "depends_on": [],
                "resources": dict(spec["resources"]),
                "checkpoint_source": bindings_by_step[int(spec["checkpoint_step"])],
                "task_names": list(spec["task_names"]), "factor_arm": spec["factor_arm"],
                "n_walkers": n_walkers, "n_draws": n_draws,
                "burn_in": int(scale["burn_in"]), "discard_draws": int(scale["discard_draws"]),
                "stride": int(scale["stride"]), "chunk_size": int(scale["chunk_size"]),
                "record_capacity": n_walkers * n_draws, "canary_protocol": CANARY_SCHEMA,
            }
            plan_stage.reject_resume_overrides(row)
            rows.append(row)
        return tuple(rows)

    scale = dict(grid["scale"])
    rows: list[dict[str, Any]] = []
    for checkpoint in grid["checkpoints"]:
        source_id = str(checkpoint["source_id"])
        step = int(checkpoint["checkpoint_step"])
        seed = int(checkpoint["evaluation_seed"])
        row = {
            "row_id": f"eval-canary-step{step:09d}",
            "index": len(rows),
            "kind": "eval",
            "stage": layout.STAGE_EVAL,
            "seed": seed,
            "checkpoint_step": step,
            "chain": 0,
            "chain_seed": seed,
            "config": str(grid["eval_config"]),
            "overrides": [f"runtime.seed={seed}", f"evaluation.seed={seed}"],
            "retained_checkpoint_steps": [],
            "depends_on": [],
            "resources": dict(grid["resources"]),
            "checkpoint_source": dict(bindings[source_id]),
        "task_names": list(grid.get("task_names", DEFAULT_TASK_NAMES)),
            "n_walkers": int(scale["n_walkers"]),
            "n_draws": int(scale["n_draws"]),
            "burn_in": int(scale["burn_in"]),
            "discard_draws": int(scale["discard_draws"]),
            "stride": int(scale["stride"]),
            "chunk_size": int(scale["chunk_size"]),
            "record_capacity": int(scale["n_walkers"]) * int(scale["n_draws"]),
            "canary_protocol": CANARY_SCHEMA,
        }
        plan_stage.reject_resume_overrides(row)
        rows.append(row)
    if len(rows) != 2 or [row["checkpoint_step"] for row in rows] != [25_000, 50_000]:
        raise CanaryError("canary expansion must produce ordered 25k and 50k rows")
    if len({str(row["row_id"]) for row in rows}) != 2:
        raise CanaryError("canary row ids must be unique")
    return tuple(rows)


def build_manifest(
    *,
    grid: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    attempt_id: str,
    results_root: str | Path,
    grid_path: str | Path,
    source_map_path: str | Path,
    evaluation_git_sha: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a generic He-v1 plan manifest carrying the canary contract."""

    evaluation_git_sha = _require_git_sha(evaluation_git_sha, "evaluation_git_sha")
    metric_namespaces = grid.get("metric_namespaces", {})
    if not isinstance(metric_namespaces, Mapping):
        raise CanaryError("grid metric_namespaces must be a mapping")
    gate_spec = (
        {METRIC_NAMESPACE_SPEC_KEY: dict(metric_namespaces)}
        if metric_namespaces
        else {}
    )
    thresholds = {
        key: value
        for key, value in gate_spec.items()
        if key != METRIC_NAMESPACE_SPEC_KEY
    }
    return {
        "schema_version": plan_stage.SCHEMA_VERSION,
        "canary_schema": CANARY_SCHEMA,
        "study": str(grid.get("study", STUDY)),
        "attempt_id": str(attempt_id),
        "created_at": created_at
        or datetime.now(ZoneInfo(plan_stage.STUDY_TIMEZONE)).isoformat(),
        "timezone": plan_stage.STUDY_TIMEZONE,
        "results_root": str(results_root),
        "grid_config_path": str(Path(grid_path).resolve()),
        "grid_config_sha256": file_sha256(grid_path),
        "grid_config": dict(grid),
        "source_map_path": str(Path(source_map_path).resolve()),
        "source_map_sha256": file_sha256(source_map_path),
        "evaluation_git_sha": evaluation_git_sha,
        "gate_spec": gate_spec,
        # Namespace bindings identify metric provenance; they are not tolerance
        # thresholds, so they must not claim that value gates are declared.
        "gate_spec_declared": bool(thresholds),
        "seed_stages": [],
        "convergence_assessment": {"status": "not_applicable"},
        "reporting_rules": {"checkpoint_reporting": "both_without_selection"},
        "unemitted_requirements": {},
        "plan_hash": plan_stage.plan_hash(rows),
        "n_rows": len(rows),
        "n_train_rows": 0,
        "n_eval_rows": len(rows),
        "resume_policy": "forbidden",
        "selection_policy": "none",
        "rows": [dict(row) for row in rows],
    }


def reconcile_manifest_sources(
    manifest: Mapping[str, Any], source_map_path: str | Path
) -> dict[str, CheckpointSource]:
    """Revalidate the exact external map and every planned source binding."""

    if manifest.get("canary_schema") != CANARY_SCHEMA or manifest.get("study") not in {
        STUDY, FROZEN_STUDY
    }:
        raise CanaryError("manifest is not the minimal He-v1 evaluation canary")
    _require_git_sha(manifest.get("evaluation_git_sha"), "evaluation_git_sha")
    if file_sha256(source_map_path) != manifest.get("source_map_sha256"):
        raise CanaryError("external checkpoint source map changed after planning")
    sources = load_source_map(source_map_path)
    bindings = reconcile_grid_sources(manifest["grid_config"], sources)
    rows = list(manifest.get("rows", []))
    if manifest.get("grid_config", {}).get("schema") == GRID_SCHEMA:
        if len(rows) != 42:
            raise CanaryError(f"frozen canary manifest requires exactly 42 rows, found {len(rows)}")
        if manifest.get("plan_hash") != plan_stage.plan_hash(rows):
            raise CanaryError("canary manifest plan hash does not bind its rows")
        expected_ids = [str(item["row_id"]) for item in manifest["grid_config"]["rows"]]
        if [str(row.get("row_id")) for row in rows] != expected_ids:
            raise CanaryError("frozen canary row order or identity changed")
        for row in rows:
            if (
                row.get("canary_protocol") != CANARY_SCHEMA
                or row.get("kind") != "eval"
                or row.get("stage") != layout.STAGE_EVAL
                or row.get("depends_on") != []
                or not isinstance(row.get("task_names"), list)
                or not row.get("task_names")
                or row.get("record_capacity") != int(row.get("n_walkers", 0)) * int(row.get("n_draws", 0))
            ):
                raise CanaryError(f"canary row {row.get('row_id')!r} changed its runtime graph")
            expected = row.get("checkpoint_source")
            if not isinstance(expected, Mapping):
                raise CanaryError(f"canary row {row.get('row_id')!r} has no source binding")
            source_id = str(expected.get("source_id") or "")
            if bindings.get(source_id) != dict(expected):
                raise CanaryError(f"canary row {row.get('row_id')!r} source binding changed after planning")
        return sources
    if len(rows) != 2:
        raise CanaryError(f"canary manifest requires exactly two rows, found {len(rows)}")
    expected_coordinates = [
        (f"eval-canary-step{step:09d}", step) for _, step in CHECKPOINT_COORDINATES
    ]
    actual_coordinates = [
        (str(row.get("row_id")), int(row.get("checkpoint_step", -1))) for row in rows
    ]
    if actual_coordinates != expected_coordinates:
        raise CanaryError(
            f"canary row coordinates changed: expected={expected_coordinates}, "
            f"actual={actual_coordinates}"
        )
    if manifest.get("plan_hash") != plan_stage.plan_hash(rows):
        raise CanaryError("canary manifest plan hash does not bind its two rows")
    for row in rows:
        if (
            row.get("canary_protocol") != CANARY_SCHEMA
            or row.get("kind") != "eval"
            or row.get("stage") != layout.STAGE_EVAL
            or row.get("depends_on") != []
            or row.get("task_names") != list(DEFAULT_TASK_NAMES)
            or row.get("record_capacity")
            != int(row.get("n_walkers", 0)) * int(row.get("n_draws", 0))
        ):
            raise CanaryError(
                f"canary row {row.get('row_id')!r} changed its minimal runtime graph"
            )
        expected = row.get("checkpoint_source")
        if not isinstance(expected, Mapping):
            raise CanaryError(f"canary row {row.get('row_id')!r} has no source binding")
        source_id = str(expected.get("source_id") or "")
        if bindings.get(source_id) != dict(expected):
            raise CanaryError(
                f"canary row {row.get('row_id')!r} source binding changed after planning"
            )
    return sources


def source_for_row(
    row: Mapping[str, Any], sources: Mapping[str, CheckpointSource]
) -> CheckpointSource:
    """Return the single typed source explicitly named by one canary row."""

    binding = row.get("checkpoint_source")
    if not isinstance(binding, Mapping):
        raise CanaryError(f"row {row.get('row_id')!r} has no checkpoint source binding")
    source_id = str(binding.get("source_id") or "")
    if source_id not in sources:
        raise CanaryError(f"row {row.get('row_id')!r} names unknown source {source_id!r}")
    source = sources[source_id]
    if source.validate() != dict(binding):
        raise CanaryError(f"row {row.get('row_id')!r} source identity no longer matches")
    return source


def _mapping_sequence(value: Any, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CanaryError(f"{field} must be a sequence of mappings")
    if not all(isinstance(item, Mapping) for item in value):
        raise CanaryError(f"{field} must contain only mappings")
    return list(value)


def _require_exact_keys(value: Mapping[str, Any], keys: frozenset[str], field: str) -> None:
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing or unknown:
        raise CanaryError(f"{field} keys mismatch: missing={missing}, unknown={unknown}")


def _require_text(value: Any, field: str) -> str:
    resolved = str(value).strip() if value is not None else ""
    if not resolved:
        raise CanaryError(f"{field} must be a non-empty string")
    return resolved


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CanaryError(f"{field} must be a positive integer")
    return int(value)


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CanaryError(f"{field} must be a non-negative integer")
    return int(value)


def _require_sha256(value: Any, field: str) -> str:
    resolved = _require_text(value, field)
    if len(resolved) != 64 or any(character not in "0123456789abcdef" for character in resolved):
        raise CanaryError(f"{field} must be a lowercase SHA-256 digest")
    return resolved


def _require_git_sha(value: Any, field: str) -> str:
    resolved = _require_text(value, field)
    if len(resolved) != 40 or any(character not in "0123456789abcdef" for character in resolved):
        raise CanaryError(f"{field} must be a full lowercase Git SHA")
    return resolved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse canary planning arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-config", required=True)
    parser.add_argument("--checkpoint-source-map", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--evaluation-git-sha", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate sources and write the evaluation plan declared by the grid."""

    args = parse_args(argv)
    grid = load_grid(args.grid_config)
    sources = load_source_map(args.checkpoint_source_map)
    bindings = reconcile_grid_sources(grid, sources)
    rows = expand_rows(grid, bindings)
    manifest = build_manifest(
        grid=grid,
        rows=rows,
        attempt_id=args.attempt_id,
        results_root=Path(args.results_root).resolve(),
        grid_path=args.grid_config,
        source_map_path=args.checkpoint_source_map,
        evaluation_git_sha=args.evaluation_git_sha,
    )
    directory = plan_stage.write_plan(manifest, results_root=args.results_root)
    print(f"[he-v1-canary] wrote {manifest['n_rows']} rows to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANARY_SCHEMA",
    "CHECKPOINT_COORDINATES",
    "CanaryError",
    "CheckpointSource",
    "METRIC_NAMESPACE_SPEC_KEY",
    "GRID_SCHEMA",
    "SOURCE_SCHEMA",
    "build_manifest",
    "canonical_sha256",
    "expand_rows",
    "file_sha256",
    "load_grid",
    "load_source_map",
    "reconcile_grid_sources",
    "reconcile_manifest_sources",
    "source_for_row",
]
