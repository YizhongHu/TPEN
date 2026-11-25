"""Immutable optimizer-pilot row identities and parent/evaluation joins.

The row layer deliberately does not import Stage-1 implementation modules.  It
owns the additive pilot roster and uses the same canonical JSON hash convention
as the Stage-1 manifest codec without changing that codec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class RowContractError(ValueError):
    """Raised when a pilot row or join violates the frozen P0 contract."""


def canonical_hash(value: Any) -> str:
    """Hash JSON-compatible data with the Stage-1 canonical serialization."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_hash(rows: list[Mapping[str, Any]]) -> str:
    """Return the canonical hash of an ordered row list."""

    return canonical_hash(rows)


def row_id(row: Mapping[str, Any]) -> str:
    """Derive the P0 extended identity from the row without ``row_id``.

    P0 sentence implemented: ``row_id = manifest_hash([row without row_id])``.
    """

    without_id = {key: value for key, value in row.items() if key != "row_id"}
    return manifest_hash([without_id])


def sha256_file(path: str | Path) -> str:
    """Hash one named file without recursively discovering its siblings."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _int(value: Any, field_name: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0) or (not positive and value < 0):
        raise RowContractError(f"{field_name} must be a {'positive ' if positive else ''}integer")
    return value


def _sha(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) not in (40, 64) or any(c not in "0123456789abcdef" for c in value):
        raise RowContractError(f"{field_name} must be lower-case hexadecimal SHA-256/SHA-1")
    return value


def _path(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not Path(value).is_absolute() or "\x00" in value:
        raise RowContractError(f"{field_name} must be an absolute path")
    return value


@dataclass(frozen=True)
class RenderedConfig:
    """One immutable rendered/resolved configuration artifact."""

    path: str
    sha256: str
    values: Mapping[str, Any]
    profile: str

    def __post_init__(self) -> None:
        _path(self.path, "rendered_config.path")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise RowContractError("rendered_config.sha256 must be lower-case SHA-256")
        if self.profile not in {"diagnostics-v1", "diagnostics-v2"}:
            raise RowContractError("unsupported diagnostics profile")


@dataclass(frozen=True)
class ParentJoin:
    """Identity of a singleton train invocation and its frozen row document."""

    campaign_id: str
    invocation_id: str
    manifest_hash: str
    train_row_hash: str
    source_sha: str
    rendered_config_sha256: str
    row_id: str
    checkpoint_steps: tuple[int, ...]
    members: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CheckpointEvidence:
    """Named, published checkpoint evidence for one parent invocation."""

    step: int
    publication_id: str
    model_sha256: str
    complete: bool
    files: Mapping[str, str]


@dataclass(frozen=True)
class CampaignMember:
    """One scientific member of an explicitly matched comparison group."""

    invocation_id: str
    family: str
    rate_role: str
    chunk_size: int | None
    runtime_seed: int
    sampler_seed: int


@dataclass(frozen=True)
class ComparisonGroup:
    """A matched group retaining invocation identity and seed membership."""

    campaign_id: str
    comparison_group_id: str
    system: str
    scientific_lineage: int
    members: tuple[CampaignMember, ...]

    def validate(self) -> None:
        """Validate explicit matching without relaxing global seed rules."""

        if not self.members:
            raise RowContractError("comparison group must have explicit members")
        runtime = {member.runtime_seed for member in self.members}
        sampler = {member.sampler_seed for member in self.members}
        if len(runtime) != 1 or len(sampler) != 1:
            raise RowContractError("matched members must share their declared seed pair")
        if next(iter(runtime)) == next(iter(sampler)):
            raise RowContractError("initialization and sampler seeds must be distinct")
        if len({member.invocation_id for member in self.members}) != len(self.members):
            raise RowContractError("comparison members must be distinct invocations")


TRAIN_KEYS = frozenset({
    "row_id", "system", "system_id", "kind", "seed", "source_sha", "config_sha256",
    "config_identity_rule", "config", "results_root", "experiment_name", "sector",
    "run_id", "updates", "walkers", "device", "row_extension", "lineage",
    "sampler_seed", "checkpoint_steps", "overlay", "overlay_sha256",
})
EVAL_KEYS = frozenset({
    "row_id", "system", "system_id", "kind", "seed", "source_sha", "config_sha256",
    "config_identity_rule", "config", "results_root", "experiment_name", "sector",
    "run_id", "updates", "walkers", "device", "row_extension", "lineage", "checkpoint",
    "stream", "draws", "expected_physical_system_id", "checkpoint_dir", "checkpoint_file",
    "checkpoint_sha256", "depends_on", "phase1_manifest_hash", "phase1_train_row_hash",
})


def validate_row(row: Mapping[str, Any], *, expected_kind: str | None = None) -> None:
    """Validate the exact additive P0 row shape and its derived identity."""

    if not isinstance(row, Mapping):
        raise RowContractError("row must be a mapping")
    kind = row.get("kind")
    if kind not in {"train", "eval"} or (expected_kind is not None and kind != expected_kind):
        raise RowContractError("extended rows must be train or eval")
    allowed = TRAIN_KEYS if kind == "train" else EVAL_KEYS
    missing = allowed - set(row)
    extra = set(row) - allowed - ({"execution", "overlay_profile"} if kind == "train" else set())
    if missing or extra:
        raise RowContractError(f"row keys mismatch; missing={sorted(missing)} extra={sorted(extra)}")
    ext = row.get("row_extension")
    if not isinstance(ext, Mapping) or set(ext) != {"version", "rendered_config", "diagnostics_profile"}:
        raise RowContractError("row_extension must contain exactly the three P0 keys")
    if ext.get("version") != "stage1-rendered-config/v1":
        raise RowContractError("unsupported row_extension version")
    rendered = ext.get("rendered_config")
    if not isinstance(rendered, Mapping) or set(rendered) != {"path", "sha256"}:
        raise RowContractError("rendered_config must contain exactly path and sha256")
    _path(rendered.get("path"), "rendered_config.path")
    if (not isinstance(rendered.get("sha256"), str) or len(rendered["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in rendered["sha256"])):
        raise RowContractError("rendered_config.sha256 must be 64 hex characters")
    # P0 sentence: “The extension branch reads, hashes and parses the SAME
    # rendered bytes (no hash-then-load window).”  A row with a stale digest is
    # not admissible even when its digest has the right shape.
    rendered_path = Path(rendered["path"])
    if not rendered_path.is_file():
        raise RowContractError("rendered_config.path must name an existing regular file")
    if sha256_file(rendered_path) != rendered["sha256"]:
        raise RowContractError("rendered_config.sha256 does not match the named bytes")
    if ext.get("diagnostics_profile") not in {"diagnostics-v1", "diagnostics-v2"}:
        raise RowContractError("invalid diagnostics_profile")
    if row.get("config_identity_rule") != "stage1-config-v1":
        raise RowContractError("extended row must retain stage1-config-v1 provenance")
    for key in ("source_sha", "config_sha256", "overlay_sha256" if kind == "train" else "checkpoint_sha256"):
        _sha(row.get(key), key)
    for key in ("seed", "updates", "walkers", "lineage"):
        _int(row.get(key), key, positive=key in {"updates", "walkers"})
    if kind == "train":
        _int(row.get("sampler_seed"), "sampler_seed")
        checkpoints = row.get("checkpoint_steps")
        if not isinstance(checkpoints, list) or not checkpoints or any(type(x) is not int or x <= 0 for x in checkpoints):
            raise RowContractError("checkpoint_steps must be positive integers")
        if len(set(checkpoints)) != len(checkpoints) or row["updates"] not in checkpoints:
            raise RowContractError("checkpoint_steps must be unique and include updates")
    else:
        for key in ("checkpoint", "stream", "draws"):
            _int(row.get(key), key, positive=key == "draws")
        if row["updates"] != row["checkpoint"]:
            raise RowContractError("eval updates must equal checkpoint")
        if row.get("checkpoint_file") != "model.pt":
            raise RowContractError("eval checkpoint_file must be model.pt")
    if row.get("device") not in {"cpu", "cuda"}:
        raise RowContractError("device must be cpu or cuda")
    for key in ("config", "results_root", "experiment_name", "sector", "run_id", "system", "system_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise RowContractError(f"{key} must be non-empty")
    expected_id = row_id(row)
    if row.get("row_id") != expected_id:
        raise RowContractError("row_id does not match row without row_id")


def make_parent_join(*, campaign_id: str, invocation_id: str, row: Mapping[str, Any], members: tuple[str, ...] = ()) -> ParentJoin:
    """Create a parent join after validating a singleton train row."""

    validate_row(row, expected_kind="train")
    ext = row["row_extension"]
    return ParentJoin(
        campaign_id=campaign_id,
        invocation_id=invocation_id,
        manifest_hash=manifest_hash([dict(row)]),
        train_row_hash=canonical_hash(dict(row)),
        source_sha=row["source_sha"],
        rendered_config_sha256=ext["rendered_config"]["sha256"],
        row_id=row["row_id"],
        checkpoint_steps=tuple(row["checkpoint_steps"]),
        members=members,
    )


def make_eval_row(*, train_row: Mapping[str, Any], parent: ParentJoin, checkpoint: CheckpointEvidence,
                  rendered: RenderedConfig, row_base: Mapping[str, Any]) -> dict[str, Any]:
    """Build an eval row joined to actual parent and publication evidence."""

    if not checkpoint.complete:
        raise RowContractError("cannot create an eval row from an incomplete publication")
    if checkpoint.step not in parent.checkpoint_steps:
        raise RowContractError("checkpoint is not in the parent schedule")
    if parent.row_id != train_row.get("row_id"):
        raise RowContractError("parent row does not match train row")
    result = dict(row_base)
    result.update({
        "kind": "eval", "system": train_row["system"], "system_id": train_row["system_id"],
        "source_sha": train_row["source_sha"], "config_sha256": train_row["config_sha256"],
        "config_identity_rule": "stage1-config-v1", "results_root": train_row["results_root"],
        "experiment_name": train_row["experiment_name"], "sector": train_row["sector"],
        "device": train_row["device"], "lineage": train_row["lineage"],
        "checkpoint": checkpoint.step, "updates": checkpoint.step,
        "checkpoint_file": "model.pt", "checkpoint_sha256": checkpoint.model_sha256,
        "depends_on": train_row["row_id"], "phase1_manifest_hash": parent.manifest_hash,
        "phase1_train_row_hash": parent.train_row_hash,
        "expected_physical_system_id": train_row["system_id"],
        "row_extension": {"version": "stage1-rendered-config/v1", "rendered_config": {"path": rendered.path, "sha256": rendered.sha256}, "diagnostics_profile": rendered.profile},
    })
    result["row_id"] = row_id(result)
    validate_row(result, expected_kind="eval")
    return result


__all__ = [
    "CampaignMember", "CheckpointEvidence", "ComparisonGroup", "EVAL_KEYS", "ParentJoin",
    "RenderedConfig", "RowContractError", "TRAIN_KEYS", "canonical_hash", "make_eval_row",
    "make_parent_join", "manifest_hash", "row_id", "sha256_file", "validate_row",
]
