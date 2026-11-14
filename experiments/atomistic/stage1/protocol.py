"""System-neutral Stage-1 protocol and identity-bound manifest primitives.

The module is intentionally independent of :mod:`tpen`.  Execution enters the
package only through the thin ``train`` and ``eval`` drivers.
"""

from __future__ import annotations

import hashlib
import json
import re
import base64
import binascii
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SYSTEMS = ("hooke1", "he", "h2")
KINDS = ("smoke", "train", "eval")
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# Seed coordinates are encoded as fixed-width mixed-radix fields.  The kind
# and system fields lead every seed, so no coordinate in one namespace can
# alias a coordinate in another namespace.
_KIND_RADIX = len(KINDS)
_SYSTEM_RADIX = len(SYSTEMS)
_COORDINATE_RADIX = 10**6
_LINEAGE_SEED_RADIX = 10**9


class ProtocolError(ValueError):
    """Raised when a Stage-1 protocol or manifest row is incomplete."""


@dataclass(frozen=True)
class SystemSpec:
    """A system-neutral descriptor used by every Stage-1 row."""

    name: str
    system_id: str

    def validate(self) -> None:
        """Reject unknown or incomplete system descriptors."""

        if self.name not in SYSTEMS or not self.system_id:
            raise ProtocolError(f"invalid system descriptor: {self!r}")


@dataclass(frozen=True)
class Protocol:
    """The complete Stage-1 sampling and lineage contract.

    All fields are required deliberately: callers cannot silently inherit a
    protocol value when constructing a manifest from a partial payload.
    """

    smoke_updates: int
    smoke_walkers: int
    lineage_updates: int
    lineage_walkers: int
    lineages: int
    eval_streams: int
    eval_walkers: int
    eval_draws: int

    def validate(self) -> None:
        """Reject non-positive dimensions and underspecified stream counts."""

        values = (self.smoke_updates, self.smoke_walkers, self.lineage_updates,
                  self.lineage_walkers, self.lineages, self.eval_streams,
                  self.eval_walkers, self.eval_draws)
        if any(value <= 0 for value in values):
            raise ProtocolError(f"protocol dimensions must be positive: {values}")
        if self.lineages < 3 or self.eval_streams < 4:
            raise ProtocolError("Stage-1 requires at least 3 lineages and 4 eval streams")


STAGE1_PROTOCOL = Protocol(1000, 256, 50000, 4096, 3, 4, 4096, 256)
SYSTEM_SPECS = tuple(SystemSpec(name, system_id) for name, system_id in (
    ("hooke1", "hooke_one_electron_omega0.5"),
    ("he", "he_atom"),
    ("h2", "h2_molecule"),
))


def _require_sha(value: object, *, length: int, field: str) -> str:
    """Return a lowercase hexadecimal digest or raise a protocol error."""

    text = value if isinstance(value, str) else ""
    pattern = _SHA1 if length == 40 else _SHA256
    if not pattern.fullmatch(text):
        raise ProtocolError(f"{field} must be {length} lowercase hex characters")
    return text


def _safe_results_root(value: object) -> str:
    """Require an absolute results root outside the source checkout."""

    if not isinstance(value, str) or not value.startswith("/"):
        raise ProtocolError("results_root must be an absolute path outside the checkout")
    path = Path(value)
    if any(part == "experiments" for part in path.parts):
        raise ProtocolError("results_root must not be inside the checkout")
    return value.rstrip("/")


def resolve_run_dir(row: Mapping[str, Any]) -> Path:
    """Resolve one row through TPEN's own artifact-directory implementation."""

    try:
        from tpen.artifacts import ArtifactManager

        admitted = row.get("run_dir")
        return ArtifactManager(
            row["results_root"], row["experiment_name"], row["sector"], row["run_id"],
            layout=row.get("run_layout", "nested"),
            logical_run_dir=None if admitted is None else Path(str(admitted)),
        ).run_dir
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"row lacks a valid run-directory binding: {row!r}") from exc


def stable_row_id(*, system: str, kind: str, lineage: int = 0,
                  checkpoint: int = 0, stream: int = 0) -> str:
    """Return a deterministic, collision-resistant row identity."""

    if system not in SYSTEMS or kind not in KINDS:
        raise ProtocolError(f"unknown row coordinates: {system=} {kind=}")
    if min(lineage, checkpoint, stream) < 0:
        raise ProtocolError("row coordinates must be non-negative")
    payload = json.dumps(["stage1/v2", system, kind, lineage, checkpoint, stream],
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def training_seed(*, system: str, lineage: int, lineage_seed: int) -> int:
    """Derive a globally injective training seed from row coordinates."""

    if (system not in SYSTEMS or lineage < 0 or lineage >= 10**6
            or lineage_seed < 0 or lineage_seed >= 10**9):
        raise ProtocolError("invalid training seed coordinates")
    return _encode_seed(kind="train", system=system, lineage=lineage,
                        lineage_seed=lineage_seed)


def smoke_seed(*, system: str) -> int:
    """Derive the dedicated smoke namespace seed for one system."""

    if system not in SYSTEMS:
        raise ProtocolError("invalid smoke seed system")
    return _encode_seed(kind="smoke", system=system)


def evaluation_seed(*, system: str, lineage: int, checkpoint: int, stream: int) -> int:
    """Derive an injective evaluation seed for arbitrary checkpoint lists."""

    if (system not in SYSTEMS or min(lineage, checkpoint, stream) < 0
            or lineage >= 10**6 or checkpoint >= 10**6 or stream >= 10**6):
        raise ProtocolError("invalid evaluation seed coordinates")
    # The checkpoint is a coordinate, not an array index, so non-default lists
    # cannot alias one another.  The leading kind/system fields separate all
    # three row kinds and all three systems.
    return _encode_seed(kind="eval", system=system, lineage=lineage,
                        checkpoint=checkpoint, stream=stream)


def _encode_seed(*, kind: str, system: str, lineage: int = 0,
                 checkpoint: int = 0, stream: int = 0,
                 lineage_seed: int = 0) -> int:
    """Encode the bounded seed coordinates with documented fixed radices."""

    if kind not in KINDS or system not in SYSTEMS:
        raise ProtocolError("invalid seed namespace")
    fields = (lineage, checkpoint, stream, lineage_seed)
    limits = (_COORDINATE_RADIX, _COORDINATE_RADIX,
              _COORDINATE_RADIX, _LINEAGE_SEED_RADIX)
    if any(value < 0 or value >= limit for value, limit in zip(fields, limits)):
        raise ProtocolError("seed coordinate exceeds its fixed field width")
    namespace = KINDS.index(kind) * _SYSTEM_RADIX + SYSTEMS.index(system)
    encoded = namespace
    for value, radix in zip(fields, limits):
        encoded = encoded * radix + value
    return encoded


def identity_expectations(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the checkpoint expectation bound by an evaluation row.

    The resolved-config identity is owned by the evaluation run and is joined
    to the record at collection time by S1-A2. This protocol slice cannot
    compute that core identity, so it must not fabricate one.
    """

    if row.get("kind") != "eval":
        raise ProtocolError("identity expectations are defined for eval rows only")
    return {
        "expected_checkpoint_sha256": _require_sha(
            row.get("checkpoint_sha256"), length=64, field="checkpoint_sha256"
        ),
    }


def decode_binding(token: str) -> dict[str, Any]:
    """Decode the launcher's complete row binding before loading a config."""

    try:
        padding = "=" * (-len(token) % 4)
        value = json.loads(base64.urlsafe_b64decode(token + padding))
    except (ValueError, TypeError, binascii.Error, json.JSONDecodeError) as exc:
        raise ProtocolError("binding is not valid URL-safe JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("binding must decode to a row object")
    return value


def validate_gateway_binding(*, binding: str, config: str, overrides: list[str], kind: str,
                             expected_checkpoint_sha256: str | None = None) -> dict[str, Any]:
    """Validate that gateway arguments still match the rendered row binding."""

    row = decode_binding(binding)
    if row.get("kind") != kind:
        raise ProtocolError("binding kind does not match gateway")
    if config != row.get("config"):
        raise ProtocolError("gateway config does not match binding")
    expected = {
        f"runtime.seed={row['seed']}",
        f"run.root={row['results_root']}",
        f"experiment.name={row['experiment_name']}",
        f"experiment.sector={row['sector']}",
        f"run.run_id={row['run_id']}",
    }
    if kind == "eval":
        checkpoint_dir = row.get("checkpoint_dir")
        checkpoint_file = row.get("checkpoint_file")
        if (not isinstance(checkpoint_dir, str) or not Path(checkpoint_dir).is_absolute()
                or not isinstance(checkpoint_file, str) or not Path(checkpoint_file).is_absolute()):
            raise ProtocolError("checkpoint_dir and checkpoint_file must be absolute")
        try:
            Path(checkpoint_file).relative_to(Path(checkpoint_dir))
        except ValueError as exc:
            raise ProtocolError("checkpoint_file must be inside checkpoint_dir") from exc
        expected.update({
            f"evaluation_sampler.n_walkers={row['walkers']}",
            f"evaluation_tasks.mcmc_energy.generator.n_draws={row['draws']}",
            f"load.path={checkpoint_dir}", "load.mode=model_only",
            "load.strict=true",
            f"trajectory_identity.checkpoint_file={checkpoint_file}",
        })
        if expected_checkpoint_sha256 is None:
            raise ProtocolError("eval requires strict restore and expected checkpoint")
        _require_sha(expected_checkpoint_sha256, length=64,
                     field="expected_checkpoint_sha256")
        if expected_checkpoint_sha256 != row.get("checkpoint_sha256"):
            raise ProtocolError("expected checkpoint sha256 does not match binding")
    else:
        expected.update({f"sampler.n_walkers={row['walkers']}",
                         f"trainer.max_steps={row['updates']}"})
    override_map: dict[str, str] = {}
    for override in overrides:
        key, separator, value = override.partition("=")
        if key.startswith(("+", "~")):
            raise ProtocolError("prefixed gateway overrides are not allowed")
        if not separator or not key:
            raise ProtocolError(f"invalid gateway override: {override!r}")
        if key in override_map:
            raise ProtocolError(f"duplicate gateway override: {key}")
        override_map[key] = value
    expected_map = {entry.partition("=")[0]: entry.partition("=")[2] for entry in expected}
    missing = sorted(key for key in expected_map if key not in override_map)
    mismatched = sorted(
        key for key, value in expected_map.items()
        if key in override_map and override_map[key] != value
    )
    if missing or mismatched:
        raise ProtocolError(
            f"binding overrides do not match: missing={missing}, mismatched={mismatched}"
        )
    return row


def build_manifest(*, protocol: Protocol, source_sha: str, config_sha256: str,
                   checkpoints: list[int], lineage_seeds: list[int],
                   checkpoint_sha256: Mapping[tuple[str, int, int], str],
                   results_root: str) -> list[dict[str, Any]]:
    """Expand systems into validated smoke, train, and eval rows.

    Parameters are explicit rather than defaulted so a missing identity or
    checkpoint binding fails before a launch script can be rendered.
    """

    protocol.validate()
    source_sha = _require_sha(source_sha, length=40, field="source_sha")
    config_sha256 = _require_sha(config_sha256, length=64, field="config_sha256")
    root = _safe_results_root(results_root)
    if len(lineage_seeds) != protocol.lineages or len(set(lineage_seeds)) != len(lineage_seeds):
        raise ProtocolError("lineage_seeds must contain one distinct seed per lineage")
    if not checkpoints or any(step <= 0 for step in checkpoints):
        raise ProtocolError("checkpoints must be a non-empty list of positive updates")
    rows: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for spec in SYSTEM_SPECS:
        spec.validate()
        seed = smoke_seed(system=spec.name)
        if seed in seen_seeds:
            raise ProtocolError("seed collision across Stage-1 rows")
        seen_seeds.add(seed)
        rows.append({
            "row_id": stable_row_id(system=spec.name, kind="smoke"),
            "system": spec.name, "system_id": spec.system_id, "kind": "smoke",
            "seed": seed, "source_sha": source_sha, "config_sha256": config_sha256,
            "config": f"configs/{spec.name}/smoke.yaml", "results_root": root,
            "experiment_name": "stage1", "sector": spec.name, "run_id": "smoke",
            "updates": protocol.smoke_updates,
            "walkers": protocol.smoke_walkers,
        })
        for lineage, seed in enumerate(lineage_seeds):
            train_seed = training_seed(system=spec.name, lineage=lineage, lineage_seed=seed)
            if train_seed in seen_seeds:
                raise ProtocolError("seed collision across Stage-1 rows")
            seen_seeds.add(train_seed)
            train_id = stable_row_id(system=spec.name, kind="train", lineage=lineage)
            rows.append({
                "row_id": train_id, "system": spec.name, "system_id": spec.system_id,
                "kind": "train", "lineage": lineage, "seed": train_seed,
                "source_sha": source_sha, "config_sha256": config_sha256,
                "config": f"configs/{spec.name}/train.yaml", "results_root": root,
                "experiment_name": "stage1", "sector": spec.name,
                "run_id": f"train-lineage-{lineage}",
                "updates": protocol.lineage_updates,
                "walkers": protocol.lineage_walkers,
            })
            train_row = rows[-1]
            for checkpoint in checkpoints:
                for stream in range(protocol.eval_streams):
                    key = (spec.name, lineage, checkpoint)
                    checkpoint_digest = _require_sha(
                        checkpoint_sha256.get(key), length=64,
                        field=f"checkpoint_sha256[{key!r}]",
                    )
                    eval_seed = evaluation_seed(system=spec.name, lineage=lineage,
                                                 checkpoint=checkpoint, stream=stream)
                    if eval_seed in seen_seeds:
                        raise ProtocolError("seed collision across Stage-1 rows")
                    seen_seeds.add(eval_seed)
                    row_id = stable_row_id(system=spec.name, kind="eval", lineage=lineage,
                                           checkpoint=checkpoint, stream=stream)
                    eval_id = f"{spec.name}-lineage-{lineage}-checkpoint-{checkpoint}-stream-{stream}"
                    checkpoint_dir = resolve_run_dir(train_row) / "checkpoints" / f"step_{checkpoint:06d}"
                    checkpoint_file = checkpoint_dir / "model.pt"
                    rows.append({
                        "row_id": row_id, "system": spec.name, "system_id": spec.system_id,
                        "kind": "eval", "lineage": lineage, "checkpoint": checkpoint,
                        "stream": stream, "seed": eval_seed, "updates": checkpoint,
                        "walkers": protocol.eval_walkers, "draws": protocol.eval_draws,
                        "source_sha": source_sha, "config_sha256": config_sha256,
                        "config": f"configs/{spec.name}/eval.yaml", "results_root": root,
                        "experiment_name": "stage1", "sector": spec.name, "run_id": eval_id,
                        "checkpoint_dir": str(checkpoint_dir), "checkpoint_file": str(checkpoint_file),
                        "checkpoint_sha256": checkpoint_digest, "depends_on": train_id,
                    })
    return rows


def manifest_hash(rows: list[Mapping[str, Any]]) -> str:
    """Hash a manifest using canonical JSON serialization."""

    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
