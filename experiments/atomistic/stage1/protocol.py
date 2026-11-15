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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.baselines.systems import load_registry
from experiments.toolkit.run_paths import resolve_run_dir

SYSTEMS = ("hooke1", "hooke2", "he", "h2")
KINDS = ("smoke", "train", "eval")
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# Coordinate bounds are retained in the manifest contract.  The resulting
# seed must also fit Torch's signed 64-bit generator API, so the coordinates
# are hashed into that range; build_manifest rejects any collision in a plan.
_COORDINATE_RADIX = 10**6
_LINEAGE_SEED_RADIX = 10**9
_TORCH_SEED_MODULUS = 2**63 - 1


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
    ("hooke2", "hooke_pair_singlet_omega0.5"),
    ("he", "he_atom"),
    ("h2", "h2_molecule"),
))

# Registry slots are execution labels; the operator identity is the physical
# contract.  Keep this mapping explicit so a same-slot operator substitution
# (for example HarmonicTrap under He's ``kinetic`` slot) cannot pass validation.
_REGISTRY_OPERATOR_IDS = {
    "kinetic": "tpen.physics:kinetic_energy",
    "harmonic_trap": "tpen.physics:harmonic_trap",
    "electron_nucleus": "tpen.physics:electron_nucleus_coulomb",
    "electron_electron": "tpen.physics:electron_electron_coulomb",
    "nucleus_nucleus": "tpen.physics:nucleus_nucleus_coulomb",
}


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


def _require_path_component(value: object, *, field: str) -> str:
    """Require one non-empty, relative path component for a run identity."""

    if (not isinstance(value, str) or not value or value in {".", ".."}
            or "/" in value or Path(value).is_absolute()):
        raise ProtocolError(f"{field} must be one relative path component")
    return value


def _checkpoint_binding(row: Mapping[str, Any]) -> tuple[Path, Path]:
    """Validate and canonicalize the checkpoint directory/file pair."""

    checkpoint_dir = row.get("checkpoint_dir")
    checkpoint_file = row.get("checkpoint_file")
    if (not isinstance(checkpoint_dir, str) or not isinstance(checkpoint_file, str)
            or not Path(checkpoint_dir).is_absolute()
            or not Path(checkpoint_file).is_absolute()):
        raise ProtocolError("checkpoint_dir and checkpoint_file must be absolute")
    canonical_dir = Path(checkpoint_dir).resolve()
    canonical_file = Path(checkpoint_file).resolve()
    if Path(checkpoint_dir) != canonical_dir or Path(checkpoint_file) != canonical_file:
        raise ProtocolError("checkpoint_dir and checkpoint_file must be canonical absolute paths")
    if canonical_file != canonical_dir / "model.pt":
        raise ProtocolError("checkpoint_file must equal checkpoint_dir/model.pt")
    return canonical_dir, canonical_file


def _validate_run_identity(row: Mapping[str, Any]) -> None:
    """Validate run identity and keep its effective directory beneath the root."""

    for key in ("experiment_name", "sector", "run_id"):
        _require_path_component(row.get(key), field=key)
    root = row.get("results_root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ProtocolError("results_root must be an absolute path")
    try:
        resolve_run_dir(row).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError) as exc:
        raise ProtocolError("effective run directory must be beneath results_root") from exc


def stable_row_id(*, system: str, kind: str, lineage: int = 0,
                  checkpoint: int = 0, stream: int = 0) -> str:
    """Return a deterministic, collision-resistant row identity."""

    if system not in SYSTEMS or kind not in KINDS:
        raise ProtocolError(f"unknown row coordinates: {system=} {kind=}")
    if min(lineage, checkpoint, stream) < 0:
        raise ProtocolError("row coordinates must be non-negative")
    payload = json.dumps(["stage1/v3", system, kind, lineage, checkpoint, stream],
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def training_seed(*, system: str, lineage: int, lineage_seed: int) -> int:
    """Derive a bounded deterministic training seed from row coordinates."""

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
    """Derive a bounded deterministic evaluation seed for any checkpoint list."""

    if (system not in SYSTEMS or min(lineage, checkpoint, stream) < 0
            or lineage >= 10**6 or checkpoint >= 10**6 or stream >= 10**6):
        raise ProtocolError("invalid evaluation seed coordinates")
    # The checkpoint is a coordinate, not an array index, so non-default lists
    # cannot alias one another.  The leading kind/system fields separate all
    # three row kinds and all four systems.
    return _encode_seed(kind="eval", system=system, lineage=lineage,
                        checkpoint=checkpoint, stream=stream)


def _encode_seed(*, kind: str, system: str, lineage: int = 0,
                 checkpoint: int = 0, stream: int = 0,
                 lineage_seed: int = 0) -> int:
    """Encode bounded coordinates as a deterministic Torch-compatible seed.

    The coordinate domain is larger than Torch's signed 64-bit seed space, so
    injectivity cannot be guaranteed by a mixed-radix integer.  The manifest
    builder therefore checks the finite plan for collisions after this stable
    digest-based encoding.
    """

    if kind not in KINDS or system not in SYSTEMS:
        raise ProtocolError("invalid seed namespace")
    fields = (lineage, checkpoint, stream, lineage_seed)
    limits = (_COORDINATE_RADIX, _COORDINATE_RADIX,
              _COORDINATE_RADIX, _LINEAGE_SEED_RADIX)
    if any(value < 0 or value >= limit for value, limit in zip(fields, limits)):
        raise ProtocolError("seed coordinate exceeds its fixed field width")
    payload = json.dumps(
        ["stage1/seed-v2", kind, system, *fields], separators=(",", ":")
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % _TORCH_SEED_MODULUS


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
    _validate_run_identity(row)
    expected = {
        f"runtime.seed={row['seed']}",
        f"run.root={row['results_root']}",
        f"experiment.name={row['experiment_name']}",
        f"experiment.sector={row['sector']}",
        f"run.run_id={row['run_id']}",
        "run.layout=nested",
    }
    if kind == "eval":
        checkpoint_dir, checkpoint_file = _checkpoint_binding(row)
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
        if row.get("checkpoint_steps"):
            steps = ",".join(str(step) for step in row["checkpoint_steps"])
            expected.add(f"checkpoint_steps=[{steps}]")
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


def physical_system_id_from_config(
    config_path: str | Path,
    *,
    train_config_path: str | Path | None = None,
    system_name: str | None = None,
) -> str:
    """Validate an eval descriptor and return its physical content ID.

    When ``train_config_path`` is supplied, the train Hamiltonian is built
    from its actual configured terms and compared with the eval descriptor
    before the eval identity is admitted to a plan.  The content ID covers
    particles, operator IDs, physical parameters, and resources; slots and
    their operator bindings are compared explicitly because slots are omitted
    from the descriptor's physical content hash.
    """

    cfg, descriptor = _load_configured_descriptor(config_path)
    registry = load_registry()
    entry = next((item for item in registry.get("systems", [])
                  if item.get("id") == cfg.system.id), None)
    if entry is None:
        raise ProtocolError(f"system {cfg.system.id!r} is absent from the baseline registry")
    particle = descriptor.particle
    for key, actual in (("n_up", particle.n_spin_up),
                        ("n_down", particle.n_spin_down),
                        ("spatial_dim", particle.spatial_dim)):
        if entry.get(key) != actual:
            raise ProtocolError(f"descriptor {key} disagrees with registry for {cfg.system.id}")
    if entry.get("n_up", 0) + entry.get("n_down", 0) != particle.n_electrons:
        raise ProtocolError(f"descriptor particle count disagrees with registry for {cfg.system.id}")
    registry_terms = entry.get("hamiltonian", {}).get("terms", [])
    configured_names = {term.slot for term in descriptor.terms}
    registry_names = {term.get("term") for term in registry_terms}
    if configured_names != registry_names:
        raise ProtocolError(f"descriptor operator set disagrees with registry for {cfg.system.id}")
    descriptor_by_slot = {term.slot: term for term in descriptor.terms}
    for registry_term in registry_terms:
        slot = registry_term.get("term")
        descriptor_term = descriptor_by_slot[slot]
        expected_operator = _REGISTRY_OPERATOR_IDS.get(slot)
        if expected_operator is None:
            raise ProtocolError(f"registry has no operator identity for slot {slot!r}")
        if str(descriptor_term.operator_id) != expected_operator:
            raise ProtocolError(
                f"descriptor operator for slot {slot!r} disagrees with registry "
                f"for {cfg.system.id}"
            )
        if slot == "harmonic_trap" and descriptor_term.parameters.get("omega") != float(registry_term["omega"]):
            raise ProtocolError(f"descriptor trap omega disagrees with registry for {cfg.system.id}")
    expected_nuclei = [term for term in registry_terms if term.get("term") == "electron_nucleus"]
    if expected_nuclei:
        nuclei = expected_nuclei[0].get("nuclei", [])
        resources = list(descriptor.resources.values())
        if len(resources) != 1:
            raise ProtocolError(f"descriptor nuclear resources disagree with registry for {cfg.system.id}")
        resource = resources[0]
        actual_positions = resource.positions.detach().cpu().tolist()
        actual_charges = resource.charges.detach().cpu().tolist()
        expected_positions = [list(n.get("position_bohr", [])) for n in nuclei]
        expected_charges = []
        for nucleus in nuclei:
            charge = nucleus.get("charge")
            if isinstance(charge, bool) or not isinstance(charge, int):
                raise ProtocolError(
                    f"registry nuclear charges must be integers for {cfg.system.id}"
                )
            expected_charges.append(charge)
        # Registry positions are in bohr.  They are compared with an absolute
        # tolerance because the instantiated default-float32 path can round a
        # decimal coordinate by a few ulps; charges remain exact integers.
        position_match = (
            len(actual_positions) == len(expected_positions)
            and all(
                len(actual) == len(expected)
                and all(math.isclose(float(a), float(e), rel_tol=0.0, abs_tol=1.0e-6)
                        for a, e in zip(actual, expected))
                for actual, expected in zip(actual_positions, expected_positions)
            )
        )
        charge_match = (
            len(actual_charges) == len(expected_charges)
            and all(float(a) == e
                    for a, e in zip(actual_charges, expected_charges))
        )
        if not position_match or not charge_match:
            raise ProtocolError(f"descriptor nuclear geometry disagrees with registry for {cfg.system.id}")
    elif any(term.resource is not None for term in descriptor.terms):
        raise ProtocolError(f"descriptor has an unexpected nuclear resource for {cfg.system.id}")
    system_id = descriptor.system_id()
    if not isinstance(system_id, str) or not _SHA256.fullmatch(system_id):
        raise ProtocolError("configured system descriptor did not produce a SHA-256 ID")
    if train_config_path is not None:
        label = system_name or str(cfg.system.id)
        try:
            train_cfg, train_descriptor = _load_configured_descriptor(train_config_path)
        except Exception as exc:
            raise ProtocolError(f"train/eval Hamiltonian mismatch for {label}") from exc
        eval_bindings = tuple(sorted((term.slot, str(term.operator_id))
                                     for term in descriptor.terms))
        train_bindings = tuple(sorted((term.slot, str(term.operator_id))
                                      for term in train_descriptor.terms))
        if (train_cfg.system.id != cfg.system.id
                or train_bindings != eval_bindings
                or train_descriptor.system_id() != system_id):
            raise ProtocolError(f"train/eval Hamiltonian mismatch for {label}")
    return system_id


def _load_configured_descriptor(config_path: str | Path) -> tuple[Any, Any]:
    """Load a config and instantiate its config-owned system descriptor."""

    from hydra.utils import instantiate
    from tpen.run import load_config

    cfg = load_config(str(config_path))
    return cfg, instantiate(cfg.system_descriptor)


def build_manifest(*, protocol: Protocol, source_sha: str, config_sha256: str,
                   checkpoints: list[int], lineage_seeds: list[int],
                   checkpoint_sha256: Mapping[tuple[str, int, int], str],
                   results_root: str,
                   expected_physical_system_id: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Expand systems into validated smoke, train, and eval rows.

    Parameters are explicit rather than defaulted so a missing identity or
    checkpoint binding fails before a launch script can be rendered.
    """

    protocol.validate()
    source_sha = _require_sha(source_sha, length=40, field="source_sha")
    config_sha256 = _require_sha(config_sha256, length=64, field="config_sha256")
    root = _safe_results_root(results_root)
    physical_ids = dict(expected_physical_system_id or {})
    if not physical_ids:
        config_root = Path(__file__).parent / "configs"
        physical_ids = {
            name: physical_system_id_from_config(
                config_root / name / "eval.yaml",
                train_config_path=config_root / name / "train.yaml",
                system_name=name,
            )
            for name in SYSTEMS
        }
    for name in SYSTEMS:
        _require_sha(physical_ids.get(name), length=64,
                     field=f"expected_physical_system_id[{name!r}]")
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
                "checkpoint_steps": list(checkpoints),
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
                        "expected_physical_system_id": physical_ids[spec.name],
                        "experiment_name": "stage1", "sector": spec.name, "run_id": eval_id,
                        "checkpoint_dir": str(checkpoint_dir), "checkpoint_file": str(checkpoint_file),
                        "checkpoint_sha256": checkpoint_digest, "depends_on": train_id,
                    })
    for row in rows:
        _validate_run_identity(row)
        if row["kind"] == "eval":
            _checkpoint_binding(row)
    return rows


def manifest_hash(rows: list[Mapping[str, Any]]) -> str:
    """Hash a manifest using canonical JSON serialization."""

    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
