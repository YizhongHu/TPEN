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

from omegaconf import OmegaConf

from experiments.baselines.systems import load_registry
from experiments.toolkit.run_paths import resolve_run_dir

SYSTEMS = ("hooke1", "hooke2", "he", "h2")
KINDS = ("smoke", "train", "eval")
CONFIG_IDENTITY_RULE = "stage1-config-v1"
LEGACY_CONFIG_IDENTITY_RULE = "legacy-config-sha256"
CHECKPOINT_EVIDENCE_SCHEMA = "stage1-checkpoint-evidence/v1"
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


STAGE1_PROTOCOL = Protocol(1000, 256, 100000, 4096, 3, 4, 4096, 256)
SYSTEM_SPECS = tuple(SystemSpec(name, system_id) for name, system_id in (
    ("hooke1", "hooke_one_electron_omega0.5"),
    ("hooke2", "hooke_pair_singlet_omega0.5"),
    ("he", "he_atom"),
    ("h2", "h2_molecule"),
))


def resolve_config_path(config: str | Path) -> Path:
    """Resolve a Stage-1 config independently of the submission directory.

    Manifest rows use paths relative to the Stage-1 study directory, which is
    the directory that owns ``configs/``. Absolute paths are accepted as-is
    so callers can pass the same bound config after resolving it once.
    """

    path = Path(config)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _config_files_for_system(system: str, config_root: str | Path) -> tuple[Path, ...]:
    """Return the canonical config inputs owned by one Stage-1 system.

    Stage-1 configs currently have empty Hydra defaults, so the two explicit
    train/eval files are the complete include closure.  Keeping this closure
    in one function makes the hashing rule fail closed if a future config
    grows a supported include mechanism.
    """

    if system not in SYSTEMS:
        raise ProtocolError(f"unknown Stage-1 system {system!r}")
    root = Path(config_root).resolve()
    paths = tuple((root / system / f"{kind}.yaml").resolve() for kind in ("train", "eval"))
    if any(not path.is_file() for path in paths):
        missing = next(path for path in paths if not path.is_file())
        raise ProtocolError(f"missing Stage-1 config input {missing}")
    return paths


def config_identity(*, config_root: str | Path, systems: tuple[str, ...] | list[str] | None = None) -> dict[str, Any]:
    """Compute the versioned, canonical identity of selected config inputs.

    The digest covers a sorted path list and each file's canonical bytes.  It
    therefore changes for either member of a system's train/eval pair and is
    unaffected by files outside the selected systems.
    """

    selected = tuple(SYSTEMS if systems is None else systems)
    if not selected or len(set(selected)) != len(selected):
        raise ProtocolError("systems must contain one or more distinct systems")
    files = sorted({path for system in selected for path in _config_files_for_system(system, config_root)},
                   key=lambda path: path.relative_to(Path(config_root).resolve()).as_posix())
    digest = hashlib.sha256()
    entries: list[str] = []
    root = Path(config_root).resolve()
    for path in files:
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(raw)).encode("ascii"))
        digest.update(b"\0")
        digest.update(raw)
        entries.append(relative)
    return {"rule": CONFIG_IDENTITY_RULE, "sha256": digest.hexdigest(), "files": entries}


def config_identities(*, config_root: str | Path, systems: tuple[str, ...] | list[str] | None = None) -> dict[str, str]:
    """Return one computed config digest per selected system."""

    selected = tuple(SYSTEMS if systems is None else systems)
    if not selected or len(set(selected)) != len(selected) or any(system not in SYSTEMS for system in selected):
        raise ProtocolError("systems must contain distinct known Stage-1 systems")
    return {system: str(config_identity(config_root=config_root, systems=(system,))["sha256"])
            for system in selected}


def _validate_config_identity_for_row(row: Mapping[str, Any], config_path: str | Path) -> None:
    """Verify a computed Stage-1 identity against the execution config tree."""

    if row.get("config_identity_rule") != CONFIG_IDENTITY_RULE:
        return
    system = row.get("system")
    if not isinstance(system, str) or system not in SYSTEMS:
        raise ProtocolError("computed config identity requires a known system")
    path = Path(config_path).resolve()
    expected_name = "eval.yaml" if row.get("kind") == "eval" else "train.yaml"
    expected_path = path.parent / expected_name
    if path != expected_path or path.parent.name != system:
        raise ProtocolError("config path does not match the bound Stage-1 system")
    actual = config_identity(config_root=path.parent.parent, systems=(system,))["sha256"]
    if actual != row.get("config_sha256"):
        raise ProtocolError("executed config identity does not match binding")


def compute_config_sha256(*, config_root: str | Path,
                          systems: tuple[str, ...] | list[str] | None = None) -> str:
    """Compatibility helper returning the digest for a selected config set."""

    return str(config_identity(config_root=config_root, systems=systems)["sha256"])

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
    row_config = row.get("config")
    if not isinstance(config, str) or not isinstance(row_config, str):
        raise ProtocolError("gateway config is invalid")
    if row.get("config_identity_rule") not in {
        None, CONFIG_IDENTITY_RULE, LEGACY_CONFIG_IDENTITY_RULE,
    }:
        raise ProtocolError("binding config identity rule is unsupported")
    if resolve_config_path(config) != resolve_config_path(row_config):
        raise ProtocolError("gateway config does not match binding")
    _validate_config_identity_for_row(row, resolve_config_path(config))
    _validate_run_identity(row)
    expected = {
        f"runtime.seed={row['seed']}",
        f"run.root={row['results_root']}",
        f"experiment.name={row['experiment_name']}",
        f"experiment.sector={row['sector']}",
        f"run.run_id={row['run_id']}",
        "run.layout=nested",
    }
    if "device" in row:
        device = row["device"]
        if device not in {"cpu", "cuda"}:
            raise ProtocolError("binding device must be 'cpu' or 'cuda'")
        expected.add(f"runtime.device={device}")
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
            raise ProtocolError(
                f"train/eval Hamiltonian mismatch for {label} config {train_config_path}"
            ) from exc
        eval_bindings = tuple(sorted((term.slot, str(term.operator_id))
                                     for term in descriptor.terms))
        train_bindings = tuple(sorted((term.slot, str(term.operator_id))
                                      for term in train_descriptor.terms))
        if (train_cfg.system.id != cfg.system.id
                or train_bindings != eval_bindings
                or train_descriptor.system_id() != system_id):
            raise ProtocolError(f"train/eval Hamiltonian mismatch for {label}")
    return system_id


def planned_physical_system_ids(
    config_root: str | Path | None = None,
) -> dict[str, str]:
    """Return train/eval-validated physical IDs for every planned system."""

    root = Path(config_root) if config_root is not None else Path(__file__).parent / "configs"
    return {
        name: physical_system_id_from_config(
            root / name / "eval.yaml",
            train_config_path=root / name / "train.yaml",
            system_name=name,
        )
        for name in SYSTEMS
    }


def _load_configured_descriptor(config_path: str | Path) -> tuple[Any, Any]:
    """Load a config and instantiate its config-owned system descriptor."""

    from hydra.utils import instantiate
    from tpen.run import load_config

    cfg = load_config(str(config_path))
    system_name = str(cfg.system.id)
    descriptor_terms = OmegaConf.to_container(cfg.system_descriptor.terms, resolve=True)
    if "runner" in cfg and "hamiltonian_terms" in cfg.runner:
        consumed_terms = cfg.runner.hamiltonian_terms
        consumer = "runner.hamiltonian_terms"
    elif "evaluation_tasks" in cfg:
        consumed_terms = cfg.evaluation_tasks.mcmc_energy.generator.hamiltonian_terms
        consumer = "evaluation_tasks.mcmc_energy.generator.hamiltonian_terms"
    else:
        raise ProtocolError(
            f"system {system_name!r} config {config_path}: "
            "no consumed hamiltonian_terms binding"
        )
    resolved_consumed_terms = OmegaConf.to_container(consumed_terms, resolve=True)
    if descriptor_terms != resolved_consumed_terms:
        raise ProtocolError(
            f"system {system_name!r} config {config_path}: "
            f"system_descriptor.terms do not match consumed {consumer}"
        )
    return cfg, instantiate(cfg.system_descriptor)


def build_manifest(*, protocol: Protocol, source_sha: str,
                   config_sha256: str | Mapping[str, str] | None,
                   checkpoints: list[int], lineage_seeds: list[int],
                   checkpoint_sha256: Mapping[tuple[str, int, int], str],
                   results_root: str,
                   expected_physical_system_id: Mapping[str, str] | None = None,
                   systems: tuple[str, ...] | list[str] | None = None,
                   phase: int = 2, device: str | None = None) -> list[dict[str, Any]]:
    """Expand systems into validated smoke, train, and eval rows.

    Parameters are explicit rather than defaulted so a missing identity or
    checkpoint binding fails before a launch script can be rendered. The
    explicit physical-ID map is retained for unit fixtures; production
    planning leaves it unset so train/eval parity is always checked here.
    """

    protocol.validate()
    source_sha = _require_sha(source_sha, length=40, field="source_sha")
    if phase not in {1, 2}:
        raise ProtocolError("manifest phase must be 1 or 2")
    selected = tuple(SYSTEMS if systems is None else systems)
    if not selected or len(set(selected)) != len(selected) or any(system not in SYSTEMS for system in selected):
        raise ProtocolError("systems must contain distinct known Stage-1 systems")
    if device is not None and device not in {"cpu", "cuda"}:
        raise ProtocolError("device must be 'cpu' or 'cuda'")
    if not checkpoints or any(type(step) is not int or step < 0 for step in checkpoints):
        raise ProtocolError("checkpoints must be a non-empty list of non-negative updates")
    if len(set(checkpoints)) != len(checkpoints):
        raise ProtocolError("checkpoints must not contain duplicates")
    if phase == 1:
        if any(step > protocol.lineage_updates for step in checkpoints):
            raise ProtocolError("checkpoints must not exceed lineage_updates")
        if protocol.lineage_updates not in checkpoints:
            raise ProtocolError("checkpoints must retain the terminal lineage_updates checkpoint")
        if 0 not in checkpoints:
            raise ProtocolError("checkpoints must retain the initial zero checkpoint")
    if isinstance(config_sha256, Mapping):
        config_ids = {system: _require_sha(config_sha256.get(system), length=64,
                                           field=f"config_sha256[{system!r}]") for system in selected}
        config_identity_rule = CONFIG_IDENTITY_RULE
    else:
        legacy_config_sha = _require_sha(config_sha256, length=64, field="config_sha256")
        config_ids = {system: legacy_config_sha for system in selected}
        config_identity_rule = LEGACY_CONFIG_IDENTITY_RULE
    root = _safe_results_root(results_root)
    physical_ids = dict(expected_physical_system_id or {})
    if phase == 2 and not physical_ids:
        physical_ids = planned_physical_system_ids()
    if phase == 2:
        for name in selected:
            _require_sha(physical_ids.get(name), length=64,
                         field=f"expected_physical_system_id[{name!r}]")
    if len(lineage_seeds) != protocol.lineages or len(set(lineage_seeds)) != len(lineage_seeds):
        raise ProtocolError("lineage_seeds must contain one distinct seed per lineage")
    # Check the complete protocol before applying a selected-system filter.
    # Otherwise an excluded system can hide a seed collision from the plan.
    protocol_seeds: set[int] = set()
    for spec in SYSTEM_SPECS:
        candidate_seeds = [smoke_seed(system=spec.name)]
        candidate_seeds.extend(
            training_seed(system=spec.name, lineage=lineage, lineage_seed=seed)
            for lineage, seed in enumerate(lineage_seeds)
        )
        candidate_seeds.extend(
            evaluation_seed(system=spec.name, lineage=lineage,
                            checkpoint=checkpoint, stream=stream)
            for lineage in range(protocol.lineages)
            for checkpoint in checkpoints
            for stream in range(protocol.eval_streams)
        )
        for candidate in candidate_seeds:
            if candidate in protocol_seeds:
                raise ProtocolError("seed collision across Stage-1 rows")
            protocol_seeds.add(candidate)
    rows: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for spec in SYSTEM_SPECS:
        if spec.name not in selected:
            continue
        spec.validate()
        seed = smoke_seed(system=spec.name)
        if seed in seen_seeds:
            raise ProtocolError("seed collision across Stage-1 rows")
        seen_seeds.add(seed)
        rows.append({
            "row_id": stable_row_id(system=spec.name, kind="smoke"),
            "system": spec.name, "system_id": spec.system_id, "kind": "smoke",
            "seed": seed, "source_sha": source_sha, "config_sha256": config_ids[spec.name],
            "config_identity_rule": config_identity_rule,
            # A smoke is the real system train config with a deliberately
            # small budget.  Keeping one config source prevents its physics
            # from drifting away from the train/eval pair.
            "config": f"configs/{spec.name}/train.yaml", "results_root": root,
            "experiment_name": "stage1", "sector": spec.name, "run_id": "smoke",
            "updates": protocol.smoke_updates,
            "walkers": protocol.smoke_walkers,
            "checkpoint_steps": [protocol.smoke_updates],
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
                "source_sha": source_sha, "config_sha256": config_ids[spec.name],
                "config_identity_rule": config_identity_rule,
                "config": f"configs/{spec.name}/train.yaml", "results_root": root,
                "experiment_name": "stage1", "sector": spec.name,
                "run_id": f"train-lineage-{lineage}",
                "updates": protocol.lineage_updates,
                "walkers": protocol.lineage_walkers,
                "checkpoint_steps": list(checkpoints),
            })
            train_row = rows[-1]
            if phase == 1:
                continue
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
                        "source_sha": source_sha, "config_sha256": config_ids[spec.name],
                        "config_identity_rule": config_identity_rule,
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
    if device is not None:
        for row in rows:
            row["device"] = device
    return rows


def bind_evaluation_manifest(*, phase1_rows: list[Mapping[str, Any]],
                             checkpoints: list[int],
                             checkpoint_evidence: Mapping[str, Any] | None = None,
                             checkpoint_sha256: Mapping[str, Any] | None = None,
                             expected_physical_system_id: Mapping[str, str] | None = None,
                             eval_streams: int = 4, eval_walkers: int = 4096,
                             eval_draws: int = 256,
                             source_sha: str | None = None,
                             config_sha256: Mapping[str, str] | None = None,
                             protocol: Protocol | None = None,
                             phase1_manifest_hash: str | None = None,
                             device: str | None = None) -> list[dict[str, Any]]:
    """Bind eval rows to an existing phase-1 manifest without re-planning it.

    The phase-1 smoke/train rows are the authority for source SHA, config
    identity, row IDs, seeds, run paths, and selected systems.  Any missing or
    inconsistent checkpoint binding is rejected before an eval row exists.
    """

    if checkpoint_evidence is not None and checkpoint_sha256 is not None:
        raise ProtocolError("phase-2 checkpoint evidence was supplied twice")
    evidence = checkpoint_evidence if checkpoint_evidence is not None else checkpoint_sha256
    if (not phase1_rows or any(not isinstance(row, Mapping) for row in phase1_rows)
            or any(row.get("kind") not in {"smoke", "train"} for row in phase1_rows)):
        raise ProtocolError("phase-2 binding requires a phase-1 smoke/train manifest")
    actual_phase1_hash = manifest_hash(phase1_rows)
    if phase1_manifest_hash is not None:
        expected_phase1_hash = _require_sha(
            phase1_manifest_hash, length=64, field="phase1_manifest_hash"
        )
        if expected_phase1_hash != actual_phase1_hash:
            raise ProtocolError("phase-1 manifest_hash does not match phase-1 rows")
    phase1_manifest_hash = actual_phase1_hash
    if len({row.get("row_id") for row in phase1_rows}) != len(phase1_rows):
        raise ProtocolError("phase-1 manifest contains duplicate row_id")
    if len({row.get("seed") for row in phase1_rows}) != len(phase1_rows):
        raise ProtocolError("phase-1 manifest contains duplicate seed")
    train_rows = [row for row in phase1_rows if row.get("kind") == "train"]
    systems = tuple(dict.fromkeys(str(row["system"]) for row in train_rows))
    train_identity_keys = [f"{row.get('system')}:{row.get('lineage')}" for row in train_rows]
    if len(set(train_identity_keys)) != len(train_identity_keys):
        raise ProtocolError("phase-1 manifest contains duplicate train identity")
    if not systems or any(row.get("config_identity_rule") != CONFIG_IDENTITY_RULE for row in train_rows):
        raise ProtocolError("phase-1 manifest has no recognized config identity rule")
    source_shas = {_require_sha(row.get("source_sha"), length=40, field="source_sha")
                   for row in train_rows}
    if len(source_shas) != 1:
        raise ProtocolError("phase-1 manifest has inconsistent source_sha")
    config_by_system: dict[str, str] = {}
    for row in train_rows:
        system = str(row["system"])
        digest = _require_sha(row.get("config_sha256"), length=64,
                              field=f"config_sha256[{system!r}]")
        previous = config_by_system.setdefault(system, digest)
        if previous != digest:
            raise ProtocolError(f"phase-1 manifest has inconsistent config_sha256 for {system}")
    if not isinstance(evidence, Mapping) or evidence.get("schema") != CHECKPOINT_EVIDENCE_SCHEMA:
        raise ProtocolError(
            "phase-2 requires structured checkpoint evidence; bare maps are not accepted"
        )
    evidence_manifest_hash = _require_sha(
        evidence.get("phase1_manifest_hash"), length=64,
        field="checkpoint evidence phase1_manifest_hash",
    )
    if evidence_manifest_hash != phase1_manifest_hash:
        raise ProtocolError("checkpoint evidence does not match phase-1 manifest")
    raw_train_hashes = evidence.get("phase1_train_row_hashes")
    if not isinstance(raw_train_hashes, Mapping):
        raise ProtocolError("checkpoint evidence must include phase-1 train identities")
    expected_train_hashes = {
        f"{row['system']}:{row['lineage']}": manifest_hash([row])
        for row in train_rows
    }
    if set(raw_train_hashes) != set(expected_train_hashes):
        raise ProtocolError("checkpoint evidence train identities do not match phase-1 rows")
    for key, expected_hash in expected_train_hashes.items():
        if _require_sha(raw_train_hashes.get(key), length=64,
                        field=f"checkpoint evidence phase1_train_row_hashes[{key!r}]") != expected_hash:
            raise ProtocolError("checkpoint evidence train identity does not match phase-1 row")
    raw_bindings = evidence.get("checkpoints")
    if not isinstance(raw_bindings, Mapping):
        raise ProtocolError("checkpoint evidence must include checkpoint digests")
    bindings: dict[tuple[str, int, int], str] = {}
    for raw_key, raw_digest in raw_bindings.items():
        if not isinstance(raw_key, str):
            raise ProtocolError("checkpoint evidence keys must be system:lineage:checkpoint")
        parts = raw_key.split(":")
        if len(parts) != 3:
            raise ProtocolError(f"invalid checkpoint evidence key {raw_key!r}")
        system, raw_lineage, raw_checkpoint = parts
        try:
            key = (system, int(raw_lineage), int(raw_checkpoint))
        except ValueError as exc:
            raise ProtocolError(f"invalid checkpoint evidence key {raw_key!r}") from exc
        if key in bindings:
            raise ProtocolError(f"duplicate checkpoint evidence key {raw_key!r}")
        bindings[key] = _require_sha(
            raw_digest, length=64, field=f"checkpoint evidence checkpoints[{raw_key!r}]"
        )
    if protocol is not None:
        eval_streams, eval_walkers, eval_draws = protocol.eval_streams, protocol.eval_walkers, protocol.eval_draws
    if eval_streams <= 0 or eval_walkers <= 0 or eval_draws <= 0:
        raise ProtocolError("evaluation dimensions must be positive")
    if not checkpoints or any(type(step) is not int or step < 0 for step in checkpoints):
        raise ProtocolError("checkpoints must be a non-empty list of non-negative updates")
    if len(set(checkpoints)) != len(checkpoints):
        raise ProtocolError("checkpoints must not contain duplicates")
    phase1_checkpoint_steps = {
        tuple(row.get("checkpoint_steps", ())) for row in train_rows
    }
    if len(phase1_checkpoint_steps) != 1 or not next(iter(phase1_checkpoint_steps)):
        raise ProtocolError("phase-1 manifest has inconsistent checkpoint_steps")
    allowed_checkpoints = set(next(iter(phase1_checkpoint_steps)))
    if any(checkpoint not in allowed_checkpoints for checkpoint in checkpoints):
        raise ProtocolError("phase-2 checkpoints must be in phase-1 checkpoint_steps")
    phase1_source_sha = next(iter(source_shas))
    if source_sha is not None and source_sha != phase1_source_sha:
        raise ProtocolError("phase-2 source_sha differs from phase-1 manifest")
    if config_sha256 is not None:
        for system, digest in config_by_system.items():
            if _require_sha(config_sha256.get(system), length=64,
                            field=f"config_sha256[{system!r}]") != digest:
                raise ProtocolError(f"phase-2 config_sha256 differs for {system}")
    phase1_devices = {row.get("device") for row in train_rows}
    if len(phase1_devices) > 1 or (phase1_devices and next(iter(phase1_devices)) not in {None, "cpu", "cuda"}):
        raise ProtocolError("phase-1 manifest has an invalid device binding")
    phase1_device = next(iter(phase1_devices), None)
    if device is not None and device not in {"cpu", "cuda"}:
        raise ProtocolError("device must be 'cpu' or 'cuda'")
    if device is not None and phase1_device not in {None, device}:
        raise ProtocolError("phase-2 device differs from phase-1 manifest")
    effective_device = phase1_device if device is None else device
    rows: list[dict[str, Any]] = []
    physical_ids = dict(expected_physical_system_id or planned_physical_system_ids())
    for system in systems:
        _require_sha(physical_ids.get(system), length=64,
                     field=f"expected_physical_system_id[{system!r}]")
    combined_row_ids = {row.get("row_id") for row in phase1_rows}
    combined_seeds = {row.get("seed") for row in phase1_rows}
    for train_row in train_rows:
        system = str(train_row["system"])
        lineage = int(train_row["lineage"])
        for checkpoint in checkpoints:
            key = (system, lineage, checkpoint)
            digest = bindings.get(key)
            if digest is None:
                raise ProtocolError(f"checkpoint evidence missing digest for {key!r}")
            checkpoint_dir = resolve_run_dir(train_row) / "checkpoints" / f"step_{checkpoint:06d}"
            for stream in range(eval_streams):
                row_id = stable_row_id(system=system, kind="eval", lineage=lineage,
                                       checkpoint=checkpoint, stream=stream)
                row: dict[str, Any] = {
                    "row_id": row_id, "system": system, "system_id": train_row["system_id"],
                    "kind": "eval", "lineage": lineage, "checkpoint": checkpoint,
                    "stream": stream, "seed": evaluation_seed(system=system, lineage=lineage,
                                                                 checkpoint=checkpoint, stream=stream),
                    "updates": checkpoint, "walkers": eval_walkers, "draws": eval_draws,
                    "source_sha": phase1_source_sha,
                    "config_sha256": config_by_system[system],
                    "config_identity_rule": CONFIG_IDENTITY_RULE,
                    "phase1_manifest_hash": phase1_manifest_hash,
                    "phase1_train_row_hash": manifest_hash([train_row]),
                    "config": f"configs/{system}/eval.yaml", "results_root": train_row["results_root"],
                    "expected_physical_system_id": physical_ids[system],
                    "experiment_name": train_row["experiment_name"], "sector": train_row["sector"],
                    "run_id": f"{system}-lineage-{lineage}-checkpoint-{checkpoint}-stream-{stream}",
                    "checkpoint_dir": str(checkpoint_dir), "checkpoint_file": str(checkpoint_dir / "model.pt"),
                    "checkpoint_sha256": digest, "depends_on": train_row["row_id"],
                }
                if row["row_id"] in combined_row_ids or row["seed"] in combined_seeds:
                    raise ProtocolError("phase-1 and phase-2 rows are not jointly unique")
                combined_row_ids.add(row["row_id"])
                combined_seeds.add(row["seed"])
                rows.append(row)
                if effective_device is not None:
                    row["device"] = effective_device
    for row in rows:
        _validate_run_identity(row)
        _checkpoint_binding(row)
    return rows


def manifest_hash(rows: list[Mapping[str, Any]]) -> str:
    """Hash a manifest using canonical JSON serialization."""

    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
