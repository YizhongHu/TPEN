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
import copy
import math
from collections import Counter
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf
from omegaconf._utils import split_key
import yaml

from experiments.baselines.systems import load_registry
from experiments.toolkit.run_paths import resolve_run_dir

SYSTEMS = ("hooke1", "hooke2", "qho2e", "he", "h2")
KINDS = ("smoke", "train", "eval")
CONFIG_IDENTITY_RULE = "stage1-config-v1"
LEGACY_CONFIG_IDENTITY_RULE = "legacy-config-sha256"
HOOKE_SCIENCE_PROFILE = "hooke-hv2-v1"
HOOKE_TRAINING_SEED_RULE = "stage1-training-seeds-v1"
HOOKE_EVAL_SELECTION = "A37"
HOOKE_EVAL_SELECTION_ID = "hooke-hv2-a37-v1"
HOOKE_A37 = tuple(range(0, 21_000, 1_000)) + tuple(range(25_000, 100_001, 5_000))
HOOKE2_ACT50_SIGMOID_PROFILE = "hooke2-sigmoid50k-v1"
HOOKE2_ACT50_TANH_PROFILE = "hooke2-tanh50k-v1"
HOOKE2_ACT50_NOLAYER_PROFILE = "hooke2-nolayer50k-v1"
HOOKE2_ACT50_PROFILES = (
    HOOKE2_ACT50_SIGMOID_PROFILE,
    HOOKE2_ACT50_TANH_PROFILE,
    HOOKE2_ACT50_NOLAYER_PROFILE,
)
HOOKE2_ACT50_HORIZON = 50_000
HOOKE2_ACT50_EVAL_SELECTION = "A27"
HOOKE2_ACT50_EVAL_SELECTION_ID = "hooke2-act50-a27-v1"
HOOKE2_ACT50_A27 = tuple(range(0, 21_000, 1_000)) + tuple(range(25_000, 50_001, 5_000))
_HOOKEE2_ACT50_PROFILE_DATA = {
    HOOKE2_ACT50_SIGMOID_PROFILE: {
        "variant": "sigmoid", "activation": "torch.nn.Sigmoid", "layers": "original",
    },
    HOOKE2_ACT50_TANH_PROFILE: {
        "variant": "tanh", "activation": "torch.nn.Tanh", "layers": "original",
    },
    HOOKE2_ACT50_NOLAYER_PROFILE: {
        "variant": "nolayer", "activation": None, "layers": "empty",
    },
}
QHO_EVAL_SELECTION = "Q21"
QHO_EVAL_SELECTION_ID = "qho2e-q21-v1"
QHO_Q21 = (0, 1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000,
           9_000, 10_000, 15_000, 20_000, 30_000, 40_000, 50_000,
           60_000, 70_000, 80_000, 90_000, 100_000)
HE_H2_EVAL_SELECTION = "B"
HE_H2_EVAL_SELECTION_ID = "he-h2-b-v1"
HE_H2_EVAL_CHECKPOINTS = (0, 1_000, 2_000, 4_000, 8_000, 16_000,
                          25_000, 40_000, 60_000, 80_000, 100_000)
HE_H2_SYSTEMS = frozenset({"he", "h2"})
HOOKE_SYSTEMS = frozenset({"hooke1", "hooke2", "qho2e"})
OVERLAY_RELATIVE_PATH = "configs/overlays/diagnostics-v1.yaml"
HOOKE_OVERLAY_RELATIVE_PATH = "configs/overlays/hooke-diagnostics-v1.yaml"
OVERLAY_V2_RELATIVE_PATH = "configs/overlays/diagnostics-v2.yaml"
HOOKE_OVERLAY_V2_RELATIVE_PATH = "configs/overlays/hooke-diagnostics-v2.yaml"
OVERLAY_PROFILE_V1 = "diagnostics-v1"
OVERLAY_PROFILE_V2 = "diagnostics-v2"
OVERLAY_PROFILES = (OVERLAY_PROFILE_V1, OVERLAY_PROFILE_V2)
OVERLAY_SYSTEMS = frozenset({"he", "h2"})
HOOKE_OVERLAY_SYSTEMS = HOOKE_SYSTEMS
OVERLAY_PATH_BY_SYSTEM = {
    **{system: OVERLAY_RELATIVE_PATH for system in OVERLAY_SYSTEMS},
    **{system: HOOKE_OVERLAY_RELATIVE_PATH for system in HOOKE_OVERLAY_SYSTEMS},
}
OVERLAY_V2_PATH_BY_SYSTEM = {
    **{system: OVERLAY_V2_RELATIVE_PATH for system in OVERLAY_SYSTEMS},
    **{system: HOOKE_OVERLAY_V2_RELATIVE_PATH for system in HOOKE_OVERLAY_SYSTEMS},
}
OVERLAY_COMPOSITION_ORDER = (
    "base-config -> declared train diagnostics overlay -> manifest overrides"
)
OVERLAY_V2_COMPOSITION_ORDER = (
    "base-config -> recorder-before-checkpoint diagnostics-v2 overlay -> manifest overrides"
)
CHECKPOINT_EVIDENCE_SCHEMA = "stage1-checkpoint-evidence/v1"
MANIFEST_SCHEMA = "stage1/v4"
LEGACY_MANIFEST_SCHEMA = "stage1/v3"
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
    retention_cadence: int | None = None

    def validate(self) -> None:
        """Reject non-positive dimensions and underspecified stream counts."""

        values = (self.smoke_updates, self.smoke_walkers, self.lineage_updates,
                  self.lineage_walkers, self.lineages, self.eval_streams,
                  self.eval_walkers, self.eval_draws)
        if any(value <= 0 for value in values):
            raise ProtocolError(f"protocol dimensions must be positive: {values}")
        if self.lineages < 3 or self.eval_streams < 4:
            raise ProtocolError("Stage-1 requires at least 3 lineages and 4 eval streams")
        if self.retention_cadence is not None and self.retention_cadence <= 0:
            raise ProtocolError("retention_cadence must be positive when declared")


STAGE1_PROTOCOL = Protocol(1000, 256, 100000, 4096, 3, 4, 4096, 256, 1000)
SYSTEM_SPECS = tuple(SystemSpec(name, system_id) for name, system_id in (
    ("hooke1", "hooke_one_electron_omega0.5"),
    ("hooke2", "hooke_pair_singlet_omega0.5"),
    ("qho2e", "qho_pair_noninteracting_singlet_omega0.5"),
    ("he", "he_atom"),
    ("h2", "h2_molecule"),
))


def hooke_evaluation_selection(system: str) -> tuple[str, str, tuple[int, ...]]:
    """Return the frozen evaluation selection for one Hooke-family system."""

    if system == "qho2e":
        return QHO_EVAL_SELECTION, QHO_EVAL_SELECTION_ID, QHO_Q21
    if system in {"hooke1", "hooke2"}:
        return HOOKE_EVAL_SELECTION, HOOKE_EVAL_SELECTION_ID, HOOKE_A37
    raise ProtocolError(f"not a Hooke-family system: {system!r}")


def hooke2_act50_profile(profile: str) -> Mapping[str, Any]:
    """Return the immutable finite definition for one Hooke-2e ACT50 profile."""

    try:
        return _HOOKEE2_ACT50_PROFILE_DATA[profile]
    except KeyError as exc:
        raise ProtocolError(f"unknown Hooke-2e ACT50 profile {profile!r}") from exc


def is_hooke2_act50_profile(profile: object) -> bool:
    """Return whether ``profile`` is one of the admitted ACT50 variants."""

    return isinstance(profile, str) and profile in HOOKE2_ACT50_PROFILES


def hooke_profile_selection(profile: str, system: str) -> tuple[str, str, tuple[int, ...]]:
    """Return the selection bound to an old or ACT50 Hooke science profile."""

    if is_hooke2_act50_profile(profile):
        if system != "hooke2":
            raise ProtocolError("Hooke-2e ACT50 profiles only admit hooke2")
        return HOOKE2_ACT50_EVAL_SELECTION, HOOKE2_ACT50_EVAL_SELECTION_ID, HOOKE2_ACT50_A27
    if profile == HOOKE_SCIENCE_PROFILE:
        return hooke_evaluation_selection(system)
    raise ProtocolError(f"unknown Hooke science profile {profile!r}")


def hooke_profile_horizon(profile: str) -> int:
    """Return the fixed training horizon for one admitted Hooke profile."""

    return HOOKE2_ACT50_HORIZON if is_hooke2_act50_profile(profile) else STAGE1_PROTOCOL.lineage_updates


def protocol_for_science_profile(protocol: Protocol, profile: str | None) -> Protocol:
    """Apply only the finite ACT50 horizon change to a protocol copy."""

    if not is_hooke2_act50_profile(profile):
        return protocol
    return Protocol(
        protocol.smoke_updates, protocol.smoke_walkers, HOOKE2_ACT50_HORIZON,
        protocol.lineage_walkers, protocol.lineages, protocol.eval_streams,
        protocol.eval_walkers, protocol.eval_draws, 1000,
    )


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


def resolve_overlay_path(overlay: str | Path) -> Path:
    """Resolve one Stage-1 overlay relative to the study directory."""

    path = Path(overlay)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def overlay_sha256(overlay: str | Path) -> str:
    """Return the SHA-256 digest of one overlay file's exact bytes."""

    path = resolve_overlay_path(overlay)
    if not path.is_file():
        raise ProtocolError(f"overlay file does not exist: {path}")
    return _sha256_file(path)


def _sha256_file(path: Path) -> str:
    """Hash one admitted file's exact bytes without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def overlay_path_for_system(system: str, *, profile: str = OVERLAY_PROFILE_V1) -> str:
    """Return the committed overlay admitted for one system and profile.

    The default deliberately remains the v1 composition so existing planners
    and manifests retain their historical identity and byte shape.
    """

    paths = OVERLAY_PATH_BY_SYSTEM if profile == OVERLAY_PROFILE_V1 else (
        OVERLAY_V2_PATH_BY_SYSTEM if profile == OVERLAY_PROFILE_V2 else None
    )
    if paths is None:
        raise ProtocolError(f"unknown diagnostics overlay profile {profile!r}")
    try:
        return paths[system]
    except KeyError as exc:
        raise ProtocolError(f"no diagnostics overlay is approved for {system!r}") from exc


def _overlay_binding(*, overlay: str | Path, expected_sha256: object,
                     system: str | None = None,
                     profile: str = OVERLAY_PROFILE_V1) -> tuple[str, str, Path]:
    """Validate one finite canonical overlay and its exact content digest.

    ``system`` is supplied at planner/gateway boundaries so a valid overlay
    cannot be replayed for a different physical system.  Composition tests may
    omit it to exercise the overlay parser independently; the finite path
    allowlist still applies in that mode.
    """

    if profile not in OVERLAY_PROFILES:
        raise ProtocolError(f"unknown diagnostics overlay profile {profile!r}")
    path = resolve_overlay_path(overlay).resolve()
    allowed = (
        {resolve_overlay_path(overlay_path_for_system(system, profile=profile)).resolve()}
        if system is not None
        else {
            resolve_overlay_path(candidate).resolve()
            for candidate in set(
                OVERLAY_PATH_BY_SYSTEM.values()
                if profile == OVERLAY_PROFILE_V1 else OVERLAY_V2_PATH_BY_SYSTEM.values()
            )
        }
    )
    if path not in allowed:
        raise ProtocolError(
            "overlay path must be the committed system-qualified diagnostics overlay"
        )
    digest = _require_sha(expected_sha256, length=64, field="overlay_sha256")
    actual = overlay_sha256(path)
    if actual != digest:
        raise ProtocolError("overlay sha256 does not match the committed overlay bytes")
    relative = path.relative_to(Path(__file__).resolve().parent).as_posix()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("overlay must contain a YAML mapping")
    _validate_overlay_payload(payload, relative=relative)
    return relative, digest, path


def _validate_overlay_payload(payload: Mapping[str, Any], *, relative: str) -> None:
    """Reject a changed or incomplete committed overlay before composition."""

    allowed = {"runtime", "sampler", "optimizer", "trainer", "callbacks", "loggers"}
    if set(payload) != allowed:
        raise ProtocolError("overlay contains unsupported or missing top-level sections")
    for key in ("runtime", "sampler", "optimizer", "trainer"):
        if not isinstance(payload[key], Mapping):
            raise ProtocolError(f"overlay section {key!r} must be a mapping")
    if dict(payload["runtime"]) != {"device": "cuda", "dtype": "float64"}:
        raise ProtocolError("diagnostics overlay runtime profile drifted")
    if dict(payload["sampler"]) != {
        "n_walkers": 4096, "burn_in": 100, "n_steps": 10, "proposal_scale": 0.5,
    }:
        raise ProtocolError("diagnostics overlay sampler profile drifted")
    if dict(payload["optimizer"]) != {
        "_target_": "torch.optim.Adam", "_partial_": True, "lr": 0.005,
    }:
        raise ProtocolError("diagnostics overlay optimizer profile drifted")
    if dict(payload["trainer"]) != {
        "max_steps": STAGE1_PROTOCOL.lineage_updates,
        "log_every_n_steps": 1,
        "return_terms": True,
    }:
        raise ProtocolError("diagnostics overlay trainer profile drifted")
    callbacks = payload["callbacks"]
    if not isinstance(callbacks, list):
        raise ProtocolError("overlay callbacks must be a list")
    expected_callbacks = _expected_overlay_callbacks(relative)
    if callbacks != expected_callbacks:
        raise ProtocolError("diagnostics overlay callback set drifted")
    if not isinstance(payload["loggers"], list) or payload["loggers"] != [
        {"_target_": "tpen.logging.JSONL", "path": "${run.dir}/metrics.jsonl"}
    ]:
        raise ProtocolError("diagnostics overlay logger profile drifted")


def _expected_overlay_callbacks(
    relative: str, *, science_profile: str | None = None,
) -> list[dict[str, Any]]:
    """Return the complete callback contract for one canonical overlay."""

    callbacks: list[dict[str, Any]] = [
        {
            "_target_": "tpen.callback.DataIntegrity",
            "every_n_steps": 1,
            "fail_fast": True,
            "max_nonfinite_energy_fraction": 0.0,
            "max_nonfinite_logabs_fraction": 0.0,
            "check_loss": True,
            "check_wavefunction_output": True,
            "check_batch": True,
            "strict_sign_values": True,
        },
        {
            "_target_": "tpen.callback.GradientStats",
            "every_n_steps": 1,
            "fail_fast": True,
            "check_finite": True,
        },
        {
            "_target_": "tpen.callback.SamplerHealth",
            "every_n_steps": 1,
            "fail_fast": False,
        },
        {
            "_target_": "tpen.callback.FactorScalars",
            "every_n_steps": 1,
            "fail_fast": False,
        },
    ]
    if relative in {HOOKE_OVERLAY_RELATIVE_PATH, HOOKE_OVERLAY_V2_RELATIVE_PATH}:
        callbacks.append({
            "_target_": "tpen.callback.AmplitudeRiskFraction",
            "thresholds": [-3, -4, -5, -7, -10],
            "every_n_steps": 100,
            "start_step": 0,
        })
    groups = {
        "embedding": ["embedding"],
        "readout": ["readout"],
        "factors": ["factors"],
        "layer_0_mixing": ["stack.layers.0.mixing"],
        "layer_0_path_aggregation": ["stack.layers.0.path_aggregation"],
    }
    if science_profile == HOOKE2_ACT50_NOLAYER_PROFILE:
        groups.pop("layer_0_mixing")
        groups.pop("layer_0_path_aggregation")
    callbacks.extend([
        {
            "_target_": "tpen.callback.ModuleNorms",
            "every_n_steps": 1,
            "groups": copy.deepcopy(groups),
        },
        {
            "_target_": "tpen.callback.OptimizerState",
            "every_n_steps": 1,
            "groups": copy.deepcopy(groups),
        },
        {
            "_target_": "tpen.callback.RawTrainingBatchRecorder",
            "path": "${run.dir}/training_history",
            "every_n_steps": 50,
            "partition_size_steps": 1000,
        },
        {
            "_target_": "tpen.callback.ParameterActivity",
            "output_path": "${run.dir}/parameter_activity.json",
        },
        {"_target_": "tpen.callback.RunTiming"},
        {
            "_target_": "tpen.callback.TrainStepTiming",
            "every_n_steps": 1,
            "rolling_window": 20,
            "accelerator_synchronize": False,
        },
    ])
    return callbacks


def _iter_mapping_leaves(value: Mapping[str, Any], prefix: str) -> Iterator[tuple[str, Any]]:
    """Yield dotted paths for every scalar leaf in a mapping.

    Overlay sections are applied one leaf at a time so a section cannot
    replace a mapping owned by the base config.  Lists remain values at their
    containing path and are handled explicitly by the append-only sections.
    """

    for key, nested in value.items():
        dotted = f"{prefix}.{key}"
        if isinstance(nested, Mapping):
            yield from _iter_mapping_leaves(nested, dotted)
        else:
            yield dotted, nested


def _act50_overlay_adjustments(cfg: Any, science_profile: str | None) -> None:
    """Apply the finite ACT50 train overlay delta after byte validation."""

    if not is_hooke2_act50_profile(science_profile):
        return
    OmegaConf.update(
        cfg, "trainer.max_steps", HOOKE2_ACT50_HORIZON, merge=False,
    )
    if science_profile != HOOKE2_ACT50_NOLAYER_PROFILE:
        return
    callbacks = OmegaConf.to_container(cfg.callbacks, resolve=False)
    for callback in callbacks:
        if not isinstance(callback, Mapping):
            continue
        if callback.get("_target_") not in {
            "tpen.callback.ModuleNorms", "tpen.callback.OptimizerState",
        }:
            continue
        groups = callback.get("groups")
        if isinstance(groups, dict):
            groups.pop("layer_0_mixing", None)
            groups.pop("layer_0_path_aggregation", None)
    OmegaConf.update(cfg, "callbacks", callbacks, merge=False)


def compose_model_profile(cfg: Any, science_profile: str | None) -> Any:
    """Compose one admitted Hooke-2e model profile in train or eval config."""

    if not is_hooke2_act50_profile(science_profile):
        return cfg
    profile = hooke2_act50_profile(str(science_profile))
    if profile["layers"] == "empty":
        OmegaConf.update(cfg, "model.layers", [], merge=False)
    else:
        activation = str(profile["activation"])
        OmegaConf.update(
            cfg, "model.embedding.activation", {"_target_": activation},
            merge=False, force_add=True,
        )
        for path in (
            "model.layers.0.mixing.activation._target_",
            "model.layers.0.path_aggregation.activation._target_",
        ):
            OmegaConf.update(cfg, path, activation, merge=False)
    OmegaConf.update(cfg, "stage1_science_profile", str(science_profile), merge=False, force_add=True)
    return cfg


def compose_train_overlay(
    cfg: Any, *, overlay: str | Path, expected_sha256: str,
    overlay_profile: str = OVERLAY_PROFILE_V1,
    science_profile: str | None = None,
) -> Any:
    """Compose the approved train overlay without touching physics sections.

    The overlay owns train-side runtime/sampler/optimizer/trainer leaves plus
    diagnostic callbacks and loggers.  Base callbacks are retained and the
    diagnostics are appended explicitly; checkpoint retention therefore stays
    manifest-owned.  A blind ``OmegaConf.merge`` is avoided so a future list
    or nested model section cannot be silently replaced.
    """

    relative, digest, path = _overlay_binding(
        overlay=overlay, expected_sha256=expected_sha256, profile=overlay_profile,
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("overlay must contain a YAML mapping")
    _validate_overlay_payload(payload, relative=relative)
    for key in ("runtime", "sampler", "optimizer", "trainer"):
        section = payload[key]
        if not isinstance(section, Mapping):
            raise ProtocolError(f"overlay section {key!r} must be a mapping")
        for dotted, value in _iter_mapping_leaves(section, key):
            OmegaConf.update(cfg, dotted, copy.deepcopy(value), merge=False)

    for key in ("callbacks", "loggers"):
        base_config = cfg.get(key)
        base_entries = (
            [] if base_config is None else OmegaConf.to_container(base_config, resolve=False)
        )
        overlay_entries = copy.deepcopy(payload[key])
        if not isinstance(base_entries, list) or not isinstance(overlay_entries, list):
            raise ProtocolError(f"overlay {key} must be lists")
        if key == "callbacks" and overlay_profile == OVERLAY_PROFILE_V2:
            recorder = [
                entry for entry in overlay_entries
                if entry.get("_target_") == "tpen.callback.RawTrainingBatchRecorder"
            ]
            if len(recorder) != 1:
                raise ProtocolError("diagnostics-v2 overlay must declare one recorder")
            checkpoint_count = sum(
                isinstance(entry, Mapping)
                and entry.get("_target_") == "tpen.callback.Checkpoint"
                for entry in base_entries
            )
            if checkpoint_count > 1:
                raise ProtocolError(
                    "diagnostics-v2 composition does not support multiple base "
                    "Checkpoint publishers"
                )
            overlay_entries = [
                entry for entry in overlay_entries
                if entry.get("_target_") != "tpen.callback.RawTrainingBatchRecorder"
            ]
            composed_entries: list[Any] = []
            for entry in base_entries:
                if isinstance(entry, Mapping) and entry.get("_target_") == "tpen.callback.Checkpoint":
                    composed_entries.extend(copy.deepcopy(recorder))
                composed_entries.append(entry)
            if checkpoint_count == 0:
                raise ProtocolError("diagnostics-v2 composition requires a base Checkpoint")
            base_entries = composed_entries
        # Both callback and logger entries are additive.  In particular, the
        # diagnostics overlay must never discard base callbacks such as the
        # resolved-config snapshot or checkpoint publisher.
        OmegaConf.update(
            cfg, key, base_entries + overlay_entries, merge=False, force_add=True,
        )

    provenance = {
        "schema": "stage1-overlay/v1" if overlay_profile == OVERLAY_PROFILE_V1
        else "stage1-overlay/v2",
        "path": relative,
        "sha256": digest,
        "composition_order": (
            OVERLAY_COMPOSITION_ORDER if overlay_profile == OVERLAY_PROFILE_V1
            else OVERLAY_V2_COMPOSITION_ORDER
        ),
        "stream": "train",
    }
    if overlay_profile == OVERLAY_PROFILE_V2:
        provenance["profile"] = OVERLAY_PROFILE_V2
    for dotted, value in _iter_mapping_leaves(provenance, "stage1_overlay"):
        OmegaConf.update(cfg, dotted, value, merge=False, force_add=True)
    _act50_overlay_adjustments(cfg, science_profile)
    return cfg


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


def config_identity(*, config_root: str | Path, systems: tuple[str, ...] | list[str] | None = None,
                    science_profile: str | None = None) -> dict[str, Any]:
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
    if science_profile is not None:
        if not is_hooke2_act50_profile(science_profile):
            raise ProtocolError(f"unknown science profile {science_profile!r}")
        marker = json.dumps(
            ["stage1-science-profile-v1", science_profile,
             dict(hooke2_act50_profile(science_profile))],
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        digest.update(b"science-profile\0")
        digest.update(str(len(marker)).encode("ascii"))
        digest.update(b"\0")
        digest.update(marker)
        entries.append(f"science-profile:{science_profile}")
    return {"rule": CONFIG_IDENTITY_RULE, "sha256": digest.hexdigest(), "files": entries}


def config_identities(*, config_root: str | Path, systems: tuple[str, ...] | list[str] | None = None,
                      science_profile: str | None = None) -> dict[str, str]:
    """Return one computed config digest per selected system."""

    selected = tuple(SYSTEMS if systems is None else systems)
    if not selected or len(set(selected)) != len(selected) or any(system not in SYSTEMS for system in selected):
        raise ProtocolError("systems must contain distinct known Stage-1 systems")
    return {system: str(config_identity(
        config_root=config_root, systems=(system,), science_profile=science_profile,
    )["sha256"])
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
    actual = config_identity(
        config_root=path.parent.parent, systems=(system,),
        science_profile=row.get("science_profile") if is_hooke2_act50_profile(
            row.get("science_profile")
        ) else None,
    )["sha256"]
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


def sampler_seed(*, system: str, lineage: int, lineage_seed: int) -> int:
    """Derive the Hooke science training-chain seed.

    This namespace is deliberately separate from :func:`training_seed`.  The
    compact JSON payload is part of the protocol, rather than an implementation
    detail, so a future seed-rule revision cannot silently reinterpret a row.
    """

    if (system not in SYSTEMS or type(lineage) is not int or type(lineage_seed) is not int
            or lineage < 0 or lineage >= 10**6
            or lineage_seed < 0 or lineage_seed >= 10**9):
        raise ProtocolError("invalid sampler seed coordinates")
    payload = json.dumps(
        ["stage1/train-sampler-v1", system, lineage, lineage_seed],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % _TORCH_SEED_MODULUS


# Descriptive aliases used by launch/analysis tooling.
training_sampler_seed = sampler_seed
hooke_sampler_seed = sampler_seed


def _is_hooke_science_row(row: Mapping[str, Any]) -> bool:
    """Return whether a row opts into the versioned Hooke science profile."""

    return row.get("science_profile") == HOOKE_SCIENCE_PROFILE or is_hooke2_act50_profile(
        row.get("science_profile")
    )


def _hooke_profile(row: Mapping[str, Any]) -> str:
    """Return and validate the finite Hooke profile carried by ``row``."""

    profile = row.get("science_profile")
    if profile == HOOKE_SCIENCE_PROFILE or is_hooke2_act50_profile(profile):
        return str(profile)
    raise ProtocolError("row does not carry an admitted Hooke science profile")


def _is_he_h2_production_protocol(protocol: Protocol) -> bool:
    """Return whether a protocol carries the released He/H2 production shape."""

    return (
        protocol.lineage_updates == STAGE1_PROTOCOL.lineage_updates
        and protocol.lineage_walkers == STAGE1_PROTOCOL.lineage_walkers
        and protocol.lineages == STAGE1_PROTOCOL.lineages
        and protocol.eval_streams == STAGE1_PROTOCOL.eval_streams
        and protocol.eval_walkers == STAGE1_PROTOCOL.eval_walkers
        and protocol.eval_draws == STAGE1_PROTOCOL.eval_draws
        and protocol.retention_cadence == STAGE1_PROTOCOL.retention_cadence
    )


def _is_he_h2_production_train_row(row: Mapping[str, Any]) -> bool:
    """Return whether a phase-1 row is the canonical He/H2 retention row."""

    return (
        row.get("system") in HE_H2_SYSTEMS
        and row.get("kind") == "train"
        and row.get("updates") == STAGE1_PROTOCOL.lineage_updates
        and row.get("walkers") == STAGE1_PROTOCOL.lineage_walkers
        and list(row.get("checkpoint_steps", ())) == list(
            range(0, STAGE1_PROTOCOL.lineage_updates + 1,
                  STAGE1_PROTOCOL.retention_cadence)
        )
    )


def _is_he_h2_production_phase(phase1_rows: list[Mapping[str, Any]]) -> bool:
    """Return whether all phase-1 train rows are canonical He/H2 rows."""

    train_rows = [row for row in phase1_rows if row.get("kind") == "train"]
    systems = {row.get("system") for row in train_rows}
    return bool(train_rows) and systems.issubset(HE_H2_SYSTEMS) and all(
        _is_he_h2_production_train_row(row) for row in train_rows
    )


def validate_he_h2_eval_grid(
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    expected_systems: tuple[str, ...] | list[str] | None = None,
    expected_profile: str | None = None,
) -> None:
    """Require the exact He/H2 roster-B x lineage x stream evaluation multiset.

    The coordinate comparison happens before collection opens any run output,
    so a missing checkpoint group or stream cannot look like a partial result.
    """

    he_h2_rows = [row for row in rows if row.get("system") in HE_H2_SYSTEMS]
    declared_systems = tuple(expected_systems) if expected_systems is not None else ()
    if any(system not in HE_H2_SYSTEMS for system in declared_systems):
        raise ProtocolError("He/H2 evaluation grid has an invalid expected system")
    if len(set(declared_systems)) != len(declared_systems):
        raise ProtocolError("He/H2 evaluation grid has duplicate expected systems")
    if not he_h2_rows and not declared_systems:
        return
    if expected_profile is None and not any(
        row.get("evaluation_selection") == HE_H2_EVAL_SELECTION for row in he_h2_rows
    ):
        return
    profile = expected_profile or HE_H2_EVAL_SELECTION
    systems = declared_systems or tuple(dict.fromkeys(str(row["system"]) for row in he_h2_rows))
    expected = [
        (system, lineage, checkpoint, stream)
        for system in systems
        for lineage in range(STAGE1_PROTOCOL.lineages)
        for checkpoint in HE_H2_EVAL_CHECKPOINTS
        for stream in range(STAGE1_PROTOCOL.eval_streams)
    ]
    actual: list[tuple[str, int, int, int]] = []
    for row in he_h2_rows:
        if row.get("kind") != "eval":
            raise ProtocolError("He/H2 roster-B grid contains a non-eval row")
        if row.get("evaluation_selection") != profile:
            raise ProtocolError("He/H2 eval row does not bind roster B")
        if row.get("evaluation_selection_id") != HE_H2_EVAL_SELECTION_ID:
            raise ProtocolError("He/H2 eval row has an invalid roster-B selection id")
        coordinates = (row.get("system"), row.get("lineage"),
                       row.get("checkpoint"), row.get("stream"))
        if any(type(value) is not int for value in coordinates[1:]):
            raise ProtocolError("He/H2 roster-B grid coordinates must be integers")
        actual.append(coordinates)  # type: ignore[arg-type]
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    if actual_counts != expected_counts:
        missing = sorted((expected_counts - actual_counts).elements())
        extra = sorted((actual_counts - expected_counts).elements())
        raise ProtocolError(
            f"He/H2 roster-B grid mismatch: expected {len(expected)} rows, got {len(actual)}; "
            f"missing={missing!r}; extra={extra!r}"
        )


def _validate_hooke_science_train_row(row: Mapping[str, Any]) -> None:
    """Validate the complete provenance and seed binding for a Hooke row."""

    profile = _hooke_profile(row)
    allowed_systems = {"hooke2"} if is_hooke2_act50_profile(profile) else HOOKE_SYSTEMS
    if row.get("system") not in allowed_systems:
        raise ProtocolError("Hooke science profile is only valid for Hooke systems")
    if row.get("kind") != "train":
        raise ProtocolError("Hooke science profile is only valid for train rows")
    horizon = hooke_profile_horizon(profile)
    if row.get("updates") != horizon:
        raise ProtocolError("Hooke science profile requires the exact production update horizon")
    if row.get("walkers") != STAGE1_PROTOCOL.lineage_walkers:
        raise ProtocolError("Hooke science profile requires the exact production walker count")
    if list(row.get("checkpoint_steps", ())) != list(range(0, horizon + 1, 1000)):
        raise ProtocolError(
            f"Hooke science profile requires {horizon // 1000 + 1} production checkpoints"
        )
    if row.get("training_seed_rule") != HOOKE_TRAINING_SEED_RULE:
        raise ProtocolError("unknown Hooke training seed rule")
    selection, selection_id, _ = hooke_profile_selection(profile, str(row["system"]))
    if row.get("evaluation_selection") != selection:
        raise ProtocolError("Hooke science row has an invalid evaluation selection")
    if row.get("evaluation_selection_id", row.get("selection_id")) != selection_id:
        raise ProtocolError("Hooke science row has an invalid evaluation selection id")
    lineage = row.get("lineage")
    lineage_seed = row.get("lineage_seed")
    if type(lineage) is not int or type(lineage_seed) is not int:
        raise ProtocolError("Hooke science row requires integer lineage coordinates")
    expected_training = training_seed(
        system=str(row["system"]), lineage=lineage, lineage_seed=lineage_seed,
    )
    expected_sampler = sampler_seed(
        system=str(row["system"]), lineage=lineage, lineage_seed=lineage_seed,
    )
    if type(row.get("seed")) is not int or row.get("seed") != expected_training:
        raise ProtocolError("Hooke training seed does not match its declared rule")
    if type(row.get("sampler_seed")) is not int or row.get("sampler_seed") != expected_sampler:
        raise ProtocolError("Hooke sampler seed does not match its declared rule")
    if expected_training == expected_sampler:
        raise ProtocolError("Hooke initialization and sampler seeds must be distinct")


def validate_hooke_eval_grid(
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    expected_systems: tuple[str, ...] | list[str] | None = None,
    expected_profile: str | None = None,
) -> None:
    """Require each Hooke system's frozen selection x lineage x stream multiset.

    The check is intentionally a multiset comparison.  Row counts and
    per-coordinate stream checks cannot detect a balanced deletion/replacement
    or a wholly absent checkpoint, which would make a partial collection look
    complete.
    """

    hooke_rows = [row for row in rows if row.get("system") in HOOKE_SYSTEMS]
    declared_systems = tuple(expected_systems) if expected_systems is not None else ()
    if any(system not in HOOKE_SYSTEMS for system in declared_systems):
        raise ProtocolError("Hooke evaluation grid has an invalid expected system")
    if len(set(declared_systems)) != len(declared_systems):
        raise ProtocolError("Hooke evaluation grid has duplicate expected systems")
    if not hooke_rows and not declared_systems:
        return
    # Historical and bounded tiny manifests remain readable under their
    # declared legacy mode.  Exact A37 admission is opt-in to the production
    # Hooke science profile.
    if expected_profile is None and not any(_is_hooke_science_row(row) for row in hooke_rows):
        return
    declared_profiles = {row.get("science_profile") for row in hooke_rows}
    profile = expected_profile or (
        next(iter(declared_profiles)) if len(declared_profiles) == 1 else None
    )
    if profile is None or not (profile == HOOKE_SCIENCE_PROFILE or is_hooke2_act50_profile(profile)):
        raise ProtocolError("Hooke evaluation rows mix or omit an admitted science profile")
    if any(row.get("science_profile") != profile for row in hooke_rows):
        raise ProtocolError("Hooke evaluation rows must declare one finite science profile")
    if is_hooke2_act50_profile(profile) and any(system != "hooke2" for system in declared_systems or {row.get("system") for row in hooke_rows}):
        raise ProtocolError("Hooke-2e ACT50 evaluation rows only admit hooke2")
    systems = declared_systems or tuple(dict.fromkeys(str(row["system"]) for row in hooke_rows))
    expected = [
        (system, lineage, checkpoint, stream)
        for system in systems
        for lineage in range(STAGE1_PROTOCOL.lineages)
        for checkpoint in hooke_profile_selection(profile, system)[2]
        for stream in range(STAGE1_PROTOCOL.eval_streams)
    ]
    actual: list[tuple[str, int, int, int]] = []
    for row in hooke_rows:
        if row.get("kind") != "eval":
            raise ProtocolError("Hooke evaluation grid contains a non-eval row")
        coordinates = (row.get("system"), row.get("lineage"),
                       row.get("checkpoint"), row.get("stream"))
        if (type(coordinates[1]) is not int or type(coordinates[2]) is not int
                or type(coordinates[3]) is not int):
            raise ProtocolError("Hooke evaluation grid coordinates must be integers")
        expected_selection, _, _ = hooke_profile_selection(profile, str(row.get("system")))
        if row.get("evaluation_selection") != expected_selection:
            raise ProtocolError("Hooke eval row does not bind its frozen selection")
        actual.append(coordinates)  # type: ignore[arg-type]
    if sorted(actual) != sorted(expected):
        selection_labels = {
            hooke_profile_selection(profile, system)[0] for system in systems
        }
        selection_label = next(iter(selection_labels)) if len(selection_labels) == 1 else "frozen"
        raise ProtocolError(
            f"Hooke {selection_label} grid mismatch: "
            f"expected {len(expected)} rows, got {len(actual)}"
        )


# Explicit names for callers that do not want to depend on the profile name.
validate_exact_hooke_eval_grid = validate_hooke_eval_grid
validate_exact_eval_grid = validate_hooke_eval_grid


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


def row_config_values(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the finite row-owned values applied by a Stage-1 gateway.

    This is deliberately a structured mapping rather than a caller-supplied
    OmegaConf dotlist.  The binding is the only application input; the
    gateway owns this small row-to-config composition step.
    """

    kind = row.get("kind")
    if kind not in KINDS:
        raise ProtocolError("manifest row has an invalid kind")
    values: dict[str, Any] = {
        "runtime": {"seed": row["seed"]},
        "run": {
            "root": row["results_root"],
            "run_id": row["run_id"],
            "layout": "nested",
        },
        "experiment": {
            "name": row["experiment_name"],
            "sector": row["sector"],
        },
    }
    if "device" in row:
        values["runtime"]["device"] = row["device"]
    if kind == "eval":
        checkpoint_dir, checkpoint_file = _checkpoint_binding(row)
        values.update({
            "evaluation_sampler": {"n_walkers": row["walkers"]},
            "evaluation_tasks": {
                "mcmc_energy": {"generator": {"n_draws": row["draws"]}},
            },
            "load": {
                "path": str(checkpoint_dir),
                "mode": "model_only",
                "strict": True,
            },
            "trajectory_identity": {"checkpoint_file": str(checkpoint_file)},
        })
    else:
        values.update({
            "sampler": {"n_walkers": row["walkers"]},
            "trainer": {"max_steps": row["updates"]},
        })
        if "sampler_seed" in row:
            values["sampler"]["seed"] = row["sampler_seed"]
        if row.get("checkpoint_steps"):
            values["checkpoint_steps"] = list(row["checkpoint_steps"])
    return values


def derive_smoke_request(
    train_row: Mapping[str, Any], *, results_root: str, stop_at: int,
) -> dict[str, Any]:
    """Derive one isolated prefix request from an admitted train row.

    A qualification request is intentionally still a ``train`` row.  Its
    science recipe, identity, seeds, horizon, overlay and checkpoint schedule
    are copied byte-for-byte; only the isolated destination and the one
    operational stop coordinate are added.  The request is kept separate from
    the science manifest so reusing a lineage seed cannot enter its collision
    domain.

    Parameters
    ----------
    train_row : mapping
        One planner-produced train row.
    results_root : str
        An isolated results root that differs from the source row's root.
    stop_at : int
        A positive retained checkpoint strictly below the fixed train horizon.
    """

    if train_row.get("kind") != "train":
        raise ProtocolError("smoke qualification must be derived from a train row")
    source_root = _safe_results_root(train_row.get("results_root"))
    isolated_root = _safe_results_root(results_root)
    if Path(source_root).resolve() == Path(isolated_root).resolve():
        raise ProtocolError("smoke qualification requires an isolated results_root")
    if type(stop_at) is not int or stop_at <= 0:
        raise ProtocolError("execution.stop_at must be a positive integer")
    horizon = train_row.get("updates")
    if type(horizon) is not int or horizon <= 0 or stop_at >= horizon:
        raise ProtocolError("execution.stop_at must be below the fixed train horizon")
    steps = train_row.get("checkpoint_steps")
    if (not isinstance(steps, list) or
            any(type(step) is not int for step in steps) or stop_at not in steps):
        raise ProtocolError("execution.stop_at must name a retained checkpoint")
    request = copy.deepcopy(dict(train_row))
    request["results_root"] = isolated_root
    request["execution"] = {"stop_at": stop_at}
    _validate_run_identity(request)
    return request


def _is_overlay_owned_override(key: str) -> bool:
    """Identify override paths reserved by the declared train overlay."""

    return (
        key == "callbacks"
        or key.startswith("callbacks.")
        or key == "loggers"
        or key.startswith("loggers.")
        or key == "stage1_overlay"
        or key.startswith("stage1_overlay.")
    )


def _canonical_override_key(key: str, *, reject_noncanonical: bool) -> str:
    """Return the path spelling used by OmegaConf's update implementation.

    ``OmegaConf.update`` resolves both dotted and getitem-style paths through
    ``split_key``.  Admission must inspect that same path, while science rows
    require the unambiguous dotted spelling at the gateway boundary.
    """

    try:
        canonical = ".".join(split_key(key))
    except Exception as exc:
        raise ProtocolError(f"invalid gateway override key: {key!r}") from exc
    if reject_noncanonical and ("[" in key or "]" in key or key.startswith(".")):
        raise ProtocolError(f"non-canonical gateway override key: {key!r}")
    return canonical


def validate_gateway_binding(*, binding: str, config: str | None = None,
                             overrides: list[str] | None = None, kind: str | None = None,
                             expected_kinds: Collection[str] | None = None,
                             expected_checkpoint_sha256: str | None = None,
                             overlay: str | None = None,
                             overlay_sha256: str | None = None) -> dict[str, Any]:
    """Validate a complete row binding before loading or constructing core state.

    ``config``/``kind``/``overrides`` remain optional only for the pinned
    restart adapter that predates the one-row gateway.  Real train/eval mains
    pass the binding alone, so no repeated application description is accepted
    on the wire.
    """

    row = decode_binding(binding)
    effective_kind = row.get("kind")
    if effective_kind not in KINDS:
        raise ProtocolError("manifest row has an invalid kind")
    if expected_kinds is not None and effective_kind not in expected_kinds:
        expected = ", ".join(sorted(expected_kinds))
        raise ProtocolError(
            f"gateway expected kind in {{{expected}}}, actual kind {effective_kind!r}"
        )
    if kind is not None and effective_kind != kind:
        raise ProtocolError("binding kind does not match gateway")
    row_config = row.get("config")
    if not isinstance(row_config, str):
        raise ProtocolError("gateway config is invalid")
    if row.get("config_identity_rule") not in {
        None, CONFIG_IDENTITY_RULE, LEGACY_CONFIG_IDENTITY_RULE,
    }:
        raise ProtocolError("binding config identity rule is unsupported")
    if config is not None and resolve_config_path(config) != resolve_config_path(row_config):
        raise ProtocolError("gateway config does not match binding")
    declared_overlay = row.get("overlay")
    declared_overlay_sha = row.get("overlay_sha256")
    overlay_profile = str(row.get("overlay_profile", OVERLAY_PROFILE_V1))
    if overlay_profile not in OVERLAY_PROFILES:
        raise ProtocolError(f"unknown diagnostics overlay profile {overlay_profile!r}")
    if (declared_overlay is None) != (declared_overlay_sha is None):
        raise ProtocolError("binding overlay declaration is incomplete")
    if declared_overlay is None or declared_overlay_sha is None:
        if overlay is not None or overlay_sha256 is not None:
            raise ProtocolError("overlay was applied without being declared in the binding")
    else:
        if effective_kind != "train":
            raise ProtocolError("overlay is supported only on train rows")
        if ((overlay is not None and overlay != declared_overlay)
                or (overlay_sha256 is not None and overlay_sha256 != declared_overlay_sha)):
            raise ProtocolError("gateway overlay does not match binding")
        _overlay_binding(
            overlay=declared_overlay,
            expected_sha256=declared_overlay_sha,
            system=str(row.get("system")),
            profile=overlay_profile,
        )
    if effective_kind == "train" and _is_hooke_science_row(row) and declared_overlay is None:
        raise ProtocolError("Hooke science train rows require the Hooke diagnostics overlay")
    config_path = resolve_config_path(row_config)
    _validate_config_identity_for_row(row, config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-1 config path does not exist: {config_path}")
    _validate_run_identity(row)
    if effective_kind == "train" and _is_hooke_science_row(row):
        _validate_hooke_science_train_row(row)
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
    if effective_kind == "eval":
        checkpoint_dir, checkpoint_file = _checkpoint_binding(row)
        expected_checkpoint_digest = _require_sha(
            row.get("checkpoint_sha256"), length=64, field="checkpoint_sha256"
        )
        if expected_kinds is not None:
            # The row-only gateways use this admission path.  The legacy
            # restart adapter continues to validate its historical override
            # contract here, while its restart lifecycle owns checkpoint I/O.
            if not checkpoint_file.is_file():
                raise ProtocolError(f"checkpoint file does not exist: {checkpoint_file}")
            actual_checkpoint_digest = _sha256_file(checkpoint_file)
            if actual_checkpoint_digest != expected_checkpoint_digest:
                raise ProtocolError(
                    "checkpoint file sha256 does not match row checkpoint_sha256: "
                    f"actual={actual_checkpoint_digest}, expected={expected_checkpoint_digest}"
                )
        expected.update({
            f"evaluation_sampler.n_walkers={row['walkers']}",
            f"evaluation_tasks.mcmc_energy.generator.n_draws={row['draws']}",
            f"load.path={checkpoint_dir}", "load.mode=model_only",
            "load.strict=true",
            f"trajectory_identity.checkpoint_file={checkpoint_file}",
        })
        if expected_checkpoint_sha256 is not None:
            _require_sha(expected_checkpoint_sha256, length=64,
                         field="expected_checkpoint_sha256")
        if (expected_checkpoint_sha256 is not None
                and expected_checkpoint_sha256 != expected_checkpoint_digest):
            raise ProtocolError("expected checkpoint sha256 does not match binding")
    else:
        expected.update({f"sampler.n_walkers={row['walkers']}",
                         f"trainer.max_steps={row['updates']}"})
        if _is_hooke_science_row(row):
            expected.add(f"sampler.seed={row['sampler_seed']}")
        if row.get("checkpoint_steps"):
            steps = ",".join(str(step) for step in row["checkpoint_steps"])
            expected.add(f"checkpoint_steps=[{steps}]")
        execution = row.get("execution")
        if execution is not None:
            if not isinstance(execution, Mapping) or set(execution) != {"stop_at"}:
                raise ProtocolError(
                    "execution must contain only the optional stop_at datum"
                )
            stop_at = execution.get("stop_at")
            if (type(stop_at) is not int or stop_at <= 0
                    or type(row.get("updates")) is not int
                    or stop_at >= row["updates"]
                    or stop_at not in row.get("checkpoint_steps", ())):
                raise ProtocolError(
                    "execution.stop_at must be a positive retained boundary below H"
                )
    if overrides is None:
        return row
    override_map: dict[str, str] = {}
    for override in overrides:
        key, separator, value = override.partition("=")
        if key.startswith(("+", "~")):
            raise ProtocolError("prefixed gateway overrides are not allowed")
        if not separator or not key:
            raise ProtocolError(f"invalid gateway override: {override!r}")
        canonical_key = _canonical_override_key(
            key, reject_noncanonical=kind in {"train", "eval"},
        )
        if declared_overlay is not None and _is_overlay_owned_override(canonical_key):
            raise ProtocolError(f"gateway override targets overlay-owned path: {key}")
        if canonical_key in override_map:
            raise ProtocolError(f"duplicate gateway override: {key}")
        override_map[canonical_key] = value
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


def validate_resolved_train_config(cfg: Any, row: Mapping[str, Any], *,
                                   base_config: Any | None = None) -> None:
    """Validate the final producer config immediately before execution.

    Gateway override admission proves the textual row binding.  This second
    boundary proves that OmegaConf composition did not later replace a bound
    science leaf, callback list, or physical model section.
    """

    if row.get("kind") != "train" or row.get("overlay") is None:
        return
    if row.get("overlay_sha256") is None:
        raise ProtocolError("overlay-backed train row is missing overlay_sha256")
    relative, digest, _ = _overlay_binding(
        overlay=str(row["overlay"]),
        expected_sha256=row["overlay_sha256"],
        system=str(row.get("system")),
        profile=str(row.get("overlay_profile", OVERLAY_PROFILE_V1)),
    )
    if _is_hooke_science_row(row):
        # The final guard is also a public boundary: callers can bypass the
        # manifest planner and present a forged science row directly here.
        _validate_hooke_science_train_row(row)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, Mapping):
        raise ProtocolError("resolved train config is not a mapping")

    def select(path: str) -> Any:
        value: Any = resolved
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise ProtocolError(f"resolved train config is missing {path}")
            value = value[part]
        return value

    if select("stage1_overlay.path") != relative or select("stage1_overlay.sha256") != digest:
        raise ProtocolError("resolved train config overlay provenance does not match binding")
    overlay_profile = str(row.get("overlay_profile", OVERLAY_PROFILE_V1))
    expected_order = (
        OVERLAY_COMPOSITION_ORDER if overlay_profile == OVERLAY_PROFILE_V1
        else OVERLAY_V2_COMPOSITION_ORDER
    )
    if select("stage1_overlay.composition_order") != expected_order:
        raise ProtocolError("resolved train config composition order is not canonical")
    if overlay_profile == OVERLAY_PROFILE_V2 and select("stage1_overlay.profile") != OVERLAY_PROFILE_V2:
        raise ProtocolError("resolved train config overlay profile does not match binding")

    unresolved = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(unresolved, Mapping):
        raise ProtocolError("unresolved train config is not a mapping")
    science_profile = str(row.get("science_profile")) if row.get("science_profile") else None
    expected_callbacks = _expected_overlay_callbacks(relative, science_profile=science_profile)
    callbacks = unresolved.get("callbacks")
    if not isinstance(callbacks, list):
        raise ProtocolError("resolved diagnostics callback profile drifted")
    if overlay_profile == OVERLAY_PROFILE_V2:
        targets = [entry.get("_target_") if isinstance(entry, Mapping) else None for entry in callbacks]
        checkpoint_indices = [index for index, target in enumerate(targets) if target == "tpen.callback.Checkpoint"]
        recorder_indices = [
            index for index, target in enumerate(targets)
            if target == "tpen.callback.RawTrainingBatchRecorder"
        ]
        if not checkpoint_indices or recorder_indices != [index - 1 for index in checkpoint_indices]:
            raise ProtocolError("resolved diagnostics callback profile drifted")
        expected_recorder = next(
            entry for entry in expected_callbacks
            if entry.get("_target_") == "tpen.callback.RawTrainingBatchRecorder"
        )
        if any(entry != expected_recorder for entry in (
            callbacks[index] for index in recorder_indices
        )):
            raise ProtocolError("resolved diagnostics callback profile drifted")
        expected_callbacks = [
            entry for entry in expected_callbacks
            if entry.get("_target_") != "tpen.callback.RawTrainingBatchRecorder"
        ]
    overlay_start = None
    for index in range(len(callbacks) - len(expected_callbacks) + 1):
        if callbacks[index:index + len(expected_callbacks)] == expected_callbacks:
            overlay_start = index
            break
    if overlay_start is None:
        raise ProtocolError("resolved diagnostics callback profile drifted")
    trailing_callbacks = callbacks[overlay_start + len(expected_callbacks):]
    if any(
        not isinstance(entry, Mapping)
        or entry.get("_target_") != "tpen.callback.InitialCheckpoint"
        or set(entry) != {"_target_", "output_dir"}
        for entry in trailing_callbacks
    ):
        raise ProtocolError("resolved diagnostics callback profile drifted")

    if not _is_hooke_science_row(row):
        return
    expected_device = row.get("device", "cuda")
    expected_values = {
        "runtime.device": expected_device,
        "runtime.dtype": "float64",
        "sampler.n_walkers": row["walkers"],
        "sampler.burn_in": 100,
        "sampler.n_steps": 10,
        "sampler.proposal_scale": 0.5,
        "sampler.seed": row["sampler_seed"],
        "trainer.max_steps": row["updates"],
        "trainer.log_every_n_steps": 1,
        "trainer.return_terms": True,
        "optimizer._target_": "torch.optim.Adam",
        "optimizer.lr": 0.005,
    }
    for path, expected in expected_values.items():
        if select(path) != expected:
            raise ProtocolError(f"resolved Hooke science config mismatch at {path}")
    checkpoint_steps = list(select("checkpoint_steps"))
    expected_steps = [step for step in row.get("checkpoint_steps", ()) if step > 0]
    if checkpoint_steps != expected_steps:
        raise ProtocolError("resolved Hooke checkpoint schedule does not match the manifest")
    callbacks = select("callbacks")
    targets = [entry.get("_target_") for entry in callbacks if isinstance(entry, Mapping)]
    if targets.count("tpen.callback.AmplitudeRiskFraction") != 1:
        raise ProtocolError("resolved Hooke config must contain one amplitude-risk callback")
    amplitude = next(
        entry for entry in callbacks
        if isinstance(entry, Mapping)
        and entry.get("_target_") == "tpen.callback.AmplitudeRiskFraction"
    )
    if (
        list(amplitude.get("thresholds", ())) != [-3, -4, -5, -7, -10]
        or amplitude.get("every_n_steps") != 100
        or amplitude.get("start_step") != 0
    ):
        raise ProtocolError("resolved Hooke amplitude-risk callback drifted")

    if base_config is not None:
        base = OmegaConf.to_container(base_config, resolve=True)
        if isinstance(base, Mapping):
            for section in ("system", "atoms", "system_descriptor", "model", "hamiltonian_terms"):
                if resolved.get(section) != base.get(section):
                    raise ProtocolError(f"overlay changed base-owned section {section}")
            base_runner = base.get("runner")
            resolved_runner = resolved.get("runner")
            if not isinstance(base_runner, Mapping) or not isinstance(resolved_runner, Mapping):
                raise ProtocolError("resolved train config is missing runner wiring")

            runner_leaf_fields = {
                "sampler": ("n_walkers", "burn_in", "n_steps", "proposal_scale", "seed"),
                "trainer": ("max_steps", "log_every_n_steps", "return_terms"),
            }
            for section, fields in runner_leaf_fields.items():
                base_section = base_runner.get(section)
                resolved_section = resolved_runner.get(section)
                if not isinstance(base_section, Mapping) or not isinstance(resolved_section, Mapping):
                    raise ProtocolError(f"resolved train config is missing runner {section} wiring")
                for field in fields:
                    root_value = select(f"{section}.{field}")
                    if (field not in base_section or field not in resolved_section
                            or resolved_section[field] != root_value):
                        raise ProtocolError(
                            f"runner {section}.{field} does not match root {section}.{field}"
                        )

            def runner_wiring(runner: Mapping[str, Any]) -> dict[str, Any]:
                """Ignore only bound sampler/trainer leaves, not runner wiring."""

                normalized = copy.deepcopy(dict(runner))
                sampler = normalized.get("sampler")
                if isinstance(sampler, Mapping):
                    sampler = dict(sampler)
                    for key in ("n_walkers", "burn_in", "n_steps", "proposal_scale", "seed"):
                        sampler.pop(key, None)
                    normalized["sampler"] = sampler
                trainer = normalized.get("trainer")
                if isinstance(trainer, Mapping):
                    trainer = dict(trainer)
                    for key in ("max_steps", "log_every_n_steps", "return_terms"):
                        trainer.pop(key, None)
                    normalized["trainer"] = trainer
                return normalized

            if runner_wiring(resolved_runner) != runner_wiring(base_runner):
                raise ProtocolError("overlay changed base-owned section runner")


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
                   phase: int = 2, device: str | None = None,
                   overlay: str | Path | None = None,
                   overlay_sha256: str | None = None,
                   science_profile: str | None = None,
                   evaluation_selection: str | None = None,
                   overlay_profile: str = OVERLAY_PROFILE_V1,
                   include_smoke: bool = True,
                   auto_overlay: bool = False) -> list[dict[str, Any]]:
    """Expand systems into validated smoke, train, and eval rows.

    Parameters are explicit rather than defaulted so a missing identity or
    checkpoint binding fails before a launch script can be rendered. The
    explicit physical-ID map is retained for unit fixtures; production
    planning leaves it unset so train/eval parity is always checked here.

    ``include_smoke`` is retained for v3 fixture and reader compatibility.
    New production planning sets it to ``False`` and derives any bounded
    qualification with :func:`derive_smoke_request`. ``auto_overlay`` makes
    the planner select each system's committed overlay instead of requiring a
    caller to repeat its path and digest.
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
    if science_profile not in {None, HOOKE_SCIENCE_PROFILE, *HOOKE2_ACT50_PROFILES}:
        raise ProtocolError("unknown science profile")
    effective_protocol = protocol_for_science_profile(protocol, science_profile)
    hooke_science_profile = science_profile is not None and (
        science_profile == HOOKE_SCIENCE_PROFILE or is_hooke2_act50_profile(science_profile)
    )
    if hooke_science_profile:
        allowed = {"hooke2"} if is_hooke2_act50_profile(science_profile) else HOOKE_SYSTEMS
        if not set(selected).issubset(allowed):
            raise ProtocolError("Hooke science profile cannot mix in He/H2 systems")
        if is_hooke2_act50_profile(science_profile):
            if evaluation_selection not in {None, HOOKE2_ACT50_EVAL_SELECTION}:
                raise ProtocolError("Hooke-2e ACT50 requires the exact A27 selection")
            evaluation_selection = HOOKE2_ACT50_EVAL_SELECTION
            expected_phase2_checkpoints = HOOKE2_ACT50_A27
        elif evaluation_selection not in {None, HOOKE_EVAL_SELECTION, QHO_EVAL_SELECTION}:
            raise ProtocolError("unknown Hooke evaluation selection")
        elif set(selected) == {"qho2e"}:
            if evaluation_selection not in {None, QHO_EVAL_SELECTION}:
                raise ProtocolError("QHO-2e requires the exact Q21 selection")
            evaluation_selection = QHO_EVAL_SELECTION
            expected_phase2_checkpoints = QHO_Q21
        else:
            if evaluation_selection not in {None, HOOKE_EVAL_SELECTION}:
                raise ProtocolError("mixed Hooke science requires the A37 selection")
            evaluation_selection = HOOKE_EVAL_SELECTION
            expected_phase2_checkpoints = HOOKE_A37
        if phase == 2 and list(checkpoints) != list(expected_phase2_checkpoints):
            raise ProtocolError(
                f"Hooke science phase 2 requires the exact {evaluation_selection} roster"
            )
    elif evaluation_selection is not None:
        raise ProtocolError("evaluation_selection requires the Hooke science profile")
    he_h2_selection = (
        phase == 2 and set(selected).issubset(HE_H2_SYSTEMS)
        and _is_he_h2_production_protocol(protocol)
    )
    if he_h2_selection and list(checkpoints) != list(HE_H2_EVAL_CHECKPOINTS):
        raise ProtocolError("He/H2 production phase 2 requires the exact roster-B checkpoints")
    if (overlay is None) != (overlay_sha256 is None):
        raise ProtocolError("overlay and overlay_sha256 must be supplied together")
    if overlay_profile not in OVERLAY_PROFILES:
        raise ProtocolError(f"unknown diagnostics overlay profile {overlay_profile!r}")
    if overlay is None and overlay_profile != OVERLAY_PROFILE_V1 and not auto_overlay:
        raise ProtocolError("diagnostics overlay profile requires an overlay")
    overlay_name: str | None = None
    overlay_digest: str | None = None
    overlays_by_system: dict[str, tuple[str, str]] = {}
    if overlay is not None:
        expected_paths = {
            overlay_path_for_system(system, profile=overlay_profile) for system in selected
        }
        if len(expected_paths) != 1:
            raise ProtocolError("one declared overlay cannot mix system-qualified profiles")
        expected_system = next(iter(selected))
        overlay_name, overlay_digest, _ = _overlay_binding(
            overlay=overlay,
            expected_sha256=overlay_sha256,
            system=expected_system,
            profile=overlay_profile,
        )
        overlays_by_system = {
            system: (overlay_name, overlay_digest) for system in selected
        }
    elif auto_overlay:
        for system in selected:
            committed = overlay_path_for_system(system, profile=overlay_profile)
            digest = _sha256_file(resolve_overlay_path(committed))
            relative, checked_digest, _ = _overlay_binding(
                overlay=committed, expected_sha256=digest, system=system,
                profile=overlay_profile,
            )
            overlays_by_system[system] = (relative, checked_digest)
    elif hooke_science_profile:
        raise ProtocolError("Hooke science train rows require the Hooke diagnostics overlay")
    if hooke_science_profile and (
        effective_protocol.lineage_walkers != STAGE1_PROTOCOL.lineage_walkers
        or effective_protocol.lineages != STAGE1_PROTOCOL.lineages
        or effective_protocol.eval_streams != STAGE1_PROTOCOL.eval_streams
        or effective_protocol.eval_walkers != STAGE1_PROTOCOL.eval_walkers
        or effective_protocol.eval_draws != STAGE1_PROTOCOL.eval_draws
        or effective_protocol.retention_cadence != 1000
    ):
        raise ProtocolError(
            "Hooke science profile requires the exact production protocol dimensions"
        )
    if not checkpoints or any(type(step) is not int or step < 0 for step in checkpoints):
        raise ProtocolError("checkpoints must be a non-empty list of non-negative updates")
    if len(set(checkpoints)) != len(checkpoints):
        raise ProtocolError("checkpoints must not contain duplicates")
    if effective_protocol.retention_cadence is not None and (
        phase == 1 or not hooke_science_profile
    ):
        expected_checkpoints = list(range(
            0, effective_protocol.lineage_updates + 1, effective_protocol.retention_cadence
        ))
        if checkpoints != expected_checkpoints:
            raise ProtocolError(
                "train checkpoints must equal the declared retention cadence"
            )
    if phase in {1, 2}:
        if any(step > effective_protocol.lineage_updates for step in checkpoints):
            raise ProtocolError("checkpoints must not exceed lineage_updates")
        if effective_protocol.lineage_updates not in checkpoints:
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
    if any(type(seed) is not int or seed < 0 or seed >= _LINEAGE_SEED_RADIX for seed in lineage_seeds):
        raise ProtocolError("lineage_seeds must be bounded integers")
    # Check the complete protocol before applying a selected-system filter.
    # Otherwise an excluded system can hide a seed collision from the plan.
    protocol_seeds: set[int] = set()
    for spec in SYSTEM_SPECS:
        candidate_seeds = []
        if include_smoke:
            candidate_seeds.append(smoke_seed(system=spec.name))
        candidate_seeds.extend(
            training_seed(system=spec.name, lineage=lineage, lineage_seed=seed)
            for lineage, seed in enumerate(lineage_seeds)
        )
        if hooke_science_profile and (
            spec.name == "hooke2" if is_hooke2_act50_profile(science_profile)
            else spec.name in HOOKE_SYSTEMS
        ):
            candidate_seeds.extend(
                sampler_seed(system=spec.name, lineage=lineage, lineage_seed=seed)
                for lineage, seed in enumerate(lineage_seeds)
            )
        candidate_checkpoints = (
            hooke_profile_selection(str(science_profile), spec.name)[2]
            if hooke_science_profile and (
                spec.name == "hooke2" if is_hooke2_act50_profile(science_profile)
                else spec.name in HOOKE_SYSTEMS
            )
            else checkpoints
        )
        candidate_seeds.extend(
            evaluation_seed(system=spec.name, lineage=lineage,
                            checkpoint=checkpoint, stream=stream)
            for lineage in range(protocol.lineages)
            for checkpoint in candidate_checkpoints
            for stream in range(protocol.eval_streams)
        )
        for candidate in candidate_seeds:
            if candidate in protocol_seeds:
                raise ProtocolError("seed collision across Stage-1 rows")
            protocol_seeds.add(candidate)
    rows: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    training_checkpoints = (
        list(range(0, effective_protocol.lineage_updates + 1, effective_protocol.retention_cadence))
        if hooke_science_profile and effective_protocol.retention_cadence is not None
        else list(checkpoints)
    )
    for spec in SYSTEM_SPECS:
        if spec.name not in selected:
            continue
        spec.validate()
        if include_smoke:
            seed = smoke_seed(system=spec.name)
            if seed in seen_seeds:
                raise ProtocolError("seed collision across Stage-1 rows")
            seen_seeds.add(seed)
            smoke_row = {
                "row_id": stable_row_id(system=spec.name, kind="smoke"),
                "system": spec.name, "system_id": spec.system_id, "kind": "smoke",
                "seed": seed, "source_sha": source_sha, "config_sha256": config_ids[spec.name],
                "config_identity_rule": config_identity_rule,
                # v3 compatibility only. New qualification requests are
                # derived from a train row and retain its full recipe.
                "config": f"configs/{spec.name}/train.yaml", "results_root": root,
                "experiment_name": "stage1", "sector": spec.name, "run_id": "smoke",
                "updates": protocol.smoke_updates,
                "walkers": protocol.smoke_walkers,
                "checkpoint_steps": [protocol.smoke_updates],
            }
            if is_hooke2_act50_profile(science_profile):
                smoke_row.update({
                    "science_profile": str(science_profile),
                    "variant": hooke2_act50_profile(str(science_profile))["variant"],
                })
            rows.append(smoke_row)
        for lineage, seed in enumerate(lineage_seeds):
            train_seed = training_seed(system=spec.name, lineage=lineage, lineage_seed=seed)
            if train_seed in seen_seeds:
                raise ProtocolError("seed collision across Stage-1 rows")
            seen_seeds.add(train_seed)
            train_id = stable_row_id(system=spec.name, kind="train", lineage=lineage)
            row: dict[str, Any] = {
                "row_id": train_id, "system": spec.name, "system_id": spec.system_id,
                "kind": "train", "lineage": lineage, "seed": train_seed,
                "source_sha": source_sha, "config_sha256": config_ids[spec.name],
                "config_identity_rule": config_identity_rule,
                "config": f"configs/{spec.name}/train.yaml", "results_root": root,
                "experiment_name": "stage1", "sector": spec.name,
                "run_id": f"train-lineage-{lineage}",
                "updates": effective_protocol.lineage_updates if hooke_science_profile else protocol.lineage_updates,
                "walkers": effective_protocol.lineage_walkers if hooke_science_profile else protocol.lineage_walkers,
                "checkpoint_steps": training_checkpoints,
            }
            if hooke_science_profile:
                selection, selection_id, _ = hooke_profile_selection(
                    str(science_profile), spec.name,
                )
                row.update({
                    "science_profile": str(science_profile),
                    "training_seed_rule": HOOKE_TRAINING_SEED_RULE,
                    "lineage_seed": seed,
                    "evaluation_selection": selection,
                    "evaluation_selection_id": selection_id,
                    "selection_id": selection_id,
                    "sampler_seed": sampler_seed(
                        system=spec.name, lineage=lineage, lineage_seed=seed,
                    ),
                })
                if is_hooke2_act50_profile(science_profile):
                    row["variant"] = hooke2_act50_profile(str(science_profile))["variant"]
                _validate_hooke_science_train_row(row)
            if spec.name in overlays_by_system:
                row["overlay"], row["overlay_sha256"] = overlays_by_system[spec.name]
                if overlay_profile != OVERLAY_PROFILE_V1:
                    row["overlay_profile"] = overlay_profile
            rows.append(row)
            train_row = rows[-1]
            if phase == 1:
                continue
            eval_checkpoints = (
                hooke_profile_selection(str(science_profile), spec.name)[2]
                if hooke_science_profile and (
                    spec.name == "hooke2" if is_hooke2_act50_profile(science_profile)
                    else spec.name in HOOKE_SYSTEMS
                )
                else tuple(checkpoints)
            )
            for checkpoint in eval_checkpoints:
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
                    eval_row = {
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
                        **({
                            "science_profile": str(science_profile),
                            "evaluation_selection": hooke_profile_selection(str(science_profile), spec.name)[0],
                            "evaluation_selection_id": hooke_profile_selection(str(science_profile), spec.name)[1],
                            "selection_id": hooke_profile_selection(str(science_profile), spec.name)[1],
                            **({"variant": hooke2_act50_profile(str(science_profile))["variant"]}
                               if is_hooke2_act50_profile(science_profile) else {}),
                        } if hooke_science_profile else {
                            "evaluation_selection": HE_H2_EVAL_SELECTION,
                            "evaluation_selection_id": HE_H2_EVAL_SELECTION_ID,
                            "selection_id": HE_H2_EVAL_SELECTION_ID,
                        } if he_h2_selection else {}),
                    }
                    rows.append(eval_row)
    for row in rows:
        _validate_run_identity(row)
        if row["kind"] == "eval":
            _checkpoint_binding(row)
    if device is not None:
        for row in rows:
            row["device"] = device
    if hooke_science_profile and phase == 2:
        validate_hooke_eval_grid([row for row in rows if row["kind"] == "eval"])
    if he_h2_selection:
        validate_he_h2_eval_grid(
            [row for row in rows if row["kind"] == "eval"],
            expected_systems=selected,
            expected_profile=HE_H2_EVAL_SELECTION,
        )
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
    he_h2_selection = _is_he_h2_production_phase(phase1_rows)
    hooke_profiles = {row.get("science_profile") for row in train_rows if _is_hooke_science_row(row)}
    hooke_science = bool(hooke_profiles)
    if hooke_science:
        if not set(systems).issubset(HOOKE_SYSTEMS):
            raise ProtocolError("Hooke science phase cannot mix in He/H2 rows")
        if len(hooke_profiles) != 1:
            raise ProtocolError("phase-1 Hooke rows must use one finite science profile")
        hooke_profile = _hooke_profile(train_rows[0])
        for row in train_rows:
            if row.get("system") in HOOKE_SYSTEMS:
                _validate_hooke_science_train_row(row)
        expected_checkpoints = hooke_profile_selection(
            hooke_profile, "hooke2" if is_hooke2_act50_profile(hooke_profile)
            else ("qho2e" if set(systems) == {"qho2e"} else "hooke1")
        )[2]
        if list(checkpoints) != list(expected_checkpoints):
            selection = hooke_profile_selection(
                hooke_profile, "hooke2" if is_hooke2_act50_profile(hooke_profile)
                else ("qho2e" if set(systems) == {"qho2e"} else "hooke1")
            )[0]
            raise ProtocolError(
                f"Hooke science phase 2 requires the exact {selection} roster"
            )
    else:
        hooke_profile = None
    if he_h2_selection and list(checkpoints) != list(HE_H2_EVAL_CHECKPOINTS):
        raise ProtocolError("He/H2 production phase 2 requires the exact roster-B checkpoints")
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
    if he_h2_selection:
        expected_bindings = {
            (system, lineage, checkpoint)
            for system in systems
            for lineage in range(STAGE1_PROTOCOL.lineages)
            for checkpoint in HE_H2_EVAL_CHECKPOINTS
        }
        if set(bindings) != expected_bindings:
            raise ProtocolError("He/H2 roster-B checkpoint evidence is incomplete or out of scope")
    if hooke_science:
        expected_bindings = {
            (system, lineage, checkpoint)
            for system in systems
            for lineage in range(STAGE1_PROTOCOL.lineages)
            for checkpoint in hooke_profile_selection(str(hooke_profile), system)[2]
        }
        if set(bindings) != expected_bindings:
            raise ProtocolError("Hooke checkpoint evidence is incomplete or out of scope")
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
    combined_seeds = {
        seed
        for row in phase1_rows
        for seed in (row.get("seed"), row.get("sampler_seed"))
    }
    for train_row in train_rows:
        system = str(train_row["system"])
        lineage = int(train_row["lineage"])
        eval_checkpoints = (
            hooke_profile_selection(str(hooke_profile), system)[2]
            if hooke_science and system in HOOKE_SYSTEMS
            else tuple(checkpoints)
        )
        for checkpoint in eval_checkpoints:
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
                    **({
                        "science_profile": str(hooke_profile),
                        "evaluation_selection": hooke_profile_selection(str(hooke_profile), system)[0],
                        "evaluation_selection_id": hooke_profile_selection(str(hooke_profile), system)[1],
                        "selection_id": hooke_profile_selection(str(hooke_profile), system)[1],
                        **({"variant": hooke2_act50_profile(str(hooke_profile))["variant"]}
                           if is_hooke2_act50_profile(hooke_profile) else {}),
                    } if hooke_science else {
                        "evaluation_selection": HE_H2_EVAL_SELECTION,
                        "evaluation_selection_id": HE_H2_EVAL_SELECTION_ID,
                        "selection_id": HE_H2_EVAL_SELECTION_ID,
                    } if he_h2_selection else {}),
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
    if hooke_science:
        validate_hooke_eval_grid(rows, expected_systems=tuple(systems), expected_profile=hooke_profile)
    if he_h2_selection:
        validate_he_h2_eval_grid(
            rows,
            expected_systems=systems,
            expected_profile=HE_H2_EVAL_SELECTION,
        )
    return rows


def manifest_hash(rows: list[Mapping[str, Any]]) -> str:
    """Hash a manifest using canonical JSON serialization."""

    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
