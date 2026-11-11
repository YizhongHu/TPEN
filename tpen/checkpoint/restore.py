"""Checkpoint restore modes."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tpen.accelerator import canonical_device

from .artifact import resolve_checkpoint_dir
from .hashing import checkpoint_hashes, file_sha256
from .replay import (
    CheckpointReplaySemantics,
    coerce_checkpoint_replay_semantics,
    verify_checkpoint_replay_semantics,
)
from .rng import apply_rng_state, require_restorable_rng_state, runtime_device
from .schema import read_manifest, validate_manifest_state

RESTORE_MODES = ("none", "model_only", "train_resume")

_FEATURE = "checkpoint restore"


@dataclass(frozen=True)
class RestoreReport:
    """Summary of the state restored from a checkpoint.

    Parameters
    ----------
    mode : str
        Restore mode that produced this report.
    checkpoint_dir : str or None, optional
        Resolved checkpoint step directory, or ``None`` for ``mode="none"``.
    schema_version : int or None, optional
        Manifest schema version that was read.
    next_iteration : int or None, optional
        Resume cursor recorded by the manifest: the iteration a resumed run
        continues from. ``None`` for ``mode="none"``.
    completed_updates : int or None, optional
        Applied optimizer updates recorded by the manifest. ``None`` for
        ``mode="none"`` and for a v1 manifest under ``model_only``, which never
        recorded the counter. Always populated under ``train_resume``, which
        only admits v2.
    loaded_model, loaded_optimizer, loaded_trainer, loaded_sampler, loaded_rng : bool, optional
        Which components were restored.
    """

    mode: str
    checkpoint_dir: str | None = None
    schema_version: int | None = None
    next_iteration: int | None = None
    completed_updates: int | None = None
    loaded_model: bool = False
    loaded_optimizer: bool = False
    loaded_trainer: bool = False
    loaded_sampler: bool = False
    loaded_rng: bool = False
    replay_semantics: CheckpointReplaySemantics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report mapping."""

        payload = {
            "mode": self.mode,
            "checkpoint_dir": self.checkpoint_dir,
            "schema_version": self.schema_version,
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
            "loaded_model": self.loaded_model,
            "loaded_optimizer": self.loaded_optimizer,
            "loaded_trainer": self.loaded_trainer,
            "loaded_sampler": self.loaded_sampler,
            "loaded_rng": self.loaded_rng,
        }
        if self.replay_semantics is not None:
            payload["replay_semantics"] = self.replay_semantics.to_dict()
        return payload


def restore_checkpoint(
    *,
    load: Any,
    model: Any,
    context: Any,
    optimizer: Any | None = None,
    trainer: Any | None = None,
    sampler: Any | None = None,
    mode: str | None = None,
    strict: bool | None = None,
    allow_protocol_mismatch: bool | None = None,
) -> RestoreReport:
    """Restore checkpoint state into explicitly configured objects."""

    config = _load_config(load)
    mode = str(mode or config.get("mode", "none"))
    if mode not in RESTORE_MODES:
        raise ValueError(f"load.mode must be one of {RESTORE_MODES}, got {mode!r}")
    if mode == "none":
        return RestoreReport(mode="none")

    path = config.get("path")
    if path in (None, ""):
        raise ValueError(f"load.path is required for mode={mode!r}")
    strict_load = bool(config.get("strict", True) if strict is None else strict)
    allow_mismatch = bool(
        config.get("allow_protocol_mismatch", False)
        if allow_protocol_mismatch is None
        else allow_protocol_mismatch
    )

    checkpoint_dir = resolve_checkpoint_dir(path)
    # Schema acceptance is mode-dependent, so the mode is decided before the
    # manifest is read: a v1 artifact is refused for `train_resume` at the gate.
    manifest = read_manifest(checkpoint_dir / "manifest.json", mode=mode)
    replay_semantics = (
        coerce_checkpoint_replay_semantics(config["replay_semantics"])
        if config.get("replay_semantics") is not None
        else None
    )
    if replay_semantics is not None:
        verify_checkpoint_replay_semantics(
            replay_semantics,
            manifest=manifest,
            checkpoint_dir=checkpoint_dir,
            model=model,
            context=context,
        )
    current_hashes = checkpoint_hashes(getattr(context, "cfg", {}))

    _verify_hash(manifest.hashes, current_hashes, "model_config", checkpoint_dir)
    if mode == "model_only":
        _verify_hash(
            manifest.hashes,
            current_hashes,
            "hamiltonian_config",
            checkpoint_dir,
            allow_mismatch=allow_mismatch,
        )
        _validate_component_files(
            checkpoint_dir,
            manifest.files,
            manifest.hashes,
            components=("model",),
        )
        _load_model(checkpoint_dir, manifest.files, model, strict=strict_load, context=context)
        return RestoreReport(
            mode=mode,
            checkpoint_dir=str(checkpoint_dir),
            schema_version=manifest.schema_version,
            next_iteration=manifest.next_iteration,
            # `None` for a v1 manifest, which never recorded the counter.
            completed_updates=manifest.completed_updates,
            loaded_model=True,
            replay_semantics=replay_semantics,
        )

    for hash_name in (
        "optimizer_config",
        "trainer_config",
        "sampler_config",
        "hamiltonian_config",
    ):
        _verify_hash(
            manifest.hashes,
            current_hashes,
            hash_name,
            checkpoint_dir,
            allow_mismatch=(hash_name == "hamiltonian_config" and allow_mismatch),
        )

    # Validate every serialized component before the first mutable load.  New
    # manifests carry namespaced content digests in the existing ``hashes``
    # mapping; older v1/v2 artifacts may not, so their presence check remains
    # the compatibility floor while current artifacts get byte validation.
    _validate_component_files(
        checkpoint_dir,
        manifest.files,
        manifest.hashes,
        components=("model", "optimizer", "trainer", "sampler", "rng"),
    )

    # The RNG payload is read and validated before anything is restored. A
    # refused resume must leave the process unmutated, and `_load_sampler` is
    # itself destructive: `MetropolisSampler.load_mcmc_state_dict` recreates its
    # generator on a device mismatch -- reseeding it, or leaving it unseeded when
    # no seed is configured -- so a refusal raised after that point would already
    # have reset the run's dominant RNG source.
    device = runtime_device(context)
    rng_state = _read_rng_state(checkpoint_dir, manifest.files)
    require_restorable_rng_state(rng_state, device, checkpoint_dir)
    # Read and validate trainer progress before loading any mutable component.
    # A permissive trainer must not turn a malformed payload into a partially
    # restored live run.
    trainer_state = _read_trainer_state(checkpoint_dir, manifest.files)
    validate_manifest_state(manifest, trainer_state, mode=mode)

    _load_model(checkpoint_dir, manifest.files, model, strict=strict_load, context=context)
    _load_optimizer(checkpoint_dir, manifest.files, optimizer)
    _load_trainer(trainer, trainer_state)
    _load_sampler(checkpoint_dir, manifest.files, sampler, context)
    apply_rng_state(rng_state, device)
    return RestoreReport(
        mode=mode,
        checkpoint_dir=str(checkpoint_dir),
        schema_version=manifest.schema_version,
        next_iteration=manifest.next_iteration,
        # The schema gate admits only v2 here, so both counters are real.
        completed_updates=manifest.completed_updates,
        loaded_model=True,
        loaded_optimizer=True,
        loaded_trainer=True,
        loaded_sampler=True,
        loaded_rng=True,
        replay_semantics=replay_semantics,
    )


def restore_checkpoint_with_events(
    *,
    load: Any,
    model: Any,
    context: Any,
    emit: Any,
    optimizer: Any | None = None,
    trainer: Any | None = None,
    sampler: Any | None = None,
) -> RestoreReport:
    """Restore a checkpoint while emitting durable load lifecycle events."""

    config = _load_config(load)
    mode = str(config.get("mode", "none"))
    if mode == "none":
        return RestoreReport(mode="none")
    path = config.get("path")
    strict = bool(config.get("strict", True))
    from .events import LoadFailed, LoadStarted, LoadSucceeded

    emit(LoadStarted(path=str(path), mode=mode, strict=strict))
    try:
        report = restore_checkpoint(
            load=load,
            model=model,
            context=context,
            optimizer=optimizer,
            trainer=trainer,
            sampler=sampler,
        )
    except Exception as exc:
        setattr(exc, "_tpen_failure_phase", "load")
        setattr(exc, "_tpen_load_path", path)
        setattr(exc, "_tpen_load_mode", mode)
        emit(LoadFailed(path=str(path), mode=mode, exception_type=type(exc).__name__, message=str(exc)))
        raise

    emit(LoadSucceeded(path=str(path), report=report))
    return report


def _load_config(load: Any) -> dict[str, Any]:
    if load is None:
        return {"mode": "none"}
    if OmegaConf.is_config(load):
        return dict(OmegaConf.to_container(load, resolve=True))
    if isinstance(load, dict):
        return dict(load)
    raise TypeError("load config must be a mapping or OmegaConf container")


def _verify_hash(
    stored: dict[str, str | None],
    current: dict[str, str | None],
    name: str,
    checkpoint_dir: Path,
    *,
    allow_mismatch: bool = False,
) -> None:
    stored_hash = stored.get(name)
    current_hash = current.get(name)
    if stored_hash is None:
        raise ValueError(f"{checkpoint_dir}: manifest missing {name}")
    if current_hash is None:
        raise ValueError(f"current config is missing {name.removesuffix('_config')} for restore")
    if stored_hash != current_hash and not allow_mismatch:
        raise ValueError(
            f"{checkpoint_dir}: {name} mismatch "
            f"(checkpoint {stored_hash}, current {current_hash})"
        )


def _validate_component_files(
    checkpoint_dir: Path,
    files: dict[str, str],
    hashes: dict[str, str | None],
    *,
    components: tuple[str, ...],
) -> None:
    """Validate required component bytes before any consumer is mutated."""

    for component in components:
        path = _required_file(checkpoint_dir, files, component)
        expected = hashes.get(f"{component}_sha256")
        if expected is None:
            # Archived manifests predate component-content digests.  Keep their
            # existing restore behavior while enforcing the stronger check on
            # every manifest written by the current saver.
            continue
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"{checkpoint_dir}: {component} checkpoint file digest mismatch "
                f"(manifest {expected}, actual {actual})"
            )


def _load_model(
    checkpoint_dir: Path,
    files: dict[str, str],
    model: Any,
    *,
    strict: bool,
    context: Any,
) -> None:
    import torch

    path = _required_file(checkpoint_dir, files, "model")
    map_location = getattr(getattr(context, "metadata", None), "device", "cpu")
    state_dict = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state_dict, strict=strict)
    _assert_model_runtime(model, context)


def _load_optimizer(checkpoint_dir: Path, files: dict[str, str], optimizer: Any) -> None:
    import torch

    if optimizer is None:
        raise ValueError("train_resume restore requires an optimizer")
    path = _required_file(checkpoint_dir, files, "optimizer")
    optimizer.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))


def _load_trainer(trainer: Any, state: Mapping[str, Any]) -> None:
    if trainer is None:
        raise ValueError("train_resume restore requires a trainer")
    load_state_dict = getattr(trainer, "load_state_dict", None)
    if not callable(load_state_dict):
        raise TypeError("trainer must expose load_state_dict() for train_resume restore")

    load_state_dict(state)


def _read_trainer_state(checkpoint_dir: Path, files: dict[str, str]) -> dict[str, Any]:
    """Read trainer progress without applying it to the live trainer."""

    path = _required_file(checkpoint_dir, files, "trainer")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_sampler(checkpoint_dir: Path, files: dict[str, str], sampler: Any, context: Any) -> None:
    import torch

    if sampler is None:
        raise ValueError("train_resume restore requires a sampler")
    load_state = getattr(sampler, "load_mcmc_state_dict", None)
    if not callable(load_state):
        raise TypeError("sampler must expose load_mcmc_state_dict() for train_resume restore")
    path = _required_file(checkpoint_dir, files, "sampler")
    state = torch.load(path, map_location="cpu", weights_only=False)
    target_device = getattr(getattr(context, "metadata", None), "device", None)
    signature = inspect.signature(load_state)
    if "device" in signature.parameters:
        load_state(state, device=target_device)
    else:
        load_state(state)


def _read_rng_state(checkpoint_dir: Path, files: dict[str, str]) -> dict[str, Any]:
    """Read ``rng.pt`` without applying it, so provenance can be checked first."""

    import torch

    path = _required_file(checkpoint_dir, files, "rng")
    return torch.load(path, map_location="cpu", weights_only=False)


def _required_file(checkpoint_dir: Path, files: dict[str, str], key: str) -> Path:
    relative = files.get(key)
    if not relative:
        raise FileNotFoundError(f"{checkpoint_dir}: checkpoint manifest lacks file entry {key!r}")
    path = checkpoint_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"{checkpoint_dir}: missing checkpoint file {relative}")
    return path


def _assert_model_runtime(model: Any, context: Any) -> None:
    import torch

    metadata = getattr(context, "metadata", None)
    expected_device = getattr(metadata, "device", None)
    expected_dtype_name = getattr(metadata, "dtype", None)
    if expected_device is None or expected_dtype_name is None:
        return
    # Index-resolved on both sides: metadata carries the config's index-free
    # device string while tensors report an indexed accelerator device, and
    # `torch.device` treats those as unequal. `canonical_device` closes that gap
    # for whatever backend is live, so this check does not silently become
    # CUDA-only.
    expected_torch_device = canonical_device(expected_device, feature=_FEATURE)
    expected_dtype = getattr(torch, str(expected_dtype_name))
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if canonical_device(tensor.device, feature=_FEATURE) != expected_torch_device:
            raise RuntimeError(
                f"checkpoint restore left model tensor {name!r} on {tensor.device}, "
                f"expected {expected_device}"
            )
        if tensor.is_floating_point() and tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"checkpoint restore left model tensor {name!r} with dtype {tensor.dtype}, "
                f"expected {expected_dtype}"
            )
