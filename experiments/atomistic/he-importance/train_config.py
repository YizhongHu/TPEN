"""Resolve one materialized HI train row into a runnable Hydra config.

The resolver is deliberately a checkout-boundary adapter.  The HI experiment
modules live under ``experiments/`` and are not included by the package build,
so callers must run from a source checkout with its repository root on
``sys.path``.  The precondition is checked explicitly rather than hidden in a
fallback importer.

This module only composes and validates configuration.  It does not launch a
process, instantiate a torch object, create a run directory, or read the
evaluation manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
import sys
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


_CONFIG_PATH = Path(__file__).with_name("configs") / "train.yaml"
_STAGE_MODULE = "experiments.atomistic.he-importance.stage_coordinate"


class TrainConfigResolutionError(ValueError):
    """A materialized train row cannot become a runnable configuration."""


class CheckoutPreconditionError(TrainConfigResolutionError):
    """The source checkout root is absent from ``sys.path``."""


class UnavailableOptimizerError(TrainConfigResolutionError):
    """A row names an optimizer cell that the build-time roster refuses."""


def _require_checkout_root() -> Path:
    """Require the repository root on ``sys.path`` before loading experiment code.

    Returns
    -------
    pathlib.Path
        The resolved repository root.

    Raises
    ------
    CheckoutPreconditionError
        If no ``sys.path`` entry resolves to the checkout root.
    """

    repo_root = Path(__file__).resolve().parents[3]
    candidates: set[Path] = set()
    for entry in sys.path:
        try:
            candidates.add(Path(entry or Path.cwd()).resolve())
        except (OSError, RuntimeError, TypeError):
            continue
    if repo_root not in candidates:
        raise CheckoutPreconditionError(
            "HI train resolver requires the repository checkout root on sys.path; "
            f"expected {repo_root}"
        )
    return repo_root


def _stage_api() -> Any:
    """Load the stage-coordinate API only after the checkout precondition."""

    _require_checkout_root()
    return import_module(_STAGE_MODULE)


def _compose_base_config() -> DictConfig:
    """Compose the committed HI train configuration without launch side effects."""

    if not _CONFIG_PATH.is_file():
        raise TrainConfigResolutionError(f"missing committed HI train config: {_CONFIG_PATH}")
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_PATH.parent.resolve())):
        return compose(config_name=_CONFIG_PATH.stem)


def _cell_from_source(source: Any) -> Any:
    """Extract a materialized cell from a cell or a training packet."""

    cell = getattr(source, "cell", source)
    required = ("manifest", "content_hash", "output_path", "seed_streams")
    if any(not hasattr(cell, name) for name in required):
        raise TrainConfigResolutionError(
            "source must be a MaterializedCell or TrainingPacket with a complete cell"
        )
    return cell


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Require a mapping at a materialized-manifest boundary."""

    if not isinstance(value, Mapping):
        raise TrainConfigResolutionError(f"{label} must be a mapping")
    return value


def _integer_seed(streams: Mapping[str, Any], name: str) -> int:
    """Return one explicit seed stream, refusing implicit RNG fallback."""

    value = streams.get(name)
    if type(value) is not int or value < 0:
        raise TrainConfigResolutionError(f"seed stream {name!r} must be a non-negative integer")
    return value


def _initializer_paths(value: Any, path: str = "") -> tuple[str, ...]:
    """Find every named initializer block in a Hydra configuration."""

    found: list[str] = []
    if isinstance(value, Mapping):
        initializer = value.get("initializer")
        if isinstance(initializer, Mapping):
            found.append(f"{path}.initializer" if path else "initializer")
        for key, nested in value.items():
            child = f"{path}.{key}" if path else str(key)
            found.extend(_initializer_paths(nested, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            found.extend(_initializer_paths(nested, f"{path}.{index}" if path else str(index)))
    return tuple(found)


def _set_named_initializer_seeds(cfg: DictConfig, seed: int) -> None:
    """Bind every declared initializer to the shared model-init seed."""

    # ``runner.model`` interpolates the same model block.  Traverse the
    # canonical model declaration once; walking the alias produces a second
    # path whose relative interpolation does not expose the named stream.
    paths = _initializer_paths(cfg.model, "model")
    if not paths:
        raise TrainConfigResolutionError("HI train config declares no named initializer")
    for path in paths:
        initializer = OmegaConf.select(cfg, path)
        if not isinstance(initializer, Mapping) or not initializer.get("stream"):
            raise TrainConfigResolutionError(f"initializer {path!r} must declare a named stream")
        OmegaConf.update(cfg, f"{path}.seed", seed, merge=False, force_add=True)


def _set_optional_execution_seeds(cfg: DictConfig, streams: Mapping[str, Any]) -> None:
    """Bind existing sampler and diagnostic seed fields when their domains exist."""

    if OmegaConf.select(cfg, "sampler.seed", default=None) is not None:
        sampler_seed = _integer_seed(streams, "training_sampler")
        OmegaConf.update(cfg, "sampler.seed", sampler_seed, merge=False)

    diagnostic_seed = streams.get("diagnostic")
    if type(diagnostic_seed) is int and diagnostic_seed >= 0:
        callbacks = OmegaConf.select(cfg, "callbacks", default=[])
        for index, callback in enumerate(callbacks):
            if not isinstance(callback, Mapping):
                continue
            checkers = callback.get("checkers")
            if not isinstance(checkers, Sequence) or isinstance(checkers, (str, bytes, bytearray)):
                continue
            for checker_index, checker in enumerate(checkers):
                if isinstance(checker, Mapping) and "seed" in checker:
                    OmegaConf.update(
                        cfg,
                        f"callbacks.{index}.checkers.{checker_index}.seed",
                        diagnostic_seed,
                        merge=False,
                    )


def _optimizer_entry(manifest: Mapping[str, Any]) -> Any:
    """Resolve the cell's optimizer through the live build-time roster."""

    identity = _mapping(manifest.get("scientific_identity"), "scientific_identity")
    optimizer_cell = _mapping(identity.get("optimizer_cell"), "scientific_identity.optimizer_cell")
    method = optimizer_cell.get("method")
    status = optimizer_cell.get("status")
    if not isinstance(method, str) or not method:
        raise TrainConfigResolutionError("optimizer cell must declare a method")
    if status not in {"available", "unavailable"}:
        raise TrainConfigResolutionError("optimizer cell must declare an availability status")

    schema = import_module("tpen.hi_schema")
    roster = tuple(schema.HI_METHOD_ROSTER)
    by_method = {entry.method: entry for entry in roster}
    entry = by_method.get(method)
    if entry is None:
        raise UnavailableOptimizerError(f"optimizer method {method!r} is absent from the HI roster")
    if not entry.admitted or status != "available" or not entry.target:
        reason = optimizer_cell.get("reason") or "not admitted by HI_METHOD_ROSTER"
        raise UnavailableOptimizerError(
            f"optimizer method {method!r} is unavailable: {reason}"
        )
    return entry


def resolve_train_config(source: Any) -> DictConfig:
    """Resolve one materialized cell or training packet into a Hydra config.

    Parameters
    ----------
    source
        A ``MaterializedCell`` or a ``TrainingPacket`` wrapping one.  The
        structural boundary is intentional so a packet imported through the
        checkout's real ``tpen.hi.train.v1`` namespace is accepted without
        duplicating the stage-coordinate classes here.

    Returns
    -------
    omegaconf.DictConfig
        A resolved, schema-validated configuration with row-specific identity,
        paths, horizon, optimizer, and explicit seed streams.

    Raises
    ------
    TrainConfigResolutionError
        If the row is malformed, unavailable, or has no train horizon.
    tpen.config_schema.ClosedSchemaError
        If the composed configuration fails the HI schema firewall.
    """

    cell = _cell_from_source(source)
    stage_api = _stage_api()
    manifest = _mapping(cell.manifest, "cell.manifest")
    stage_api.validate_materialized_manifest(manifest)
    if stage_api.content_hash(manifest) != cell.content_hash:
        raise TrainConfigResolutionError("cell content hash does not bind its manifest")

    stage = stage_api.stage_definition(manifest["stage"])
    if stage.updates is None:
        raise TrainConfigResolutionError(
            f"stage {stage.code!r} has no fixed training horizon"
        )
    output_path = Path(cell.output_path)
    if not output_path.is_absolute() or output_path.parent.parent == output_path:
        raise TrainConfigResolutionError("cell output_path must be an absolute stage/hash path")

    optimizer_entry = _optimizer_entry(manifest)
    streams = _mapping(cell.seed_streams, "cell.seed_streams")
    model_seed = _integer_seed(streams, "model_initialization")

    cfg = _compose_base_config()
    OmegaConf.update(cfg, "trainer.max_steps", stage.updates, merge=False)
    OmegaConf.update(cfg, "run.root", str(output_path.parent.parent), merge=False)
    OmegaConf.update(cfg, "run.run_id", str(cell.content_hash), merge=False)
    OmegaConf.update(cfg, "run.layout", "flat", merge=False, force_add=True)
    OmegaConf.update(cfg, "run.dir", str(output_path), merge=False)
    OmegaConf.update(cfg, "runtime.seed", model_seed, merge=False)
    OmegaConf.update(cfg, "optimizer._target_", optimizer_entry.target, merge=False)
    _set_named_initializer_seeds(cfg, model_seed)
    _set_optional_execution_seeds(cfg, streams)
    OmegaConf.resolve(cfg)

    schema = import_module("tpen.hi_schema")
    schema.validate_hi_train_config(cfg)
    return cfg


resolve_training_config = resolve_train_config


__all__ = [
    "CheckoutPreconditionError",
    "TrainConfigResolutionError",
    "UnavailableOptimizerError",
    "resolve_train_config",
    "resolve_training_config",
]
