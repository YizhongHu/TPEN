"""Structured checkpoint saving."""

from __future__ import annotations

import shutil
import socket
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tpen import __version__ as tpen_version

from .artifact import checkpoint_step_dir_name, write_latest
from .catalog import CheckpointCatalog, publication_catalog_path
from .hashing import checkpoint_hashes, file_sha256
from .manifest import CHECKPOINT_KIND, CHECKPOINT_SCHEMA_VERSION, CheckpointManifest
from .payload import CheckpointPayload, ModelOnly, TrainResume
from .receipt import publication_receipt_path, record_publication_receipt
from .reference import CheckpointRef
from .rng import rng_state_dict, runtime_device


def save_checkpoint(
    *,
    output_dir: str | Path,
    next_iteration: int,
    completed_updates: int,
    model: Any,
    context: Any,
    optimizer: Any | None = None,
    trainer: Any | None = None,
    sampler: Any | None = None,
    payload: CheckpointPayload | None = None,
    save_optimizer: bool | None = None,
    save_trainer: bool | None = None,
    save_sampler: bool | None = None,
    save_rng: bool | None = None,
    keep_last: int | None = None,
    publication_catalog: str | Path | CheckpointCatalog | None = None,
) -> Path:
    """Write one complete directory checkpoint and update ``latest.json``.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Checkpoint root the step directory is written under.
    next_iteration : int
        Trainer resume cursor. Names the checkpoint directory and is recorded
        in the manifest as the checkpoint's identity.
    completed_updates : int
        Applied optimizer updates at write time. Required independently of
        ``save_trainer``: the v2 manifest always records both counters, so the
        count must not depend on whether ``trainer.json`` is written.
    model : Any
        Module whose ``state_dict()`` is written as ``model.pt``.
    context : Any
        Run context supplying ``cfg``, ``metadata``, and ``run_dir``.
    optimizer, trainer, sampler : Any or None, optional
        Train-resume components, required when their ``save_*`` flag is set.
    payload : CheckpointPayload or None, optional
        Explicit payload profile.  When supplied, its required components own
        the defaults and conflicting save flags are rejected.
    save_optimizer, save_trainer, save_sampler, save_rng : bool or None, optional
        Which train-resume components to include.  ``None`` lets ``payload``
        choose; without an explicit payload the historical default is all
        components enabled.
    keep_last : int or None, optional
        Compatibility input accepted and ignored. All committed checkpoints
        are retained.
    publication_catalog : str, pathlib.Path, CheckpointCatalog, or None, optional
        Append-only publication catalog.  When omitted, the catalog is
        ``output_dir/publications.jsonl``.  The ref is constructed and
        appended only after the temporary directory has been atomically
        renamed to its final ``step_*`` directory.

    Returns
    -------
    pathlib.Path
        The completed checkpoint step directory.

    Notes
    -----
    **Publication sequence.** In order: every component file is written into
    ``<step>.tmp``; ``manifest.json`` and ``COMPLETE`` are written last, into
    the same temporary directory; ``tmp_dir.rename(final_dir)`` commits the
    checkpoint; a :class:`~tpen.checkpoint.reference.CheckpointRef` is built
    from the committed directory and appended to the publication catalog;
    ``output_dir/latest.json`` is atomically updated to point at the new
    directory; finally, a :class:`~tpen.checkpoint.receipt.CheckpointPublicationReceipt`
    is built from the committed directory and appended to
    ``output_dir/publication_receipts.jsonl``. Each step after the rename is
    additive: a later step never re-runs, undoes, or reorders an earlier one.

    The catalog publish and the ``latest.json`` update are load-bearing and
    fail loud: an exception there propagates out of this function, though the
    checkpoint remains committed and published (see
    ``tpen.checkpoint.catalog.reconcile_publication`` for repairing a catalog
    row or ``latest.json`` that failed to write). The receipt append is
    deliberately NOT load-bearing: it is best-effort, catching ``OSError``
    and logging a WARNING instead of raising, so a receipts-log failure (for
    example, a quota failure) can never turn an already-committed, published
    checkpoint into a reported failure. See
    :func:`tpen.checkpoint.receipt.record_publication_receipt`. A checkpoint
    whose receipt append failed this way can have it backfilled later via
    ``tpen.checkpoint.catalog.reconcile_publication``, which records sizes
    only -- never fabricated durations. See :mod:`tpen.checkpoint.receipt`
    for the receipt's field-by-field semantics, including the exact instants
    its two durations bracket and when they are explicitly absent instead.
    """

    import torch

    cfg = _require_config(context)
    save_flags = _resolve_save_flags(
        payload,
        save_optimizer=save_optimizer,
        save_trainer=save_trainer,
        save_sampler=save_sampler,
        save_rng=save_rng,
    )
    save_optimizer = save_flags["optimizer"]
    save_trainer = save_flags["trainer"]
    save_sampler = save_flags["sampler"]
    save_rng = save_flags["rng"]
    effective_payload = payload if payload is not None else _infer_payload(save_flags)
    if effective_payload is not None:
        # A public base payload may be structurally valid but noncanonical.
        # Canonicalize it before creating the output root, so a rejected
        # manifest cannot leave a complete directory outside the catalog.
        effective_payload = CheckpointPayload.from_manifest(
            effective_payload.to_manifest()
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    created_at = time.time()
    final_dir = root / checkpoint_step_dir_name(next_iteration)
    tmp_dir = root / f"{final_dir.name}.tmp"
    if final_dir.exists():
        raise FileExistsError(f"checkpoint already exists: {final_dir}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    # Wall-clock duration bracket for the publication receipt. perf_counter is
    # used instead of `created_at`'s time.time() so a system clock adjustment
    # mid-write cannot corrupt the measured duration.
    write_start = time.perf_counter()
    files: dict[str, str] = {}
    try:
        # INSIDE the try, deliberately. It used to run just above it, which
        # left a window: an interrupt after the directory existed but before
        # the try was entered escaped cleanup entirely and stranded the tmp
        # directory -- the residue that then collides with the next attempt.
        # The window was small, not absent, and "small" is not a guarantee.
        tmp_dir.mkdir(parents=True)
        _write_resolved_config(tmp_dir / "resolved_config.yaml", cfg)
        files["resolved_config"] = "resolved_config.yaml"

        torch.save(model.state_dict(), tmp_dir / "model.pt")
        files["model"] = "model.pt"

        if save_optimizer:
            if optimizer is None:
                raise ValueError("save_optimizer=True requires optimizer in the checkpoint event")
            torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
            files["optimizer"] = "optimizer.pt"

        if save_trainer:
            trainer_state = _state_dict_from(trainer, "trainer")
            if effective_payload is not None:
                effective_payload.validate_state(trainer_state)
            _write_json_mapping(tmp_dir / "trainer.json", trainer_state)
            files["trainer"] = "trainer.json"

        if save_sampler:
            sampler_state = _sampler_state_dict(sampler)
            torch.save(sampler_state, tmp_dir / "sampler.pt")
            files["sampler"] = "sampler.pt"

        if save_rng:
            # The run's declared device is recorded with the state, so a resume
            # onto a different device is refused instead of silently continuing
            # on a different random stream.
            torch.save(rng_state_dict(runtime_device(context)), tmp_dir / "rng.pt")
            files["rng"] = "rng.pt"

        hashes = checkpoint_hashes(cfg)
        # Keep the component content digests beside the existing config
        # digests.  They are computed only after every payload file is
        # written, so restore can reject changed bytes before any consumer is
        # mutated.
        hashes.update(_component_file_hashes(tmp_dir, files))
        manifest = CheckpointManifest(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            kind=CHECKPOINT_KIND,
            next_iteration=int(next_iteration),
            completed_updates=int(completed_updates),
            created_at_unix=created_at,
            files=files,
            hashes=hashes,
            runtime=_runtime_metadata(context),
            provenance=_provenance_metadata(context),
            payload=(
                None if effective_payload is None else effective_payload.to_manifest()
            ),
        )
        manifest.write(tmp_dir / "manifest.json")
        (tmp_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
        tmp_dir.rename(final_dir)
        write_end = time.perf_counter()
        # The rename is the checkpoint commit.  Publication is deliberately
        # after it, so a catalog can never name a tmp or partially written
        # directory as a CheckpointRef.
        ref = CheckpointRef.from_directory(final_dir)
        catalog = (
            publication_catalog
            if isinstance(publication_catalog, CheckpointCatalog)
            else CheckpointCatalog(
                publication_catalog_path(root)
                if publication_catalog is None
                else publication_catalog
            )
        )
        catalog.publish(ref)
        # `latest.json` stays minimal: a pointer plus the directory's own step
        # number. The manifest is the place that carries both counters.
        write_latest(root, final_dir, step=int(next_iteration), created_at_unix=created_at)
        publish_end = time.perf_counter()
        # Additive last step: appends a new index entry after the existing
        # rename -> publish -> write_latest sequence without altering it.
        # Best-effort: the checkpoint is already committed and published, so
        # an OSError here (e.g. a quota failure on the receipts log) is
        # logged rather than allowed to report a failed save. See
        # tpen.checkpoint.receipt.record_publication_receipt.
        record_publication_receipt(
            ref,
            final_dir,
            files,
            publication_receipt_path(root),
            write_duration_sec=write_end - write_start,
            publish_duration_sec=publish_end - write_end,
        )
    finally:
        # `finally`, NOT `except Exception`. This was an `except Exception`,
        # which does not run for `KeyboardInterrupt` or `SystemExit` -- both
        # derive from BaseException -- so a Ctrl-C or an interpreter shutdown
        # during a write left the temporary directory behind. That residue is
        # what collides with the next attempt on NFS.
        #
        # Safe on the success path: the rename above moves the directory, so
        # `exists()` is already False and this is a no-op.
        #
        # WHAT THIS STILL DOES NOT COVER, so the guarantee is not overread: a
        # `SIGKILL`, or a `SIGTERM` under Python's default handler, terminates
        # the process WITHOUT unwinding, so no `finally` runs. A scheduler
        # timeout is exactly that case. Interruption safety here is therefore
        # "the interpreter got a chance to unwind", not "the directory is never
        # left behind".
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return final_dir


def _require_config(context: Any) -> Any:
    cfg = getattr(context, "cfg", None)
    if cfg is None:
        raise ValueError("checkpoint saving requires event.context.cfg")
    return cfg


def _resolve_save_flags(
    payload: CheckpointPayload | None,
    *,
    save_optimizer: bool | None,
    save_trainer: bool | None,
    save_sampler: bool | None,
    save_rng: bool | None,
) -> dict[str, bool]:
    """Resolve component flags, letting an explicit payload own defaults."""

    defaults = {
        "optimizer": True,
        "trainer": True,
        "sampler": True,
        "rng": True,
    }
    supplied = {
        "optimizer": save_optimizer,
        "trainer": save_trainer,
        "sampler": save_sampler,
        "rng": save_rng,
    }
    if payload is not None:
        defaults = {
            component: component in payload.required_files
            for component in defaults
        }
    resolved = {
        component: defaults[component] if value is None else value
        for component, value in supplied.items()
    }
    if payload is not None:
        payload.validate_save_flags(
            {"model": True, **resolved}
        )
    return resolved


def _infer_payload(flags: Mapping[str, bool]) -> CheckpointPayload | None:
    """Annotate the two complete built-in flag sets without guessing partial ones."""

    if all(flags[component] for component in ("optimizer", "trainer", "sampler", "rng")):
        return TrainResume()
    if not any(flags.values()):
        return ModelOnly()
    # A partial payload is retained for compatibility with callers that
    # intentionally construct malformed artifacts to test restore failures.
    return None


def _write_resolved_config(path: Path, cfg: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if OmegaConf.is_config(cfg):
        OmegaConf.save(config=cfg, f=path, resolve=True)
        return
    OmegaConf.save(config=OmegaConf.create(cfg), f=path, resolve=True)


def _write_json_mapping(path: Path, data: Mapping[str, Any]) -> None:
    from tpen.artifacts import write_json

    write_json(path, data)


def _component_file_hashes(
    checkpoint_dir: Path, files: Mapping[str, str]
) -> dict[str, str]:
    """Return SHA-256 digests for the files named by a manifest.

    The manifest already has a ``hashes`` mapping for config digests.  The
    namespaced ``<component>_sha256`` entries bind each serialized file to the
    manifest without adding a second schema field, while retaining the older
    config-hash keys used by restore compatibility checks.
    """

    return {
        f"{component}_sha256": file_sha256(checkpoint_dir / relative)
        for component, relative in files.items()
    }


def _state_dict_from(value: Any, owner: str) -> Mapping[str, Any]:
    if value is None:
        raise ValueError(f"save_{owner}=True requires {owner} in the checkpoint event")
    state_dict = getattr(value, "state_dict", None)
    if not callable(state_dict):
        raise TypeError(f"{owner} must expose state_dict() for checkpoint saving")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError(f"{owner}.state_dict() must return a mapping")
    return state


def _sampler_state_dict(sampler: Any) -> Mapping[str, Any]:
    if sampler is None:
        raise ValueError("save_sampler=True requires sampler in the checkpoint event")
    state_dict = getattr(sampler, "mcmc_state_dict", None)
    if not callable(state_dict):
        raise TypeError("sampler must expose mcmc_state_dict() for checkpoint saving")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("sampler.mcmc_state_dict() must return a mapping")
    return state


def _runtime_metadata(context: Any) -> dict[str, Any]:
    import torch

    metadata = getattr(context, "metadata", None)
    return {
        "dtype": getattr(metadata, "dtype", None),
        "device": getattr(metadata, "device", None),
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
    }


def _provenance_metadata(context: Any) -> dict[str, Any]:
    cfg = _require_config(context)
    metadata = getattr(context, "metadata", None)
    extra = getattr(metadata, "extra", None) or {}
    study = OmegaConf.select(cfg, "study", default={}) or {}
    if OmegaConf.is_config(study):
        study = OmegaConf.to_container(study, resolve=True)
    if not isinstance(study, Mapping):
        study = {}
    return {
        "run_id": getattr(metadata, "run_id", None),
        "run_dir": str(getattr(context, "run_dir", "")),
        "config_id": study.get("config_id"),
        "study_name": study.get("name"),
        "git_sha": getattr(metadata, "git_commit", None),
        "git_branch": getattr(metadata, "git_branch", None),
        "git_dirty": getattr(metadata, "dirty_worktree", None),
        "command": getattr(metadata, "command", None),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "tpen_version": tpen_version,
        "slurm": extra.get("slurm", {}),
    }
