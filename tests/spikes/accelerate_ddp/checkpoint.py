"""Accelerate payload transport under a TPEN-owned publication protocol.

THIS MODULE IS WHERE THE PROGRAM'S SEAM IS ACTUALLY TESTED. The seam says the
runtime may own "model/optimizer byte transport" while TPEN retains its
"checkpoint control plane" and its "exact rank-local sampler/walker/RNG resume
policy". Accelerate's ``save_state``/``load_state`` want more than that: they
also capture and restore per-process RNG. So the two are in genuine tension, and
this module's job is to find out whether the seam as written is achievable.

The design under test: Accelerate writes bytes into a STAGING directory and
nothing else. TPEN validates digests, decides when a generation is complete,
publishes exactly once by rename, and owns the rank-local sidecar. Accelerate is
reduced to byte transport.

TWO CONSEQUENCES ESTABLISHED BY GATE, NOT ASSUMED, and both recorded even though
one is unflattering to this candidate:

1. ``save_state`` is a MULTI-WRITER call -- every process writes into the target
   directory. A TPEN publication protocol therefore cannot simply wrap "the
   writer"; it has to bound a directory that N processes are writing into. That
   is why staging exists here and why validation happens after a barrier.
2. ``load_state`` RESTORES Accelerate's own captured RNG. If TPEN's sidecar were
   applied first, Accelerate would silently overwrite it and the rank-local
   resume policy the program reserves to TPEN would be a fiction. The ordering
   below is therefore load-bearing, not stylistic, and A-G3b is what would catch
   it being wrong.

Proposal-local. Production TPEN does not import this, and it does not modify
TPEN's real checkpoint path.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tests.spikes.accelerate_ddp.model_access import SemanticWavefunction
from tests.spikes.accelerate_ddp.runtime import DistributedRuntime


class CheckpointTopologyMismatch(RuntimeError):
    """Raised BEFORE any mutation when a checkpoint's world size differs."""


class CheckpointCorrupt(RuntimeError):
    """Raised when a rank-local sidecar fails its recorded digest."""


@dataclass(frozen=True)
class FileDigest:
    """Closed-file size and SHA-256 digest used by publication validation."""

    relative_path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _digest(path: Path, *, root: Path) -> FileDigest:
    payload = path.read_bytes()
    return FileDigest(
        relative_path=str(path.relative_to(root)),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _file_digests(root: Path) -> list[FileDigest]:
    return [_digest(p, root=root) for p in sorted(root.rglob("*")) if p.is_file()]


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON through a same-directory temp plus rename.

    A kill mid-write can truncate the temp path but never the destination, since
    rename is a single syscall on the filesystems this runs on.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def _jsonable(value: Any) -> Any:
    """Convert tensors and byte payloads into JSON-safe, losslessly reloadable form."""

    if isinstance(value, torch.Tensor):
        return {"__tensor__": True, "dtype": str(value.dtype), "data": value.tolist()}
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": True, "hex": bytes(value).hex()}
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _unjsonable(value: Any) -> Any:
    """Inverse of :func:`_jsonable`."""

    if isinstance(value, dict) and value.get("__tensor__"):
        dtype = getattr(torch, str(value["dtype"]).removeprefix("torch."))
        return torch.tensor(value["data"], dtype=dtype)
    if isinstance(value, dict) and value.get("__bytes__"):
        return bytes.fromhex(value["hex"])
    if isinstance(value, dict):
        return {key: _unjsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unjsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class CheckpointPayloadStore:
    """Proposed typed checkpoint adapter surface, satisfied by Accelerate.

    Deliberately the SAME surface name the native spike proposed, so DG0 compares
    two satisfactions of one boundary rather than two boundaries.
    """

    root: Path
    runtime: DistributedRuntime

    def save(
        self,
        model: SemanticWavefunction,
        optimizer: torch.optim.Optimizer,
        *,
        generation: int,
        sampler_state: dict[str, Any],
        rng_state: dict[str, Any],
        completed_updates: int,
        failure_rank: int | None = None,
        delay_rank: int | None = None,
        delay_seconds: float = 0.0,
    ) -> Path:
        """Stage Accelerate's bytes, validate every sidecar, then publish once."""

        if generation < 1:
            raise ValueError("checkpoint generation must be positive")
        stage = self.root / "staging" / f"gen-{generation:06d}"
        final = self.root / "generations" / f"gen-{generation:06d}"
        if self.runtime.rank == 0:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "staging").mkdir(parents=True, exist_ok=True)
            (self.root / "generations").mkdir(parents=True, exist_ok=True)
            if stage.exists() or final.exists():
                raise FileExistsError(f"checkpoint generation already exists: {generation}")
            stage.mkdir()
        self.runtime.barrier()

        # Accelerate as BYTE TRANSPORT ONLY, into staging. Every process writes
        # here -- this call is not single-writer -- which is exactly why the
        # publication decision below is TPEN's and happens after a barrier.
        self.runtime.accelerator.save_state(str(stage / "accelerate"))

        if self.runtime.rank == failure_rank:
            os._exit(7)
        if self.runtime.rank == delay_rank:
            import time

            time.sleep(delay_seconds)

        # TPEN's own rank-local sidecar. Deliberately SEPARATE from whatever
        # Accelerate wrote: the program reserves the exact rank-local
        # sampler/walker/RNG resume policy to TPEN, so TPEN must hold its own
        # copy rather than trusting a runtime-owned artifact it does not control.
        sidecar = stage / "sidecars" / f"rank-{self.runtime.rank:05d}.json"
        _atomic_write_json(
            sidecar,
            {
                "rank": self.runtime.rank,
                "world_size": self.runtime.world_size,
                "completed_updates": completed_updates,
                "sampler_state": _jsonable(sampler_state),
                "rng_state": _jsonable(rng_state),
            },
        )
        local_digest = _digest(sidecar, root=stage)
        self.runtime.barrier()
        gathered_digests = self.runtime.all_gather_objects(local_digest.as_dict())

        if self.runtime.rank == 0:
            # Re-digest from disk rather than trusting the gathered claim: a rank
            # reporting a digest is not the same evidence as the closed file
            # having it, and publication must rest on the file.
            for rank, payload in enumerate(gathered_digests):
                expected_path = stage / "sidecars" / f"rank-{rank:05d}.json"
                actual = _digest(expected_path, root=stage)
                if payload != actual.as_dict():
                    raise CheckpointCorrupt(
                        f"rank {rank} sidecar digest changed before publication"
                    )
            manifest = {
                "generation": generation,
                "world_size": self.runtime.world_size,
                "completed_updates": completed_updates,
                "publisher_rank": 0,
                # No `module.` prefix: the canonical keys are the raw semantic
                # module's, not the wrapper's.
                "canonical_model_keys": list(model.state_dict().keys()),
                "accelerate_files": [
                    digest.as_dict() for digest in _file_digests(stage / "accelerate")
                ],
                "files": [digest.as_dict() for digest in _file_digests(stage)],
            }
            _atomic_write_json(stage / "manifest.json", manifest)
            (stage / "COMPLETE").write_text("COMPLETE\n")
            # The publication itself: one rename, by one rank, after validation.
            stage.rename(final)
            _atomic_write_json(
                self.root / "latest.json",
                {"generation": generation, "path": str(final.relative_to(self.root))},
            )
        self.runtime.barrier()
        return final

    def load(
        self,
        model: SemanticWavefunction,
        optimizer: torch.optim.Optimizer,
        *,
        generation: int,
    ) -> dict[str, Any]:
        """Restore one generation, with TPEN's rank-local policy applied LAST."""

        final = self.root / "generations" / f"gen-{generation:06d}"
        manifest = json.loads((final / "manifest.json").read_text())

        # Topology is refused BEFORE any mutation. A partially-restored model
        # under a mismatched world size is worse than a clean refusal.
        if int(manifest["world_size"]) != self.runtime.world_size:
            raise CheckpointTopologyMismatch(
                f"checkpoint world_size {manifest['world_size']} != runtime "
                f"world_size {self.runtime.world_size}"
            )

        sidecar_path = final / "sidecars" / f"rank-{self.runtime.rank:05d}.json"
        recorded = next(
            entry
            for entry in manifest["files"]
            if entry["relative_path"] == str(sidecar_path.relative_to(final))
        )
        actual = _digest(sidecar_path, root=final)
        if actual.as_dict() != recorded:
            raise CheckpointCorrupt(
                f"rank {self.runtime.rank} sidecar failed its recorded digest"
            )
        sidecar = json.loads(sidecar_path.read_text())

        # ORDER IS LOAD-BEARING. Accelerate's load_state restores ITS OWN captured
        # per-process RNG, so it must run FIRST; applying TPEN's sidecar before it
        # would let Accelerate silently overwrite the rank-local resume policy the
        # program reserves to TPEN. Establishing that this ordering is REQUIRED --
        # rather than merely tidy -- is a DG0 input, and A-G3b is the arm that
        # would catch the reverse order.
        self.runtime.accelerator.load_state(str(final / "accelerate"))

        return {
            "completed_updates": int(sidecar["completed_updates"]),
            "sampler_state": _unjsonable(sidecar["sampler_state"]),
            "rng_state": _unjsonable(sidecar["rng_state"]),
            "canonical_model_keys": list(manifest["canonical_model_keys"]),
        }

    def selectable_generations(self) -> list[int]:
        """Return generations a reader may select: published AND marked COMPLETE.

        A staged-but-unpublished generation is invisible here by construction --
        it lives under ``staging/`` and never appears in ``generations/`` unless
        the publisher renamed it.
        """

        generations_root = self.root / "generations"
        if not generations_root.exists():
            return []
        selectable = []
        for path in sorted(generations_root.iterdir()):
            if (path / "COMPLETE").exists() and (path / "manifest.json").exists():
                selectable.append(int(path.name.removeprefix("gen-")))
        return selectable


__all__ = [
    "CheckpointCorrupt",
    "CheckpointPayloadStore",
    "CheckpointTopologyMismatch",
    "FileDigest",
]
