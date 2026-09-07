"""Accelerate-backed distributed runtime for the DS-A0 spike.

Deliberately exposes the SAME surface name and the SAME methods as
``tests/spikes/native_ddp/runtime.py``'s ``DistributedRuntime``. That is not
cosmetic: DG0's job is to choose between two satisfactions of ONE proposed
adapter boundary, and two prototypes that expose different surfaces would give it
two different questions instead of one comparison. Where Accelerate makes a
method mean something different, the difference is documented on that method
rather than hidden behind a matching signature.

Proposal-local. Owns no application state, and production TPEN does not import
it. Landing it authorises no production surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedRuntime:
    """Immutable identity and collective facade over one ``Accelerator``."""

    rank: int
    world_size: int
    process_group_timeout_seconds: float
    accelerator: Any
    backend: str = "gloo"

    def __post_init__(self) -> None:
        if self.backend != "gloo":
            raise ValueError("the Accelerate spike is CPU/Gloo-only")
        if type(self.rank) is not int or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be an integer in the runtime world")
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if self.process_group_timeout_seconds <= 0:
            raise ValueError("process_group_timeout_seconds must be positive")

    @classmethod
    def initialize(
        cls,
        *,
        rank: int,
        world_size: int,
        rendezvous_file: Path,
        process_group_timeout_seconds: float,
    ) -> "DistributedRuntime":
        """Bring up one ``Accelerator`` over a TPEN-arranged FileStore rendezvous.

        THE OWNERSHIP DIFFERENCE FROM THE NATIVE SPIKE, which is a DG0 input:
        here ACCELERATE calls ``init_process_group``, not TPEN. TPEN supplies only
        the rendezvous address and the rank identity. Probe 44956566 established
        that this works at world_size 2 on CPU/Gloo without a launcher binary and
        without ``MASTER_ADDR``/``MASTER_PORT``.

        Rank identity is published into the environment because Accelerate reads
        it from there rather than from constructor arguments -- it normally sits
        downstream of a launcher. ``LOCAL_RANK`` equals ``RANK`` because this
        harness places every rank on one node.
        """

        import os

        from accelerate import Accelerator, InitProcessGroupKwargs

        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
        # Explicit, so the result is a property of Accelerate rather than of
        # whatever device the node happens to expose.
        os.environ["ACCELERATE_USE_CPU"] = "1"

        kwargs = InitProcessGroupKwargs(
            backend="gloo",
            init_method=f"file://{rendezvous_file}",
            timeout=timedelta(seconds=process_group_timeout_seconds),
        )
        accelerator = Accelerator(cpu=True, kwargs_handlers=[kwargs])

        # Agreement is CHECKED, never assumed. A runtime that silently renumbered
        # ranks would corrupt every rank-local sampler, walker and RNG decision
        # downstream, and would surface far from its cause.
        if int(accelerator.process_index) != rank or int(accelerator.num_processes) != world_size:
            raise RuntimeError(
                "Accelerate disagreed with the harness about rank identity: "
                f"harness=({rank}, {world_size}) accelerate=("
                f"{accelerator.process_index}, {accelerator.num_processes})"
            )

        return cls(
            rank=rank,
            world_size=world_size,
            process_group_timeout_seconds=process_group_timeout_seconds,
            accelerator=accelerator,
        )

    def barrier(self) -> None:
        """Synchronize every rank through Accelerate's own primitive."""

        self.accelerator.wait_for_everyone()

    def all_gather_objects(self, value: Any) -> list[Any]:
        """Gather one Python value from every rank in rank order.

        Uses ``accelerate.utils.gather_object`` rather than ``dist`` directly, so
        this measures what adopting Accelerate would actually buy. Rank ordering
        was verified in probe 44956566 and is asserted here rather than trusted:
        the statistics reducer and the checkpoint coordinator both depend on it,
        and an unstable order would change scientific results without erroring.
        """

        from accelerate.utils import gather_object

        gathered = list(gather_object([value]))
        if len(gathered) != self.world_size:
            raise RuntimeError(
                f"gather_object returned {len(gathered)} entries for world_size {self.world_size}"
            )
        return gathered

    def broadcast_object(self, value: Any | None, *, source: int = 0) -> Any:
        """Broadcast one Python value from ``source`` and return it everywhere."""

        from accelerate.utils import broadcast_object_list

        payload = [value if self.rank == source else None]
        broadcast_object_list(payload, from_process=source)
        return payload[0]

    def collective_scalar_sum(self, value: float) -> float:
        """Return a scalar sum used only for non-gradient diagnostics.

        Deliberately NOT ``accelerator.reduce``: that would route a diagnostic
        through the same machinery the gradient path uses, and A-G6 counts
        gradient reductions. Keeping diagnostics on a plainly separate call is
        what lets the reduction counter mean what it says.
        """

        tensor = torch.tensor(float(value), dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def close(self) -> None:
        """Release the process group when it is still live.

        ``Accelerator`` brought the group up, so the teardown goes through
        Accelerate's own ``end_training`` first; the ``dist`` call afterwards is
        the backstop for the case where Accelerate leaves the group standing.
        Both are guarded, because a poisoned communicator must never be reused
        and a teardown that raises would mask the failure that preceded it.
        """

        try:
            self.accelerator.end_training()
        except Exception:  # noqa: BLE001 - teardown must not mask a real failure
            pass
        if dist.is_initialized():
            dist.destroy_process_group()


__all__ = ["DistributedRuntime"]
