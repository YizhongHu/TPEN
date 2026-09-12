"""Round-three review probes for the L4a inference-packet boundary.

Each probe prints one verdict line of the form ``[ADMITTED]``,
``[REFUSED:<member value>]`` or ``[OTHER:<ExceptionType>: <message>]``.
The serializability probe (P5) prints one matrix line per cell instead.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import pickle
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_TARGET = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "atomistic"
    / "he-importance"
    / "inference_packet.py"
)

# Probe-only module key: never collide with the contract suite's alias.
_spec = importlib.util.spec_from_file_location("l4a_r3_probe_target", _TARGET)
assert _spec is not None and _spec.loader is not None
ip = importlib.util.module_from_spec(_spec)
sys.modules["l4a_r3_probe_target"] = ip
_spec.loader.exec_module(ip)

_HASH = "a" * 64


def verdict(label: str, thunk) -> Any:
    """Run ``thunk`` and print exactly one verdict line for it."""

    try:
        value = thunk()
    except ip.InferencePacketError as error:
        print(f"{label} [REFUSED:{error.refusal.value}]")
        return None
    except BaseException as error:  # noqa: BLE001 - the probe measures crashes
        print(f"{label} [OTHER:{type(error).__name__}: {error}]")
        return None
    print(f"{label} [ADMITTED]")
    return value


def tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="l4a-r3-")).resolve()


def checkpoint(root: Path, topology: object = None) -> Any:
    parent = (root / "O1" / _HASH).resolve()
    return ip.CheckpointReference(
        parent_cell_path=parent,
        checkpoint_path=parent / "checkpoints" / "update-00050000",
        source_content_hash=_HASH,
        topology_provenance=(
            {"world_size": 1, "launcher": "probe"} if topology is None else topology
        ),
    )


def seeds(index: int) -> Any:
    return ip.ChainSeedProvenance(
        training_seed=100 + index,
        calibration_seed=200 + index,
        inference_seed=300 + index,
        chain_seed=400 + index,
    )


_CREATED = datetime(2026, 9, 9, 4, 30, tzinfo=UTC)


def chain(index: int) -> Any:
    status = ip.ChainStatus(
        state=ip.ChainState.IDLE, created_at=_CREATED, last_activity_at=_CREATED
    )
    return ip.IndependentChain(
        chain_id=f"chain-{index}",
        seeds=seeds(index),
        interval=ip.SamplingInterval(10_000, 10, 32_768),
        status=status,
    )


def source(reference: Any, inputs: object | None = None) -> Any:
    return SimpleNamespace(
        checkpoint_path=reference.checkpoint_path,
        source_content_hash=reference.source_content_hash,
        independent_sampler_inputs=(
            {"sampler": {"walkers": 4_096}} if inputs is None else inputs
        ),
    )


# --------------------------------------------------------------------------
# P1 - a mutable tzinfo re-reads differently after construction
# --------------------------------------------------------------------------


class RecordingTZ(tzinfo):
    """Fixed -10h zone that records every ``utcoffset`` interrogation."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def utcoffset(self, dt):  # noqa: D102 - tzinfo protocol
        self.calls.append(dt)
        return timedelta(hours=-10)

    def dst(self, dt):  # noqa: D102 - tzinfo protocol
        return timedelta(0)

    def tzname(self, dt):  # noqa: D102 - tzinfo protocol
        return "rec"


class FlipTZ(tzinfo):
    """-10h for the first ``flip_after`` reads, then ``late`` for every later one."""

    def __init__(self, flip_after: int, late: timedelta | None) -> None:
        self.flip_after = flip_after
        self.late = late
        self.calls = 0

    def utcoffset(self, dt):  # noqa: D102 - tzinfo protocol
        self.calls += 1
        if self.calls <= self.flip_after:
            return timedelta(hours=-10)
        return self.late

    def dst(self, dt):  # noqa: D102 - tzinfo protocol
        return timedelta(0)

    def tzname(self, dt):  # noqa: D102 - tzinfo protocol
        return "flip"


def completed_status(finished_tz: tzinfo) -> Any:
    return ip.ChainStatus(
        state=ip.ChainState.COMPLETED,
        created_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        started_at=datetime(2026, 1, 2, 0, 0, tzinfo=UTC),
        last_activity_at=datetime(2026, 1, 2, 11, 0, tzinfo=UTC),
        finished_at=datetime(2026, 1, 2, 0, 30, tzinfo=finished_tz),
        terminal_reason=None,
    )


def probe_p1() -> None:
    recorder = RecordingTZ()
    calibrated = verdict("P1-calibrate", lambda: completed_status(recorder))
    count = len(recorder.calls)
    print(f"P1-calibrate utcoffset-call-count N={count} (admitted={calibrated is not None})")

    flip = FlipTZ(count, timedelta(hours=+10))
    status = verdict("P1-flip", lambda: completed_status(flip))
    if status is not None:
        finished = status.finished_at.astimezone(UTC)
        started = status.started_at.astimezone(UTC)
        print(f"P1-flip stored finished_at(UTC)={finished.isoformat()}")
        print(f"P1-flip stored started_at(UTC)={started.isoformat()}")
        print(f"P1-flip finish_precedes_start_after_construction={finished < started}")

    flip_none = FlipTZ(count, None)
    status_none = verdict("P1-flip-none", lambda: completed_status(flip_none))
    if status_none is not None:
        try:
            read = status_none.finished_at.astimezone(UTC)
        except BaseException as error:  # noqa: BLE001
            print(f"P1-flip-none finished_at.astimezone(UTC) raised {type(error).__name__}: {error}")
        else:
            print(f"P1-flip-none finished_at.astimezone(UTC)={read.isoformat()}")
            print(f"P1-flip-none reinterpreted_in_host_local_time={read.isoformat()}")


# --------------------------------------------------------------------------
# P2 - extreme aware timestamps overflow the instant policy
# --------------------------------------------------------------------------


def probe_p2() -> None:
    east = timezone(timedelta(hours=1))
    west = timezone(timedelta(hours=-1))
    floor = datetime.min.replace(tzinfo=east)
    verdict(
        "P2-idle-min",
        lambda: ip.ChainStatus(
            state=ip.ChainState.IDLE, created_at=floor, last_activity_at=floor
        ),
    )
    ceiling = datetime.max.replace(tzinfo=west)
    verdict(
        "P2-terminal-max",
        lambda: ip.ChainStatus(
            state=ip.ChainState.COMPLETED,
            created_at=ceiling,
            started_at=ceiling,
            last_activity_at=ceiling,
            finished_at=ceiling,
            terminal_reason=None,
        ),
    )


# --------------------------------------------------------------------------
# P3 - activity-before-finish reported under a finish-precedes-start member
# --------------------------------------------------------------------------


def probe_p3() -> None:
    verdict(
        "P3-member",
        lambda: ip.ChainStatus(
            state=ip.ChainState.COMPLETED,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            last_activity_at=datetime(2026, 1, 5, tzinfo=UTC),
            started_at=datetime(2026, 1, 2, tzinfo=UTC),
            finished_at=datetime(2026, 1, 10, tzinfo=UTC),
            terminal_reason=None,
        ),
    )


# --------------------------------------------------------------------------
# P4 - a running chain with activity four days before its start
# --------------------------------------------------------------------------


def probe_p4() -> None:
    verdict(
        "P4-running-gap",
        lambda: ip.ChainStatus(
            state=ip.ChainState.RUNNING,
            created_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            last_activity_at=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            started_at=datetime(2026, 1, 5, 0, 0, tzinfo=UTC),
        ),
    )


# --------------------------------------------------------------------------
# P5 - serializability matrix over the packet's object graph
# --------------------------------------------------------------------------


def build_distributed_packet() -> Any:
    root = tmpdir()
    reference = checkpoint(root)
    artifacts = {
        rank: reference.checkpoint_path / f"rank-{rank}.pt" for rank in range(2)
    }
    distributed = ip.DistributedCheckpointProvenance(2, artifacts)
    return ip.launch_inference_packet(
        source(reference),
        reference,
        (chain(0), chain(1)),
        distributed_checkpoint=distributed,
    )


def cell(operation: str, thunk) -> str:
    try:
        thunk()
    except BaseException as error:  # noqa: BLE001
        return f"{operation}=FAIL[{type(error).__name__}: {error}]"
    return f"{operation}=OK"


def probe_p5() -> None:
    packet = build_distributed_packet()
    targets = [
        ("packet", packet),
        ("checkpoint", packet.checkpoint),
        ("distributed_checkpoint", packet.distributed_checkpoint),
        ("chain", packet.chains[0]),
        ("chain.status", packet.chains[0].status),
    ]
    for name, obj in targets:
        cells = [
            cell("pickle.dumps", lambda o=obj: pickle.dumps(o)),
            cell("copy.deepcopy", lambda o=obj: copy.deepcopy(o)),
            cell("copy.copy", lambda o=obj: copy.copy(o)),
            cell("hash", lambda o=obj: hash(o)),
        ]
        if dataclasses.is_dataclass(obj):
            cells.append(cell("dataclasses.asdict", lambda o=obj: dataclasses.asdict(o)))
            cells.append(cell("dataclasses.replace", lambda o=obj: dataclasses.replace(o)))
        try:
            blob = pickle.dumps(obj)
        except BaseException:  # noqa: BLE001
            cells.append("pickle.loads=SKIPPED[dumps failed]")
        else:
            cells.append(cell("pickle.loads", lambda b=blob: pickle.loads(b)))
        for entry in cells:
            print(f"P5-matrix {name} {entry}")


# --------------------------------------------------------------------------
# P6 - a functional getattr-backed structural source
# --------------------------------------------------------------------------


class GetattrSource:
    """A structural source whose only interface is ``__getattr__``."""

    def __init__(self, fields: dict[str, Any]) -> None:
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str) -> Any:
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return fields[name]
        raise AttributeError(name)


def probe_p6() -> None:
    root = tmpdir()
    reference = checkpoint(root)
    obj = GetattrSource(
        {
            "checkpoint_path": reference.checkpoint_path,
            "source_content_hash": reference.source_content_hash,
            "independent_sampler_inputs": {"sampler": {"walkers": 4_096}},
        }
    )
    print(f"P6-precheck getattr(obj, 'checkpoint_path')={getattr(obj, 'checkpoint_path')}")
    verdict(
        "P6-getattr-source",
        lambda: ip.launch_inference_packet(obj, reference, (chain(0),)),
    )


# --------------------------------------------------------------------------
# P7 - unattributed crash channels
# --------------------------------------------------------------------------


class ThreeTupleItemsMapping(Mapping):
    """A Mapping whose ``items()`` yields three-tuples instead of pairs."""

    def __getitem__(self, key):
        return {"walkers": 4_096}[key]

    def __iter__(self):
        return iter(("walkers",))

    def __len__(self) -> int:
        return 1

    def items(self):
        return [("walkers", 4_096, 7)]


def probe_p7() -> None:
    root = tmpdir()
    reference = checkpoint(root)

    bad_source = SimpleNamespace(
        checkpoint_path=5,
        source_content_hash=reference.source_content_hash,
        independent_sampler_inputs={"sampler": {"walkers": 4_096}},
    )
    verdict(
        "P7a-source-path-int",
        lambda: ip.launch_inference_packet(bad_source, reference, (chain(0),)),
    )
    verdict(
        "P7b-rank-artifact-int",
        lambda: ip.DistributedCheckpointProvenance(1, {0: 5}),
    )
    verdict(
        "P7c-chains-int",
        lambda: ip.InferencePacket(reference, {"walkers": 4_096}, 5, None),
    )
    verdict(
        "P7d-three-tuple-items",
        lambda: ip.InferencePacket(
            reference, ThreeTupleItemsMapping(), (chain(0),), None
        ),
    )


# --------------------------------------------------------------------------
# P8 - topology provenance contradicting the distributed world size
# --------------------------------------------------------------------------


def probe_p8() -> None:
    root = tmpdir()
    reference = checkpoint(root, topology={"world_size": 1})
    artifacts = {
        rank: reference.checkpoint_path / f"rank-{rank}.pt" for rank in range(2)
    }
    distributed = ip.DistributedCheckpointProvenance(2, artifacts)
    verdict(
        "P8-world-size-contradiction",
        lambda: ip.launch_inference_packet(
            source(reference),
            reference,
            (chain(0), chain(1)),
            distributed_checkpoint=distributed,
        ),
    )


def main() -> None:
    print(f"probe run at {datetime.now(UTC).isoformat()}")
    print(f"target module {_TARGET}")
    probe_p1()
    probe_p2()
    probe_p3()
    probe_p4()
    probe_p5()
    probe_p6()
    probe_p7()
    probe_p8()


if __name__ == "__main__":
    main()
