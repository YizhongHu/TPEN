"""Child-process worker for the run-identity agreement harness.

Invoked as ``python -m tests.helpers.run_id_agreement_worker <args>``, always as
a genuinely separate OS process launched by
:mod:`tests.helpers.run_id_agreement_harness` -- never in-process and never via
``torch.multiprocessing.spawn``, so the pytest process itself is never a
distributed worker.

DIVERGENT ID FACTORIES, load-bearing. Before the process group exists, this
worker replaces both inputs to a derived run id -- the run clock and the
``uuid4`` draw -- with values derived from its own global rank. Without that,
two ranks that happened to start inside the same wall-clock second and drew
different uuids would produce ids differing only in the suffix, and a test
asserting agreement could pass on a broken build by coincidence. With it, an
unbroadcast id CANNOT match across ranks, and the agreed id is checkable against
rank 0's factory output specifically -- so the test also witnesses WHICH rank
derived it, not merely that the ranks concur.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch.distributed as dist
from omegaconf import OmegaConf

import tpen.artifacts as artifacts
from tpen.distributed import ExecutionTopology

# Any rank's derived id is a pure function of its rank through these two.
_CLOCK_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_UUID_BASE = 0xAA0000

RUN_NAME = "rank-agreement"
EXPERIMENT_NAME = "run-identity"
SECTOR = "agreement"

#: Sentinel for "this rank's config leaves ``run.run_id`` null". A bare absent
#: flag cannot express it, because the harness must also be able to give ONE
#: rank a null id while its peers carry explicit ones.
NULL_RUN_ID = "@null"


def rank_timestamp(rank: int) -> str:
    """Return the timestamp component this rank's clock would produce."""

    return (_CLOCK_EPOCH + timedelta(days=rank)).strftime("%Y-%m-%d_%H%M%S")


def rank_uuid_suffix(rank: int) -> str:
    """Return the uuid suffix component this rank's draw would produce."""

    return _rank_uuid(rank).hex[:6]


def expected_derived_run_id(rank: int) -> str:
    """Return the exact run id this rank would derive on its own.

    The harness compares the agreed id against ``expected_derived_run_id(0)``,
    which is what makes the test witness the coordinator as the derivation site
    rather than merely witnessing that the ranks agree on something.
    """

    return f"{rank_timestamp(rank)}_{RUN_NAME}_{rank_uuid_suffix(rank)}"


def _rank_uuid(rank: int) -> uuid.UUID:
    # Rank goes in the LEADING hex digits: ``generate_run_id`` keeps only
    # ``hex[:6]``, so a rank encoded in the trailing digits would be truncated
    # away and every rank would draw an identical suffix -- a fixture that
    # manufactures the agreement it is supposed to test.
    return uuid.UUID(hex=f"{_UUID_BASE + rank:06x}" + "0" * 26)


def _install_rank_derived_factories(rank: int) -> None:
    """Make this process's derived run id a pure function of ``rank``."""

    fixed_now = _CLOCK_EPOCH + timedelta(days=rank)
    # Patched on the class, not on one instance: `prepare_run_context` builds
    # its own `RunClock` from the config and never accepts one from a caller.
    artifacts.RunClock.now = lambda self: fixed_now  # type: ignore[method-assign]
    artifacts.uuid4 = lambda: _rank_uuid(rank)  # type: ignore[assignment]


def _build_cfg(run_root: Path, configured_run_id: str | None) -> object:
    return OmegaConf.create(
        {
            "experiment": {
                "name": EXPERIMENT_NAME,
                "sector": SECTOR,
                "run_name": RUN_NAME,
            },
            "run": {
                "root": str(run_root),
                "run_id": configured_run_id,
                "dir": None,
                "layout": "nested",
            },
            "runtime": {"device": "cpu", "dtype": "float64"},
            "callbacks": [],
            "loggers": [],
        }
    )


def _topology(rank: int, world_size: int) -> ExecutionTopology:
    # `ExecutionTopology` validates rank < size, so a DECLARED size below this
    # rank's real rank cannot be expressed -- the mismatch test therefore
    # over-declares (say 3 for a real world of 2) rather than under-declaring.
    return ExecutionTopology(
        global_rank=rank,
        global_size=world_size,
        local_rank=rank,
        local_size=world_size,
        node_rank=0,
        node_size=1,
        host=socket.gethostname(),
        pid=os.getpid(),
        device="cpu",
    )


def main(argv: list[str] | None = None) -> int:
    """Run one rank of the agreement harness and write its receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rendezvous-file", required=True)
    parser.add_argument("--receipt-path", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--pg-timeout", type=float, required=True)
    parser.add_argument("--mode", choices=("resolve", "context"), required=True)
    parser.add_argument(
        "--declared-world-size",
        type=int,
        default=None,
        help=(
            "global_size to put in the supplied ExecutionTopology. Defaults to the "
            "real world size; setting it differently makes this rank a launcher "
            "that contradicts itself, which run identity must refuse."
        ),
    )
    parser.add_argument(
        "--configured-run-id",
        required=True,
        help=f"This rank's configured run.run_id, or {NULL_RUN_ID!r} for null.",
    )
    args = parser.parse_args(argv)

    rank = int(args.rank)
    configured = None if args.configured_run_id == NULL_RUN_ID else args.configured_run_id
    receipt: dict[str, object] = {"rank": rank, "mode": args.mode}

    _install_rank_derived_factories(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{args.rendezvous_file}",
        rank=rank,
        world_size=int(args.world_size),
        timeout=timedelta(seconds=float(args.pg_timeout)),
    )
    try:
        run_root = Path(args.run_root)
        declared = int(args.world_size if args.declared_world_size is None else args.declared_world_size)
        topology = _topology(rank, declared)
        try:
            if args.mode == "resolve":
                run_id = artifacts.resolve_run_id(
                    configured,
                    RUN_NAME,
                    clock=artifacts.RunClock(timezone="UTC", tzinfo=UTC),
                    topology=topology,
                )
                receipt["run_id"] = run_id
            else:
                # Imported here so `resolve` mode does not depend on the whole
                # run-setup import graph (Hydra, loggers, callbacks) at all.
                from tpen.run import prepare_run_context

                context = prepare_run_context(
                    _build_cfg(run_root, configured),
                    config_path="run_id_agreement_worker",
                    command="pytest",
                    topology=topology,
                )
                receipt["run_id"] = context.metadata.run_id
                receipt["run_dir"] = str(context.run_dir)
        except Exception as exc:  # noqa: BLE001 - every rank's failure is evidence
            # Recorded rather than raised: a rank that dies without a receipt is
            # indistinguishable from a rank that hung, and the negative tests
            # need to assert that EVERY rank refused, not just that the group
            # failed somehow.
            receipt["error_type"] = type(exc).__name__
            receipt["error"] = str(exc)
        Path(args.receipt_path).write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
