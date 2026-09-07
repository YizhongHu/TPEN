"""Serial and no-process-group behaviour of run-identity resolution.

The multi-rank half of the contract lives in
``tests/unit/test_run_id_rank_agreement.py``, which needs real subprocesses and
a Gloo build. Everything here runs in one torch-free-capable process: the serial
path that local and smoke runs take, and the refusal that closes the multi-rank
launch which never initialized a process group at all.
"""

from __future__ import annotations

import json
import re
import sys

import pytest
from omegaconf import OmegaConf

import tpen.artifacts as artifacts
from tpen.artifacts import (
    MULTIPLICITY_ANNOUNCEMENT_KEYS,
    RunIdentityError,
    collect_launcher_multiplicity_announcements,
    generate_run_id,
    resolve_run_id,
)
from tpen.distributed import ExecutionTopology
from tpen.run import prepare_run_context

#: The historical id shape, pinned so the fix cannot quietly change it: the
#: random suffix is KEPT, because the defect was per-rank divergence rather than
#: suffix randomness.
_RUN_ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_[A-Za-z0-9_.-]+_[0-9a-f]{6}$")


def _multi_rank_topology(global_size: int = 2) -> ExecutionTopology:
    """Return a launcher-supplied topology that declares several ranks."""

    return ExecutionTopology(
        global_rank=0,
        global_size=global_size,
        local_rank=0,
        local_size=global_size,
        node_rank=0,
        node_size=1,
        host="node-a",
        pid=42,
        device="cpu",
    )


def _cfg(tmp_path, run_id):
    return OmegaConf.create(
        {
            "experiment": {"name": "identity", "sector": "unit", "run_name": "serial-run"},
            "run": {"root": str(tmp_path), "run_id": run_id, "dir": None},
            "runtime": {"device": "cpu", "dtype": "float64"},
            "callbacks": [],
            "loggers": [],
        }
    )


def test_serial_null_id_keeps_the_historical_derived_shape() -> None:
    """C1: one process, null id -- unchanged behaviour, suffix still random."""

    first = resolve_run_id(None, "serial run")
    second = resolve_run_id(None, "serial run")

    assert _RUN_ID_PATTERN.match(first), first
    # Two resolutions in one process must still differ. Collision resistance
    # across runs is the property the random suffix buys, and the fix keeps it.
    assert first != second


def test_serial_explicit_id_is_returned_unchanged(monkeypatch) -> None:
    """C3: an explicit id is honored as text and nothing is derived."""

    def forbidden(*args, **kwargs):  # pragma: no cover - asserted not to run
        raise AssertionError("an explicit run id must not be regenerated")

    monkeypatch.setattr(artifacts, "generate_run_id", forbidden)

    assert resolve_run_id("2026-01-01_000000_fixed_abcdef", "serial run") == (
        "2026-01-01_000000_fixed_abcdef"
    )


def test_serial_resolution_survives_an_absent_torch_distributed(monkeypatch) -> None:
    """A torch-free environment is a serial run, not an error.

    `tpen.artifacts` is importable without torch and run setup must stay usable,
    so the agreement probe treats an unimportable ``torch.distributed`` as "no
    channel" rather than propagating.
    """

    # A ``None`` entry in sys.modules makes the import statement itself raise
    # ImportError, which is exactly what a torch-free environment does.
    monkeypatch.setitem(sys.modules, "torch.distributed", None)

    assert _RUN_ID_PATTERN.match(resolve_run_id(None, "serial run"))


def test_single_rank_topology_still_derives_locally() -> None:
    """A one-rank launcher topology is the serial path, not a refusal."""

    topology = ExecutionTopology.single_process(device="cpu")

    assert _RUN_ID_PATTERN.match(resolve_run_id(None, "serial run", topology=topology))


def test_multi_rank_topology_without_a_process_group_refuses_a_derived_id() -> None:
    """C5: the reachable scatter path -- several ranks, nothing to agree over.

    ``ExecutionTopology`` is launcher-supplied and torch-free, so a launcher can
    declare three ranks with no process group initialized. There is no channel
    to broadcast over, and deriving locally is precisely the defect, so the only
    safe answer is a loud refusal.
    """

    with pytest.raises(RunIdentityError) as excinfo:
        resolve_run_id(None, "serial run", topology=_multi_rank_topology(3))

    message = str(excinfo.value)
    assert "3 ranks" in message
    assert "no distributed process group is initialized" in message


def test_multi_rank_topology_accepts_an_explicit_id_without_a_process_group() -> None:
    """The refusal must not close a legitimate configuration.

    An explicit id is ACCEPTED without a channel, on the assumption -- not the
    fact -- that every rank resolved the same config to the same value. This is
    the over-restriction direction of the C6 mutation pair: widening the refusal
    to cover explicit ids turns this red. What the acceptance does not verify is
    pinned separately in
    ``test_channel_less_mixed_launch_is_asymmetric_and_knowingly_undetectable``.
    """

    agreed = resolve_run_id("explicit-id", "serial run", topology=_multi_rank_topology(4))

    assert agreed == "explicit-id"


def test_channel_less_mixed_launch_is_asymmetric_and_knowingly_undetectable() -> None:
    """Pin the one shape this slice cannot detect, instead of claiming it cannot occur.

    A multi-rank launch with NO process group, where some ranks configure an
    explicit id and others leave it null: the null ranks refuse, the explicit
    ranks proceed. That asymmetry is real and is not fixable here -- with no
    channel there is nothing to compare against, so no rank can learn that its
    peers decided differently. It is pinned rather than left implicit because
    the code used to assert the opposite ("an explicit id ... is already common
    to every rank"), and a false claim in a comment outlives the reviewer who
    would have caught it.

    Both halves ARE detected once a channel exists; see
    ``test_run_id_rank_agreement.py::test_mixed_null_and_explicit_ids_fail_on_every_rank``.
    """

    topology = _multi_rank_topology(2)

    # The null-id rank refuses...
    with pytest.raises(RunIdentityError):
        resolve_run_id(None, "serial run", topology=topology)

    # ...while its explicit-id peer proceeds, unverified. Asserted so that a
    # future change to either half has to confront the asymmetry deliberately.
    assert resolve_run_id("explicit-id", "serial run", topology=topology) == "explicit-id"


def test_prepare_run_context_still_fills_a_null_id_for_a_serial_run(tmp_path) -> None:
    """C1 through the real construction path, not just the helper."""

    context = prepare_run_context(_cfg(tmp_path, None), config_path="test.yaml", command="pytest")

    assert _RUN_ID_PATTERN.match(context.metadata.run_id)
    assert str(OmegaConf.select(context.cfg, "run.run_id")) == context.metadata.run_id
    assert context.run_dir.is_dir()


def test_prepare_run_context_refuses_multi_rank_null_id_before_creating_anything(tmp_path) -> None:
    """C5 through the real construction path: refuse while the tree is absent.

    The ordering is the load-bearing part. A refusal raised after ``make_dirs``
    would leave behind exactly the per-rank directories the agreement exists to
    prevent, so the assertion is on the empty filesystem, not only on the raise.
    """

    with pytest.raises(RunIdentityError):
        prepare_run_context(
            _cfg(tmp_path, None),
            config_path="test.yaml",
            command="pytest",
            topology=_multi_rank_topology(2),
        )

    assert list(tmp_path.iterdir()) == []


def test_generate_run_id_remains_process_local() -> None:
    """The raw generator is deliberately unchanged and still non-agreeing.

    Pinned because `tests/unit/test_hi_schema.py` reasons about this property,
    and because the fix must not smuggle a collective into a function whose
    contract is a fresh draw per call.

    Note for whoever reads that HI cross-reference next: its rationale still
    describes `prepare_run_context` calling `generate_run_id` directly, which
    stopped being true here. Retiring or rewording that rule is HI L1's call,
    not this slice's -- `tpen/hi_schema.py` is outside this write surface.
    """

    assert generate_run_id("hi") != generate_run_id("hi")


# ---------------------------------------------------------------------------
# Launcher multiplicity DISCLOSURE. Recorded, never read.
#
# The bound these tests exist to enforce is bound 1: no control flow may consult
# the recorded value. That is not checkable by reading the code once -- it is a
# property a future edit can quietly break -- so it is pinned by behaviour: the
# resolved run id must be INDIFFERENT to every announcement.
# ---------------------------------------------------------------------------


def test_multiplicity_disclosure_records_presence_absence_and_what_was_consulted(
    monkeypatch,
) -> None:
    """Absence is data: it is what tells the launch shapes apart.

    Measured on Cannon 2026-09-06 -- ``SLURM_NTASKS`` is allocation-scoped and
    reads 4 for a genuinely single process inside a 4-task allocation, while
    ``SLURM_STEP_NUM_TASKS`` appears only inside an ``srun`` step. A record of
    the counts alone would be actively misleading, so the record names what was
    consulted, what answered, and what did not.
    """

    for key in MULTIPLICITY_ANNOUNCEMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_PROCID", "0")

    record = collect_launcher_multiplicity_announcements()

    assert record["present"] == {"SLURM_NTASKS": "4", "SLURM_PROCID": "0"}
    assert "SLURM_STEP_NUM_TASKS" in record["absent"]
    assert "OMPI_COMM_WORLD_SIZE" in record["absent"]
    assert record["consulted"] == list(MULTIPLICITY_ANNOUNCEMENT_KEYS)
    # Every consulted name is accounted for exactly once, so the record cannot
    # silently omit a name it looked at.
    assert sorted([*record["absent"], *record["present"]]) == sorted(record["consulted"])


def test_multiplicity_disclosure_never_decides_anything(monkeypatch) -> None:
    """BOUND 1, enforced by behaviour rather than by inspection.

    An environment loudly announcing four ranks must not change the resolved run
    id by one character. If a future edit routes this observation into control
    flow -- a refusal, a warning that alters a path, a different derivation --
    this test goes red, which is the whole point of writing it as an
    indifference assertion instead of a comment saying "do not read this".
    """

    for key in MULTIPLICITY_ANNOUNCEMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    quiet = resolve_run_id("explicit-id", "serial run")
    quiet_derived_shape = _RUN_ID_PATTERN.match(resolve_run_id(None, "serial run")) is not None

    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_NPROCS", "4")
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "4")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setenv("WORLD_SIZE", "4")

    assert resolve_run_id("explicit-id", "serial run") == quiet
    assert (_RUN_ID_PATTERN.match(resolve_run_id(None, "serial run")) is not None) == (
        quiet_derived_shape
    )
    # And the multi-rank refusal still keys on the DECLARED topology only: a
    # four-rank announcement with a one-rank topology is not a refusal.
    assert _RUN_ID_PATTERN.match(
        resolve_run_id(None, "serial run", topology=ExecutionTopology.single_process(device="cpu"))
    )


def test_run_start_artifact_carries_the_disclosure(tmp_path, monkeypatch) -> None:
    """The record reaches the artifact a human actually reads."""

    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.delenv("SLURM_STEP_NUM_TASKS", raising=False)

    context = prepare_run_context(_cfg(tmp_path, None), config_path="test.yaml", command="pytest")

    run_start = json.loads((context.run_dir / "run_start.json").read_text())
    disclosure = run_start["launcher_multiplicity"]
    assert disclosure["present"]["SLURM_NTASKS"] == "4"
    assert "SLURM_STEP_NUM_TASKS" in disclosure["absent"]
