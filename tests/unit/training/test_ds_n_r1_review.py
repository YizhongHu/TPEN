"""Independent contract-first round-one review probes for PR #470.

All distributed tests are capability-gated and are intended for Cannon.  This
module is an evidence instrument only; it does not authorize production use.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from tests.helpers.ddp_capability import missing_capability_reason, probe_gloo_capability
from tests.helpers.ddp_fault_injection import FaultKind, FaultPhase, FaultPlan
from tests.helpers.ddp_subprocess_harness import HarnessBounds, run_gloo_subprocess_group
from tests.spikes.native_ddp import checkpoint as checkpoint_module
from tests.spikes.native_ddp.checkpoint import CheckpointPayloadStore, CheckpointTopologyMismatch
from tests.spikes.native_ddp.model_access import SemanticWavefunction
from tests.unit.training.test_ds_n_native_ddp_spike import _decode, _run_native, _states


_CAPABILITY = probe_gloo_capability()
_FAULT_BOUNDS = HarnessBounds(process_group_timeout=2.0, watchdog_timeout=12.0)


def _run_review_native(tmp_path: Path, *, extra_args: tuple[str, ...] = ()):
    """Run the reviewed worker through the independent call-counting wrapper."""

    return run_gloo_subprocess_group(
        2,
        None,
        HarnessBounds(process_group_timeout=6.0, watchdog_timeout=20.0),
        tmp_path,
        worker_module="tests.spikes.native_ddp.review_probe",
        worker_extra_args=extra_args,
    )


@pytest.fixture(autouse=True)
def _require_gloo_subprocess_capability() -> None:
    """Make every Cannon-dependent probe's capability decision explicit."""

    if not _CAPABILITY.gloo_available:
        pytest.skip(missing_capability_reason(_CAPABILITY, "gloo_available"))
    if not _CAPABILITY.subprocess_spawn_available:
        pytest.skip(missing_capability_reason(_CAPABILITY, "subprocess_spawn_available"))


def _assert_no_rank_success(states: list[dict]) -> None:
    assert states
    assert all(state.get("status") != "success" for state in states)


def test_r1_closure_state_is_globally_equal_on_nonidentical_shards(tmp_path: Path) -> None:
    """A closure cannot hide rank-divergent parameter or optimizer state."""

    result = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "scientific", "--fixture", "regular", "--optimizer", "closure",
            "--closure-inner-iterates", "3",
        ),
    )
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert result.publication_observed is True
    assert result.exit_codes == (0, 0)
    states = _states(result)
    assert states[0]["parameters_after"] == states[1]["parameters_after"]
    assert states[0]["optimizer_state_after"] == states[1]["optimizer_state_after"]
    assert states[0]["ddp_gradient_reductions_per_update"] == states[1]["ddp_gradient_reductions_per_update"]

    features = torch.tensor(
        [[0.2, 0.1], [-0.5, 0.3], [0.7, -0.2], [0.4, 0.9],
         [-0.3, 0.8], [0.4, -0.1], [1.1, 0.2]], dtype=torch.float64
    )
    energy = torch.tensor([1.0, 2.0, 0.5, float("nan"), 3.0, -1.0, 2.0], dtype=torch.float64)
    finite = torch.isfinite(energy)
    reference_model = SemanticWavefunction()
    reference_optimizer = torch.optim.LBFGS(
        reference_model.parameters(), lr=0.25, max_iter=3, history_size=5,
        tolerance_grad=0.0, tolerance_change=0.0,
    )
    def closure() -> torch.Tensor:
        reference_optimizer.zero_grad()
        logabs = reference_model(features)
        mean = energy[finite].mean()
        loss = 2.0 * ((energy[finite] - mean) * logabs[finite]).sum() / finite.sum()
        loss.backward()
        return loss
    reference_optimizer.step(closure)
    for state in states:
        assert state["parameters_after"]["weight"] == pytest.approx(reference_model.weight.detach().tolist())
        assert state["parameters_after"]["bias"] == pytest.approx(reference_model.bias.detach().tolist())
        _assert_nested_equal(_decode(state["optimizer_state_after"]), reference_optimizer.state_dict())


def test_r2_m2_observes_each_rank_energy_tensor_and_finite_mask(tmp_path: Path) -> None:
    """Global repeated totals must not replace the required shard observation."""

    result = _run_native(
        tmp_path,
        world_size=3,
        extra_args=("--experiment", "scientific", "--fixture", "m2", "--optimizer", "sgd"),
    )
    states = _states(result)
    raw = [json.loads((Path(result.invocation_dir) / f"state_{rank}.json").read_text())["energy"] for rank in range(3)]
    assert [len(item["__tensor__"]) for item in raw] == [5, 3, 7]
    expected = [[1.0, 2.0, -1.0], [], [3.0, -2.0, 1.0, 0.5, 2.5]]
    for item, expected_finite in zip(raw, expected, strict=True):
        finite = [value for value in item["__tensor__"] if not (isinstance(value, float) and math.isnan(value))]
        assert finite == expected_finite
    assert [sum(not (isinstance(value, float) and math.isnan(value)) for value in item["__tensor__"]) for item in raw] == [3, 0, 5]
    assert all(state["status"] == "success" for state in states)


def test_r3_global_zero_has_no_ddp_forward_gradient_or_update(tmp_path: Path) -> None:
    """Refusal is a behavioral no-op, not merely a pair of boolean labels."""

    result = _run_native(
        tmp_path,
        world_size=2,
        extra_args=("--experiment", "scientific", "--fixture", "all_invalid"),
    )
    for state in _states(result):
        assert state["status"] == "refused"
        assert state["ddp_forward_calls"] == 0
        assert state["counter_before"] == state["counter_after"] == 0
        assert state["optimizer_state_before"] == state["optimizer_state_after"]
        assert "gradients" not in state
        assert "parameters_after" not in state


def test_r4_topology_refusal_precedes_dcp_load_and_preserves_state(tmp_path: Path, monkeypatch) -> None:
    """The topology gate must precede both DCP load and state application."""

    model = SemanticWavefunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    checkpoint_dir = tmp_path / "generations" / "gen-000001"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "COMPLETE").write_text("COMPLETE\n")
    sidecar = checkpoint_dir / "sidecars" / "rank-00001.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({"rank": 1, "world_size": 3}))
    sidecar_digest = checkpoint_module._digest(sidecar, root=checkpoint_dir).as_dict()
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps({
            "world_size": 2,
            "path": "generations/gen-000001",
            "files": [sidecar_digest],
        })
    )
    runtime = Mock(rank=1, world_size=3)
    runtime.broadcast_object.return_value = {"world_size": 2, "path": "generations/gen-000001"}
    runtime.broadcast_object.return_value["files"] = [sidecar_digest]
    runtime.all_gather_objects.return_value = [None, None, None]
    store = CheckpointPayloadStore(root=tmp_path, runtime=runtime)
    load = Mock(side_effect=AssertionError("DCP load must not be entered"))
    apply = Mock(side_effect=AssertionError("set_state_dict must not be entered"))
    monkeypatch.setattr(checkpoint_module.dcp, "load", load)
    monkeypatch.setattr(checkpoint_module, "set_state_dict", apply)
    with pytest.raises(CheckpointTopologyMismatch):
        store.load(model, optimizer, generation=1)
    assert load.call_count == 0
    assert apply.call_count == 0
    for key, value in before_model.items():
        assert torch.equal(model.state_dict()[key], value)
    assert optimizer.state_dict() == before_optimizer


def test_r5_peer_fault_has_no_successful_rank_or_checkpoint_publication(tmp_path: Path) -> None:
    """A peer fault invalidates provisional local success globally."""

    root = tmp_path / "peer-fault"
    result = _run_native(
        tmp_path,
        world_size=2,
        bounds=_FAULT_BOUNDS,
        extra_args=(
            "--experiment", "scientific", "--fixture", "regular", "--iterations", "1",
            "--checkpoint-root", str(root), "--checkpoint-generation", "1",
            "--checkpoint-failure-rank", "1",
        ),
    )
    _assert_no_rank_success(_states(result))
    assert result.publication_observed is False
    assert not (root / "latest.json").exists()
    assert not (root / "generations" / "gen-000001" / "COMPLETE").exists()
    assert all(code is not None and code != 0 for code in result.exit_codes)


def test_r6_post_digest_sidecar_corruption_blocks_publication(tmp_path: Path, monkeypatch) -> None:
    """A sidecar changed after its local digest must fail coordinator validation."""

    runtime = Mock(rank=0, world_size=2)
    runtime.barrier.return_value = None
    runtime.all_gather_objects.side_effect = lambda value: [value, value]
    model = SemanticWavefunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    original_digest = checkpoint_module._digest
    digest_calls = 0

    def corrupt_after_local_digest(path: Path, *, root: Path):
        nonlocal digest_calls
        digest_calls += 1
        digest = original_digest(path, root=root)
        if digest_calls == 1:
            rank_one = root / "sidecars" / "rank-00001.json"
            rank_one.parent.mkdir(parents=True, exist_ok=True)
            rank_one.write_text("{}")
            path.write_text('{"corrupted": true}')
        return digest

    monkeypatch.setattr(checkpoint_module, "_digest", corrupt_after_local_digest)
    monkeypatch.setattr(checkpoint_module, "get_state_dict", lambda *args: ({}, {}))
    monkeypatch.setattr(
        checkpoint_module.dcp,
        "save",
        lambda payload, storage_writer: None,
    )
    with pytest.raises(checkpoint_module.CheckpointCorrupt, match="digest changed"):
        CheckpointPayloadStore(root=tmp_path, runtime=runtime).save(
            model, optimizer, generation=1, sampler_state={}, rng_state={}, completed_updates=1
        )
    assert not (tmp_path / "generations" / "gen-000001" / "COMPLETE").exists()


def test_r7_probe_counts_executed_sampling_and_kinetic_raw_calls(tmp_path: Path) -> None:
    """Worker-path raw calls scale with actual short/long workloads."""

    short = _run_review_native(
        tmp_path,
        extra_args=("--experiment", "scientific", "--fixture", "regular", "--mcmc-steps", "1", "--kinetic-forwards", "1"),
    )
    long = _run_review_native(
        tmp_path,
        extra_args=("--experiment", "scientific", "--fixture", "regular", "--mcmc-steps", "5", "--kinetic-forwards", "5"),
    )
    short_states = _states(short)
    long_states = _states(long)
    assert [state["review_raw_model_calls"] for state in short_states] == [6, 6]
    assert [state["review_raw_model_calls"] for state in long_states] == [14, 14]
    assert all(long_state["review_raw_model_calls"] > short_state["review_raw_model_calls"] for short_state, long_state in zip(short_states, long_states, strict=True))


def test_r8_delay_beyond_outer_bound_fires_watchdog(tmp_path: Path) -> None:
    """The reviewed 2/12-second bounds distinguish nominal and over-bound work."""

    nominal_plan = FaultPlan(
        target_rank=1,
        kind=FaultKind.STALL_BEFORE_COLLECTIVE,
        phase=FaultPhase.BEFORE_COLLECTIVE,
        delay_seconds=6.0,
    )
    nominal = _run_native(tmp_path, world_size=2, fault_plan=nominal_plan, bounds=_FAULT_BOUNDS)
    assert nominal.watchdog_fired is False
    assert nominal.all_reaped is True
    assert nominal.publication_observed is False

    over_bound_plan = FaultPlan(
        target_rank=1,
        kind=FaultKind.STALL_BEFORE_COLLECTIVE,
        phase=FaultPhase.BEFORE_COLLECTIVE,
        delay_seconds=13.0,
    )
    result = _run_native(tmp_path, world_size=2, fault_plan=over_bound_plan, bounds=_FAULT_BOUNDS)
    assert result.watchdog_fired is True
    assert result.all_reaped is True
    assert result.publication_observed is False
