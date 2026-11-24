"""Contract tests for the consolidated 377f6b0f/02859027 score-row policy.

Both :class:`~tpen.training.sr.StochasticReconfigurationUpdate` and
:class:`~tpen.training.spring.SPRINGUpdate` enforce
:data:`~tpen.training.score_geometry.NONFINITE_SCORE_ROW_POLICY`
(``"fail_on_pinned"``) through
:func:`~tpen.training.score_geometry.select_score_update_rows`:

1. Each step pins its finite row mask at the first (local-energy) evaluation.
2. Any pinned row carrying a non-finite parameter score raises rather than
   silently re-masking, under both ``"fail"`` and ``"mask"`` local-energy
   policies.
3. No opt-out boolean flag exists on the selector.
4. Both policies travel through method descriptions, step telemetry,
   trainer-method compatibility validation, and versioned checkpoint state.
"""

from __future__ import annotations

import copy
import inspect
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tests.helpers.hooke_models import build_tiny_hamiltonian_terms, build_tiny_spenn
from tests.helpers.spring_dense_oracle import spring_trajectory
from tests.helpers.sr_dense_oracle import sr_direction
from tests.unit.training.test_sr_trainer_integration import (
    LEARNING_RATE as TRAINER_LEARNING_RATE,
    _FixedSampler,
    build_connected_model,
)
from tests.unit.training.test_update_binding import (
    _FixedSampler as _CheckpointFixedSampler,
    _checkpoint_context,
)
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext
from tpen.checkpoint import TrainResume, restore_checkpoint, save_checkpoint
from tpen.checkpoint.hashing import file_sha256
from tpen.data.batch import (
    ElectronBatch,
    MaterializedParameterLogScores,
    ParameterBinding,
    ParameterLayout,
    ParameterScoreForwardPacket,
    ParameterSlot,
    WavefunctionOutput,
)
from tpen.training.qgt import DampingPolicy
from tpen.training.score_geometry import (
    NONFINITE_SCORE_ROW_POLICY,
    ScoreConventions,
    select_score_update_rows,
)
from tpen.training.spring import (
    SPRING_STATE_VERSION,
    SPRINGPolicy,
    SPRINGUpdate,
)
from tpen.training.sr import (
    SR_STATE_VERSION,
    SRPolicy,
    StochasticReconfigurationUpdate,
)
from tpen.training.trainer import VMCTrainer
from tpen.training.update import ModelParameterBinding, ScoreUpdateInput

LEARNING_RATE = 0.05
ORACLE_RTOL = 1.0e-9
ORACLE_ATOL = 1.0e-9


def _layout(n_parameters: int) -> ParameterLayout:
    slot = ParameterSlot(
        ordinal=0,
        shape=(n_parameters,),
        numel=n_parameters,
        dtype=torch.float64,
    )
    return ParameterLayout(slots=(slot,))


def _step_input(
    parameter: torch.nn.Parameter,
    features: torch.Tensor,
    energies: torch.Tensor,
    *,
    step: int = 0,
    n_electrons: int = 2,
) -> ScoreUpdateInput:
    n_samples = int(features.shape[0])
    positions = torch.zeros((n_samples, n_electrons, 1), dtype=torch.float64)
    spins = torch.ones((n_samples, n_electrons), dtype=torch.float64)
    batch = ElectronBatch(positions=positions, spins=spins)
    # Keep WavefunctionOutput.logabs finite when tests inject non-finite score entries
    # into `features`, so WavefunctionOutput.__post_init__ does not reject `-inf`
    # logabs with `sign == 1` before `method.update(...)` sees the poisoned scores.
    clean_features = torch.where(
        torch.isfinite(features), features, torch.zeros_like(features)
    )
    output = WavefunctionOutput(
        logabs=(clean_features @ parameter).detach(),
        sign=torch.ones(n_samples, dtype=torch.float64),
    )
    layout = _layout(int(features.shape[1]))
    return ScoreUpdateInput(
        batch=batch,
        wavefunction=output,
        local_energy=energies,
        step=step,
        parameter_scores=MaterializedParameterLogScores(layout=layout, blocks=(features,)),
        parameter_binding=ParameterBinding(layout=layout, parameters=(parameter,)),
    )


def _problem(
    *,
    n_samples: int = 7,
    n_parameters: int = 4,
    seed: int = 101,
) -> tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    parameter = torch.nn.Parameter(
        torch.randn(n_parameters, generator=generator, dtype=torch.float64)
    )
    features = torch.randn(
        (n_samples, n_parameters), generator=generator, dtype=torch.float64
    )
    energies = torch.randn(n_samples, generator=generator, dtype=torch.float64)
    return parameter, features, energies


def _build_method(
    kind: str,
    parameter: torch.nn.Parameter,
    *,
    nonfinite_local_energy_policy: str = "mask",
) -> StochasticReconfigurationUpdate | SPRINGUpdate:
    optimizer = torch.optim.SGD([parameter], lr=LEARNING_RATE)
    binding = ModelParameterBinding(parameters=(parameter,))
    if kind in ("sr_parameter", "sr_sample"):
        space = "parameter" if kind == "sr_parameter" else "sample"
        policy = SRPolicy(
            solve_space=space,
            damping=DampingPolicy(absolute=0.0, relative=1.0e-2),
            learning_rate=LEARNING_RATE,
        )
        return StochasticReconfigurationUpdate(
            optimizer,
            model_parameters=binding,
            policy=policy,
            conventions=ScoreConventions(),
            nonfinite_local_energy_policy=nonfinite_local_energy_policy,
        )
    if kind == "spring":
        base = SRPolicy(
            solve_space="sample",
            damping=DampingPolicy(absolute=0.0, relative=1.0e-2),
            learning_rate=LEARNING_RATE,
        )
        return SPRINGUpdate(
            optimizer,
            model_parameters=binding,
            policy=SPRINGPolicy(base=base, history_decay=0.35),
            conventions=ScoreConventions(),
            nonfinite_local_energy_policy=nonfinite_local_energy_policy,
        )
    raise ValueError(f"unknown method kind {kind!r}")


def _build_trainer_method(
    kind: str,
    model: torch.nn.Module,
    *,
    nonfinite_local_energy_policy: str,
) -> StochasticReconfigurationUpdate | SPRINGUpdate:
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=TRAINER_LEARNING_RATE)
    binding = ModelParameterBinding(parameters=parameters)
    base = SRPolicy(
        solve_space="sample",
        damping=DampingPolicy(absolute=0.0, relative=1.0e-2, minimum=1.0e-12),
        learning_rate=TRAINER_LEARNING_RATE,
    )
    if kind == "sr":
        return StochasticReconfigurationUpdate(
            optimizer,
            model_parameters=binding,
            policy=base,
            conventions=ScoreConventions(solve_dtype=torch.float64),
            nonfinite_local_energy_policy=nonfinite_local_energy_policy,
        )
    if kind == "spring":
        return SPRINGUpdate(
            optimizer,
            model_parameters=binding,
            policy=SPRINGPolicy(base=base, history_decay=0.35),
            conventions=ScoreConventions(solve_dtype=torch.float64),
            nonfinite_local_energy_policy=nonfinite_local_energy_policy,
        )
    raise ValueError(f"unknown trainer method kind {kind!r}")


# --------------------------------------------------------------------------
# Selector signature pin and validation
# --------------------------------------------------------------------------


def test_select_score_update_rows_signature_pins_no_opt_out_flag() -> None:
    """No opt-out boolean exists: the selector accepts exactly 4 keyword-only args."""

    assert NONFINITE_SCORE_ROW_POLICY == "fail_on_pinned"
    signature = inspect.signature(select_score_update_rows)
    assert set(signature.parameters) == {
        "local_energy",
        "rows",
        "nonfinite_local_energy_policy",
        "method_name",
    }
    for param in signature.parameters.values():
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty


def test_select_score_update_rows_validates_inputs_and_caps_bad_index_preview() -> None:
    """Shape/type/policy validation is loud, and >10 bad indices cap the preview."""

    energies = torch.zeros(4, dtype=torch.float64)
    rows = torch.zeros((4, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="must be 1-D"):
        select_score_update_rows(
            local_energy=energies.reshape(2, 2),
            rows=rows,
            nonfinite_local_energy_policy="mask",
            method_name="SR",
        )
    with pytest.raises(ValueError, match="must be 2-D"):
        select_score_update_rows(
            local_energy=energies,
            rows=rows.reshape(-1),
            nonfinite_local_energy_policy="mask",
            method_name="SR",
        )
    with pytest.raises(ValueError, match="one score row per local energy"):
        select_score_update_rows(
            local_energy=energies,
            rows=torch.zeros((3, 2), dtype=torch.float64),
            nonfinite_local_energy_policy="mask",
            method_name="SR",
        )
    with pytest.raises(ValueError, match="nonfinite local-energy policy"):
        select_score_update_rows(
            local_energy=energies,
            rows=rows,
            nonfinite_local_energy_policy="drop",
            method_name="SR",
        )

    many_energies = torch.zeros(12, dtype=torch.float64)
    many_bad_rows = torch.full((12, 2), float("nan"), dtype=torch.float64)
    with pytest.raises(
        RuntimeError,
        match=r"12 of 12 pinned.*\[0, 1, 2, 3, 4, 5, 6, 7, 8, 9\].*first 10 of 12",
    ):
        select_score_update_rows(
            local_energy=many_energies,
            rows=many_bad_rows,
            nonfinite_local_energy_policy="mask",
            method_name="SR",
        )


# --------------------------------------------------------------------------
# SR (parameter + sample) and SPRING row-policy contracts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sr_parameter", "sr_sample", "spring"])
@pytest.mark.parametrize("policy", ["fail", "mask"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_score_on_pinned_row_raises_and_leaves_state_unchanged(
    kind: str,
    policy: str,
    bad_value: float,
) -> None:
    """nan/+inf/-inf on a pinned score row refuses under both fail and mask."""

    parameter, features, energies = _problem(seed=201)
    method = _build_method(kind, parameter, nonfinite_local_energy_policy=policy)

    # Take one clean step first so completed_updates == 1 and SPRING history is nonzero.
    warmup = method.update(_step_input(parameter, features, energies, step=0))
    assert warmup.applied is True
    assert method.completed_updates == 1
    before_param = parameter.detach().clone()
    before_history = method.history.detach().clone() if isinstance(method, SPRINGUpdate) else None
    if before_history is not None:
        assert torch.linalg.vector_norm(before_history).item() > 0.0

    poisoned = features.clone()
    poisoned[2, 1] = bad_value
    with pytest.raises(
        RuntimeError,
        match=rf"non-finite parameter scores.*row indices \[2\].*fail_on_pinned.*{policy!r}",
    ):
        method.update(_step_input(parameter, poisoned, energies, step=1))

    assert torch.equal(parameter.detach(), before_param)
    assert method.completed_updates == 1
    if isinstance(method, SPRINGUpdate):
        assert before_history is not None
        assert torch.equal(method.history, before_history)


@pytest.mark.parametrize("kind", ["sr_parameter", "sr_sample", "spring"])
def test_mixed_batch_under_mask_names_only_the_pinned_bad_score_row(kind: str) -> None:
    """Unpinned row 0 (NaN energy) is excluded; pinned row 2 (NaN score) triggers raise."""

    parameter, features, energies = _problem(seed=202)
    method = _build_method(kind, parameter, nonfinite_local_energy_policy="mask")
    before_param = parameter.detach().clone()
    before_history = method.history.detach().clone() if isinstance(method, SPRINGUpdate) else None

    poisoned_energies = energies.clone()
    poisoned_energies[0] = float("nan")
    poisoned_features = features.clone()
    poisoned_features[0, 0] = float("inf")  # unpinned row: not a pinned violation
    poisoned_features[2, 1] = float("nan")  # pinned row: must refuse and name [2] only

    with pytest.raises(RuntimeError) as exc_info:
        method.update(_step_input(parameter, poisoned_features, poisoned_energies, step=0))

    message = str(exc_info.value)
    assert "non-finite parameter scores" in message
    assert "row indices [2]" in message
    assert "0," not in message and "[0" not in message
    assert torch.equal(parameter.detach(), before_param)
    assert method.completed_updates == 0
    if isinstance(method, SPRINGUpdate):
        assert before_history is not None
        assert torch.equal(method.history, before_history)


@pytest.mark.parametrize("kind", ["sr_parameter", "sr_sample", "spring"])
def test_cooccurring_bad_energy_and_score_on_same_row_masks_once_under_mask_and_fails_under_fail(
    kind: str,
) -> None:
    """When one row has BOTH non-finite energy and score, mask excludes it once; fail raises."""

    parameter, features, energies = _problem(seed=203)
    poisoned_energies = energies.clone()
    poisoned_energies[1] = float("nan")
    poisoned_features = features.clone()
    poisoned_features[1, 2] = float("-inf")
    keep = [0, 2, 3, 4, 5, 6]

    # Mask arm: excluded once by energy_finite, matches independent NumPy oracle on survivors.
    mask_param = torch.nn.Parameter(parameter.detach().clone())
    before = mask_param.detach().clone()
    mask_method = _build_method(kind, mask_param, nonfinite_local_energy_policy="mask")
    result = mask_method.update(_step_input(mask_param, poisoned_features, poisoned_energies))

    assert result.applied is True
    assert mask_method.last_telemetry is not None
    assert mask_method.last_telemetry.n_samples == 7
    assert mask_method.last_telemetry.n_finite_samples == 6
    assert mask_method.last_telemetry.n_nonfinite_energy_samples == 1
    assert mask_method.last_telemetry.n_nonfinite_score_samples == 1
    assert mask_method.last_telemetry.nonfinite_local_energy_policy == "mask"
    assert mask_method.last_telemetry.nonfinite_score_row_policy == "fail_on_pinned"

    prefix = "spring" if kind == "spring" else "sr"
    metrics = mask_method.last_telemetry.as_metrics()
    assert metrics[f"{prefix}_nonfinite_energy_samples"] == 1
    assert metrics[f"{prefix}_nonfinite_score_samples"] == 1
    assert metrics[f"{prefix}_nonfinite_local_energy_policy"] == "mask"
    assert metrics[f"{prefix}_nonfinite_score_row_policy"] == "fail_on_pinned"

    if kind == "spring":
        oracle_step = spring_trajectory(
            (features.numpy()[keep],),
            (energies.numpy()[keep],),
            history_decay=0.35,
            relative=1.0e-2,
        )[0]
        expected_step = LEARNING_RATE * oracle_step.history
    else:
        oracle = sr_direction(
            features.numpy()[keep],
            energies.numpy()[keep],
            absolute=0.0,
            relative=1.0e-2,
        )
        expected_step = LEARNING_RATE * oracle.direction

    np.testing.assert_allclose(
        (before - mask_param.detach()).numpy(),
        expected_step,
        rtol=ORACLE_RTOL,
        atol=ORACLE_ATOL,
    )

    # Fail arm: raises on local-energy non-finiteness before inspecting scores.
    fail_param = torch.nn.Parameter(parameter.detach().clone())
    fail_before = fail_param.detach().clone()
    fail_method = _build_method(kind, fail_param, nonfinite_local_energy_policy="fail")
    with pytest.raises(
        RuntimeError,
        match=r"1 of 7 local-energy samples are non-finite and the active policy is 'fail'",
    ):
        fail_method.update(_step_input(fail_param, poisoned_features, poisoned_energies))
    assert torch.equal(fail_param.detach(), fail_before)
    assert fail_method.completed_updates == 0


@pytest.mark.parametrize("kind", ["sr_parameter", "sr_sample", "spring"])
@pytest.mark.parametrize("policy", ["fail", "mask"])
def test_all_finite_energies_with_all_nonfinite_scores_names_parameter_scores(
    kind: str,
    policy: str,
) -> None:
    """Must report non-finite parameter scores, not 'no finite local-energy sample'."""

    parameter, features, energies = _problem(seed=204)
    method = _build_method(kind, parameter, nonfinite_local_energy_policy=policy)
    all_bad_scores = torch.full_like(features, float("nan"))

    with pytest.raises(RuntimeError) as exc_info:
        method.update(_step_input(parameter, all_bad_scores, energies))

    message = str(exc_info.value)
    assert "non-finite parameter scores" in message
    assert "no finite local-energy sample" not in message


@pytest.mark.parametrize("kind", ["sr_parameter", "sr_sample", "spring"])
def test_clean_batch_produces_identical_updates_under_fail_and_mask(kind: str) -> None:
    """Over-restriction control: clean batches behave identically under both policies."""

    parameter, features, energies = _problem(seed=205)
    param_fail = torch.nn.Parameter(parameter.detach().clone())
    param_mask = torch.nn.Parameter(parameter.detach().clone())

    method_fail = _build_method(kind, param_fail, nonfinite_local_energy_policy="fail")
    method_mask = _build_method(kind, param_mask, nonfinite_local_energy_policy="mask")

    result_fail = method_fail.update(_step_input(param_fail, features, energies))
    result_mask = method_mask.update(_step_input(param_mask, features, energies))

    assert result_fail.applied is True
    assert result_mask.applied is True
    assert torch.equal(param_fail.detach(), param_mask.detach())
    if isinstance(method_fail, SPRINGUpdate) and isinstance(method_mask, SPRINGUpdate):
        assert torch.equal(method_fail.history, method_mask.history)


# --------------------------------------------------------------------------
# Checkpoint / method_state_dict provenance & resume conflicts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sr_parameter", "spring"])
def test_method_state_dict_provenance_and_resume_policy_conflicts(kind: str) -> None:
    """State envelopes record both policies, JSON-round-trip, and refuse drift atomically."""

    parameter, features, energies = _problem(seed=301)
    saved_param = torch.nn.Parameter(parameter.detach().clone())
    saved_method = _build_method(kind, saved_param, nonfinite_local_energy_policy="fail")
    assert saved_method.update(_step_input(saved_param, features, energies, step=0)).applied

    raw_state = dict(saved_method.method_state_dict())
    expected_version = SPRING_STATE_VERSION if kind == "spring" else SR_STATE_VERSION
    assert raw_state["version"] == expected_version
    assert raw_state["nonfinite_local_energy_policy"] == "fail"
    assert raw_state["nonfinite_score_row_policy"] == "fail_on_pinned"

    round_tripped = json.loads(json.dumps(raw_state))
    assert round_tripped == raw_state

    # Clean matching restore succeeds.
    clean_param = torch.nn.Parameter(saved_param.detach().clone())
    clean_target = _build_method(kind, clean_param, nonfinite_local_energy_policy="fail")
    clean_target.load_method_state_dict(round_tripped)
    assert clean_target.completed_updates == 1

    # Seed a live target with distinct state (completed_updates=2, nonzero history)
    # so every refusal below proves the target was NOT mutated.
    live_param = torch.nn.Parameter(parameter.detach().clone())
    live_fail = _build_method(kind, live_param, nonfinite_local_energy_policy="fail")
    live_fail.update(_step_input(live_param, features, energies, step=0))
    live_fail.update(_step_input(live_param, features, energies, step=1))
    assert live_fail.completed_updates == 2
    prior_history_fail = (
        live_fail.history.detach().clone() if isinstance(live_fail, SPRINGUpdate) else None
    )

    legacy_version = "spring-state-1" if kind == "spring" else "sr-state-1"
    legacy_state = copy.deepcopy(round_tripped)
    legacy_state["version"] = legacy_version
    with pytest.raises(ValueError, match="unsupported .* state version"):
        live_fail.load_method_state_dict(legacy_state)

    missing_energy_policy = copy.deepcopy(round_tripped)
    del missing_energy_policy["nonfinite_local_energy_policy"]
    with pytest.raises(ValueError, match="nonfinite_local_energy_policy"):
        live_fail.load_method_state_dict(missing_energy_policy)

    mismatched_energy_policy = copy.deepcopy(round_tripped)
    mismatched_energy_policy["nonfinite_local_energy_policy"] = "mask"
    with pytest.raises(ValueError, match=r"nonfinite_local_energy_policy.*'mask'.*'fail'"):
        live_fail.load_method_state_dict(mismatched_energy_policy)

    missing_score_policy = copy.deepcopy(round_tripped)
    del missing_score_policy["nonfinite_score_row_policy"]
    with pytest.raises(ValueError, match="nonfinite_score_row_policy"):
        live_fail.load_method_state_dict(missing_score_policy)

    mismatched_score_policy = copy.deepcopy(round_tripped)
    mismatched_score_policy["nonfinite_score_row_policy"] = "mask"
    with pytest.raises(
        ValueError, match=r"nonfinite_score_row_policy.*'mask'.*'fail_on_pinned'"
    ):
        live_fail.load_method_state_dict(mismatched_score_policy)

    # All refusals left live_fail completely unmutated.
    assert live_fail.completed_updates == 2
    if isinstance(live_fail, SPRINGUpdate):
        assert prior_history_fail is not None
        assert torch.equal(live_fail.history, prior_history_fail)

    # Reverse direction: checkpoint recorded "fail", live method configured for "mask".
    live_mask_param = torch.nn.Parameter(parameter.detach().clone())
    live_mask = _build_method(kind, live_mask_param, nonfinite_local_energy_policy="mask")
    live_mask.update(_step_input(live_mask_param, features, energies, step=0))
    prior_history_mask = (
        live_mask.history.detach().clone() if isinstance(live_mask, SPRINGUpdate) else None
    )
    with pytest.raises(ValueError, match=r"nonfinite_local_energy_policy.*'fail'.*'mask'"):
        live_mask.load_method_state_dict(round_tripped)
    assert live_mask.completed_updates == 1
    if isinstance(live_mask, SPRINGUpdate):
        assert prior_history_mask is not None
        assert torch.equal(live_mask.history, prior_history_mask)


# --------------------------------------------------------------------------
# End-to-end VMCTrainer integration (real TPENWaveFunction + real score packet)
# --------------------------------------------------------------------------


class _CountingFixedSampler(_FixedSampler):
    """Fixed sampler that counts ``collect_samples`` calls to pin pre-sampling refusal."""

    def __init__(self, *, n_walkers: int = 6, seed: int = 5) -> None:
        super().__init__(n_walkers=n_walkers, seed=seed)
        self.calls = 0

    def collect_samples(self, model, *, reset=False, device=None, emitter=None):
        self.calls += 1
        return super().collect_samples(model, reset=reset, device=device, emitter=emitter)


@pytest.mark.parametrize("kind", ["sr", "spring"])
@pytest.mark.parametrize(
    ("trainer_policy", "method_policy"),
    [("fail", "mask"), ("mask", "fail")],
)
def test_trainer_rejects_disagreeing_local_energy_policy_before_sampling(
    kind: str,
    trainer_policy: str,
    method_policy: str,
) -> None:
    """Mismatched trainer vs SR/SPRING policy raises at resolve_update_state and fit."""

    torch.manual_seed(0)
    model = build_connected_model()
    method = _build_trainer_method(
        kind, model, nonfinite_local_energy_policy=method_policy
    )
    trainer_for_resolve = VMCTrainer(
        max_steps=1,
        update_method=method,
        nonfinite_local_energy_policy=trainer_policy,
    )
    with pytest.raises(
        ValueError,
        match=r"update method nonfinite_local_energy_policy disagrees with VMCTrainer",
    ):
        trainer_for_resolve.resolve_update_state(model=model, optimizer=method.optimizer)

    sampler = _CountingFixedSampler()
    trainer_for_fit = VMCTrainer(
        max_steps=1,
        update_method=method,
        nonfinite_local_energy_policy=trainer_policy,
    )
    with pytest.raises(
        ValueError,
        match=r"update method nonfinite_local_energy_policy disagrees with VMCTrainer",
    ):
        trainer_for_fit.fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=build_tiny_hamiltonian_terms(),
            optimizer=method.optimizer,
            context=_StubContext(),
            emit=lambda **_: None,
        )
    assert sampler.calls == 0


@pytest.mark.parametrize("kind", ["sr", "spring"])
@pytest.mark.parametrize("policy", ["fail", "mask"])
def test_matching_policies_emit_provenance_in_description_metrics_and_state_dict(
    kind: str,
    policy: str,
) -> None:
    """Matching trainer and SR/SPRING policies surface both policies across all records."""

    torch.manual_seed(0)
    model = build_connected_model()
    method = _build_trainer_method(kind, model, nonfinite_local_energy_policy=policy)
    context = _StubContext()
    trainer = VMCTrainer(
        max_steps=1,
        log_every_n_steps=1,
        update_method=method,
        nonfinite_local_energy_policy=policy,
    )
    trainer.fit(
        model=model,
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=method.optimizer,
        context=context,
        emit=lambda **_: None,
    )

    description_records = [
        metrics for ns, metrics in context.records if ns == "train/update_method"
    ]
    assert len(description_records) == 1
    described = description_records[0]
    assert described["update_method_setting_nonfinite_local_energy_policy"] == policy
    assert (
        described["update_method_setting_nonfinite_score_row_policy"]
        == NONFINITE_SCORE_ROW_POLICY
    )

    train_records = [metrics for ns, metrics in context.records if ns == "train"]
    assert len(train_records) == 1
    step_metrics = train_records[0]
    prefix = "spring" if kind == "spring" else "sr"
    assert step_metrics[f"{prefix}_nonfinite_energy_samples"] == 0
    assert step_metrics[f"{prefix}_nonfinite_score_samples"] == 0
    assert step_metrics[f"{prefix}_nonfinite_local_energy_policy"] == policy
    assert step_metrics[f"{prefix}_nonfinite_score_row_policy"] == NONFINITE_SCORE_ROW_POLICY

    state = trainer.state_dict()
    assert state["nonfinite_local_energy_policy"] == policy
    assert state["update_method"]["nonfinite_local_energy_policy"] == policy
    assert state["update_method"]["nonfinite_score_row_policy"] == NONFINITE_SCORE_ROW_POLICY


@pytest.mark.parametrize("kind", ["sr", "spring"])
def test_nonfinite_score_injected_during_trainer_fit_raises_and_preserves_state(
    kind: str,
) -> None:
    """A pinned bad score during VMCTrainer.fit aborts cleanly without advancing state."""

    torch.manual_seed(0)
    model = build_connected_model()
    method = _build_trainer_method(kind, model, nonfinite_local_energy_policy="mask")
    sampler = _FixedSampler()
    terms = build_tiny_hamiltonian_terms()

    # Step 0 runs cleanly so post-step-0 parameters and SPRING history are nontrivial.
    trainer = VMCTrainer(
        max_steps=2,
        log_every_n_steps=1,
        update_method=method,
        nonfinite_local_energy_policy="mask",
    )
    original_eval = model.evaluate_materialized_parameter_score_request
    call_count = 0
    post_step0_params: list[torch.Tensor] = []
    post_step0_history: torch.Tensor | None = None

    def poison_on_second_step(*args: Any, **kwargs: Any) -> ParameterScoreForwardPacket:
        nonlocal call_count, post_step0_params, post_step0_history
        packet = original_eval(*args, **kwargs)
        call_count += 1
        if call_count == 1:
            return packet
        # Snapshot the state entering step 1 (after step 0 completed cleanly).
        post_step0_params = [p.detach().clone() for p in model.parameters()]
        if isinstance(method, SPRINGUpdate):
            post_step0_history = method.history.detach().clone()
        blocks = list(packet.parameter_scores.blocks)
        poisoned_first = blocks[0].clone()
        poisoned_first[2] = float("nan")
        blocks[0] = poisoned_first
        return ParameterScoreForwardPacket(
            output=packet.output,
            parameter_scores=MaterializedParameterLogScores(
                layout=packet.parameter_scores.layout,
                blocks=tuple(blocks),
            ),
        )

    model.evaluate_materialized_parameter_score_request = poison_on_second_step  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match=r"non-finite parameter scores.*row indices \[2\]"):
        trainer.fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=terms,
            optimizer=method.optimizer,
            context=_StubContext(),
            emit=lambda **_: None,
        )

    assert call_count == 2
    assert trainer.next_iteration == 1
    assert trainer.completed_updates == 1
    assert method.completed_updates == 1
    assert post_step0_params
    for expected_param, current_param in zip(
        post_step0_params, model.parameters(), strict=True
    ):
        assert torch.equal(current_param.detach(), expected_param)
    if isinstance(method, SPRINGUpdate):
        assert post_step0_history is not None
        assert torch.linalg.vector_norm(post_step0_history).item() > 0.0
        assert torch.equal(method.history, post_step0_history)


# --------------------------------------------------------------------------
# Consumer-side save_checkpoint -> restore_checkpoint v2 policy round-trip
# --------------------------------------------------------------------------


def _build_checkpoint_run(
    kind: str,
    *,
    max_steps: int,
    seed: int,
    nonfinite_local_energy_policy: str,
) -> tuple[
    torch.nn.Module,
    StochasticReconfigurationUpdate | SPRINGUpdate,
    VMCTrainer,
    _CheckpointFixedSampler,
]:
    """Build one seeded tiny model/method/trainer/sampler bundle for checkpoint resume."""

    torch.manual_seed(seed)
    model = build_tiny_spenn()
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=TRAINER_LEARNING_RATE)
    binding = ModelParameterBinding(parameters=parameters)
    if kind == "sr":
        method: StochasticReconfigurationUpdate | SPRINGUpdate = (
            StochasticReconfigurationUpdate(
                optimizer,
                model_parameters=binding,
                policy=SRPolicy(
                    solve_space="parameter",
                    damping=DampingPolicy(absolute=0.0, relative=1.0e-2, minimum=1.0e-12),
                    learning_rate=TRAINER_LEARNING_RATE,
                ),
                conventions=ScoreConventions(solve_dtype=torch.float64),
                nonfinite_local_energy_policy=nonfinite_local_energy_policy,
            )
        )
    elif kind == "spring":
        base = SRPolicy(
            solve_space="sample",
            damping=DampingPolicy(absolute=0.0, relative=1.0e-2, minimum=1.0e-12),
            learning_rate=TRAINER_LEARNING_RATE,
        )
        method = SPRINGUpdate(
            optimizer,
            model_parameters=binding,
            policy=SPRINGPolicy(base=base, history_decay=0.35),
            conventions=ScoreConventions(solve_dtype=torch.float64),
            nonfinite_local_energy_policy=nonfinite_local_energy_policy,
        )
    else:
        raise ValueError(f"unknown checkpoint method kind {kind!r}")
    trainer = VMCTrainer(
        max_steps=max_steps,
        log_every_n_steps=1,
        update_method=method,
        nonfinite_local_energy_policy=nonfinite_local_energy_policy,
    )
    return model, method, trainer, _CheckpointFixedSampler()


def _rewrite_trainer_json_and_digest(
    src_dir: Path,
    dst_dir: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    """Clone a checkpoint dir, mutate ``trainer.json``, and recompute ``trainer_sha256``."""

    shutil.copytree(src_dir, dst_dir)
    trainer_path = dst_dir / "trainer.json"
    manifest_path = dst_dir / "manifest.json"
    trainer_state = json.loads(trainer_path.read_text(encoding="utf-8"))
    mutate(trainer_state)
    trainer_path.write_text(
        json.dumps(trainer_state, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hashes"]["trainer_sha256"] = file_sha256(trainer_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return dst_dir


@pytest.mark.parametrize("kind", ["sr", "spring"])
def test_checkpoint_save_restore_v2_policy_roundtrip_and_refusals(
    kind: str,
    tmp_path: Path,
) -> None:
    """Real ``save_checkpoint`` -> ``restore_checkpoint`` pins v2 schema and policy gates."""

    terms = build_tiny_hamiltonian_terms()
    ckpt_context = _checkpoint_context(tmp_path)

    # 1. Uninterrupted 2-step "fail" reference vs 1-step save + train_resume + step 2.
    straight_model, straight_method, straight_trainer, straight_sampler = (
        _build_checkpoint_run(
            kind, max_steps=2, seed=0, nonfinite_local_energy_policy="fail"
        )
    )
    straight_trainer.resolve_update_state(
        model=straight_model, optimizer=straight_method.optimizer
    )
    straight_trainer.fit(
        model=straight_model,
        sampler=straight_sampler,
        hamiltonian_terms=terms,
        optimizer=straight_method.optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    assert straight_trainer.next_iteration == 2
    assert straight_trainer.completed_updates == 2
    assert straight_method.completed_updates == 2
    straight_params = [p.detach().clone() for p in straight_model.parameters()]

    model, method, trainer, sampler = _build_checkpoint_run(
        kind, max_steps=1, seed=0, nonfinite_local_energy_policy="fail"
    )
    trainer.resolve_update_state(model=model, optimizer=method.optimizer)
    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=method.optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    assert trainer.next_iteration == 1
    assert trainer.completed_updates == 1
    assert method.completed_updates == 1
    step1_params = [p.detach().clone() for p in model.parameters()]
    assert any(
        not torch.equal(first, second)
        for first, second in zip(step1_params, straight_params, strict=True)
    ), "step 2 must move parameters so resumed step parity is non-vacuous"
    if isinstance(method, SPRINGUpdate):
        assert torch.linalg.vector_norm(method.history).item() > 0.0

    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=trainer.next_iteration,
        completed_updates=trainer.completed_updates,
        model=model,
        context=ckpt_context,
        optimizer=method.optimizer,
        trainer=trainer,
        sampler=sampler,
        payload=TrainResume(),
    )

    expected_version = SPRING_STATE_VERSION if kind == "spring" else SR_STATE_VERSION
    on_disk_trainer = json.loads(
        (checkpoint_dir / "trainer.json").read_text(encoding="utf-8")
    )
    assert on_disk_trainer["nonfinite_local_energy_policy"] == "fail"
    on_disk_method = on_disk_trainer["update_method"]
    assert on_disk_method["version"] == expected_version
    assert on_disk_method["nonfinite_local_energy_policy"] == "fail"
    assert on_disk_method["nonfinite_score_row_policy"] == "fail_on_pinned"
    assert on_disk_method["completed_updates"] == 1

    resumed_model, resumed_method, resumed_trainer, resumed_sampler = (
        _build_checkpoint_run(
            kind, max_steps=2, seed=99, nonfinite_local_energy_policy="fail"
        )
    )
    resumed_trainer.resolve_update_state(
        model=resumed_model, optimizer=resumed_method.optimizer
    )
    report = restore_checkpoint(
        load={"mode": "train_resume", "path": str(checkpoint_dir)},
        model=resumed_model,
        context=ckpt_context,
        optimizer=resumed_method.optimizer,
        trainer=resumed_trainer,
        sampler=resumed_sampler,
        mode="train_resume",
    )
    assert report.loaded_trainer is True
    assert report.completed_updates == 1
    assert resumed_trainer.next_iteration == 1
    assert resumed_trainer.completed_updates == 1
    assert resumed_method.completed_updates == 1
    for expected_param, actual_param in zip(
        step1_params, resumed_model.parameters(), strict=True
    ):
        assert torch.equal(actual_param.detach(), expected_param)
    if isinstance(resumed_method, SPRINGUpdate) and isinstance(method, SPRINGUpdate):
        assert torch.equal(resumed_method.history, method.history)

    resumed_trainer.rebuild_update_state(model=resumed_model)
    resumed_trainer.fit(
        model=resumed_model,
        sampler=resumed_sampler,
        hamiltonian_terms=terms,
        optimizer=resumed_method.optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    assert resumed_trainer.next_iteration == 2
    assert resumed_trainer.completed_updates == 2
    assert resumed_method.completed_updates == 2
    for expected_param, actual_param in zip(
        straight_params, resumed_model.parameters(), strict=True
    ):
        assert torch.equal(actual_param.detach(), expected_param)
    if isinstance(resumed_method, SPRINGUpdate) and isinstance(
        straight_method, SPRINGUpdate
    ):
        assert torch.linalg.vector_norm(straight_method.history).item() > 0.0
        assert torch.equal(resumed_method.history, straight_method.history)

    # 2. Legacy v1 schema refusal via restore_checkpoint (updated trainer_sha256 so the
    #    method version gate fires rather than the digest gate or trainer policy gate).
    legacy_version = "spring-state-1" if kind == "spring" else "sr-state-1"

    def _to_legacy_v1(state: dict[str, Any]) -> None:
        state["nonfinite_local_energy_policy"] = "mask"
        state["update_method"]["version"] = legacy_version
        state["update_method"].pop("nonfinite_local_energy_policy", None)
        state["update_method"].pop("nonfinite_score_row_policy", None)

    legacy_dir = _rewrite_trainer_json_and_digest(
        checkpoint_dir, tmp_path / "legacy-v1", _to_legacy_v1
    )
    legacy_model, legacy_method, legacy_trainer, legacy_sampler = (
        _build_checkpoint_run(
            kind, max_steps=2, seed=101, nonfinite_local_energy_policy="mask"
        )
    )
    legacy_trainer.resolve_update_state(
        model=legacy_model, optimizer=legacy_method.optimizer
    )
    with pytest.raises(ValueError, match=r"unsupported (SR|SPRING) state version"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(legacy_dir)},
            model=legacy_model,
            context=ckpt_context,
            optimizer=legacy_method.optimizer,
            trainer=legacy_trainer,
            sampler=legacy_sampler,
            mode="train_resume",
        )
    assert legacy_trainer.next_iteration == 0
    assert legacy_trainer.completed_updates == 0
    assert legacy_method.completed_updates == 0
    if isinstance(legacy_method, SPRINGUpdate):
        assert torch.equal(legacy_method.history, torch.zeros_like(legacy_method.history))

    # 3a. Unmodified "fail" checkpoint fed into a fresh "mask" trainer+method:
    #     same ckpt_context passes config-hash checks so trainer policy refusal fires.
    mask_model, mask_method, mask_trainer, mask_sampler = _build_checkpoint_run(
        kind, max_steps=2, seed=102, nonfinite_local_energy_policy="mask"
    )
    mask_trainer.resolve_update_state(
        model=mask_model, optimizer=mask_method.optimizer
    )
    with pytest.raises(
        ValueError,
        match=r"non-finite local-energy policy disagrees with the checkpoint",
    ):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(checkpoint_dir)},
            model=mask_model,
            context=ckpt_context,
            optimizer=mask_method.optimizer,
            trainer=mask_trainer,
            sampler=mask_sampler,
            mode="train_resume",
        )
    assert mask_trainer.next_iteration == 0
    assert mask_trainer.completed_updates == 0
    assert mask_method.completed_updates == 0

    # 3b. Method-level nonfinite_local_energy_policy drift inside trainer.json["update_method"]
    #     (top-level trainer policy matches "fail" so load_method_state_dict's check fires).
    def _drift_method_energy_policy(state: dict[str, Any]) -> None:
        state["update_method"]["nonfinite_local_energy_policy"] = "mask"

    energy_drift_dir = _rewrite_trainer_json_and_digest(
        checkpoint_dir, tmp_path / "method-energy-drift", _drift_method_energy_policy
    )
    ed_model, ed_method, ed_trainer, ed_sampler = _build_checkpoint_run(
        kind, max_steps=2, seed=103, nonfinite_local_energy_policy="fail"
    )
    ed_trainer.resolve_update_state(model=ed_model, optimizer=ed_method.optimizer)
    with pytest.raises(
        ValueError,
        match=(
            r"(SR|SPRING) checkpoint nonfinite_local_energy_policy does not match "
            r"the live method: recorded 'mask', live 'fail'"
        ),
    ):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(energy_drift_dir)},
            model=ed_model,
            context=ckpt_context,
            optimizer=ed_method.optimizer,
            trainer=ed_trainer,
            sampler=ed_sampler,
            mode="train_resume",
        )
    assert ed_trainer.next_iteration == 0
    assert ed_trainer.completed_updates == 0
    assert ed_method.completed_updates == 0

    # 3c. Method-level nonfinite_score_row_policy drift inside trainer.json["update_method"].
    def _drift_score_row_policy(state: dict[str, Any]) -> None:
        state["update_method"]["nonfinite_score_row_policy"] = "mask"

    score_drift_dir = _rewrite_trainer_json_and_digest(
        checkpoint_dir, tmp_path / "score-policy-drift", _drift_score_row_policy
    )
    sd_model, sd_method, sd_trainer, sd_sampler = _build_checkpoint_run(
        kind, max_steps=2, seed=104, nonfinite_local_energy_policy="fail"
    )
    sd_trainer.resolve_update_state(model=sd_model, optimizer=sd_method.optimizer)
    with pytest.raises(
        ValueError,
        match=(
            r"(SR|SPRING) checkpoint nonfinite_score_row_policy does not match "
            r"the live method: recorded 'mask', live 'fail_on_pinned'"
        ),
    ):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(score_drift_dir)},
            model=sd_model,
            context=ckpt_context,
            optimizer=sd_method.optimizer,
            trainer=sd_trainer,
            sampler=sd_sampler,
            mode="train_resume",
        )
    assert sd_trainer.next_iteration == 0
    assert sd_trainer.completed_updates == 0
    assert sd_method.completed_updates == 0
