"""Real restore atomicity and method-state prevalidation contracts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from tpen.checkpoint import TrainResume, restore_checkpoint, save_checkpoint
from tpen.training.block_ng import BlockDiagonalNaturalGradientUpdate, BlockNGPolicy
from tpen.training.kfac import KFACPolicy, NativeKFACUpdate
from tpen.training.score_geometry import ScoreConventions
from tpen.training.spring import SPRINGPolicy, SPRINGUpdate
from tpen.training.sr import SRPolicy, StochasticReconfigurationUpdate
from tpen.training.update import ModelParameterBinding
from tests.unit.training.test_nonfinite_score_row_policy import _build_checkpoint_run, _checkpoint_context, _rewrite_trainer_json_and_digest
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _snapshot(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "parameters": [parameter.detach().clone() for parameter in model.parameters()],
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }


def _assert_snapshot_unchanged(snapshot: dict[str, Any], model, optimizer) -> None:
    assert all(torch.equal(before, after.detach()) for before, after in zip(snapshot["parameters"], model.parameters(), strict=True))
    assert _equal(snapshot["optimizer"], optimizer.state_dict())


def _checkpoint(tmp_path: Path, kind: str):
    model, method, trainer, sampler = _build_checkpoint_run(
        kind, max_steps=1, seed=0, nonfinite_local_energy_policy="fail"
    )
    trainer.resolve_update_state(model=model, optimizer=method.optimizer)
    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms={},
        optimizer=method.optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    method.optimizer.state[next(iter(model.parameters()))]["sentinel"] = torch.tensor(
        17.0, dtype=torch.float64
    )
    return save_checkpoint(
        output_dir=tmp_path / kind,
        next_iteration=trainer.next_iteration,
        completed_updates=trainer.completed_updates,
        model=model,
        optimizer=method.optimizer,
        trainer=trainer,
        sampler=sampler,
        context=_checkpoint_context(tmp_path),
        payload=TrainResume(),
    )


@pytest.mark.parametrize(
    ("kind", "mutate", "message"),
    [
        ("sr", lambda state: state["update_method"]["policy"].update(learning_rate=9.0), "SR checkpoint policy"),
        ("spring", lambda state: state["update_method"]["policy"].update(history_decay=0.9), "SPRING checkpoint policy"),
    ],
)
def test_train_resume_method_refusal_leaves_model_and_optimizer_bit_identical(
    kind: str, mutate: Callable[[dict[str, Any]], None], message: str, tmp_path: Path
) -> None:
    checkpoint = _checkpoint(tmp_path, kind)
    drift = _rewrite_trainer_json_and_digest(checkpoint, tmp_path / f"{kind}-drift", mutate)
    model, method, trainer, sampler = _build_checkpoint_run(
        kind, max_steps=2, seed=99, nonfinite_local_energy_policy="fail"
    )
    trainer.resolve_update_state(model=model, optimizer=method.optimizer)
    method.optimizer.state[next(iter(model.parameters()))]["sentinel"] = torch.tensor(
        -23.0, dtype=torch.float64
    )
    before = _snapshot(model, method.optimizer)
    with pytest.raises(ValueError, match=message):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(drift)},
            model=model,
            optimizer=method.optimizer,
            trainer=trainer,
            sampler=sampler,
            context=_checkpoint_context(tmp_path),
        )
    _assert_snapshot_unchanged(before, model, method.optimizer)


def test_train_resume_rejects_pre_857_sr_state_shape_with_valid_digest(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, "sr")

    def remove_squared_bound(state: dict[str, Any]) -> None:
        del state["update_method"]["policy"]["squared_update_norm_bound"]

    drift = _rewrite_trainer_json_and_digest(checkpoint, tmp_path / "sr-state-2-producer", remove_squared_bound)
    model, method, trainer, sampler = _build_checkpoint_run(
        "sr", max_steps=2, seed=99, nonfinite_local_energy_policy="fail"
    )
    trainer.resolve_update_state(model=model, optimizer=method.optimizer)
    method.optimizer.state[next(iter(model.parameters()))]["sentinel"] = torch.tensor(
        -23.0, dtype=torch.float64
    )
    before = _snapshot(model, method.optimizer)
    with pytest.raises(ValueError, match="SR checkpoint policy"):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(drift)},
            model=model,
            optimizer=method.optimizer,
            trainer=trainer,
            sampler=sampler,
            context=_checkpoint_context(tmp_path),
        )
    _assert_snapshot_unchanged(before, model, method.optimizer)


def test_train_resume_matching_policy_restores_method_state(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, "sr")
    model, method, trainer, sampler = _build_checkpoint_run(
        "sr", max_steps=2, seed=99, nonfinite_local_energy_policy="fail"
    )
    trainer.resolve_update_state(model=model, optimizer=method.optimizer)
    report = restore_checkpoint(
        load={"mode": "train_resume", "path": str(checkpoint)},
        model=model,
        optimizer=method.optimizer,
        trainer=trainer,
        sampler=sampler,
        context=_checkpoint_context(tmp_path),
    )
    assert report.loaded_trainer is True
    assert method.completed_updates == 1
    assert trainer.completed_updates == 1


@pytest.mark.parametrize("kind", ["sr", "spring"])
def test_json_method_validation_is_non_mutating_and_loader_reuses_it(kind: str) -> None:
    model, method, _, _ = _build_checkpoint_run(kind, max_steps=1, seed=0, nonfinite_local_energy_policy="fail")
    state = dict(method.method_state_dict())
    invalid = copy.deepcopy(state)
    invalid["policy"] = {"drift": True}
    before = method.method_state_dict()
    with pytest.raises(ValueError):
        method.validate_method_state_dict(invalid)
    assert method.method_state_dict() == before
    with pytest.raises(ValueError):
        method.load_method_state_dict(invalid)


@pytest.mark.parametrize("factory", [
    lambda parameter: NativeKFACUpdate(torch.optim.SGD([parameter], lr=0.03), model_parameters=ModelParameterBinding(parameters=(parameter,)), policy=KFACPolicy()),
    lambda parameter: BlockDiagonalNaturalGradientUpdate(torch.optim.SGD([parameter], lr=0.03), model_parameters=ModelParameterBinding(parameters=(parameter,)), policy=BlockNGPolicy(damping=0.1, learning_rate=0.03)),
])
def test_other_stateful_method_validation_is_non_mutating_and_loader_reuses_it(factory) -> None:
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    method = factory(parameter)
    invalid = copy.deepcopy(method.method_state_dict())
    invalid["policy"] = {"drift": True}
    before = method.method_state_dict()
    with pytest.raises(ValueError):
        method.validate_method_state_dict(invalid)
    assert method.method_state_dict() == before
    with pytest.raises(ValueError):
        method.load_method_state_dict(invalid)
