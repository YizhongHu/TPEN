"""Refusal qualification for dense Linear Method HI-shaped bindings.

These tests intentionally construct the real O-raw/O-composite model layouts,
but stop at the O(P) admission boundary.  They are not a dense-solve or HI
training test.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import tpen.sampling.metropolis as metropolis
import tpen.training.linear_method as linear_method
import tpen.training.trainer as trainer
from tpen.config_schema import ClosedSchemaError
from tpen.artifacts import ArtifactManager
from tpen.run import load_config, run_from_config
from tpen.training.linear_method import (
    LinearMethodMemoryLimitError,
    LinearMethodUpdate,
    dense_matrix_working_bytes,
    dense_step_working_bytes,
)
from tpen.training.update import ModelParameterBinding


ROOT = Path(__file__).resolve().parents[2]
HI_ROOT = ROOT / "experiments" / "atomistic" / "he-importance"
HI_CONFIG = HI_ROOT / "configs" / "train.yaml"


def _load_context_mapping() -> Any:
    """Load the experiment-owned composer without making it a package import."""

    path = HI_ROOT / "context_mapping.py"
    spec = importlib.util.spec_from_file_location("hi_context_mapping", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load context owner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_config(context: str) -> Any:
    """Return the committed model config for one HI O context."""

    cfg = load_config(str(HI_CONFIG))
    cfg.runtime.device = "cpu"
    cfg.runtime.dtype = "float64"
    if context == "O-composite":
        owner = _load_context_mapping()
        cfg.model = owner.compose_architecture_package(
            cfg.model,
            producer_path="hybrid/coordinate-neighbor",
            activation="channel-MLP,channel-MLP",
            channels="(32,32)",
            embedding="(128,1)",
            update="residual",
            producer_initialization=".50",
            electron_electron="1",
            electron_nucleus="(.01,1)",
            readout_initialization="equal",
            seed=0,
            package="B11",
        )
    return cfg


@pytest.fixture(scope="module")
def models() -> dict[str, torch.nn.Module]:
    """Build both real model bindings once, on CPU in float64."""

    built: dict[str, torch.nn.Module] = {}
    for context in ("O-raw", "O-composite"):
        cfg = _model_config(context)
        model = instantiate(cfg.model)
        model = model.to(device="cpu", dtype=torch.float64)
        model.eval()
        built[context] = model
    return built


@pytest.mark.parametrize(
    ("context", "expected_parameters"),
    [("O-raw", 961284), ("O-composite", 998630)],
)
def test_hi_models_have_authoritative_full_parameter_bindings(
    models: dict[str, torch.nn.Module], context: str, expected_parameters: int
) -> None:
    """The guard counts every named trainable slot, including inactive slots."""

    model = models[context]
    named = tuple((name, parameter) for name, parameter in model.named_parameters())
    binding = ModelParameterBinding.from_parameters(tuple(parameter for _, parameter in named))

    assert named
    assert all(parameter.requires_grad for _, parameter in named)
    assert binding.layout is not None
    assert binding.layout.total_numel == expected_parameters
    assert tuple(binding.parameters) == tuple(model.parameters())
    assert len(binding.parameters) == len(named)
    assert {name for name, _ in named}  # named coverage is explicit, not a synthetic vector.


@pytest.mark.parametrize(
    ("p_count", "expected_dense", "expected_step"),
    [
        (961284, 59140406478400, {2048: 59171905832512, 4096: 59203405186624}),
        (998630, 63824887946304, {2048: 63857611054144, 4096: 63890334161984}),
    ],
)
def test_hi_dense_byte_envelope_is_exact(
    p_count: int, expected_dense: int, expected_step: dict[int, int]
) -> None:
    """Public estimators match the reviewed integer envelope for Q and O1."""

    assert dense_matrix_working_bytes(p_count, torch.float64) == expected_dense
    for n_samples, expected in expected_step.items():
        assert dense_step_working_bytes(p_count, n_samples, torch.float64) == expected


@pytest.mark.parametrize("context", ["O-raw", "O-composite"])
def test_hi_constructor_refuses_before_prohibited_work(
    models: dict[str, torch.nn.Module], context: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real bindings fail at constructor preflight, before any tangent work."""

    model = models[context]
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))
    optimizer = torch.optim.SGD(binding.parameters, lr=1.0)
    with torch.no_grad():
        for index, parameter in enumerate(binding.parameters, start=1):
            parameter.grad = torch.full_like(parameter, 0.125 * index)
    parameter_values = tuple(parameter.detach().clone() for parameter in binding.parameters)
    parameter_grads = tuple(parameter.grad.detach().clone() for parameter in binding.parameters)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    optimizer_groups = tuple(
        {
            **{
                key: copy.deepcopy(value)
                for key, value in group.items()
                if key != "params"
            },
            "params": tuple(id(parameter) for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    )

    def fail_if_entered(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dense LM refusal entered prohibited work")

    for name in (
        "flatten_parameter_score_blocks",
        "flatten_local_energy_derivative_blocks",
        "assemble_dense_linear_method_matrices",
    ):
        monkeypatch.setattr(linear_method, name, fail_if_entered)
    monkeypatch.setattr(torch.linalg, "eig", fail_if_entered)
    monkeypatch.setattr(metropolis.MetropolisSampler, "collect_samples", fail_if_entered)
    monkeypatch.setattr(
        trainer._TrainerStepProducer,
        "produce_materialized_hamiltonian_tangent_step",
        fail_if_entered,
    )

    with pytest.raises(LinearMethodMemoryLimitError) as caught:
        LinearMethodUpdate(optimizer, model_parameters=binding)
    message = str(caught.value)
    assert type(caught.value) is LinearMethodMemoryLimitError
    assert f"P={binding.layout.total_numel}" in message
    assert "estimated working set" in message
    assert "1073741824" in message
    assert all(
        torch.equal(parameter.detach(), expected)
        for parameter, expected in zip(binding.parameters, parameter_values)
    )
    assert all(
        torch.equal(parameter.grad.detach(), expected)
        for parameter, expected in zip(binding.parameters, parameter_grads)
    )
    assert optimizer.state_dict() == optimizer_state
    assert optimizer_groups == tuple(
        {
            **{
                key: copy.deepcopy(value)
                for key, value in group.items()
                if key != "params"
            },
            "params": tuple(id(parameter) for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    )


def test_under_budget_control_constructs_without_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """A small real binding remains constructible under the default budget."""

    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    binding = ModelParameterBinding.from_parameters((parameter,))
    optimizer = torch.optim.SGD(binding.parameters, lr=1.0)
    method = LinearMethodUpdate(optimizer, model_parameters=binding)
    assert method.model_parameters.compare(binding)[0]


def test_hi_schema_refuses_current_linear_method_before_core_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The current HI-named config remains unavailable at its schema boundary."""

    cfg = load_config(str(HI_CONFIG))
    cfg.trainer.update_method = {
        "_target_": "tpen.training.linear_method.LinearMethodUpdate",
        "_partial_": True,
    }
    def fail_if_constructed(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("HI schema refusal entered Linear Method construction")

    monkeypatch.setattr(LinearMethodUpdate, "__init__", fail_if_constructed)
    with pytest.raises(ClosedSchemaError) as caught:
        run_from_config(cfg, config_path=str(HI_CONFIG), raise_exceptions=True)
    assert any(rejection.rule == "unadmitted-update-method" for rejection in caught.value.rejections)


def _qualification_config(tmp_path: Path) -> tuple[Any, Path]:
    """Write a reference-free local block-swap config outside the HI family."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = _model_config("O-raw")
    cfg.pop("schema", None)
    cfg.experiment.name = "linear_method_refusal_qualification"
    cfg.experiment.sector = "atomistic"
    cfg.run.root = str(tmp_path / "runs")
    cfg.run.dir = str(tmp_path / "run")
    cfg.run.run_id = "qualification"
    cfg.run.layout = "nested"
    cfg.runtime.device = "cpu"
    cfg.optimizer = {"_target_": "torch.optim.SGD", "_partial_": True, "lr": 1.0}
    cfg.trainer.max_steps = 1
    cfg.trainer.update_method = {
        "_target_": "tpen.training.linear_method.LinearMethodUpdate",
        "_partial_": True,
    }
    # Qualification is intentionally local: no HI schema, manifest, reference,
    # or HI-named config is used to reach the core Train -> VMCTrainer boundary.
    cfg.callbacks = []
    path = tmp_path / "qualification.yaml"
    OmegaConf.save(cfg, path)
    return load_config(str(path)), path


def _qualification_error_path(cfg: Any) -> Path:
    """Resolve the failure artifact through the production run layout."""

    manager = ArtifactManager(
        cfg.run.root,
        cfg.experiment.name,
        cfg.experiment.sector,
        cfg.run.run_id,
        layout=cfg.run.layout,
    )
    return manager.path("error.json")


def test_production_entrypoint_refuses_real_hi_shape_before_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both public run_from_config surfaces preserve the typed refusal."""

    cfg, path = _qualification_config(tmp_path)
    def fail_if_entered(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("production path entered prohibited work before refusal")

    monkeypatch.setattr(metropolis.MetropolisSampler, "collect_samples", fail_if_entered)
    monkeypatch.setattr(
        trainer._TrainerStepProducer,
        "produce_materialized_hamiltonian_tangent_step",
        fail_if_entered,
    )
    for name in (
        "flatten_parameter_score_blocks",
        "flatten_local_energy_derivative_blocks",
        "assemble_dense_linear_method_matrices",
    ):
        monkeypatch.setattr(linear_method, name, fail_if_entered)
    monkeypatch.setattr(torch.linalg, "eig", fail_if_entered)
    with pytest.raises(LinearMethodMemoryLimitError) as caught:
        run_from_config(cfg, config_path=str(path), raise_exceptions=True)
    assert type(caught.value) is LinearMethodMemoryLimitError
    assert "P=961284" in str(caught.value)

    cfg, path = _qualification_config(tmp_path / "return-code")
    assert run_from_config(cfg, config_path=str(path), raise_exceptions=False) == 1
    error = _qualification_error_path(cfg)
    assert error.exists()
    payload = json.loads(error.read_text(encoding="utf-8"))
    assert payload["exception_type"] == "LinearMethodMemoryLimitError"
