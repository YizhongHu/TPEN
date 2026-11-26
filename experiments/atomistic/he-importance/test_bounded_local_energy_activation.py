"""Focused regression tests for the canonical Stage-Q chunk-size route."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
from hydra.utils import instantiate
import pytest


stage_coordinate = importlib.import_module("experiments.atomistic.he-importance.stage_coordinate")
optimizer_grid = importlib.import_module("experiments.atomistic.he-importance.optimizer_grid")
architecture_stage_family = importlib.import_module(
    "experiments.atomistic.he-importance.architecture_stage_family"
)
train_config = importlib.import_module("experiments.atomistic.he-importance.train_config")
run_stage_q = importlib.import_module("experiments.atomistic.he-importance.run_stage_q")


def _cell(tmp_path: Path) -> object:
    cells = stage_coordinate.materialize_intended_configurations(
        tmp_path, availability=optimizer_grid.roster_availability(), stages=("Q",)
    )
    return next(
        cell
        for cell in cells
        if cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["seed_identity"]["label"] == 810001
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
    )


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "3"])
def test_invalid_chunk_bounds_refuse_before_source_access(value: object) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match="exact positive integer"):
        train_config.resolve_train_config(object(), local_energy_chunk_size=value)  # type: ignore[arg-type]


def test_resolver_binds_both_consumed_limbs_and_default_is_unbounded(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    bounded = train_config.resolve_train_config(cell, local_energy_chunk_size=3)
    assert bounded.trainer.local_energy_chunk_size == 3
    assert bounded.runner.trainer.local_energy_chunk_size == 3

    legacy = train_config.resolve_train_config(cell)
    assert train_config.OmegaConf.select(legacy, "trainer.local_energy_chunk_size") is None
    assert train_config.OmegaConf.select(legacy, "runner.trainer.local_energy_chunk_size") is None


def test_truthfulness_guard_rejects_consumed_limb_drift(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    config = train_config.resolve_train_config(cell, local_energy_chunk_size=3)
    config.runner.trainer.local_energy_chunk_size = None
    with pytest.raises(run_stage_q.StageQSelectionError, match="runner.trainer= None|runner.trainer=None"):
        run_stage_q._require_truthful_cell(cell, config, 3)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_trainer_override_has_the_same_strict_validation(tmp_path: Path, value: object) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match="exact positive integer"):
        train_config.resolve_train_config(
            _cell(tmp_path), overrides={"trainer.local_energy_chunk_size": value}
        )


def test_explicit_bound_cannot_conflict_with_trainer_override(tmp_path: Path) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match="conflicts"):
        train_config.resolve_train_config(
            _cell(tmp_path),
            local_energy_chunk_size=3,
            overrides={"trainer.local_energy_chunk_size": 2},
        )


def test_duplicate_dotlist_override_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(train_config.TrainConfigResolutionError, match="exact positive integer"):
        train_config.resolve_train_config(
            _cell(tmp_path),
            overrides=[
                "trainer.local_energy_chunk_size=3",
                "trainer.local_energy_chunk_size=0",
            ],
        )
    with pytest.raises(train_config.TrainConfigResolutionError, match="duplicate"):
        train_config.resolve_train_config(
            _cell(tmp_path / "valid"),
            overrides=[
                "trainer.local_energy_chunk_size=3",
                "trainer.local_energy_chunk_size=2",
            ],
        )


def _entrypoint_args(tmp_path: Path, bound: int | None) -> object:
    argv = [
        "--stage", "Q", "--context", "O-raw", "--optimizer", "adam",
        "--seed-label", "810001", "--walkers", "8", "--output-root", str(tmp_path),
    ]
    if bound is not None:
        argv.extend(["--local-energy-chunk-size", str(bound)])
    return run_stage_q.build_parser().parse_args(argv)


def _run_entrypoint_with_stubbed_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: int | None
) -> object:
    original_overrides = stage_coordinate.sampler_override_values
    monkeypatch.setattr(
        stage_coordinate,
        "sampler_override_values",
        lambda stage, coordinate: (8,)
        if stage == "Q" and coordinate == "n_walkers"
        else original_overrides(stage, coordinate),
    )
    captured: list[object] = []

    def fake_runner(config: object, *, topology: object, runtime_callbacks: object = ()) -> int:
        del topology, runtime_callbacks
        captured.append(config)
        path = Path(config.run.dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config), encoding="utf-8")
        return 0

    monkeypatch.setattr(
        run_stage_q,
        "_execution_facts",
        lambda: {
            "global_rank": 0, "global_size": 1, "local_rank": 0, "local_size": 1,
            "node_rank": 0, "node_size": 1, "host": "cpu-test", "pid": 1,
            "device": "cpu", "job_id": "cpu-test", "device_identity": None,
        },
    )
    monkeypatch.setattr(run_stage_q.launch, "run_from_config", fake_runner)
    assert run_stage_q.execute(_entrypoint_args(tmp_path, bound)) == 0
    assert len(captured) == 1
    return captured[0]


def test_real_stage_q_entrypoint_records_both_limbs_and_legacy_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bounded = _run_entrypoint_with_stubbed_runner(tmp_path / "bounded", monkeypatch, 3)
    assert bounded.trainer.local_energy_chunk_size == 3
    assert bounded.runner.trainer.local_energy_chunk_size == 3
    receipt = next((tmp_path / "bounded").glob("cells/*/selection_receipt.json"))
    report = json.loads(receipt.read_text(encoding="utf-8"))
    assert report["local_energy_chunking"] == {
        "activation_path": "stage_q_cli", "effective_chunk_size": 3
    }

    legacy = _run_entrypoint_with_stubbed_runner(tmp_path / "legacy", monkeypatch, None)
    assert OmegaConf.select(legacy, "trainer.local_energy_chunk_size") is None
    legacy_receipt = next((tmp_path / "legacy").glob("cells/*/selection_receipt.json"))
    legacy_report = json.loads(legacy_receipt.read_text(encoding="utf-8"))
    assert legacy_report["local_energy_chunking"] == {
        "activation_path": "legacy_default", "effective_chunk_size": None
    }


def test_family_stage_resolver_and_launch_adapter_forward_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_name = architecture_stage_family.family_stage_codes()[0]
    family_cells = architecture_stage_family.materialize_family_stage(stage_name, tmp_path)
    cell = next(
        candidate
        for candidate in family_cells
        if candidate.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
    )
    descriptor = architecture_stage_family.descriptor_for(stage_name)
    resolved = descriptor.resolver(cell, local_energy_chunk_size=3)
    assert resolved.trainer.local_energy_chunk_size == 3
    assert resolved.runner.trainer.local_energy_chunk_size == 3
    calls: list[object] = []
    launch = importlib.import_module("experiments.atomistic.he-importance.launch")
    monkeypatch.setattr(
        launch, "launch_train", lambda *args, **kwargs: calls.append(kwargs) or 0
    )
    assert descriptor.launch_adapter(cell, {}, local_energy_chunk_size=3) == 0
    assert calls[0]["local_energy_chunk_size"] == 3


def test_real_hydra_runner_constructs_and_executes_bounded_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu_config = _run_entrypoint_with_stubbed_runner(tmp_path / "runner", monkeypatch, 3)
    cpu_config.runtime.device = "cpu"
    cpu_config.runner.trainer.max_steps = 1
    cpu_config.runner.sampler.burn_in = 0
    cpu_config.runner.sampler.n_steps = 1
    torch = importlib.import_module("torch")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        constructed = instantiate(cpu_config.runner)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert constructed.trainer.local_energy_chunk_size == 3

    trainer_mod = sys.modules[type(constructed.trainer).__module__]
    helper_mod = sys.modules[trainer_mod.evaluate_local_energy_in_chunks.__module__]
    chunk_calls: list[int] = []
    row_sizes: list[int] = []
    local_calls: list[int] = []
    real_chunked = trainer_mod.evaluate_local_energy_in_chunks
    real_local = trainer_mod.local_energy
    real_slice = helper_mod.slice_flat_batch

    def spy_chunked(*args: object, **kwargs: object) -> object:
        chunk_calls.append(int(kwargs["chunk_size"]))
        return real_chunked(*args, **kwargs)

    def spy_local(terms: object, model: object, batch: object, **kwargs: object) -> object:
        local_calls.append(int(batch.batch_size))
        return real_local(terms, model, batch, **kwargs)

    def spy_slice(batch: object, start: int, stop: int) -> object:
        row_sizes.append(stop - start)
        return real_slice(batch, start, stop)

    monkeypatch.setattr(trainer_mod, "evaluate_local_energy_in_chunks", spy_chunked)
    monkeypatch.setattr(trainer_mod, "local_energy", spy_local)
    monkeypatch.setattr(helper_mod, "slice_flat_batch", spy_slice)
    context_helper = importlib.import_module("tests.helpers.run_context")
    context = context_helper.make_run_context(tmp_path / "context", device="cpu")
    walkers = constructed.sampler.initialize(8, device="cpu")
    producer = trainer_mod._TrainerStepProducer(
        trainer=constructed.trainer,
        model=constructed.model,
        hamiltonian_terms=constructed.hamiltonian_terms,
        batch=walkers.make_batch(),
        step=0,
        context=context,
        state=trainer_mod.TrainerState(
            model=constructed.model, trainer=constructed.trainer, sampler=constructed.sampler
        ),
    )
    producer._evaluate_detached_local_energy_with_output()
    assert chunk_calls == [3]
    assert tuple(row_sizes) == (3, 3, 2)
    assert local_calls == []


def test_real_hydra_runner_legacy_helper_is_single_unchunked_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_config = _run_entrypoint_with_stubbed_runner(tmp_path / "legacy-runner", monkeypatch, None)
    legacy_config.runtime.device = "cpu"
    torch = importlib.import_module("torch")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        constructed = instantiate(legacy_config.runner)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert constructed.trainer.local_energy_chunk_size is None

    trainer_mod = sys.modules[type(constructed.trainer).__module__]
    helper_mod = sys.modules[trainer_mod.evaluate_local_energy_in_chunks.__module__]
    chunk_calls: list[int] = []
    local_calls: list[int] = []
    real_chunked = trainer_mod.evaluate_local_energy_in_chunks
    real_local = trainer_mod.local_energy

    def spy_chunked(*args: object, **kwargs: object) -> object:
        chunk_calls.append(int(kwargs["chunk_size"]))
        return real_chunked(*args, **kwargs)

    def spy_local(terms: object, model: object, batch: object, **kwargs: object) -> object:
        local_calls.append(int(batch.batch_size))
        return real_local(terms, model, batch, **kwargs)

    monkeypatch.setattr(trainer_mod, "evaluate_local_energy_in_chunks", spy_chunked)
    monkeypatch.setattr(trainer_mod, "local_energy", spy_local)
    context_helper = importlib.import_module("tests.helpers.run_context")
    context = context_helper.make_run_context(tmp_path / "context", device="cpu")
    walkers = constructed.sampler.initialize(8, device="cpu")
    producer = trainer_mod._TrainerStepProducer(
        trainer=constructed.trainer,
        model=constructed.model,
        hamiltonian_terms=constructed.hamiltonian_terms,
        batch=walkers.make_batch(),
        step=0,
        context=context,
        state=trainer_mod.TrainerState(
            model=constructed.model, trainer=constructed.trainer, sampler=constructed.sampler
        ),
    )
    producer._evaluate_detached_local_energy_with_output()
    assert chunk_calls == []
    assert local_calls == [8]


def test_dropped_bound_is_refused_before_production_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_overrides = stage_coordinate.sampler_override_values
    monkeypatch.setattr(
        stage_coordinate,
        "sampler_override_values",
        lambda stage, coordinate: (8,)
        if stage == "Q" and coordinate == "n_walkers"
        else original_overrides(stage, coordinate),
    )
    facts = {
        "global_rank": 0, "global_size": 1, "local_rank": 0, "local_size": 1,
        "node_rank": 0, "node_size": 1, "host": "cpu-test", "pid": 1,
        "device": "cpu", "job_id": "cpu-test", "device_identity": None,
    }
    runner_calls: list[object] = []
    monkeypatch.setattr(run_stage_q, "_execution_facts", lambda: facts)

    def late_resolve(packet: object, _facts: object, *, runner: object, **kwargs: object) -> int:
        del _facts, kwargs
        dropped = train_config.resolve_train_config(packet)
        return runner(dropped, topology=facts)

    monkeypatch.setattr(run_stage_q.launch, "launch_train", late_resolve)
    monkeypatch.setattr(
        run_stage_q.launch, "run_from_config", lambda *args, **kwargs: runner_calls.append(args) or 0
    )
    with pytest.raises(run_stage_q.StageQSelectionError, match="local_energy_chunk_size"):
        run_stage_q.execute(_entrypoint_args(tmp_path, 3))
    assert not runner_calls


@pytest.mark.parametrize("argv", [["--local-energy-chunk-size", "0"], ["--local-energy-chunk-size", "-1"]])
def test_stage_q_invalid_cli_bounds_refuse_before_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    args = run_stage_q.build_parser().parse_args([*argv, "--output-root", str(tmp_path)])
    monkeypatch.setattr(run_stage_q, "select_cell", lambda *a, **k: pytest.fail("selection ran"))
    with pytest.raises(run_stage_q.StageQSelectionError, match="exact positive integer"):
        run_stage_q.execute(args)
    assert not tuple(tmp_path.iterdir())

