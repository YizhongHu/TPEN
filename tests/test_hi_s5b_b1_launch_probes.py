"""Reviewer probes for HI S5b-B1 continuation transport and refusal ordering."""

from __future__ import annotations

from importlib import import_module
import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf


stage_coordinate = import_module("experiments.atomistic.he-importance.stage_coordinate")
optimizer_grid = import_module("experiments.atomistic.he-importance.optimizer_grid")
launch = import_module("experiments.atomistic.he-importance.launch")


def _topology() -> dict[str, object]:
    return {
        "global_rank": 0,
        "global_size": 1,
        "local_rank": 0,
        "local_size": 1,
        "node_rank": 0,
        "node_size": 1,
        "host": "b1-probe-host",
        "pid": 1,
        "device": "cpu",
        "job_id": "b1-probe-job",
        "device_identity": None,
    }


def _o1_adam_rows(tmp_path: Path) -> dict[str, object]:
    cells = stage_coordinate.materialize_intended_configurations(
        tmp_path / "o1",
        availability=optimizer_grid.roster_availability(),
        stages=("O1",),
    )
    rows = {
        cell.manifest["scientific_identity"]["context"]["name"]: cell
        for cell in cells
        if cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
    }
    assert set(rows) >= {"O-raw", "O-composite"}
    return {name: rows[name] for name in ("O-raw", "O-composite")}


def _tree_snapshot(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("api", ["prepare", "launch"])
def test_restore_refusal_does_not_resolve_standard_override(
    tmp_path: Path, api: str
) -> None:
    """A non-None restore request must gate before ordinary interpolation."""

    cell = _o1_adam_rows(tmp_path)["O-raw"]
    predecessor = tmp_path / "predecessor"
    predecessor.mkdir()
    (predecessor / "state").write_bytes(b"opaque predecessor")
    before = _tree_snapshot(predecessor)
    resolver_calls: list[object] = []
    runner_calls: list[object] = []

    OmegaConf.register_new_resolver(
        "review_b1",
        lambda: resolver_calls.append(object()) or "float64",
        replace=True,
    )
    try:
        outcome: BaseException | None = None
        try:
            if api == "prepare":
                launch.prepare_train_launch(
                    cell,
                    _topology(),
                    overrides=["runtime.dtype=${review_b1:}"],
                    restore_request={"predecessor": predecessor},
                )
            else:
                launch.launch_train(
                    cell,
                    _topology(),
                    overrides=["runtime.dtype=${review_b1:}"],
                    restore_request={"predecessor": predecessor},
                    runner=lambda config, **kwargs: runner_calls.append((config, kwargs)) or 0,
                )
        except BaseException as error:  # assert resolver activity before refusal classification
            outcome = error
    finally:
        OmegaConf.clear_resolver("review_b1")

    assert resolver_calls == []
    assert isinstance(outcome, launch.HIContinuationRefusal)
    assert runner_calls == []
    assert _tree_snapshot(predecessor) == before


def test_real_o1_adam_rows_forward_opaque_operational_values(
    tmp_path: Path,
) -> None:
    """Real O1 producer rows preserve opaque values into an injected runner."""

    restore_request = object()
    artifact_destination = object()
    for context, cell in _o1_adam_rows(tmp_path).items():
        received: list[tuple[object, object, object]] = []

        def runner(config: object, **kwargs: object) -> int:
            received.append((config, kwargs["restore_request"], kwargs["artifact_destination"]))
            return 0

        plan = launch.prepare_train_launch(
            cell,
            _topology(),
            restore_request=restore_request,
            artifact_destination=artifact_destination,
        )
        assert plan.restore_request is restore_request, context
        assert plan.artifact_destination is artifact_destination, context
        assert launch.launch_train(
            cell,
            _topology(),
            runner=runner,
            restore_request=restore_request,
            artifact_destination=artifact_destination,
        ) == 0
        assert len(received) == 1, context
        assert received[0][1] is restore_request, context
        assert received[0][2] is artifact_destination, context


def test_launcher_admission_failure_is_typed_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launcher must surface the existing full-tree admission failure."""

    cell = _o1_adam_rows(tmp_path)["O-raw"]
    inventory = import_module(
        "experiments.atomistic.he-importance.mutable_state_inventory"
    )

    def refuse(*args: object, **kwargs: object) -> object:
        raise ValueError("synthetic admission mismatch")

    monkeypatch.setattr(inventory, "validate_hi_continuation_admission", refuse)
    with pytest.raises(launch.HIContinuationRefusal, match="synthetic admission mismatch"):
        launch.prepare_train_launch(cell, _topology(), restore_request=object())


@pytest.mark.parametrize("keyword", ["restore_request", "artifact_destination"])
@pytest.mark.parametrize("shape", ["positional-only", "args-only"])
def test_nonkeyword_runner_capabilities_refuse_without_call(
    tmp_path: Path, keyword: str, shape: str
) -> None:
    cell = _o1_adam_rows(tmp_path)["O-raw"]
    calls: list[object] = []

    if shape == "positional-only":
        if keyword == "restore_request":
            def candidate(config: object, restore_request: object = None, /) -> int:
                calls.append((config, restore_request))
                return 0
        else:
            def candidate(config: object, artifact_destination: object = None, /) -> int:
                calls.append((config, artifact_destination))
                return 0
    else:
        if keyword == "restore_request":
            def candidate(config: object, *restore_request: object) -> int:
                calls.append((config, restore_request))
                return 0
        else:
            def candidate(config: object, *artifact_destination: object) -> int:
                calls.append((config, artifact_destination))
                return 0

    with pytest.raises(launch.LaunchValidationError, match=keyword):
        launch.launch_train(
            cell,
            _topology(),
            runner=candidate,
            **{keyword: object()},
        )
    assert calls == []


@pytest.mark.parametrize("keyword", ["restore_request", "artifact_destination"])
def test_uninspectable_runner_refuses_each_operational_keyword(
    tmp_path: Path, keyword: str
) -> None:
    cell = _o1_adam_rows(tmp_path)["O-raw"]
    calls: list[object] = []

    class Uninspectable:
        @property
        def __signature__(self) -> inspect.Signature:
            raise ValueError("signature unavailable")

        def __call__(self, *args: object, **kwargs: object) -> int:
            calls.append(args)
            return 0

    with pytest.raises(launch.LaunchValidationError, match=keyword):
        launch.launch_train(
            cell,
            _topology(),
            runner=Uninspectable(),
            **{keyword: object()},
        )
    assert calls == []


def test_keyword_capable_runner_forwards_identity_and_fresh_config_only_succeeds(
    tmp_path: Path,
) -> None:
    cell = _o1_adam_rows(tmp_path)["O-raw"]
    restore_request = object()
    artifact_destination = object()
    received: list[dict[str, object]] = []

    def runner(config: object, *, restore_request: object, artifact_destination: object) -> int:
        received.append(
            {
                "config": config,
                "restore_request": restore_request,
                "artifact_destination": artifact_destination,
            }
        )
        return 0

    assert launch.launch_train(
        cell,
        _topology(),
        runner=runner,
        restore_request=restore_request,
        artifact_destination=artifact_destination,
    ) == 0
    assert received[0]["restore_request"] is restore_request
    assert received[0]["artifact_destination"] is artifact_destination

    fresh_calls: list[object] = []

    def config_only(config: object) -> int:
        fresh_calls.append(config)
        return 0

    assert launch.launch_train(cell, _topology(), runner=config_only) == 0
    assert len(fresh_calls) == 1


def test_default_runner_path_preserves_operational_kwargs_topology_and_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _o1_adam_rows(tmp_path)["O-raw"]
    restore_request = object()
    artifact_destination = object()
    runtime_callbacks = (object(),)
    received: list[dict[str, object]] = []

    def recording_default(
        config: object,
        *,
        topology: object,
        runtime_callbacks: object,
        restore_request: object,
        artifact_destination: object,
    ) -> int:
        received.append(
            {
                "config": config,
                "topology": topology,
                "runtime_callbacks": runtime_callbacks,
                "restore_request": restore_request,
                "artifact_destination": artifact_destination,
            }
        )
        return 0

    monkeypatch.setattr(launch, "run_from_config", recording_default)
    assert launch.launch_train(
        cell,
        _topology(),
        runner=recording_default,
        runtime_callbacks=runtime_callbacks,
        restore_request=restore_request,
        artifact_destination=artifact_destination,
    ) == 0
    assert received[0]["restore_request"] is restore_request
    assert received[0]["artifact_destination"] is artifact_destination
    assert received[0]["runtime_callbacks"] is runtime_callbacks
    assert received[0]["topology"] == _topology()
