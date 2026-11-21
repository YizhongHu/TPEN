"""Reviewer tests for Stage-1 restart lease and publication boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Barrier, Event, Thread, current_thread
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1.restart import (
    RestartApproval,
    RestartRefusal,
    execute_restart,
    select_restart,
)
from experiments.atomistic.stage1.protocol import STAGE1_PROTOCOL
from experiments.toolkit.run_paths import resolve_run_dir
from tpen.checkpoint import TrainResume, save_checkpoint

from tests.integration.test_stage1_restartable import (
    CONFIG_ROOT,
    _production_cfg,
    _row,
)


def _publish(run_dir: Path, *, step: int = 1000, run_id: str = "stage1-chain") -> None:
    """Publish a real full-state checkpoint with controlled provenance."""

    import torch

    model = torch.nn.Linear(1, 1, bias=False)
    context = SimpleNamespace(
        cfg=OmegaConf.create({"trainer": {"max_steps": STAGE1_PROTOCOL.lineage_updates}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id=run_id, extra={}),
        run_dir=run_dir,
    )
    save_checkpoint(
        output_dir=run_dir / "checkpoints",
        next_iteration=step,
        completed_updates=step,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        trainer=SimpleNamespace(
            state_dict=lambda: {"next_iteration": step, "completed_updates": step}
        ),
        sampler=SimpleNamespace(mcmc_state_dict=lambda: {"walker": step}),
        payload=TrainResume(),
        context=context,
    )


def _attempt_cfg(row: dict[str, object], system: str):
    return _production_cfg(row, CONFIG_ROOT / system / "train.yaml")


@pytest.mark.integration
def test_inflight_initial_lease_blocks_second_driver_after_publication(tmp_path: Path) -> None:
    """A publication during an admitted initial attempt does not admit a peer."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    first_cfg = _attempt_cfg(row, system)
    second_cfg = _attempt_cfg(row, system)
    published = Event()
    release = Event()
    second_calls: list[object] = []
    first_errors: list[BaseException] = []

    def first_driver(_cfg, **_kwargs):
        _publish(run_dir)
        published.set()
        assert release.wait(timeout=10)
        (run_dir / "status.json").write_text(
            json.dumps({"status": "planned_yield"}) + "\n", encoding="utf-8"
        )
        return 7

    def first_attempt() -> None:
        try:
            execute_restart(
                cfg=first_cfg, row=row, config_path=config_path,
                run=first_driver, run_dir=run_dir,
            )
        except BaseException as exc:  # pragma: no cover - diagnostic propagation
            first_errors.append(exc)

    thread = Thread(target=first_attempt)
    thread.start()
    assert published.wait(timeout=10)

    def second_driver(_cfg, **_kwargs):
        second_calls.append(True)
        return 0

    try:
        with pytest.raises(RestartRefusal, match="occupied predecessor lease"):
            execute_restart(
                cfg=second_cfg, row=row, config_path=config_path,
                run=second_driver, run_dir=run_dir,
            )
        assert second_calls == []
    finally:
        release.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert first_errors == []


@pytest.mark.integration
def test_lineage_claim_serializes_stale_and_current_selections(
    tmp_path: Path, monkeypatch,
) -> None:
    """A stale selection and a current selection cannot both enter the driver."""

    from experiments.atomistic.stage1 import restart

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    first_cfg = _attempt_cfg(row, system)
    second_cfg = _attempt_cfg(row, system)

    publisher_at_1000 = Event()
    release_publisher = Event()
    first_selected = Event()
    release_first = Event()
    scans = Barrier(2)
    select_counts: dict[str, int] = {}
    scan_counts: dict[str, int] = {}
    driver_calls: list[str] = []
    errors: list[BaseException] = []
    stale_revalidated = Event()
    original_select = restart.select_restart
    original_active = restart._active_attempt_lease_paths

    def delayed_select(**kwargs):
        selected = original_select(**kwargs)
        name = current_thread().name
        select_counts[name] = select_counts.get(name, 0) + 1
        if name == "stale-contender" and select_counts[name] == 1:
            assert selected.predecessor is not None
            assert selected.predecessor.next_iteration == 1000
            first_selected.set()
            assert release_first.wait(timeout=10)
        elif name == "stale-contender":
            assert selected.predecessor is not None
            assert selected.predecessor.next_iteration == 2000
            stale_revalidated.set()
        return selected

    def synchronized_scan(attempts: Path):
        paths = original_active(attempts)
        name = current_thread().name
        scan_counts[name] = scan_counts.get(name, 0) + 1
        if name in {"stale-contender", "current-contender"} and scan_counts[name] == 1:
            scans.wait(timeout=10)
        return paths

    monkeypatch.setattr(restart, "select_restart", delayed_select)
    monkeypatch.setattr(restart, "_active_attempt_lease_paths", synchronized_scan)

    def publisher_driver(_cfg, **_kwargs):
        _publish(run_dir, step=1000)
        publisher_at_1000.set()
        assert release_publisher.wait(timeout=10)
        _publish(run_dir, step=2000)
        (run_dir / "status.json").write_text(
            json.dumps({"status": "planned_yield"}) + "\n", encoding="utf-8"
        )
        return 0

    def contender(name: str, cfg: object) -> None:
        try:
            execute_restart(
                cfg=cfg, row=row, config_path=config_path,
                run=lambda *_args, **_kwargs: driver_calls.append(name) or 0,
                run_dir=run_dir,
            )
        except BaseException as exc:  # pragma: no cover - diagnostic propagation
            errors.append(exc)

    publisher = Thread(
        target=lambda: execute_restart(
            cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
            run=publisher_driver, run_dir=run_dir,
        ),
        name="publisher",
    )
    publisher.start()
    assert publisher_at_1000.wait(timeout=10)

    stale = Thread(target=contender, args=("stale", first_cfg), name="stale-contender")
    stale.start()
    assert first_selected.wait(timeout=10)

    # The publisher completes after the first contender selected step 1000.
    release_publisher.set()
    publisher.join(timeout=10)
    assert not publisher.is_alive()
    current = Thread(target=contender, args=("current", second_cfg), name="current-contender")
    current.start()
    release_first.set()
    stale.join(timeout=10)
    current.join(timeout=10)
    assert not stale.is_alive() and not current.is_alive()
    assert len(driver_calls) <= 1
    assert stale_revalidated.is_set()
    assert all(
        isinstance(error, RestartRefusal)
        and error.cause.value
        in {"lineage_claim_lost", "selection_stale_after_claim"}
        for error in errors
    ), [repr(error) for error in errors]


@pytest.mark.integration
def test_invalid_initial_takeover_approval_refuses_before_takeover_lease(tmp_path: Path) -> None:
    """An occupied fresh lease still requires a complete takeover approval."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    attempts = run_dir / "attempts"
    attempts.mkdir(parents=True)
    (attempts / "initial.json").write_text("{}\n", encoding="utf-8")
    cfg = _attempt_cfg(row, system)
    calls: list[object] = []
    approval = RestartApproval(
        predecessor_content_id="",
        terminal_pbs_proof="",
        quiescent=False,
        adopted_budget="",
        lineage_id="wrong-lineage",
        config_identity="wrong-config",
        overlay_identity="wrong-overlay",
        target_iterations=-1,
    )

    with pytest.raises(RestartRefusal):
        execute_restart(
            cfg=cfg, row=row, config_path=config_path,
            run=lambda *_args, **_kwargs: calls.append(True) or 0,
            run_dir=run_dir, takeover=approval,
        )
    assert calls == []
    assert not (attempts / "initial.takeover.json").exists()


@pytest.mark.integration
def test_approved_takeover_releases_abandoned_initial_owner_for_successor(
    tmp_path: Path,
) -> None:
    """A durable takeover outcome must quiesce the abandoned initial owner."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    attempts = run_dir / "attempts"
    attempts.mkdir(parents=True)
    # Simulate a crashed initial owner: its lease is present, but no outcome
    # receipt was published before the successor checkpoint became visible.
    (attempts / "initial.json").write_text("{}\n", encoding="utf-8")
    cfg = _attempt_cfg(row, system)
    _publish(run_dir, step=1000)
    selected = select_restart(cfg=cfg, row=row, run_dir=run_dir)
    assert selected.decision.value == "restore"
    assert selected.predecessor is not None
    config_identity = row["config_sha256"]
    approval = RestartApproval(
        predecessor_content_id=selected.predecessor.content_id,
        terminal_pbs_proof="pbs-terminal-approval",
        quiescent=True,
        adopted_budget="budget-replacement-1",
        lineage_id=row["row_id"],
        config_identity=config_identity,
        overlay_identity="none",
        target_iterations=row["updates"],
    )
    takeover_calls: list[object] = []

    def takeover_driver(_cfg, **_kwargs):
        takeover_calls.append(True)
        _publish(run_dir, step=2000)
        (run_dir / "status.json").write_text(
            json.dumps({"status": "planned_yield"}) + "\n", encoding="utf-8"
        )
        return 0

    takeover_result = execute_restart(
        cfg=cfg, row=row, config_path=config_path,
        run=takeover_driver, run_dir=run_dir, takeover=approval,
    )
    assert takeover_calls == [True]
    assert takeover_result.outcome == "planned_yield"
    assert takeover_result.outcome_path.is_file()
    assert json.loads(takeover_result.outcome_path.read_text())[
        "outcome"
    ] == "planned_yield"

    successor_calls: list[object] = []

    def successor_driver(_cfg, **_kwargs):
        successor_calls.append(True)
        _publish(run_dir, step=3000)
        (run_dir / "status.json").write_text(
            json.dumps({"status": "planned_yield"}) + "\n", encoding="utf-8"
        )
        return 0

    successor_result = execute_restart(
        cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
        run=successor_driver, run_dir=run_dir,
    )
    assert successor_calls == [True]
    assert successor_result.outcome == "planned_yield"


@pytest.mark.integration
def test_fresh_positional_train_resume_is_refused_before_driver_or_lease(
    tmp_path: Path,
) -> None:
    """A concrete Train sixth positional load is still a configured restore."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    cfg = _attempt_cfg(row, system)
    OmegaConf.update(
        cfg, "load", {"mode": "train_resume", "path": str(tmp_path / "foreign")},
        merge=False, force_add=True,
    )
    cfg.runner = {
        "_target_": "tpen.runner.Train",
        "_args_": [
            "${model}", "${sampler}", "${hamiltonian_terms}",
            "${optimizer}", "${trainer}", "${load}",
        ],
    }
    calls: list[object] = []
    try:
        result = execute_restart(
            cfg=cfg, row=row, config_path=config_path,
            run=lambda *_args, **_kwargs: calls.append(True) or 0,
            run_dir=run_dir,
        )
    except RestartRefusal as exc:
        assert "restore" in str(exc) or "runner.load" in str(exc)
        result = None
    else:
        assert result.outcome == "parked"
    assert calls == []
    assert not (run_dir / "attempts").exists()


@pytest.mark.integration
@pytest.mark.parametrize(
    "configured_load",
    [pytest.param(None, id="missing"), {}, {"mode": "none", "path": None}],
)
def test_inactive_named_load_does_not_block_fresh_driver(
    tmp_path: Path, configured_load: object,
) -> None:
    """Inactive named load forms are not concrete typed restore requests."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    cfg = _attempt_cfg(row, system)
    if configured_load is not None:
        OmegaConf.update(
            cfg, "runner.load", configured_load, merge=False, force_add=True,
        )
    calls: list[object] = []
    result = execute_restart(
        cfg=cfg, row=row, config_path=config_path,
        run=lambda *_args, **_kwargs: calls.append(True) or 0,
        run_dir=run_dir,
    )
    assert calls == [True]
    assert result.outcome == "parked"
    assert result.outcome_path.is_file()


@pytest.mark.integration
def test_fresh_attempt_does_not_forward_raw_runner_load(tmp_path: Path) -> None:
    """Fresh admission must not let configured load state bypass typed restore."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    cfg = _attempt_cfg(row, system)
    OmegaConf.update(
        cfg, "runner.load", {"mode": "train_resume", "path": "/foreign/checkpoint"},
        merge=False, force_add=True,
    )
    calls: list[object] = []

    try:
        result = execute_restart(
            cfg=cfg, row=row, config_path=config_path,
            run=lambda run_cfg, **_kwargs: calls.append(OmegaConf.select(run_cfg, "runner.load")) or 0,
            run_dir=run_dir,
        )
    except RestartRefusal as exc:
        assert "runner.load" in str(exc)
        result = None
    assert calls == []
    assert result is None or result.outcome == "parked"
    assert not (run_dir / "attempts").exists()


@pytest.mark.integration
def test_failed_terminal_status_cannot_be_reported_completed(tmp_path: Path) -> None:
    """A terminal checkpoint does not erase a durable failed status."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    _publish(run_dir, step=STAGE1_PROTOCOL.lineage_updates)
    (run_dir / "status.json").write_text(
        json.dumps({"status": "failed"}) + "\n", encoding="utf-8"
    )
    cfg = _attempt_cfg(row, system)
    calls: list[object] = []

    result = execute_restart(
        cfg=cfg, row=row, config_path=config_path,
        run=lambda *_args, **_kwargs: calls.append(True) or 0,
        run_dir=run_dir,
    )
    assert calls == []
    assert result.outcome != "completed"
