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
    RestartCause,
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
    stale_claimed = Event()
    scans = Barrier(2)
    select_counts: dict[str, int] = {}
    scan_counts: dict[str, int] = {}
    driver_calls: list[str] = []
    errors: list[BaseException] = []
    stale_revalidated = Event()
    original_select = restart.select_restart
    original_active = restart._active_attempt_lease_paths
    original_claim = restart._claim_attempt

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

    def deterministic_claim(**kwargs):
        claimed = original_claim(**kwargs)
        if current_thread().name == "stale-contender":
            stale_claimed.set()
        elif current_thread().name == "current-contender":
            assert stale_claimed.wait(timeout=10)
        return claimed

    monkeypatch.setattr(restart, "select_restart", delayed_select)
    monkeypatch.setattr(restart, "_active_attempt_lease_paths", synchronized_scan)
    monkeypatch.setattr(restart, "_claim_attempt", deterministic_claim)

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
    assert stale_claimed.is_set()
    assert stale_revalidated.is_set()
    assert all(
        isinstance(error, RestartRefusal)
        and error.cause.value
        in {"lineage_claim_lost", "selection_stale_after_claim"}
        for error in errors
    ), [repr(error) for error in errors]


@pytest.mark.integration
def test_lineage_claim_collision_refuses_loser_without_orphan_lease(
    tmp_path: Path, monkeypatch,
) -> None:
    """A same-sequence collision refuses the loser and leaves no orphan lease."""

    from experiments.atomistic.stage1 import restart

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    cfg = _attempt_cfg(row, system)
    claim_gate = Barrier(2)
    lease_calls: list[Path] = []
    original_create = restart.create_json_receipt

    def synchronized_create(path: Path, payload: object) -> None:
        path = Path(path)
        is_lease = (
            path.parent == run_dir / "attempts"
            and path.name.startswith("attempt-")
            and path.name.endswith(".json")
            and not path.name.endswith(".outcome.json")
        )
        if is_lease:
            lease_calls.append(path)
            # Both contenders must compute attempt-000000 before either
            # exclusive create is allowed to run.  A retry-next mutant then
            # necessarily writes a second lease, which the assertions expose.
            if len(lease_calls) <= 2:
                claim_gate.wait(timeout=10)
        original_create(path, payload)

    monkeypatch.setattr(restart, "create_json_receipt", synchronized_create)
    errors: list[tuple[str, BaseException]] = []
    driver_calls: list[str] = []

    def contender(name: str) -> None:
        try:
            execute_restart(
                cfg=cfg, row=row, config_path=config_path,
                run=lambda *_args, **_kwargs: driver_calls.append(name) or 0,
                run_dir=run_dir,
            )
        except BaseException as exc:  # pragma: no cover - diagnostic propagation
            errors.append((name, exc))

    contenders = [
        Thread(target=contender, args=(name,), name=f"{name}-contender")
        for name in ("first", "second")
    ]
    for thread in contenders:
        thread.start()
    for thread in contenders:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in contenders)

    attempts = run_dir / "attempts"
    lease_files = sorted(
        path for path in attempts.glob("attempt-*.json")
        if not path.name.endswith(".outcome.json")
    )
    outcome_files = sorted(attempts.glob("attempt-*.outcome.json"))
    assert lease_calls == [attempts / "attempt-000000.json"] * 2
    assert [path.name for path in lease_files] == ["attempt-000000.json"]
    assert len(driver_calls) == 1
    assert len(errors) == 1
    loser_error = errors[0][1]
    assert isinstance(loser_error, RestartRefusal)
    assert loser_error.cause is RestartCause.LINEAGE_CLAIM_LOST

    # The winner may publish its ordinary parked outcome.  A refusal receipt,
    # if the implementation chooses to persist one, must still reference the
    # winning lease and carry a typed collision cause; it cannot create an
    # outcome for a non-existent retry lease.
    assert outcome_files
    winner = lease_files[0]
    for outcome_path in outcome_files:
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        assert outcome["lease"] == str(winner)
        assert outcome["attempt"] == winner.stem
        if outcome.get("cause") == RestartCause.LINEAGE_CLAIM_LOST.value:
            assert outcome["outcome"] == "parked"

    # Once the winner's outcome is durable, an ordinary restart must be able
    # to claim the next sequence; only an orphan lease would block it.
    successor_calls: list[bool] = []
    successor = execute_restart(
        cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
        run=lambda *_args, **_kwargs: successor_calls.append(True) or 0,
        run_dir=run_dir,
    )
    assert successor_calls == [True]
    assert successor.lease_path.name == "attempt-000001.json"


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
        superseded_attempt_keys=("initial.json",),
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
        superseded_attempt_keys=("initial.json",),
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
def test_reused_takeover_approval_refuses_while_first_driver_is_live(
    tmp_path: Path,
) -> None:
    """One approved takeover cannot admit a concurrent replacement driver."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    attempts = run_dir / "attempts"

    # Create the predecessor publication and a real immutable orphan lease by
    # letting the initial owner terminate outside execute_restart's Exception
    # handler.  This is the crash state that requires an approved takeover.
    with pytest.raises(SystemExit):
        execute_restart(
            cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
            run=lambda _cfg, **_kwargs: (
                _publish(run_dir, step=1000) or (_ for _ in ()).throw(SystemExit(17))
            ),
            run_dir=run_dir,
        )
    assert (attempts / "attempt-000000.json").is_file()
    assert not (attempts / "attempt-000000.outcome.json").exists()

    selected = select_restart(
        cfg=_attempt_cfg(row, system), row=row, run_dir=run_dir,
    )
    assert selected.predecessor is not None
    approval = RestartApproval(
        predecessor_content_id=selected.predecessor.content_id,
        terminal_pbs_proof="pbs-terminal-approval",
        quiescent=True,
        adopted_budget="budget-replacement-1",
        lineage_id=row["row_id"],
        config_identity=row["config_sha256"],
        overlay_identity="none",
        target_iterations=row["updates"],
        superseded_attempt_keys=("attempt-000000.json",),
    )

    first_entered = Event()
    release_first = Event()
    first_errors: list[BaseException] = []
    first_results: list[object] = []
    second_calls: list[object] = []

    def first_driver(_cfg, **_kwargs):
        first_entered.set()
        assert release_first.wait(timeout=10), "first driver was not released"
        _publish(run_dir, step=2000)
        (run_dir / "status.json").write_text(
            json.dumps({"status": "planned_yield"}) + "\n", encoding="utf-8"
        )
        return 0

    def first_attempt() -> None:
        try:
            first_results.append(execute_restart(
                cfg=_attempt_cfg(row, system), row=row,
                config_path=config_path, run=first_driver,
                run_dir=run_dir, takeover=approval,
            ))
        except BaseException as exc:  # pragma: no cover - diagnostic propagation
            first_errors.append(exc)

    first = Thread(target=first_attempt, name="approved-takeover-owner")
    first.start()
    assert first_entered.wait(timeout=10), "first approved driver did not start"

    try:
        second_error: BaseException | None = None
        try:
            execute_restart(
                cfg=_attempt_cfg(row, system), row=row,
                config_path=config_path,
                run=lambda *_args, **_kwargs: second_calls.append(True) or 0,
                run_dir=run_dir, takeover=approval,
            )
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            second_error = exc
        assert isinstance(second_error, RestartRefusal), repr(second_error)
        assert "takeover" in str(second_error) or "occupied" in str(second_error)
        assert second_calls == [], (
            "reused approval entered a second driver; first_alive="
            f"{first.is_alive()} first_errors={[repr(error) for error in first_errors]}"
        )
        assert first.is_alive(), (
            "first takeover stopped before the duplicate approval was checked; "
            f"results={first_results} errors={[repr(error) for error in first_errors]}"
        )
    finally:
        release_first.set()
        first.join(timeout=10)

    assert not first.is_alive(), (
        "first takeover thread did not finish; "
        f"results={first_results} errors={[repr(error) for error in first_errors]}"
    )
    assert first_errors == []
    assert len(first_results) == 1


@pytest.mark.integration
def test_takeover_approval_scope_revalidated_after_claim(tmp_path: Path, monkeypatch) -> None:
    """A concurrent reuse is refused even after the first replacement finishes."""

    from experiments.atomistic.stage1 import restart

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    with pytest.raises(SystemExit):
        execute_restart(
            cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
            run=lambda _cfg, **_kwargs: (
                _publish(run_dir, step=1000) or (_ for _ in ()).throw(SystemExit(17))
            ),
            run_dir=run_dir,
        )

    selected = select_restart(cfg=_attempt_cfg(row, system), row=row, run_dir=run_dir)
    assert selected.predecessor is not None
    approval = RestartApproval(
        predecessor_content_id=selected.predecessor.content_id,
        terminal_pbs_proof="pbs-terminal-approval",
        quiescent=True,
        adopted_budget="budget-replacement-1",
        lineage_id=row["row_id"],
        config_identity=row["config_sha256"],
        overlay_identity="none",
        target_iterations=row["updates"],
        superseded_attempt_keys=("attempt-000000.json",),
    )
    second_admitted = Event()
    release_second_claim = Event()
    first_errors: list[BaseException] = []
    first_results: list[object] = []
    second_calls: list[bool] = []
    second_errors: list[BaseException] = []
    original_claim = restart._claim_attempt

    def second_driver(_cfg, **_kwargs):
        second_calls.append(True)
        return 0

    def second_attempt() -> None:
        try:
            execute_restart(
                cfg=_attempt_cfg(row, system), row=row,
                config_path=config_path, run=second_driver,
                run_dir=run_dir, takeover=approval,
            )
        except BaseException as exc:  # pragma: no cover - diagnostic propagation
            second_errors.append(exc)

    def interleaved_claim(**kwargs):
        if current_thread().name == "second-takeover":
            second_admitted.set()
            assert release_second_claim.wait(timeout=10)
        return original_claim(**kwargs)

    monkeypatch.setattr(restart, "_claim_attempt", interleaved_claim)

    second = Thread(target=second_attempt, name="second-takeover")
    second.start()
    assert second_admitted.wait(timeout=10), "second takeover did not reach its claim gate"

    def first_driver(_cfg, **_kwargs):
        return 0

    def first_attempt() -> None:
        try:
            first_results.append(execute_restart(
                cfg=_attempt_cfg(row, system), row=row,
                config_path=config_path, run=first_driver,
                run_dir=run_dir, takeover=approval,
            ))
        except BaseException as exc:  # pragma: no cover - diagnostic propagation
            first_errors.append(exc)
        finally:
            release_second_claim.set()

    first = Thread(target=first_attempt, name="first-takeover")
    first.start()
    first.join(timeout=10)
    second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    assert first_errors == []
    assert len(first_results) == 1
    assert second_calls == []
    assert len(second_errors) == 1
    assert isinstance(second_errors[0], RestartRefusal)
    assert second_errors[0].cause is RestartCause.TAKEOVER_APPROVAL_SCOPE_MISMATCH


@pytest.mark.integration
def test_crashed_replacement_requires_fresh_approval_scope(tmp_path: Path) -> None:
    """A crashed replacement must be named by its next approval."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    with pytest.raises(SystemExit):
        execute_restart(
            cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
            run=lambda _cfg, **_kwargs: (
                _publish(run_dir, step=1000) or (_ for _ in ()).throw(SystemExit(17))
            ),
            run_dir=run_dir,
        )

    selected = select_restart(cfg=_attempt_cfg(row, system), row=row, run_dir=run_dir)
    assert selected.predecessor is not None
    approval = RestartApproval(
        predecessor_content_id=selected.predecessor.content_id,
        terminal_pbs_proof="pbs-terminal-approval",
        quiescent=True,
        adopted_budget="budget-replacement-1",
        lineage_id=row["row_id"],
        config_identity=row["config_sha256"],
        overlay_identity="none",
        target_iterations=row["updates"],
        superseded_attempt_keys=("attempt-000000.json",),
    )
    with pytest.raises(SystemExit):
        execute_restart(
            cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
            run=lambda _cfg, **_kwargs: (
                _publish(run_dir, step=2000) or (_ for _ in ()).throw(SystemExit(18))
            ),
            run_dir=run_dir, takeover=approval,
        )

    stale_calls: list[bool] = []
    with pytest.raises(RestartRefusal) as stale_error:
        execute_restart(
            cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
            run=lambda *_args, **_kwargs: stale_calls.append(True) or 0,
            run_dir=run_dir, takeover=approval,
        )
    assert "predecessor_content_id" in str(stale_error.value)
    assert stale_calls == []
    assert not (run_dir / "attempts" / "attempt-000002.json").exists()

    replacement = select_restart(
        cfg=_attempt_cfg(row, system), row=row, run_dir=run_dir,
    )
    assert replacement.predecessor is not None
    fresh_approval = RestartApproval(
        predecessor_content_id=replacement.predecessor.content_id,
        terminal_pbs_proof="pbs-terminal-approval-2",
        quiescent=True,
        adopted_budget="budget-replacement-2",
        lineage_id=row["row_id"],
        config_identity=row["config_sha256"],
        overlay_identity="none",
        target_iterations=row["updates"],
        superseded_attempt_keys=("attempt-000001.json",),
    )
    fresh_calls: list[bool] = []
    result = execute_restart(
        cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
        run=lambda _cfg, **_kwargs: (
            fresh_calls.append(True)
            or _publish(run_dir, step=3000)
            or (run_dir / "status.json").write_text(
                json.dumps({"status": "planned_yield"}) + "\n", encoding="utf-8"
            )
            or 0
        ),
        run_dir=run_dir, takeover=fresh_approval,
    )
    assert fresh_calls == [True]
    assert result.outcome == "planned_yield"


@pytest.mark.integration
def test_crashed_unpublished_replacement_requires_fresh_approval_scope(
    tmp_path: Path,
) -> None:
    """A crashed replacement without a publication needs a fresh lease approval."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    with pytest.raises(SystemExit):
        execute_restart(
            cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
            run=lambda _cfg, **_kwargs: (
                _publish(run_dir, step=1000) or (_ for _ in ()).throw(SystemExit(17))
            ),
            run_dir=run_dir,
        )

    selected = select_restart(cfg=_attempt_cfg(row, system), row=row, run_dir=run_dir)
    assert selected.predecessor is not None
    approval = RestartApproval(
        predecessor_content_id=selected.predecessor.content_id,
        terminal_pbs_proof="pbs-terminal-approval",
        quiescent=True,
        adopted_budget="budget-replacement-1",
        lineage_id=row["row_id"],
        config_identity=row["config_sha256"],
        overlay_identity="none",
        target_iterations=row["updates"],
        superseded_attempt_keys=("attempt-000000.json",),
    )
    with pytest.raises(SystemExit):
        execute_restart(
            cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
            run=lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit(18)),
            run_dir=run_dir, takeover=approval,
        )

    attempts = run_dir / "attempts"
    assert (attempts / "attempt-000001.json").is_file()
    assert not (attempts / "attempt-000001.outcome.json").exists()
    stale_calls: list[bool] = []
    with pytest.raises(RestartRefusal) as stale_error:
        execute_restart(
            cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
            run=lambda *_args, **_kwargs: stale_calls.append(True) or 0,
            run_dir=run_dir, takeover=approval,
        )
    assert stale_error.value.cause is RestartCause.TAKEOVER_APPROVAL_SCOPE_MISMATCH
    assert stale_calls == []
    assert not (attempts / "attempt-000002.json").exists()

    fresh_approval = RestartApproval(
        predecessor_content_id=selected.predecessor.content_id,
        terminal_pbs_proof="pbs-terminal-approval-2",
        quiescent=True,
        adopted_budget="budget-replacement-2",
        lineage_id=row["row_id"],
        config_identity=row["config_sha256"],
        overlay_identity="none",
        target_iterations=row["updates"],
        superseded_attempt_keys=("attempt-000001.json",),
    )
    fresh_calls: list[bool] = []
    result = execute_restart(
        cfg=_attempt_cfg(row, system), row=row, config_path=config_path,
        run=lambda _cfg, **_kwargs: (
            fresh_calls.append(True)
            or _publish(run_dir, step=2000)
            or (run_dir / "status.json").write_text(
                json.dumps({"status": "planned_yield"}) + "\n", encoding="utf-8"
            )
            or 0
        ),
        run_dir=run_dir, takeover=fresh_approval,
    )
    assert fresh_calls == [True]
    assert result.outcome == "planned_yield"


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
