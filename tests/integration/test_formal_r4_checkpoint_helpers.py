"""R4 checkpoint staging controls for the CR native adapter."""

from __future__ import annotations

from pathlib import Path

from tests.integration.test_native_checkpoint_recovery import _final_checkpoint, _wait_for_checkpoint


def test_r4_checkpoint_helpers_ignore_staged_tmp_checkpoint(tmp_path: Path) -> None:
    """Staged temporary state must not displace a committed checkpoint."""

    outputs = tmp_path / "outputs"
    run_dir = outputs / "run-1"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    committed = checkpoints / "step_000001"
    committed.mkdir()
    (committed / "manifest.json").write_text("{}\n", encoding="utf-8")
    (committed / "COMPLETE").write_text("complete\n", encoding="utf-8")
    staged = checkpoints / "step_000001.tmp"
    staged.mkdir()
    (staged / "manifest.json").write_text("{}\n", encoding="utf-8")
    (staged / "COMPLETE").write_text("complete\n", encoding="utf-8")
    (run_dir / "status.json").write_text("{}\n", encoding="utf-8")

    assert _wait_for_checkpoint(outputs, target=2) == run_dir
    assert _final_checkpoint(run_dir) == committed
