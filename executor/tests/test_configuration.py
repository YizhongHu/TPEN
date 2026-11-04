"""Acceptance tests for unresolved Hydra program capture."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from tpen_executor.configuration import StagedHydra
from tpen_executor.programs import WorkerProgram

from helpers import staged_targets


def _write_config(directory: Path, label: str = "base") -> None:
    """Write a temporary config with an absent submitter-side interpolation."""

    (directory / "config.yaml").write_text(
        f"""program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  process:
    _target_: helpers.staged_targets.CountedAllocation
    _recursive_: false
    worker:
      _target_: tpen_executor.programs.WorkerProgram
      _recursive_: false
      runner:
        _target_: helpers.staged_targets.CountedWorker
        _recursive_: false
        command: ${{remote_only.value}}
  allocation_label: {label}
outside: base
""",
        encoding="utf-8",
    )


def _staged(directory: Path, *overrides: str):
    """Compose a temporary config through Hydra's config-dir API."""

    return StagedHydra(config_module="", config_dir=str(directory)).compose_config("config", overrides)


def test_capture_preserves_absent_remote_interpolation(tmp_path: Path) -> None:
    """A missing submitter variable remains literal in the captured spec."""

    _write_config(tmp_path)
    staged = _staged(tmp_path)
    raw = staged.program.spec.config_ref  # type: ignore[union-attr]
    assert raw["process"]["worker"]["runner"]["command"] == "${remote_only.value}"
    worker = staged.programs["program.process.worker"]
    assert worker.spec.config_ref["runner"]["command"] == "${remote_only.value}"


def test_capture_does_not_construct_and_materialize_constructs_once(tmp_path: Path) -> None:
    """Nested worker descriptions stay opaque until process-local materialization."""

    _write_config(tmp_path)
    staged_targets.reset()
    staged = _staged(tmp_path)
    assert staged_targets.CONSTRUCTIONS == 0

    result = staged.program.materialize({"remote_only": {"value": "worker-value"}})  # type: ignore[union-attr]
    assert staged_targets.CONSTRUCTIONS == 1
    assert isinstance(result.process.worker, WorkerProgram)
    worker = staged.programs["program.process.worker"]
    assert result["process"].worker.spec.content_digest == worker.spec.content_digest

    runner = worker.materialize({"remote_only": {"value": "worker-value"}})
    assert staged_targets.CONSTRUCTIONS == 2
    assert runner.runner.command == "worker-value"


def test_program_digest_is_stable_local_and_sensitive_only_to_program_changes(tmp_path: Path) -> None:
    """Program digests are stable and ignore unrelated root configuration."""

    _write_config(tmp_path)
    first = _staged(tmp_path).program.spec.content_digest  # type: ignore[union-attr]
    second = _staged(tmp_path).program.spec.content_digest  # type: ignore[union-attr]
    outside = _staged(tmp_path, "outside=changed").program.spec.content_digest  # type: ignore[union-attr]
    changed = _staged(tmp_path, "program.allocation_label=changed").program.spec.content_digest  # type: ignore[union-attr]
    assert first == second == outside
    assert changed != first


def test_non_program_subtree_named_program_is_not_captured(tmp_path: Path) -> None:
    """A same-named subtree is not staged without a recognized target."""

    (tmp_path / "config.yaml").write_text(
        """program:
  _target_: helpers.staged_targets.CountedAllocation
  _recursive_: false
  command: literal
outside: base
""",
        encoding="utf-8",
    )
    staged = _staged(tmp_path)
    assert staged.program is None
    assert staged.programs == {}


def test_packaged_executor_config_is_a_package_resource() -> None:
    """The example config is shipped as package data for wheel consumers."""

    config = resources.files("tpen_executor").joinpath("conf", "executor.yaml")
    assert config.is_file()
    pyproject = Path(__file__).parents[1].joinpath("pyproject.toml").read_text(encoding="utf-8")
    assert 'tpen_executor = ["conf/*.yaml"]' in pyproject
