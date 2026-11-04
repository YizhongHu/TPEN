from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from helpers import review_targets
from tpen_executor.configuration import StagedHydra
from tpen_executor.programs import WorkerProgram


def _compose(directory: Path):
    return StagedHydra(config_module="", config_dir=str(directory)).compose_config("config")


def _write(directory: Path, text: str) -> None:
    (directory / "config.yaml").write_text(text, encoding="utf-8")


def test_r2_t1_nested_worker_absolute_self_reference(tmp_path: Path) -> None:
    """R2-1a: nested worker absolute self-reference must retain its marker root."""
    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  payload:
    _target_: helpers.review_targets.Dispatcher
    _recursive_: false
    worker:
      _target_: tpen_executor.programs.WorkerProgram
      _recursive_: false
      label: wl
      runner:
        _target_: helpers.review_targets.Runner
        _recursive_: false
        command: ${program.payload.worker.label}
""",
    )
    staged = _compose(tmp_path)
    assert staged.programs["program.payload.worker"].spec.config_ref["runner"]["command"] == "${program.payload.worker.label}"
    runner = staged.programs["program.payload.worker"].materialize({})
    assert runner.command == "wl"


def test_r2_t2_marker_under_different_top_level_key(tmp_path: Path) -> None:
    """R2-1b: a marker's root alias must follow its actual top-level key."""
    _write(
        tmp_path,
        """allocation:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  label: al
  payload:
    _target_: helpers.review_targets.Environment
    _recursive_: false
    python: ${allocation.label}
""",
    )
    staged = _compose(tmp_path)
    assert "allocation" in staged.programs
    assert staged.program is None
    assert staged.programs["allocation"].materialize({}).python == "al"


def test_r2_t3_environment_shadows_marker_alias(tmp_path: Path) -> None:
    """R2-1c: an environment key colliding with the marker alias is rejected or isolated."""
    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  allocation_label: lbl
  payload:
    _target_: helpers.review_targets.Environment
    extra: ${program.allocation_label}
""",
    )
    staged = _compose(tmp_path)
    try:
        result = staged.program.materialize({"program": {"unrelated": 1}})
    except InterpolationResolutionError:
        raise
    except ValueError:
        return
    assert result.extra == "lbl"


def test_r2_t4_relative_reference_climbs_out_of_payload(tmp_path: Path) -> None:
    """R2-2: relative references must retain the marker/payload parent relationship."""
    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  allocation_label: lbl
  payload:
    _target_: helpers.review_targets.Environment
    _recursive_: false
    extra: ${..allocation_label}
""",
    )
    staged = _compose(tmp_path)
    assert staged.program.spec.config_ref["payload"]["extra"] == "${..allocation_label}"
    assert staged.program.materialize({}).extra == "lbl"


def test_r2_t5_key_reference_hidden_in_resolver_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R2-3: resolver-argument key references are deferred to materialization."""
    monkeypatch.delenv("TPEN_P4_R2_UNSET", raising=False)
    _write(
        tmp_path,
        """outside: known
program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  payload:
    _target_: helpers.review_targets.Environment
    _recursive_: false
    python: ${oc.env:TPEN_P4_R2_UNSET,${outside}}
""",
    )
    staged = _compose(tmp_path)
    with pytest.raises(InterpolationResolutionError):
        staged.program.materialize({})


def test_r2_t6_submitter_existence_check_does_not_resolve_resolvers(tmp_path: Path) -> None:
    """R2-4: submitter-key existence checks must not evaluate resolver nodes."""
    calls: list[int] = []
    OmegaConf.register_new_resolver(
        "p4_r2_probe",
        lambda: calls.append(1) or {"control_python": "/sub"},
        replace=True,
    )
    try:
        _write(
            tmp_path,
            """paths: ${p4_r2_probe:}
program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  payload:
    _target_: helpers.review_targets.Environment
    _recursive_: false
    python: ${paths.control_python}
""",
        )
        try:
            _compose(tmp_path)
        except ValueError:
            pass
        assert calls == []
    finally:
        OmegaConf.clear_resolver("p4_r2_probe")


def test_r2_g1_nested_boundary_is_independent(tmp_path: Path) -> None:
    """R2-2 green guard: nested markers form their own validation boundary."""
    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  allocation_label: lbl
  payload:
    _target_: helpers.review_targets.Dispatcher
    _recursive_: false
    worker:
      _target_: tpen_executor.programs.WorkerProgram
      _recursive_: false
      runner:
        _target_: helpers.review_targets.Runner
        _recursive_: false
        command: ${remote_only.value}
        extra: ${program.allocation_label}
""",
    )
    with pytest.raises(ValueError, match="program.allocation_label"):
        _compose(tmp_path)

    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  allocation_label: lbl
  payload:
    _target_: helpers.review_targets.Dispatcher
    _recursive_: false
    worker:
      _target_: tpen_executor.programs.WorkerProgram
      _recursive_: false
      runner:
        _target_: helpers.review_targets.Runner
        _recursive_: false
        command: ${remote_only.value}
""",
    )
    staged = _compose(tmp_path)
    assert set(staged.programs) == {"program", "program.payload.worker"}


def test_r2_g2_nested_handles_inside_list(tmp_path: Path) -> None:
    """R2-3 green guard: nested program handles are substituted inside lists."""
    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  payload:
    _target_: helpers.review_targets.Dispatcher
    _recursive_: false
    workers:
      - _target_: tpen_executor.programs.WorkerProgram
        _recursive_: false
        runner:
          _target_: helpers.review_targets.Runner
          _recursive_: false
          command: a
      - _target_: tpen_executor.programs.WorkerProgram
        _recursive_: false
        runner:
          _target_: helpers.review_targets.Runner
          _recursive_: false
          command: b
""",
    )
    staged = _compose(tmp_path)
    assert set(staged.programs) == {"program", "program.payload.workers[0]", "program.payload.workers[1]"}
    review_targets.reset()
    result = staged.program.materialize({})
    assert review_targets.CONSTRUCTIONS == 1
    workers = result.options["workers"]
    assert len(workers) == 2
    assert all(isinstance(worker, WorkerProgram) for worker in workers)
    assert workers[1].spec.content_digest == staged.programs["program.payload.workers[1]"].spec.content_digest
    assert workers[1].materialize({}).command == "b"
    assert review_targets.CONSTRUCTIONS == 2


def test_r2_g3_nested_handles_inside_list_containers(tmp_path: Path) -> None:
    """R2-6b: recurse through list elements that contain nested lists/maps."""
    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  payload:
    _target_: helpers.review_targets.Dispatcher
    _recursive_: false
    workers:
      - slot:
          _target_: tpen_executor.programs.WorkerProgram
          _recursive_: false
          runner:
            _target_: helpers.review_targets.Runner
            _recursive_: false
            command: mapping-in-list
    groups:
      - - _target_: tpen_executor.programs.WorkerProgram
          _recursive_: false
          runner:
            _target_: helpers.review_targets.Runner
            _recursive_: false
            command: list-in-list
""",
    )
    staged = _compose(tmp_path)
    mapping_path = "program.payload.workers[0].slot"
    nested_list_path = "program.payload.groups[0][0]"
    assert set(staged.programs) == {"program", mapping_path, nested_list_path}

    review_targets.reset()
    result = staged.program.materialize({})
    assert review_targets.CONSTRUCTIONS == 1
    mapping_worker = result.options["workers"][0]["slot"]
    nested_list_worker = result.options["groups"][0][0]
    assert isinstance(mapping_worker, WorkerProgram)
    assert isinstance(nested_list_worker, WorkerProgram)
    assert mapping_worker.path == mapping_path
    assert nested_list_worker.path == nested_list_path

    assert mapping_worker.materialize({}).command == "mapping-in-list"
    assert review_targets.CONSTRUCTIONS == 2
    assert nested_list_worker.materialize({}).command == "list-in-list"
    assert review_targets.CONSTRUCTIONS == 3
