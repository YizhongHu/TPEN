"""Independent P4 review probes for staged Hydra configuration."""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import pytest

from helpers import review_targets, staged_targets
from tpen_executor._codec import digest, thaw
from tpen_executor.configuration import StagedHydra
from tpen_executor.contracts import ProgramSpecRef
from tpen_executor.programs import WorkerProgram


def _compose(directory: Path, *overrides: str):
    return StagedHydra(config_module="", config_dir=str(directory)).compose_config("config", overrides)


def _write(directory: Path, text: str) -> None:
    (directory / "config.yaml").write_text(text, encoding="utf-8")


def test_t1_environment_does_not_leak_into_target_kwargs(tmp_path: Path) -> None:
    """finding 1"""
    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  payload:
    _target_: helpers.review_targets.StrictAllocation
    _recursive_: false
    label: ${remote_only.value}
""",
    )
    review_targets.reset()
    staged = _compose(tmp_path)
    result = staged.program.materialize({"remote_only": {"value": "x"}})
    assert result.label == "x"
    assert review_targets.CONSTRUCTIONS == 1


def test_t1b_materialize_cannot_hijack_target(tmp_path: Path) -> None:
    """finding 1"""
    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  payload:
    _target_: helpers.review_targets.StrictAllocation
    _recursive_: false
    label: ${remote_only.value}
""",
    )
    review_targets.reset()
    staged = _compose(tmp_path)
    try:
        staged.program.materialize(
            {"remote_only": {"value": "x"}, "_target_": "helpers.review_targets.Hijacked"}
        )
    except (ValueError, TypeError):
        pass
    assert review_targets.CONSTRUCTIONS == 0


def test_t2_submitter_interpolations_are_rejected_but_subtree_refs_resolve(tmp_path: Path) -> None:
    """Submitter-known outside refs fail at compose; internal refs remain legal."""
    _write(
        tmp_path,
        """outside: known-on-submitter
program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  allocation_label: lbl
  payload:
    _target_: helpers.review_targets.Environment
    _recursive_: false
    python: ${outside}
    extra: ${program.allocation_label}
""",
    )
    with pytest.raises(ValueError, match="outside.*submitter key"):
        _compose(tmp_path)

    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  allocation_label: lbl
  payload:
    _target_: helpers.review_targets.Environment
    _recursive_: false
    extra: ${program.allocation_label}
""",
    )
    staged = _compose(tmp_path)
    raw = staged.program.spec.config_ref
    assert raw["payload"]["extra"] == "${program.allocation_label}"
    review_targets.reset()
    result = staged.program.materialize({})
    assert result.extra == "lbl"


def test_t3_nested_program_marker_becomes_handle_before_parent_instantiate(tmp_path: Path) -> None:
    """finding 5: the parent receives a WorkerProgram handle, not a DictConfig."""
    _write(
        tmp_path,
        """paths:
program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
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
    review_targets.reset()
    result = staged.program.materialize({"remote_only": {"value": "cmd"}})
    assert review_targets.CONSTRUCTIONS == 1
    assert isinstance(result.worker, WorkerProgram)


def test_orchestrator_stage_captures_already_composed_config(tmp_path: Path) -> None:
    """The stage entry point shares the compose capture path."""
    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  payload:
    _target_: helpers.review_targets.Environment
""",
    )
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(tmp_path)):
        config = compose(config_name="config")
    staged = StagedHydra.stage(config)
    assert staged.program is not None


def test_materialize_rejects_hydra_reserved_environment_keys(tmp_path: Path) -> None:
    """Environment keys cannot alter Hydra's target control fields."""
    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  payload:
    _target_: helpers.review_targets.Environment
""",
    )
    staged = _compose(tmp_path)
    with pytest.raises(ValueError, match="reserved keys"):
        staged.program.materialize({"_target_": "helpers.review_targets.Hijacked"})


def test_root_program_marker_is_rejected(tmp_path: Path) -> None:
    """A root marker cannot be addressed through StagedConfig.program."""
    _write(
        tmp_path,
        """_target_: tpen_executor.programs.AllocationProgram
_recursive_: false
payload:
  _target_: helpers.review_targets.Environment
""",
    )
    with pytest.raises(ValueError, match="configuration root"):
        _compose(tmp_path)


def test_no_config_source_is_rejected() -> None:
    """The constructor reports a missing config source clearly."""
    with pytest.raises(ValueError, match="either config_module or config_dir"):
        StagedHydra(config_module="")


def test_t4_unmarked_program_target_is_rejected(tmp_path: Path) -> None:
    """finding 4"""
    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  payload:
    _target_: helpers.review_targets.Environment
""",
    )
    with pytest.raises(ValueError):
        _compose(tmp_path)


def test_t5_program_digest_is_canonical_and_round_trips(tmp_path: Path) -> None:
    """Digest canonicality probe."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write(
        first,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  payload:
    _target_: helpers.review_targets.Environment
    _recursive_: false
    python: base
    values: [one, two]
  allocation_label: base
""",
    )
    _write(
        second,
        """program:
  allocation_label: base
  payload:
    values: [one, two]
    _recursive_: false
    python: base
    _target_: helpers.review_targets.Environment
  _recursive_: false
  _target_: tpen_executor.programs.AllocationProgram
""",
    )
    ref = _compose(first).program.spec
    reordered = _compose(second).program.spec
    assert ref.content_digest == reordered.content_digest

    _write(
        second,
        (second / "config.yaml").read_text(encoding="utf-8").replace("[one, two]", "[two, one]"),
    )
    assert _compose(second).program.spec.content_digest != ref.content_digest
    assert _compose(first, "program.allocation_label=base").program.spec.content_digest == ref.content_digest
    assert _compose(first, "+program.extra=1").program.spec.content_digest != ref.content_digest

    decoded = ProgramSpecRef.decode(ref.encode())
    assert decoded.content_digest == ref.content_digest
    assert digest(thaw(decoded.config_ref)) == ref.content_digest


def test_t6_nested_worker_materializes_independently(tmp_path: Path) -> None:
    """Nested worker records can be consumed without composition."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        """program:
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
        command: ${remote_only.value}
""",
        encoding="utf-8",
    )
    staged = _compose(config_dir)
    ref_path = tmp_path / "worker.ref"
    ref_path.write_bytes(staged.programs["program.process.worker"].spec.encode())
    src = Path(__file__).parents[1] / "src"
    tests = Path(__file__).parents[0]
    code = """import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
sys.path.insert(0, sys.argv[3])
from tpen_executor.contracts import ProgramSpecRef
from tpen_executor.programs import WorkerProgram
from helpers import staged_targets
staged_targets.reset()
ref = ProgramSpecRef.decode(Path(sys.argv[1]).read_bytes())
result = WorkerProgram(ref).materialize({"remote_only": {"value": "w"}})
print(staged_targets.CONSTRUCTIONS)
print(result.command)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ref_path), str(src), str(tests)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["1", "w"]


def test_t7_capture_precedes_resolver_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture remains literal until the owner process resolves it."""
    variable = "TPEN_P4_REVIEW_UNSET_VAR"
    monkeypatch.delenv(variable, raising=False)
    _write(
        tmp_path,
        """program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  payload:
    _target_: helpers.staged_targets.CountedAllocation
    _recursive_: false
    secret: ${oc.env:TPEN_P4_REVIEW_UNSET_VAR}
    extra: ${remote_only.other}
""",
    )
    staged = _compose(tmp_path)
    assert staged.program.spec.config_ref["payload"]["secret"] == "${oc.env:TPEN_P4_REVIEW_UNSET_VAR}"
    assert staged.program.spec.config_ref["payload"]["extra"] == "${remote_only.other}"
    monkeypatch.setenv(variable, "v")
    result = staged.program.materialize({"remote_only": {"other": "o"}})
    assert result.options["secret"] == "v"
    assert result.options["extra"] == "o"


def test_t8_contracts_import_without_hydra_or_omegaconf(tmp_path: Path) -> None:
    """finding 7"""
    src = Path(__file__).parents[1] / "src"
    code = """import sys
sys.modules["hydra"] = None
sys.modules["omegaconf"] = None
sys.path.insert(0, sys.argv[1])
import tpen_executor.contracts
"""
    result = subprocess.run([sys.executable, "-c", code, str(src)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

