"""P5a PR #581 review round 2 probes at 79c67d39: slot-name rules at compose, F3 over-reach guard.

Reviewer-proposed tests.  Each docstring states the expected outcome on the reviewed head
79c67d39: GREEN tests guard behaviour the round-1 fixes must keep; RED tests assert a property
the reviewer proposes and the orchestrator decides.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.configuration import StagedConfig, StagedHydra
from tpen_executor.contracts import ProgramSpecRef
from tpen_executor.programs import AllocationProgram, WorkerProgram


def _compose(tmp_path: Path, text: str, name: str = "executor") -> StagedConfig:
    """Write one YAML config into ``tmp_path`` and compose it through StagedHydra."""

    (tmp_path / f"{name}.yaml").write_text(textwrap.dedent(text), encoding="utf-8")
    return StagedHydra(config_module="", config_dir=str(tmp_path)).compose_config(name)


_RESERVED_SLOT = """
program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  keys:
    _target_: tpen_executor.programs.DummyRunner
    _recursive_: false
"""


def test_p5a_r2_t1_reserved_slot_name_is_rejected_at_compose(tmp_path: Path) -> None:
    """RED on 79c67d39.  R2-1: the F6 reserved-name rule runs only in
    ``MaterializedProgram.__post_init__`` (owner process).  A marker whose slot is named ``keys``
    composes cleanly on the submitter and fails at materialize, the late-failure shape F4 removed."""

    with pytest.raises(ValueError, match="keys"):
        _compose(tmp_path, _RESERVED_SLOT)


_INT_SLOT = """
program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  1:
    _target_: tpen_executor.programs.DummyRunner
    _recursive_: false
  alpha:
    _target_: tpen_executor.programs.DummyProcess
    _recursive_: false
"""


def test_p5a_r2_t2_non_string_slot_name_fails_at_compose_with_a_clear_error(tmp_path: Path) -> None:
    """RED on 79c67d39.  R2-1: an integer YAML key beside a string slot raises ``TypeError`` from
    ``sorted()`` inside ``_payload_entries`` ("'<' not supported between instances of 'str' and
    'int'"), and an all-integer marker sorts fine and fails only at materialize.  Slot names should
    be validated as non-empty strings at compose with an error naming the marker."""

    with pytest.raises(ValueError, match="program"):
        _compose(tmp_path, _INT_SLOT)


_LEGITIMATE = """
program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  metadata:
    owner: control-team
    tags: [a, b]
  dispatcher:
    _target_: tpen_executor.programs.DummyDispatcher
    _recursive_: false
    label: ${program.metadata.owner}
    pools:
      a:
        _target_: tpen_executor.programs.WorkerProgram
        _recursive_: false
        runner:
          _target_: tpen_executor.programs.DummyRunner
          _recursive_: false
"""


def test_p5a_r2_t3_plan_shapes_survive_the_compose_time_shape_rule(tmp_path: Path) -> None:
    """GREEN on 79c67d39.  Guards F3 against over-reach: a marker-level plain mapping of scalars is
    still retained and interpolable, and a nested marker inside a grouping mapping *within* a
    concrete slot's interior is still captured and handed back as a handle."""

    staged = _compose(tmp_path, _LEGITIMATE)
    assert set(staged.programs) == {"program", "program.dispatcher.pools.a"}
    assert staged.program is not None
    result = staged.program.materialize({})
    assert list(result) == ["dispatcher"]
    assert result.dispatcher.options["label"] == "control-team"
    pool = result.dispatcher.options["pools"]["a"]
    assert isinstance(pool, WorkerProgram)
    assert pool.path == "program.dispatcher.pools.a"
    assert list(pool.materialize({})) == ["runner"]


def test_p5a_r2_t4_materialize_applies_the_shape_rule_to_a_ref_that_bypassed_capture() -> None:
    """RED on 79c67d39.  R2-2: ``_validate_marker_shape`` lives in ``configuration.py`` and runs only
    at capture.  A ``ProgramSpecRef`` built or decoded without ``StagedHydra`` still materializes
    with a direct nested marker silently dropped (slots == ['environment'])."""

    raw = {
        "_target_": "tpen_executor.programs.AllocationProgram",
        "_recursive_": False,
        "environment": {"_target_": "tpen_executor.programs.DummyProcess", "_recursive_": False},
        "worker": {
            "_target_": "tpen_executor.programs.WorkerProgram",
            "_recursive_": False,
            "runner": {"_target_": "tpen_executor.programs.DummyRunner", "_recursive_": False},
        },
    }
    handle = AllocationProgram(ProgramSpecRef(raw, digest(raw)), "program")
    with pytest.raises(ValueError, match="program.worker"):
        handle.materialize({})
