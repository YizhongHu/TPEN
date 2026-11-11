"""P5a PR #581 review round 1 probes: slot contract, descriptor controls, authoring paths.

Reviewer-proposed tests.  Outcome on the reviewed head 33d2a6b4 is stated in each
docstring: GREEN tests document or guard current behaviour; RED tests assert a
contract property the reviewer proposes and the orchestrator decides.
"""

from __future__ import annotations

import ast
import builtins
import io
import os
import textwrap
from pathlib import Path

import pytest
from hydra.errors import InstantiationException

from tpen_executor._codec import digest
from tpen_executor.configuration import StagedConfig, StagedHydra
from tpen_executor.contracts import ProgramSpecRef
from tpen_executor.delivery import PreparedInputBundle
from tpen_executor.environments import PreparedPythonEnvironment, SourceScript
from tpen_executor.programs import AllocationProgram, MaterializedProgram


def _compose(tmp_path: Path, text: str, name: str = "executor") -> StagedConfig:
    """Write one YAML config into ``tmp_path`` and compose it through StagedHydra."""

    (tmp_path / f"{name}.yaml").write_text(textwrap.dedent(text), encoding="utf-8")
    return StagedHydra(config_module="", config_dir=str(tmp_path)).compose_config(name)


_TWO_SLOTS = """
program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  zeta:
    _target_: tpen_executor.programs.DummyRunner
    _recursive_: false
  alpha:
    _target_: tpen_executor.programs.DummyProcess
    _recursive_: false
"""


def test_p5a_r1_t1_slot_order_is_identical_before_and_after_wire_round_trip(tmp_path: Path) -> None:
    """RED on 33d2a6b4.  F1: ``MaterializedProgram.__iter__`` promises source order, but a
    handle rebuilt from the canonical (sorted-key) wire form iterates and instantiates its
    slots in lexicographic order while the submitter-side handle uses YAML order, under one
    identical content digest."""

    staged = _compose(tmp_path, _TWO_SLOTS)
    live = staged.program
    assert live is not None
    decoded = AllocationProgram(ProgramSpecRef.decode(live.spec.encode()), live.path)
    assert decoded.spec.content_digest == live.spec.content_digest
    assert list(live.materialize({})) == list(decoded.materialize({}))


_FS_PROBES = [
    (builtins, "open"), (io, "open"), (os, "open"), (os, "stat"), (os, "lstat"), (os, "access"),
    (os, "listdir"), (os, "scandir"), (os, "mkdir"), (os, "makedirs"), (os, "remove"), (os, "unlink"),
    (os, "rename"), (os, "replace"), (os, "rmdir"), (os, "symlink"), (os, "link"), (os, "chmod"),
    (os, "utime"), (os.path, "exists"), (os.path, "lexists"), (os.path, "isfile"), (os.path, "isdir"),
    (os.path, "islink"), (os.path, "getsize"), (os.path, "realpath"), (os.path, "samefile"),
]


def test_p5a_r1_t2a_descriptors_never_touch_the_filesystem() -> None:
    """GREEN on 33d2a6b4.  F2: runtime control for acceptance D.  Every filesystem entry point a
    descriptor could reach (reads, existence checks, writes) is recorded while each descriptor is
    constructed and round-tripped; the only path operation allowed is the lexical ``os.path.isabs``.
    The wrappers delegate to the real functions and every patch is undone before the assertion, so
    a kill is reported by pytest cleanly.  Kills the existence-check and bare-``open`` mutants that
    the static AST control misses."""

    calls: list[str] = []

    def _recording(name: str, original):
        def _wrapped(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        return _wrapped

    with pytest.MonkeyPatch.context() as patch:
        for module, name in _FS_PROBES:
            patch.setattr(module, name, _recording(f"{module.__name__}.{name}", getattr(module, name)))
        script = SourceScript("/prepared/does-not-exist/setup.sh")
        environment = PreparedPythonEnvironment("/prepared/does-not-exist/bin/python", script, {"MODE": "control"})
        bundle = PreparedInputBundle("/prepared/does-not-exist/inputs", "a" * 64)
        records = (script, environment, bundle)
        decoded = [(type(record).decode(record.encode()), type(record).from_mapping(record.to_dict())) for record in records]
    for record, (from_bytes, from_mapping) in zip(records, decoded):
        assert from_bytes == record
        assert from_mapping == record
    assert calls == [], f"descriptor reached the filesystem via {sorted(set(calls))}"


_DESCRIPTOR_MODULES = ("environments.py", "delivery.py", "_records.py")
_ALLOWED_IMPORTS = {"__future__", "os", "collections.abc", "dataclasses", "types", "typing", "tpen_executor._codec", "tpen_executor._records"}
_BANNED_NAME_CALLS = {"open", "exec", "eval", "compile", "__import__"}
_ALLOWED_OS_ATTRIBUTES = {"os.path", "os.path.isabs"}


def _dotted(node: ast.AST) -> str | None:
    """Return the dotted name of a Name/Attribute chain, or None for anything else."""

    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def test_p5a_r1_t2b_descriptor_modules_reach_nothing_beyond_os_path_isabs() -> None:
    """GREEN on 33d2a6b4.  F2: static control stronger than the shipped one.  Descriptor modules
    may import only a closed allowlist, may reference no ``os`` attribute other than
    ``os.path.isabs``, and may not call bare ``open``/``exec``/``eval``/``compile``/``__import__``.
    The shipped test inspects attribute calls only, so ``open(...)`` and ``os.path.exists(...)``
    pass it."""

    source_root = Path(__file__).parents[1] / "src" / "tpen_executor"
    violations: list[str] = []
    for name in _DESCRIPTOR_MODULES:
        tree = ast.parse((source_root / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in _ALLOWED_IMPORTS:
                        violations.append(f"{name}:{node.lineno}:import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                resolved = f"tpen_executor.{module}" if node.level else module
                if resolved not in _ALLOWED_IMPORTS:
                    violations.append(f"{name}:{node.lineno}:from {resolved}")
            elif isinstance(node, ast.Call):
                dotted = _dotted(node.func)
                if dotted in _BANNED_NAME_CALLS:
                    violations.append(f"{name}:{node.lineno}:{dotted}()")
            elif isinstance(node, ast.Attribute):
                dotted = _dotted(node)
                if dotted is not None and dotted.split(".")[0] == "os" and dotted not in _ALLOWED_OS_ATTRIBUTES:
                    violations.append(f"{name}:{node.lineno}:{dotted}")
    assert not violations, "descriptor reach beyond os.path.isabs: " + ", ".join(violations)


_NON_SLOT_CHILDREN = """
program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  environment:
    _target_: tpen_executor.environments.PreparedPythonEnvironment
    python: /prepared/bin/python
  worker:
    _target_: tpen_executor.programs.WorkerProgram
    _recursive_: false
    runner:
      _target_: tpen_executor.programs.DummyRunner
      _recursive_: false
  evidence_list:
    - _target_: tpen_executor.programs.DummyEvidence
      _recursive_: false
  grouped:
    inner:
      _target_: tpen_executor.programs.DummyEvidence
      _recursive_: false
"""


@pytest.mark.parametrize(
    ("child_name", "child"),
    [
        (
            "worker",
            """worker:
    _target_: tpen_executor.programs.WorkerProgram
    _recursive_: false
    runner:
      _target_: tpen_executor.programs.DummyRunner
      _recursive_: false
""",
        ),
        (
            "evidence_list",
            """evidence_list:
    - _target_: tpen_executor.programs.DummyEvidence
      _recursive_: false
""",
        ),
        (
            "grouped",
            """grouped:
    inner:
      _target_: tpen_executor.programs.DummyEvidence
      _recursive_: false
""",
        ),
    ],
)
def test_p5a_r1_t3_direct_nested_marker_list_and_grouped_targets_are_rejected(
    tmp_path: Path, child_name: str, child: str
) -> None:
    """F3: unsupported direct children fail at compose and name their marker path."""

    text = f"""
program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  environment:
    _target_: tpen_executor.environments.PreparedPythonEnvironment
    python: /prepared/bin/python
  {child.rstrip().replace(chr(10), chr(10) + '  ')}
"""
    with pytest.raises(ValueError, match=rf"program\.{child_name}"):
        _compose(tmp_path, text)


_NO_SLOTS = """
program:
  _target_: tpen_executor.programs.AllocationProgram
  _recursive_: false
  label: control
"""


def test_p5a_r1_t4_marker_without_concrete_slots_is_rejected_at_compose(tmp_path: Path) -> None:
    """RED on 33d2a6b4.  F4: a marker with no concrete slot composes cleanly on the submitter and
    fails only in the owner process (``ValueError`` from ``materialize``).  The reviewer proposes
    rejecting it at capture, where every other marker-shape error is already raised."""

    with pytest.raises(ValueError):
        _compose(tmp_path, _NO_SLOTS)


def test_p5a_r1_t5_prepared_environment_is_hashable_like_its_sibling_descriptors() -> None:
    """RED on 33d2a6b4.  F5: ``SourceScript`` and ``PreparedInputBundle`` hash; ``PreparedPythonEnvironment``
    raises ``TypeError`` because its generated ``__hash__`` hashes the ``env`` mapping proxy.  Equal
    records (same bindings, any env insertion order) should hash equal so consumers can key on them."""

    first = PreparedPythonEnvironment("/prepared/bin/python", SourceScript("/prepared/setup.sh"), {"A": "1", "B": "2"})
    second = PreparedPythonEnvironment("/prepared/bin/python", SourceScript("/prepared/setup.sh"), {"B": "2", "A": "1"})
    assert first == second
    assert hash(first) == hash(second)
    payload = {"_target_": "example.Target"}
    ref = ProgramSpecRef(payload, digest(payload))
    assert hash(ref) == hash(ProgramSpecRef.decode(ref.encode()))
    assert len({first, second, SourceScript("/prepared/setup.sh"), PreparedInputBundle("/prepared/inputs"), ref}) == 4


def test_p5a_r1_t6_slot_names_colliding_with_record_attributes_are_rejected() -> None:
    """F6: record attributes, mapping protocol names, and dunders are not valid slots."""

    marker = object()
    for name in ("slots", "keys", "__dict__"):
        with pytest.raises(ValueError, match="collide"):
            MaterializedProgram({name: marker})


def test_p5a_r1_t7_authoring_comments_state_the_interpolation_and_env_rules() -> None:
    """F7: shipped authoring guidance matches the accepted R3-2 wording."""

    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    config = (root / "src" / "tpen_executor" / "conf" / "executor.yaml").read_text(encoding="utf-8")
    readme = " ".join(readme.split())
    config_comments = " ".join(
        line.lstrip()[1:]
        for line in config.splitlines()
        if line.lstrip().startswith("#")
    )
    config_comments = " ".join(config_comments.split())
    assert "${oc.env:NAME}" in readme
    assert "on keys referenced by a program" in readme
    assert "on keys referenced by a program" in config_comments
    assert "env: values must be strings" in config_comments


_CONFIG_DRIVEN = """
program:
  _target_: tpen_executor.programs.WorkerProgram
  _recursive_: false
  environment:
    _target_: tpen_executor.environments.PreparedPythonEnvironment
    python: ${paths.science_python}
    setup:
      _target_: tpen_executor.environments.SourceScript
      path: ${paths.science_setup}
    env:
      OMP_NUM_THREADS: "4"
      TPEN_RUN_TAG: ${paths.run_tag}
  inputs:
    _target_: tpen_executor.delivery.PreparedInputBundle
    root: ${paths.input_bundle}
    manifest_digest: ${paths.input_digest}
"""


def test_p5a_r1_t8_yaml_authored_descriptors_materialize_as_frozen_plain_values(tmp_path: Path) -> None:
    """GREEN on 33d2a6b4.  The packaged config carries no ``env:`` block and no bundle, so the
    Hydra path into ``PreparedPythonEnvironment.env`` (a ``DictConfig`` at instantiate time) and
    into ``PreparedInputBundle`` is otherwise untested.  Values must arrive as plain frozen strings
    with no OmegaConf container leaking into the wire form, and an unquoted YAML integer must be
    rejected rather than coerced."""

    staged = _compose(tmp_path, _CONFIG_DRIVEN)
    env = {
        "paths": {
            "science_python": "/prepared/science/bin/python",
            "science_setup": "/prepared/science/setup.sh",
            "run_tag": "r1",
            "input_bundle": "/prepared/inputs",
            "input_digest": "b" * 64,
        }
    }
    assert staged.program is not None
    result = staged.program.materialize(env)
    environment = result.environment
    assert environment.setup == SourceScript("/prepared/science/setup.sh")
    assert dict(environment.env) == {"OMP_NUM_THREADS": "4", "TPEN_RUN_TAG": "r1"}
    assert type(environment.to_dict()["env"]) is dict
    assert PreparedPythonEnvironment.decode(environment.encode()) == environment
    bundle = result.inputs
    assert bundle == PreparedInputBundle("/prepared/inputs", "b" * 64)
    assert PreparedInputBundle.decode(bundle.encode()) == bundle

    integer_valued = _compose(tmp_path, _CONFIG_DRIVEN.replace('OMP_NUM_THREADS: "4"', "OMP_NUM_THREADS: 4"), name="ints")
    assert integer_valued.program is not None
    with pytest.raises(InstantiationException) as info:
        integer_valued.program.materialize(env)
    assert isinstance(info.value.__cause__, TypeError)
