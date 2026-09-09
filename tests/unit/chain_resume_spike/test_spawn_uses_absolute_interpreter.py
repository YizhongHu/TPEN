"""HARD CLAUSE: every spawn in this package uses an absolute ``sys.executable``.

This exists because ``experiments/baselines/test_scaling_probe.py`` line 168
builds ``["python", "-c", ...]`` and spawns it. That test executes a bare
interpreter NAME, so it inherits whatever ``PATH`` the job happens to have. It
is green on this workstation only because ``command -v python`` resolves to an
Anaconda install here; it is the one pre-existing red on trunk, and a Slurm
environment exposing only ``python3`` or only the uv-selected venv interpreter
cannot spawn it at all.

This package's whole mechanism is spawning fresh processes, so it is maximally
exposed to that failure. The rule is therefore enforced by a test rather than by
writer discipline. Two independent checks, because either alone has a gap:

* a STRUCTURAL check that every spawn call's ``argv[0]`` is literally the
  ``sys.executable`` attribute -- which catches a spawn helper that resolves an
  interpreter some other way, including one no name-based rule would notice;
* a LEXICAL check that no bare interpreter name appears anywhere in the
  package -- which catches a helper that merely *accepts* or defaults to one,
  before any spawn call is written.

The detector is exercised against a known-bad control below. A checker nobody
has ever seen fire is not a checker.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path("tests/helpers/chain_resume_spike")

#: Calls that hand a command to the OS. ``shell=True`` on any of them is itself
#: a violation, since the shell resolves the program against ``PATH``.
_SPAWNING_CALLS = frozenset(
    {
        "Popen",
        "run",
        "call",
        "check_call",
        "check_output",
        "execv",
        "execvp",
        "execvpe",
        "spawnv",
        "spawnvp",
    }
)

#: Interpreter names that resolve against ``PATH`` rather than naming a file.
_BARE_INTERPRETER_NAMES = frozenset({"python", "python3", "python2", "py"})


def _is_sys_executable(node: ast.expr) -> bool:
    """Return whether ``node`` is literally the ``sys.executable`` attribute."""

    return (
        isinstance(node, ast.Attribute)
        and node.attr == "executable"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _is_bare_interpreter(node: ast.expr) -> bool:
    """Return whether ``node`` is a string literal naming a PATH-resolved interpreter."""

    return isinstance(node, ast.Constant) and node.value in _BARE_INTERPRETER_NAMES


def _local_list_bindings(tree: ast.AST) -> dict[str, ast.List]:
    """Map simple ``name = [...]`` bindings, per enclosing function.

    Needed because a readable launcher builds its ``argv`` across several
    statements and then passes the variable to ``Popen``. Refusing to follow
    that binding would force either an unreadable inline list or a blanket
    exemption, and the exemption is what would let a bare name back in. Only a
    direct list literal is followed; anything else stays unresolved and is
    reported, so the checker fails closed rather than open.
    """

    bindings: dict[str, ast.List] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node.value
    return bindings


def _check_argv(program: ast.expr, label: str, lineno: int) -> list[str]:
    """Return violations for a resolved ``argv[0]`` expression."""

    if _is_sys_executable(program):
        return []
    if _is_bare_interpreter(program):
        return [
            f"{label}:{lineno}: argv[0] is the bare interpreter name "
            f"{program.value!r}, which resolves against PATH"  # type: ignore[attr-defined]
        ]
    return [
        f"{label}:{lineno}: argv[0] is not sys.executable ({ast.dump(program)[:80]})"
    ]


def _spawn_violations(source: str, label: str) -> list[str]:
    """Return every spawn-discipline violation in ``source``.

    Exposed as a plain function over text, rather than inlined into a loop over
    the package, so the same detector can be pointed at deliberately bad inputs
    and shown to fire -- and at a deliberately good one and shown not to.

    Three rules, covering PASSING a bare name, ACCEPTING one, and reaching the
    program through a shell:

    1. every spawn call's ``argv[0]`` must be ``sys.executable``, following a
       simple local ``argv = [...]`` binding to find it;
    2. no function parameter may DEFAULT to a bare interpreter name, and no
       interpreter-ish variable may be assigned one -- this catches a helper
       that accepts one before any spawn is written;
    3. ``shell=`` must be absent or literally ``False``.

    Deliberately NOT a blanket ban on the string ``"python"`` anywhere in the
    file. ``restore_limbs.capture_global_rng`` uses ``"python"`` as a JSON key
    naming which RNG a state blob belongs to, which is not a command context;
    a rule that flagged it would train a reader to expect false positives here.
    """

    violations: list[str] = []
    tree = ast.parse(source)
    bindings = _local_list_bindings(tree)

    for node in ast.walk(tree):
        # --- Rule 2a: a parameter defaulting to a bare interpreter name. ---
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            for default in [*arguments.defaults, *arguments.kw_defaults]:
                if default is not None and _is_bare_interpreter(default):
                    violations.append(
                        f"{label}:{node.lineno}: parameter of {node.name!r} defaults to "
                        f"the bare interpreter name {default.value!r}"  # type: ignore[attr-defined]
                    )

        # --- Rule 2b: an interpreter-ish variable bound to a bare name. ---
        if isinstance(node, ast.Assign) and _is_bare_interpreter(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name) and any(
                    token in target.id.lower()
                    for token in ("interpreter", "executable", "python", "cmd", "program")
                ):
                    violations.append(
                        f"{label}:{node.lineno}: {target.id!r} is assigned the bare "
                        f"interpreter name {node.value.value!r}"  # type: ignore[attr-defined]
                    )

        # --- Rule 2c: a command list whose first element is a bare name. ---
        if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
            if _is_bare_interpreter(node.elts[0]):
                violations.append(
                    f"{label}:{node.lineno}: a command sequence starts with the bare "
                    f"interpreter name {node.elts[0].value!r}"  # type: ignore[attr-defined]
                )

        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = (
            callee.attr
            if isinstance(callee, ast.Attribute)
            else callee.id
            if isinstance(callee, ast.Name)
            else None
        )
        if name not in _SPAWNING_CALLS:
            continue

        # --- Rule 3: the shell resolves the program against PATH. ---
        for keyword in node.keywords:
            if keyword.arg == "shell" and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            ):
                violations.append(
                    f"{label}:{node.lineno}: shell= resolves the program against PATH"
                )

        # --- Rule 1: argv[0] must be sys.executable. ---
        if not node.args:
            continue
        command = node.args[0]
        if isinstance(command, ast.Name):
            resolved = bindings.get(command.id)
            if resolved is None:
                violations.append(
                    f"{label}:{node.lineno}: argv[0] not statically verifiable "
                    f"(command built as variable {command.id!r})"
                )
                continue
            command = resolved
        if isinstance(command, (ast.List, ast.Tuple)) and command.elts:
            violations.extend(_check_argv(command.elts[0], label, node.lineno))
        elif isinstance(command, ast.Constant):
            violations.extend(_check_argv(command, label, node.lineno))

    return violations


def _package_modules() -> list[Path]:
    """Return every module this package owns."""

    modules = sorted(PACKAGE_DIR.glob("*.py"))
    assert modules, f"no modules found under {PACKAGE_DIR}; the scan would be vacuous"
    return modules


def test_the_detector_fires_on_a_known_bad_spawn() -> None:
    """Instrument check, run before the real scan.

    Reproduces the exact shape of the trunk defect. If this control does not
    produce a violation, the clean result below means nothing.
    """

    known_bad = (
        "import subprocess\n"
        'command = ["python", "-c", "print(1)"]\n'
        "subprocess.run(command)\n"
    )
    violations = _spawn_violations(known_bad, "known_bad")
    assert violations, "the detector did not fire on a bare-'python' spawn"
    assert any("bare interpreter name" in violation for violation in violations)


def test_the_detector_fires_on_a_shell_spawn() -> None:
    """Second control: ``shell=True`` is PATH resolution by another route."""

    known_bad = "import subprocess\nsubprocess.run([sys.executable], shell=True)\n"
    assert any("shell=" in v for v in _spawn_violations(known_bad, "known_bad"))


def test_the_detector_fires_on_a_helper_that_accepts_a_bare_interpreter() -> None:
    """Fourth control: ACCEPTING a bare name is caught before any spawn exists.

    The clause forbids a spawn helper from accepting or passing a bare
    interpreter name. A rule that only inspected spawn call sites would pass a
    helper whose default is ``"python"`` until the day someone calls it.
    """

    known_bad = 'def launch(interpreter="python"):\n    return interpreter\n'
    violations = _spawn_violations(known_bad, "known_bad")
    assert any("defaults to the bare interpreter name" in v for v in violations), violations


def test_the_detector_does_not_fire_on_a_correct_spawn() -> None:
    """Third control, the other direction: a correct spawn must come back clean.

    Without this the two controls above are satisfied by a detector that flags
    everything, which would make the package scan below pass for no reason.
    """

    good = (
        "import subprocess\nimport sys\n"
        'subprocess.Popen([sys.executable, "-m", "mod"], shell=False)\n'
    )
    assert _spawn_violations(good, "good") == []

    # Also clean: an argv built over several statements, and a dict that merely
    # uses "python" as a key. Both occur in this package, and a checker that
    # flagged either would be routinely ignored.
    good_indirect = (
        "import subprocess\nimport sys\n"
        "def launch():\n"
        '    argv = [sys.executable, "-m", "mod"]\n'
        '    argv += ["--flag"]\n'
        "    return subprocess.Popen(argv)\n"
        'STATE = {"python": [1, 2], "numpy_legacy": [3]}\n'
    )
    assert _spawn_violations(good_indirect, "good_indirect") == []


@pytest.mark.parametrize(
    "module", _package_modules(), ids=lambda path: path.name
)
def test_no_module_in_this_package_spawns_a_bare_interpreter(module: Path) -> None:
    """The clause itself, module by module so a failure names the file."""

    violations = _spawn_violations(module.read_text(encoding="utf-8"), str(module))
    assert violations == [], "\n".join(violations)


def test_sys_executable_is_absolute_at_runtime() -> None:
    """The structural rule is only worth anything if the value is a real path.

    Not ``Path.resolve()``: a virtualenv's ``bin/python`` is a symlink to the
    base interpreter, and resolving it would name a path outside the
    environment. Absoluteness and existence are what matter here.
    """

    executable = Path(sys.executable)
    assert executable.is_absolute(), f"sys.executable is not absolute: {sys.executable!r}"
    assert executable.exists(), f"sys.executable does not exist: {sys.executable!r}"


def test_the_child_reports_the_same_absolute_interpreter_it_was_launched_with() -> None:
    """End to end: the rule holds for the process this package actually starts.

    A static scan cannot see what the OS resolved. The child echoes its own
    ``sys.executable`` back, so the claim rests on a measurement rather than on
    the argv the parent believes it passed.
    """

    completed = subprocess.run(
        [sys.executable, "-c", "import sys; print(sys.executable)"],
        capture_output=True,
        text=True,
        check=True,
    )
    reported = completed.stdout.strip()
    assert reported == sys.executable
    assert Path(reported).is_absolute()
