"""Review round 1 controls for the shared RunContext double in test_launch.py.

Each control loads a MUTATED COPY of ``test_launch.py`` from ``tmp_path`` and
asks the PR's own drift arms (field arm and structural arm) to reject it. A
control is RED while the arms accept a double that bypasses the production
constructor. This module imports no ``tpen`` name: the mutant copy imports
``tpen`` at load time from outside the boundary scanner's source walk.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

_LAUNCH_TESTS = Path(__file__).with_name("test_launch.py")


def _mutated_source(*, append: str = "", replace: tuple[str, str] | None = None) -> str:
    """Return test_launch.py's source with one mutation applied, guarded as applied."""

    original = _LAUNCH_TESTS.read_text(encoding="utf-8")
    source = original
    if replace is not None:
        needle, replacement = replace
        # Unapplied-mutation guard: a needle that is absent or ambiguous would
        # otherwise produce a "mutant" identical to the original and a false result.
        assert source.count(needle) == 1, f"needle count != 1: {needle!r}"
        source = source.replace(needle, replacement)
    source += append
    assert source != original, "mutation did not change the source"
    return source


def _load(tmp_path: Path, name: str, source: str) -> ModuleType:
    """Load *source* as a throwaway module named ``review_r1_<name>``."""

    path = tmp_path / f"review_r1_{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"review_r1_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _arms_reject(module: ModuleType) -> bool:
    """Return whether either PR drift arm fails on *module*."""

    arms = (
        module.test_fake_context_tracks_run_context_init_fields,
        module.test_fake_context_doubles_use_shared_contract,
    )
    for arm in arms:
        try:
            arm()
        except AssertionError:
            return True
    return False


_NESTED_SAME_NAME = '''

def test_review_r1_nested_same_name_double() -> None:
    class _FakeContext(RunContext):  # bypasses RunContext.__init__
        def __init__(self) -> None:
            self.cfg = None

    assert _FakeContext().cfg is None
'''

_SUBCLASS_OF_DOUBLE = '''

class _ReviewVariant(_FakeContext):  # bypasses RunContext.__init__
    def __init__(self) -> None:
        self.cfg = None
'''

_NESTED_OTHER_NAME = '''

def test_review_r1_nested_other_name_double() -> None:
    class FakeContext(RunContext):  # bypasses RunContext.__init__
        def __init__(self) -> None:
            self.cfg = None

    assert FakeContext().cfg is None
'''

# Make _FakeContext skip RunContext.__init__ while keeping the call text parseable.
_SKIP_INIT = (
    "            profile_writer=None,\n        )\n",
    "            profile_writer=None,\n        ) if False else None\n",
)

# Test 4's handled-failure runner replaced by one whose construction raises.
_RAISING_RUNNER = (
    "        lambda _: HandledFailureRunner(),\n",
    "        lambda _: (_ for _ in ()).throw(RuntimeError('review r1: runner construction raised')),\n",
)


# RED at 4eac71dd (issue I1): a nested double named _FakeContext defeats both arms.
def test_drift_arms_reject_nested_same_name_double(tmp_path: Path) -> None:
    module = _load(tmp_path, "nested_same_name", _mutated_source(append=_NESTED_SAME_NAME))
    assert _arms_reject(module)


# RED at 4eac71dd (issue I2): a subclass of _FakeContext that skips __init__ defeats both arms.
def test_drift_arms_reject_subclass_of_shared_double(tmp_path: Path) -> None:
    module = _load(tmp_path, "subclass_of_double", _mutated_source(append=_SUBCLASS_OF_DOUBLE))
    assert _arms_reject(module)


# RED at 4eac71dd (issue I4): test 4 passes even when run_from_config raises instead of
# returning the handled failure, because the shared double now survives the exception path.
def test_handled_failure_control_rejects_raising_runner(tmp_path: Path) -> None:
    module = _load(tmp_path, "raising_runner", _mutated_source(replace=_RAISING_RUNNER))
    cell_root = tmp_path / "cell"
    cell_root.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        with pytest.raises(AssertionError):
            module.test_launch_propagates_success_and_handled_failure_exit_codes(
                cell_root, monkeypatch
            )


# CONTROL, GREEN at 4eac71dd: the writer's mutant-B shape (nested, other name) IS caught.
def test_control_drift_arms_reject_nested_other_name_double(tmp_path: Path) -> None:
    module = _load(tmp_path, "nested_other_name", _mutated_source(append=_NESTED_OTHER_NAME))
    assert _arms_reject(module)


# CONTROL, GREEN at 4eac71dd: a _FakeContext that skips RunContext.__init__ IS caught.
def test_control_drift_arms_reject_skipped_init(tmp_path: Path) -> None:
    module = _load(tmp_path, "skip_init", _mutated_source(replace=_SKIP_INIT))
    assert _arms_reject(module)


# CONTROL, GREEN at 4eac71dd: the unmutated file is accepted by both arms.
def test_control_drift_arms_accept_unmutated_source(tmp_path: Path) -> None:
    module = _load(tmp_path, "unmutated", _LAUNCH_TESTS.read_text(encoding="utf-8"))
    assert not _arms_reject(module)


# RED at 77f9d7be (round-2 F1): the structural arm's visitor pushes only ClassDef
# nodes onto its parent stack, so a double nested inside a FUNCTION is reported
# as module-level. Duplicates are still caught by count, but the arm's
# "exactly one module-level _FakeContext" claim is not what it measures.
def test_structural_arm_reports_function_nested_double_as_nested(tmp_path: Path) -> None:
    module = _load(tmp_path, "function_nested_flag", _mutated_source(append=_NESTED_SAME_NAME))
    with pytest.raises(AssertionError) as excinfo:
        module.test_fake_context_doubles_use_shared_contract()
    assert "('_FakeContext', False)" in str(excinfo.value), str(excinfo.value)
