from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Siblings are loaded study-scoped, not by bare import: experiments/ has several
# same-named modules and the first study loaded would otherwise own the bare name
# for every study after it. See experiments/toolkit/study_imports.py.
#
# The loader is reached BY PATH rather than by putting the repository root on
# sys.path. A study directory that mutates sys.path is the mechanism behind the
# very defect this import exists to fix, and he-cutover's gateway test forbids it
# outright -- so the fix must not reintroduce it in order to install itself.
import importlib.util as _tpen_importlib  # noqa: E402
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

if "_tpen_study_imports" not in _tpen_sys.modules:
    _tpen_spec = _tpen_importlib.spec_from_file_location(
        "_tpen_study_imports",
        _TpenPath(__file__).resolve().parents[3] / "experiments" / "toolkit" / "study_imports.py",
    )
    _tpen_module = _tpen_importlib.module_from_spec(_tpen_spec)
    _tpen_sys.modules["_tpen_study_imports"] = _tpen_module
    _tpen_spec.loader.exec_module(_tpen_module)
sibling = _tpen_sys.modules["_tpen_study_imports"].sibling

admission = sibling(__file__, 'admission')
cutover_plan = sibling(__file__, 'cutover_plan')


def _plan(tmp_path: Path):
    grid = cutover_plan.load_grid(Path(__file__).with_name("smoke_grid.yaml"))
    return cutover_plan.build_plans(grid, facility="cannon", results_root=tmp_path, plan_id="p")[1]


def test_admission_preserves_unresolved_python_and_aligns_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = tmp_path / "venv"
    executable = prefix / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "executable", str(executable))
    dispatches = admission.admit_plan(_plan(tmp_path), admission_id="a", cwd=tmp_path, environment={})
    assert len(dispatches) == 2
    assert all(item.argv[0] == str(executable) for item in dispatches)
    assert all(item.argv[0] != str(executable.resolve()) for item in dispatches)
    assert [item.run_id for item in dispatches] == ["seed-000-chain-00", "seed-000-chain-01"]
    assert {item.runtime for item in dispatches} == {"tpen-cu126"}


def test_admission_rejects_interpreter_outside_selected_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    with pytest.raises(ValueError, match="outside sys.prefix"):
        admission.admit_plan(_plan(tmp_path), admission_id="a", cwd=tmp_path, environment={}, python=str(Path(sys.executable).resolve()))


def test_admission_rejects_visibility_environment_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "dispatch_specs.jsonl"
    error = None
    try:
        rows = admission.admit_plan(_plan(tmp_path), admission_id="a", cwd=tmp_path, environment={"CUDA_VISIBLE_DEVICES": "0"})
        admission.write_dispatch_specs(output, rows)
    except ValueError as exc:
        error = exc
    assert error is not None
    assert "allocation binding" in str(error)
    assert not output.exists()
