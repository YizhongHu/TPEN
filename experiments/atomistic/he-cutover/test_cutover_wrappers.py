from __future__ import annotations

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

run_train_row = sibling(__file__, 'run_train_row')


def test_allocation_detection_accepts_slurm() -> None:
    assert run_train_row.require_allocation({"SLURM_JOB_ID": "123"}) == "123"


def test_allocation_detection_accepts_pbs() -> None:
    assert run_train_row.require_allocation({"PBS_JOBID": "456.server"}) == "456.server"


def test_allocation_detection_refuses_login_node() -> None:
    with pytest.raises(RuntimeError, match="allocation required"):
        run_train_row.require_allocation({})

