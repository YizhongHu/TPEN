"""Deliberate-red probes for the current H-v2 boundary scanner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("h_v2_boundary_under_test", HERE / "test_import_boundary.py")
assert spec is not None and spec.loader is not None
boundary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = boundary
spec.loader.exec_module(boundary)


@pytest.mark.parametrize(
    ("number", "source"),
    [
        (1, 'import importlib\nm = importlib.import_module("tpen.physics.kinetic")'),
        (2, 'k = __import__("tpen.physics.kinetic")'),
        (3, 'import pytest\nk = pytest.importorskip("tpen.physics.kinetic")'),
        (4, 'import hydra\nobj = hydra.utils.instantiate({})'),
        (5, 'from hydra import utils\nobj = utils.instantiate({})'),
        (6, 'from hydra.utils import call'),
        (7, 'from hydra.utils import get_class'),
        (8, 'import importlib.util\nspec = importlib.util.spec_from_file_location("k", "../../../tpen/physics/kinetic.py")'),
    ],
)
def test_current_scanner_rejects_dynamic_forms(number: int, source: str, tmp_path: Path) -> None:
    """Each form should be rejected; failures expose a scanner gap."""

    path = tmp_path / f"probe_{number}.py"
    path.write_text(source, encoding="utf-8")
    assert boundary._violations(path), f"probe {number} was missed"


def test_current_scanner_rejects_subdirectory_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The scanner must still catch a nested violation when the sweep is recursive."""

    for name in ("protocol.py", "collect.py", "test_pipeline.py", "test_collect.py", "conftest.py"):
        (tmp_path / name).write_text("", encoding="utf-8")
    probe = tmp_path / "_probe_sub" / "x.py"
    probe.parent.mkdir()
    probe.write_text("import tpen\n", encoding="utf-8")
    monkeypatch.setattr(boundary, "HERE", tmp_path)
    with pytest.raises(AssertionError):
        boundary.test_experiment_files_only_reach_core_through_run_from_config()


def test_current_scanner_rejects_empty_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty sweep should fail closed rather than pass vacuously."""

    monkeypatch.setattr(boundary, "HERE", tmp_path)
    with pytest.raises(AssertionError):
        boundary.test_experiment_files_only_reach_core_through_run_from_config()


def test_current_scanner_controls(tmp_path: Path) -> None:
    """The intended direct-import control pair remains meaningful."""

    flagged = tmp_path / "_control_flagged.py"
    allowed = tmp_path / "_control_allowed.py"
    flagged.write_text("from tpen.data.batch import ElectronBatch\n", encoding="utf-8")
    allowed.write_text("from tpen.run import run_from_config\n", encoding="utf-8")
    assert boundary._violations(flagged)
    assert not boundary._violations(allowed)
