"""Tests for run metadata provenance collection."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from omegaconf import DictConfig, OmegaConf

import tpen.artifacts as artifacts
from tpen.run import prepare_run_context


class FakeCuda:
    """Small fake ``torch.cuda`` surface for metadata tests."""

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_properties(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(
            name="NVIDIA A100-SXM4-40GB",
            total_memory=40 * 1024**3,
            major=8,
            minor=0,
        )


class FakeTorch:
    """Small fake torch module for metadata tests."""

    __version__ = "2.9.0"
    version = SimpleNamespace(cuda="12.8")
    cuda = FakeCuda()


class FakePartialTorch:
    """Torch-like module missing optional CUDA and NN surfaces."""

    __version__ = "partial"


def test_artifacts_imports_first_in_a_fresh_interpreter() -> None:
    """Keep checkpoint package initialization from reintroducing an import cycle."""

    result = subprocess.run(
        [sys.executable, "-B", "-c", "import tpen.artifacts; print('import-ok')"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import-ok"


def test_collect_hardware_metadata_records_runtime_hardware_and_slurm(
    monkeypatch,
) -> None:
    monkeypatch.setattr(artifacts, "import_module", lambda name: FakeTorch)
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "kozinsky_gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    metadata = artifacts.collect_hardware_metadata(device="cuda", dtype="float64")

    assert metadata["runtime"]["device"] == "cuda"
    assert metadata["runtime"]["dtype"] == "float64"
    assert metadata["runtime"]["torch_version"] == "2.9.0"
    assert metadata["runtime"]["torch_cuda_version"] == "12.8"
    assert metadata["runtime"]["cuda_visible_devices"] == "0"
    assert metadata["hardware"]["hostname"]
    assert metadata["hardware"]["cpu_count_logical"] is not None
    assert metadata["hardware"]["cuda_available"] is True
    assert metadata["hardware"]["cuda_device_count"] == 1
    assert metadata["hardware"]["cuda_devices"] == [
        {
            "index": 0,
            "name": "NVIDIA A100-SXM4-40GB",
            "total_memory_bytes": 40 * 1024**3,
            "capability": "8.0",
        }
    ]
    for key, value in {
        "job_id": "123456",
        "array_task_id": "7",
        "cpus_per_task": "8",
        "job_partition": "kozinsky_gpu",
    }.items():
        assert metadata["slurm"][key] == value


def test_collect_hardware_metadata_tolerates_partial_torch(monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "import_module", lambda name: FakePartialTorch)

    metadata = artifacts.collect_hardware_metadata(device="cpu", dtype="float64")

    assert metadata["runtime"]["torch_version"] == "partial"
    assert metadata["runtime"]["torch_cuda_version"] is None
    assert metadata["hardware"]["cuda_available"] is False
    assert metadata["hardware"]["cuda_device_count"] == 0
    assert metadata["hardware"]["cuda_devices"] == []


def test_prepare_run_context_defaults_to_utc_timezone(tmp_path: Path) -> None:
    context = prepare_run_context(_run_cfg(tmp_path), config_path="test.yaml", command="pytest")

    assert context.clock.timezone == "UTC"
    assert context.cfg.run.timezone == "UTC"
    assert context.source_cfg.run.timezone == "UTC"
    assert context.metadata.timezone == "UTC"
    assert datetime.fromisoformat(context.metadata.timestamp).utcoffset().total_seconds() == 0.0


def test_prepare_run_context_enforces_configured_timezone(tmp_path: Path) -> None:
    cfg = _run_cfg(tmp_path)
    cfg.run.timezone = "America/New_York"

    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    timestamp = datetime.fromisoformat(context.metadata.timestamp)
    assert context.clock.timezone == "America/New_York"
    assert context.cfg.run.timezone == "America/New_York"
    assert context.metadata.timezone == "America/New_York"
    assert timestamp.utcoffset() == ZoneInfo("America/New_York").utcoffset(timestamp)


def test_prepare_run_context_writes_run_start_breadcrumb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    cfg = _run_cfg(tmp_path)
    cfg.study = {"name": "study", "config_id": "cfg"}

    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest command")

    run_start = json.loads((context.run_dir / "run_start.json").read_text())
    assert run_start["run_id"] == context.metadata.run_id
    assert run_start["run_dir"] == context.metadata.run_dir
    assert run_start["study"] == {"name": "study", "config_id": "cfg"}
    assert run_start["hostname"]
    assert run_start["command"] == "pytest command"
    assert run_start["git"]["sha"] == context.metadata.git_commit
    assert run_start["slurm"]["job_id"] == "123456"
    assert run_start["slurm"]["array_task_id"] == "7"
    assert run_start["environment"]["SLURM_JOB_ID"] == "123456"
    assert "WANDB_API_KEY" not in run_start["environment"]
    assert "WANDB_API_KEY" not in json.dumps(run_start)
    assert isinstance(run_start["start_time_unix"], float)


def test_prepare_run_context_rejects_invalid_timezone(tmp_path: Path) -> None:
    cfg = _run_cfg(tmp_path)
    cfg.run.timezone = "Mars/Base"

    with pytest.raises(ValueError, match="run.timezone"):
        prepare_run_context(cfg, config_path="test.yaml", command="pytest")


def test_prepare_run_context_default_run_name_is_tpen_run(tmp_path: Path) -> None:
    """The unnamed-run fallback is a DURABLE public name, so it is pinned here.

    ``run_name`` reaches run directories, ``run_start.json``, and durable metadata, so
    a silent drift in this default renames artifacts for every config that omits both
    ``experiment.run_name`` and ``experiment.name``. It was ``spenn_run`` before the
    v0.3.0 rename and nothing pinned it, which is exactly how that residue survived.
    """

    cfg = _run_cfg(tmp_path)
    del cfg.experiment.run_name
    del cfg.experiment.name

    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    assert context.metadata.run_name == "tpen_run"


def test_prepare_run_context_null_run_name_falls_back_to_experiment_name(tmp_path: Path) -> None:
    cfg = _run_cfg(tmp_path)
    cfg.experiment.run_name = None

    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    assert context.metadata.run_name == "metadata"
    assert context.cfg.experiment.run_name == "metadata"
    assert context.metadata.run_id.split("_")[2] == "metadata"


@pytest.mark.parametrize("run_name", [None, "", "   "])
@pytest.mark.parametrize("experiment_name", [None, "", "   "])
def test_prepare_run_context_normalizes_blank_durable_names(
    tmp_path: Path, run_name: str | None, experiment_name: str | None
) -> None:
    cfg = _run_cfg(tmp_path)
    cfg.experiment.run_name = run_name
    cfg.experiment.name = experiment_name

    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    assert context.metadata.run_name == "tpen_run"
    assert context.cfg.experiment.run_name == "tpen_run"
    assert "_tpen_run_" in context.metadata.run_id
    run_start = json.loads((context.run_dir / "run_start.json").read_text())
    assert run_start["run_id"] == context.metadata.run_id
    assert all(part and part != "None" for part in context.run_dir.parts)
    assert context.artifact_manager.experiment == "experiment"


@pytest.mark.parametrize("run_name", [None, "", "   "])
def test_prepare_run_context_uses_nonblank_experiment_name_for_namespace(
    tmp_path: Path, run_name: str | None
) -> None:
    cfg = _run_cfg(tmp_path)
    cfg.experiment.run_name = run_name
    cfg.experiment.name = "  named experiment  "

    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    assert context.metadata.run_name == "named experiment"
    assert context.artifact_manager.experiment == "named experiment"
    assert context.run_dir.parts[-3] == "named experiment"
    assert "_named_experiment_" in context.metadata.run_id


def test_prepare_run_context_prefers_nonblank_run_name_and_strips_it(tmp_path: Path) -> None:
    cfg = _run_cfg(tmp_path)
    cfg.experiment.run_name = "  configured run  "
    cfg.experiment.name = "experiment family"

    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    assert context.metadata.run_name == "configured run"
    assert context.cfg.experiment.run_name == "configured run"
    assert context.artifact_manager.experiment == "experiment family"
    assert "_configured_run_" in context.metadata.run_id
    run_start = json.loads((context.run_dir / "run_start.json").read_text())
    assert run_start["run_id"] == context.metadata.run_id


def _run_cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "experiment": {"name": "metadata", "sector": "unit", "run_name": "metadata_unit"},
            "run": {"root": str(tmp_path), "run_id": None, "dir": None},
            "runtime": {"device": "cpu", "dtype": "float64"},
            "callbacks": [],
            "loggers": [],
        }
    )
