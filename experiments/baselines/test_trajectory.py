"""Offline tests for the common baseline trajectory reader."""

import csv
import json
from pathlib import Path

import pytest

from experiments.baselines.adapters import ferminet
from experiments.baselines.errors import AdapterError
from experiments.baselines.trajectory import main, read_trajectory, summarize


def _write_ferminet(path: Path) -> None:
    path.write_text(
        "step,energy,ewmean,ewvar,pmove\n"
        "10,-1.0,-1.0,0.25,0.5\n"
        "11,-2.0,-1.5,0.16,0.5\n"
        "12,nan,nan,nan,0.5\n"
        "13,-4.0,-2.0,0.04,0.5\n",
        encoding="utf-8",
    )


def test_ferminet_keeps_ewvar_out_of_walker_variance(tmp_path: Path):
    _write_ferminet(tmp_path / ferminet.TRAIN_STATS_FILENAME)
    rows = read_trajectory("ferminet", tmp_path)
    assert rows[0].walker_local_energy_variance is None
    assert rows[0].energy_time_ewvar == 0.25
    assert rows[2].finite is False
    summary = summarize(rows, window_fraction=0.5)
    assert summary["non_finite_count"] == 1
    assert summary["first_non_finite_step"] == 12
    assert summary["walker_variance_ratio_last_over_first"] is None
    assert "unavailable" in summary["walker_variance_unavailable_reason"]


def test_deepqmc_squares_std_and_cli_writes_requested_outputs(tmp_path: Path):
    h5py = pytest.importorskip("h5py", reason="h5py is not a TPEN dependency")
    training = tmp_path / "training"
    training.mkdir()
    with h5py.File(training / "result.h5", "w") as handle:
        handle.create_dataset("local_energy/mean", data=[[-1.0], [-2.0], [-3.0], [-4.0]])
        handle.create_dataset("local_energy/std", data=[[2.0], [1.0], [0.5], [0.25]])
    rows = read_trajectory("deepqmc", tmp_path)
    assert [row.step for row in rows] == [0, 1, 2, 3]
    assert [row.walker_local_energy_variance for row in rows] == [4.0, 1.0, 0.25, 0.0625]
    csv_path, json_path = tmp_path / "trajectory.csv", tmp_path / "summary.json"
    assert main(["--code", "deepqmc", str(tmp_path), "--out", str(csv_path), "--json", str(json_path)]) == 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.DictReader(handle))["walker_local_energy_variance"] == "4.0"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["window_size"] == 1
    assert payload["walker_variance_ratio_last_over_first"] == pytest.approx(0.015625)


def test_missing_trajectory_column_or_dataset_is_refused(tmp_path: Path):
    (tmp_path / ferminet.TRAIN_STATS_FILENAME).write_text("step,energy\n1,-1\n", encoding="utf-8")
    with pytest.raises(AdapterError, match="ewvar"):
        read_trajectory("ferminet", tmp_path)

    h5py = pytest.importorskip("h5py", reason="h5py is not a TPEN dependency")
    training = tmp_path / "training"
    training.mkdir()
    with h5py.File(training / "result.h5", "w") as handle:
        handle.create_dataset("local_energy/mean", data=[[-1.0]])
    with pytest.raises(AdapterError, match="local_energy/std"):
        read_trajectory("deepqmc", tmp_path)


@pytest.mark.parametrize(
    ("mean_shape", "std_shape"),
    [((2, 2, 1), (2, 2, 1)), ((2, 2, 1), (4, 1, 1))],
)
def test_deepqmc_rejects_molecule_state_axes_and_shape_mismatch(
    tmp_path: Path, mean_shape: tuple[int, ...], std_shape: tuple[int, ...]
):
    h5py = pytest.importorskip("h5py", reason="h5py is not a TPEN dependency")
    training = tmp_path / "training"
    training.mkdir()
    with h5py.File(training / "result.h5", "w") as handle:
        handle.create_dataset("local_energy/mean", data=[-1.0] * 4, shape=mean_shape)
        handle.create_dataset("local_energy/std", data=[1.0] * 4, shape=std_shape)
    with pytest.raises(AdapterError):
        read_trajectory("deepqmc", tmp_path)


def test_deepqmc_singleton_axes_and_persisted_steps_are_metadata(tmp_path: Path):
    h5py = pytest.importorskip("h5py", reason="h5py is not a TPEN dependency")
    training = tmp_path / "training"
    training.mkdir()
    with h5py.File(training / "result.h5", "w") as handle:
        handle.create_dataset("local_energy/mean", data=[-1.0, -2.0], shape=(2, 1, 1))
        handle.create_dataset("local_energy/std", data=[1.0, 0.5], shape=(2, 1, 1))
        handle.create_dataset("step", data=[100, 101])
    rows = read_trajectory("deepqmc", tmp_path)
    assert [row.step for row in rows] == [100, 101]
    assert rows.step_basis == "optimizer_step"
    assert summarize(rows)["step_basis"] == "optimizer_step"


def test_deepqmc_without_steps_uses_observation_index(tmp_path: Path):
    h5py = pytest.importorskip("h5py", reason="h5py is not a TPEN dependency")
    training = tmp_path / "training"
    training.mkdir()
    with h5py.File(training / "result.h5", "w") as handle:
        handle.create_dataset("local_energy/mean", data=[-1.0, -2.0], shape=(2, 1))
        handle.create_dataset("local_energy/std", data=[1.0, 0.5], shape=(2, 1))
    rows = read_trajectory("deepqmc", tmp_path)
    assert [row.step for row in rows] == [0, 1]


def test_ferminet_sparse_steps_have_optimizer_basis_and_strict_summary_json(tmp_path: Path):
    _write_ferminet(tmp_path / ferminet.TRAIN_STATS_FILENAME)
    rows = read_trajectory("ferminet", tmp_path)
    assert [row.step for row in rows] == [10, 11, 12, 13]
    summary = summarize(rows)
    json.dumps(summary, allow_nan=False)
    assert main(["--code", "ferminet", str(tmp_path)]) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == [ferminet.TRAIN_STATS_FILENAME]

