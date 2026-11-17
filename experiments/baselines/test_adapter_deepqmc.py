"""Tests for the DeepQMC run-directory adapter.

Most tests exercise :func:`record_from_series`, which takes an energy list and
therefore needs no HDF5 file and no ``h5py``. That split is deliberate: ``h5py``
is not a TPEN dependency, so tests that require it would be skipped in this
environment, and a skipped test protects nothing. The HDF5-reading tests are
gated on ``h5py`` and are the ones that must be run in the DeepQMC virtualenv on
the cluster.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from experiments.baselines.errors import AdapterError
import experiments.baselines.adapters.deepqmc as deepqmc
from experiments.baselines.records import BaselineRecord
from experiments.baselines.adapters.deepqmc import (
    DEFAULT_TAIL_FRACTION,
    ENERGY_DATASET,
    RECORD_FILENAME,
    build_record,
    main,
    parse_device,
    parse_wall_clock_seconds,
    read_energies,
    record_from_series,
    result_path,
    write_record,
)

LOG_TEXT = """ansatz=lapnet host=holygpu8a25406 job=39639503 start=2026-08-16T20:13:16-04:00
NVIDIA H200, GPU-1a4626a2-1754-c017-fdb7-6d7d682f4165, 143771 MiB
jax 0.8.3 [CudaDevice(id=0)]
end=2026-08-17T15:43:32-04:00
"""


def _converged(count: int = 40000, seed: int = 5) -> list[float]:
    """A plateaued series: noise about a fixed mean, no drift."""

    rng = random.Random(seed)
    return [-2.9037 + rng.gauss(0.0, 1e-5) for _ in range(count)]


# --------------------------------------------------------------------------
# code identity -- the defect this adapter exists to avoid
# --------------------------------------------------------------------------


def test_code_is_deepqmc_never_ferminet() -> None:
    """A DeepQMC record must never claim the FermiNet codebase.

    DeepQMC ships an ansatz named ``ferminet``, but it is a reimplementation.
    ``code="ferminet", ansatz="ferminet"`` claims a run of google-deepmind's
    code; this must claim DeepQMC's. The mistake is invisible to a reader,
    because the ansatz field looks right either way.
    """

    record = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="ferminet", run_id="r"
    )
    assert record.code == "deepqmc"
    assert record.ansatz == "ferminet"


def test_ansatz_is_recorded_not_assumed() -> None:
    """Nothing about a DeepQMC run directory reveals which ansatz produced it."""

    for ansatz in ("lapnet", "psiformer", "deeperwin", "default"):
        record = record_from_series(
            _converged(), system_id="he_atom", batch_size=4096, ansatz=ansatz, run_id="r"
        )
        assert record.ansatz == ansatz


def test_reimplementation_is_flagged_in_notes_but_native_default_is_not() -> None:
    """`default` is PauliNet's own code; the others are reimplementations.

    DeepQMC *is* the PauliNet repository, so that one asymmetry must not be
    flattened into "everything here is a reimplementation".
    """

    reimpl = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    native = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="default", run_id="r"
    )
    assert "REIMPLEMENTATION" in (reimpl.notes or "")
    assert "native" in (native.notes or "")
    assert "REIMPLEMENTATION" not in (native.notes or "")


# --------------------------------------------------------------------------
# estimator and convergence reporting
# --------------------------------------------------------------------------


def test_default_tail_is_long() -> None:
    """A short tail produced impossible below-exact energies four times here."""

    assert DEFAULT_TAIL_FRACTION >= 0.25


def test_notes_report_a_monotone_tail_as_possibly_unconverged() -> None:
    """A still-descending run must be flagged, not reported as a clean number.

    This is the failure that invalidated the largest effect in this program's
    six-system comparison: a tail average looked perfectly stable while the run
    was still improving at its final step.
    """

    descending = [-7.4779 - 1e-9 * i for i in range(80000)]
    record = record_from_series(
        descending, system_id="li_atom", batch_size=4096, ansatz="ferminet", run_id="r"
    )
    assert "MONOTONE" in (record.notes or "")
    assert "may not have converged" in (record.notes or "")


def test_notes_report_a_plateaued_tail_as_noise() -> None:
    record = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    assert "not monotone" in (record.notes or "")
    assert "MONOTONE" not in (record.notes or "")


def test_a_tail_below_the_block_floor_is_refused_not_given_a_zero_bar() -> None:
    """The predecessor of this test accepted the record this one refuses.

    It drove six values through the adapter and asserted only that the NOTES
    said ``UNASSESSED`` and ``provisional``. The record it accepted carried
    ``energy_stderr_hartree = 0.0``, because a window that cannot fill one
    blocking level used to return the loop's zero initialiser. Prose caveats do
    not travel: the 0.0 would have been read as infinite precision by anything
    that quoted the field instead of the sentence. A bar that cannot be assessed
    is not a wide bar, so no record is emitted at all.
    """

    with pytest.raises(AdapterError) as caught:
        record_from_series(
            [-2.9 + 1e-6 * i for i in range(6)],
            system_id="he_atom",
            batch_size=16,
            ansatz="lapnet",
            run_id="r",
            tail_fraction=1.0,
            allow_short_tail=True,
        )
    # THIS ADAPTER must be the layer that refused, and the assert this block
    # replaced could not say so. `assert "32-block" in message` alone passes
    # whether this adapter refuses or the layer underneath does, because both
    # refusals carry that substring:
    #   statistics.py   f"window of {len(data)} values cannot fill the "
    #                   f"{min_blocks}-block minimum "
    #   this adapter    f"tail of {len(tail)} steps is below the "
    #                   f"{MIN_BLOCKS}-block minimum for "
    # That sentence describes the REPLACED line, not the ones below it.
    #
    # Measured in job 41112543 at tree dc1216e over the mutant set
    # M2+M-WRAP -- M2 withholds the allow_below_floor flag so the layer below
    # raises instead; M-WRAP wraps and re-raises the lower layer's error:
    #   full arm (pos+negA+negB)  M2 KILL at the positive, M-WRAP KILL at negA
    #   pos alone                 M2 KILL, M-WRAP SURVIVE
    #   negA alone                both KILL
    #   negB alone                both KILL
    #   DISPENSABILITY dispensable=[negA, negB, pos] sufficient_alone=[negA, negB]
    # So over THAT set no single assert is indispensable, and the positive one
    # is not sufficient alone: an `in` check is satisfied by a superstring, so a
    # wrapped re-raise passes it. Dispensability is mutant-set-scoped -- do not
    # read these three lines as redundant against a set nobody has named.
    #
    # The message must also name an action a CLI user can actually take, so
    # naming a keyword argument no command line can pass would describe the flag
    # this caller already used. That is the second reason the last line stays.
    message = str(caught.value)
    assert "is below the" in message and "-block minimum for" in message
    assert "cannot fill the" not in message
    assert "allow_below_floor" not in message


def test_the_short_tail_opt_in_still_buys_a_real_bar_above_the_block_floor() -> None:
    """The opt-in is not revoked, only bounded.

    A run shorter than ``MIN_TAIL_STEPS`` but long enough to block is still
    emitted, with the window's provisional status stated. Refusing this too
    would make the flag dead rather than narrower.
    """

    record = record_from_series(
        _converged(count=4000),
        system_id="he_atom",
        batch_size=16,
        ansatz="lapnet",
        run_id="r",
        tail_fraction=1.0,
        allow_short_tail=True,
    )
    assert record.energy_stderr_hartree is not None
    assert record.energy_stderr_hartree > 0.0
    assert "provisional" in (record.notes or "")


def test_no_emitted_field_or_note_carries_the_word_none() -> None:
    """A content check, not an exception check.

    ``f"{factor:.2f}x"`` raises TypeError when the factor is missing, which is
    not an ``AdapterError`` and so is caught by no handler here; ``f"{factor}x"``
    raises nothing at all and renders the word into the record. Only reading the
    emitted payload catches the second case.
    """

    record = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    assert "None" not in json.dumps(record.to_json_dict())
    assert "None" not in (record.notes or "")


# --------------------------------------------------------------------------
# the command line -- the surface a human actually runs
# --------------------------------------------------------------------------
# Before this section, `grep -c 'main(' ` on this file returned 0: every test
# called the library and none had ever crossed the entry point, so no test in
# it could observe an exit status, a stream, or a file on disk however green the
# file looked. These tests monkeypatch `read_energies` rather than write HDF5,
# so they are NOT gated on h5py -- the CLI contract being asserted here is exit
# status, stderr, and absence of a file, none of which involve reading a file
# format. A test skipped for a missing dependency protects nothing.


def _cli_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                 series: list[float]) -> Path:
    """A run directory whose energy series is supplied, not read from HDF5."""

    run_dir = tmp_path / "run39411090"
    run_dir.mkdir()
    (run_dir / "training" / ".hydra").mkdir(parents=True)
    (run_dir / "training" / ".hydra" / "config.yaml").write_text(
        "task:\n  seed: 0\n", encoding="utf-8"
    )
    monkeypatch.setattr(deepqmc, "read_energies", lambda _: list(series))
    return run_dir


def test_cli_refuses_with_status_one_and_writes_no_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refusal is a non-zero exit, a message on stderr, and nothing on disk.

    All four are asserted together. An exit status alone would not catch a
    partial record left behind, and a record left behind is the failure that
    matters: a file on disk outlives the shell that printed the warning.
    """

    run_dir = _cli_run_dir(tmp_path, monkeypatch, [-2.9 + 1e-6 * i for i in range(6)])
    status = main(
        [
            "--run-dir", str(run_dir),
            "--system-id", "he_atom",
            "--batch-size", "16",
            "--ansatz", "lapnet",
            "--tail-fraction", "1.0",
            "--allow-short-tail",
        ]
    )
    assert status == 1
    captured = capsys.readouterr()
    assert "32-block" in captured.err
    # No record-shaped text on stdout either: a caller redirecting stdout to a
    # file would otherwise capture a refusal as though it were a record.
    assert "energy_hartree" not in captured.out
    assert not (run_dir / RECORD_FILENAME).exists()


def test_cli_writes_a_record_and_exits_zero_when_the_bar_is_assessable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal above must not be the only outcome the CLI can produce."""

    run_dir = _cli_run_dir(tmp_path, monkeypatch, _converged())
    status = main(
        [
            "--run-dir", str(run_dir),
            "--system-id", "he_atom",
            "--batch-size", "4096",
            "--ansatz", "lapnet",
        ]
    )
    assert status == 0
    written = json.loads((run_dir / RECORD_FILENAME).read_text())
    assert written["energy_stderr_hartree"] > 0.0


def test_the_short_tail_flag_help_names_the_block_floor_it_does_not_relax(
    capsys: pytest.CaptureFixture[str]
) -> None:
    """The help text is part of the contract, not commentary on it.

    ``--allow-short-tail`` reads as "accept a short window" while the blocking
    floor it does not relax is what will actually refuse the run. A user who
    passes the flag and is then refused anyway needs the reason to be in the
    text of the flag they read.
    """

    with pytest.raises(SystemExit):
        main(["--help"])
    assert "32-block" in capsys.readouterr().out


def test_notes_carry_the_autocorrelation_inflation_ratio() -> None:
    record = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    assert "inflation" in (record.notes or "")


def test_estimator_distinguishes_training_tail_from_inference() -> None:
    train = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    infer = record_from_series(
        _converged(),
        system_id="he_atom",
        batch_size=4096,
        ansatz="lapnet",
        run_id="r",
        estimator="inference",
        tail_fraction=1.0,
    )
    assert train.estimator == "training_tail"
    assert infer.estimator == "inference"
    assert "Training-tail average" in (train.notes or "")
    assert "Fixed-parameter inference" in (infer.notes or "")


# --------------------------------------------------------------------------
# arithmetic and validation
# --------------------------------------------------------------------------


def test_steps_and_samples_count_the_whole_run_not_the_tail() -> None:
    """`steps` is the run length; the tail is only the estimator window."""

    record = record_from_series(
        _converged(count=40000), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    assert record.steps == 40000
    assert record.samples == 40000 * 4096


def test_energy_is_the_tail_mean_not_the_final_value() -> None:
    """A final row is one step's walker mean and is pure noise at this scale."""

    series = [-1.0] * 39999 + [999.0]
    record = record_from_series(
        series, system_id="he_atom", batch_size=16, ansatz="lapnet", run_id="r", tail_fraction=0.5
    )
    assert record.energy_hartree == pytest.approx(-1.0, abs=0.1)


def test_rejects_out_of_range_tail_fraction() -> None:
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(AdapterError, match="tail fraction must be in"):
            record_from_series(
                _converged(), system_id="he_atom", batch_size=16, ansatz="lapnet", run_id="r",
                tail_fraction=bad,
            )


def test_rejects_a_run_shorter_than_the_floor() -> None:
    """A run too short for the minimum window must refuse, not quietly shrink.

    Naming both the run length and the floor is deliberate: "too short" is not
    actionable without them.
    """

    with pytest.raises(AdapterError, match="run of 3 steps cannot fill the 10000-step"):
        record_from_series(
            [-1.0, -1.1, -1.2],
            system_id="he_atom",
            batch_size=16,
            ansatz="lapnet",
            run_id="r",
        )


def test_floor_overrides_a_fraction_that_would_be_too_small() -> None:
    """0.25 of 20000 is 5000 steps, which produced impossible energies.

    The floor must win, so the window is 10000 and the notes say so in STEPS.
    """

    record = record_from_series(
        _converged(count=20000), system_id="he_atom", batch_size=4096,
        ansatz="psiformer", run_id="r", tail_fraction=0.25,
    )
    assert "last 10000 of 20000 steps" in (record.notes or "")
    assert "provisional" not in (record.notes or "")


def test_fraction_wins_when_it_exceeds_the_floor() -> None:
    """On a long run the fraction still governs; the floor is only a minimum."""

    record = record_from_series(
        _converged(count=200000), system_id="he_atom", batch_size=4096,
        ansatz="psiformer", run_id="r", tail_fraction=0.25,
    )
    assert "last 50000 of 200000 steps" in (record.notes or "")


def test_constant_window_is_refused_rather_than_given_a_zero_bar() -> None:
    """A zero error bar is the most reassuring number an emission can publish.

    Blocking returns 0.0 for a zero-variance series -- its running maximum is
    initialised at 0.0 and never beaten -- and ``BaselineRecord`` accepts 0.0 as
    a non-negative bar, so nothing downstream of the adapter stops the row. The
    adapter is the last place that knows a record is about to be written.
    """

    with pytest.raises(AdapterError, match="constant"):
        record_from_series(
            [-2.9037] * 40000, system_id="he_atom", batch_size=4096,
            ansatz="lapnet", run_id="r",
        )


def test_constant_window_is_refused_even_with_the_short_tail_opt_in() -> None:
    """The opt-in buys a short window, never an unmeasurable one.

    ``allow_short_tail`` says "this run is shorter than the standard window";
    it does not say "publish a bar you could not estimate".
    """

    with pytest.raises(AdapterError, match="constant"):
        record_from_series(
            [-2.9037] * 40, system_id="he_atom", batch_size=16,
            ansatz="lapnet", run_id="r", tail_fraction=1.0, allow_short_tail=True,
        )


def test_the_refusal_is_the_only_thing_stopping_a_zero_bar() -> None:
    """Pin the downstream permissiveness the guard above exists to cover.

    If this ever starts failing, ``records.py`` has begun rejecting a zero bar
    itself and the adapter guard's justification has changed -- which is worth
    knowing, not worth silently keeping.
    """

    record = BaselineRecord(
        system_id="he_atom", code="deepqmc", code_commit="0" * 40, ansatz="lapnet",
        energy_hartree=-2.9037, energy_stderr_hartree=0.0, steps=40000,
        samples=40000 * 4096, estimator="training_tail", run_id="r",
    )
    assert record.energy_stderr_hartree == 0.0


# --------------------------------------------------------------------------
# operator caveats -- facts the numbers cannot carry
# --------------------------------------------------------------------------


def test_note_is_appended_without_displacing_the_generated_account() -> None:
    """A caveat extends the provenance text; it never replaces any of it."""

    kwargs = dict(
        system_id="lih_molecule", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    plain = record_from_series(_converged(), **kwargs)
    caveated = record_from_series(
        _converged(), **kwargs, note="Ran at R=3.09913 bohr, registry is 3.015 bohr."
    )

    assert (plain.notes or "") in (caveated.notes or "")
    assert (caveated.notes or "").endswith(
        " Ran at R=3.09913 bohr, registry is 3.015 bohr."
    )


def test_note_changes_no_number() -> None:
    """The caveat is documentation; every estimated field must be untouched."""

    kwargs = dict(
        system_id="lih_molecule", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    plain = record_from_series(_converged(), **kwargs)
    caveated = record_from_series(_converged(), **kwargs, note="geometry deviates")

    assert caveated.energy_hartree == plain.energy_hartree
    assert caveated.energy_stderr_hartree == plain.energy_stderr_hartree
    assert caveated.steps == plain.steps
    assert caveated.samples == plain.samples


def test_omitting_the_note_leaves_the_record_unchanged() -> None:
    """Default behaviour is the pre-change adapter's, character for character."""

    kwargs = dict(
        system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    assert (
        record_from_series(_converged(), **kwargs, note=None).notes
        == record_from_series(_converged(), **kwargs).notes
    )


def test_whitespace_only_note_is_refused_rather_than_dropped() -> None:
    """A caveat that silently vanishes is worse than no argument at all."""

    for blank in ("", "   ", "\t\n"):
        with pytest.raises(AdapterError, match="empty"):
            record_from_series(
                _converged(), system_id="he_atom", batch_size=4096,
                ansatz="lapnet", run_id="r", note=blank,
            )


def test_device_fields_stay_none_without_a_log() -> None:
    """Unknown provenance is None, never an invented value."""

    record = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    assert record.device_type is None
    assert record.gpu_model is None
    assert record.wall_clock_seconds is None
    assert record.dtype is None


def test_parse_device_reads_the_delivered_card() -> None:
    """`seas_gpu` mixes H200 and A100, so the partition does not imply hardware."""

    assert parse_device(LOG_TEXT) == ("cuda", "NVIDIA H200")


def test_parse_device_absent_yields_none_not_a_guess() -> None:
    assert parse_device("no device line here") == (None, None)


def test_parse_wall_clock_seconds() -> None:
    assert parse_wall_clock_seconds(LOG_TEXT) == pytest.approx(19 * 3600 + 30 * 60 + 16)


def test_record_round_trips_to_json(tmp_path: Path) -> None:
    record = record_from_series(
        _converged(), system_id="he_atom", batch_size=4096, ansatz="lapnet", run_id="r"
    )
    path = write_record(record, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["code"] == "deepqmc"
    assert payload["ansatz"] == "lapnet"
    assert payload["run_dir"] is None


# --------------------------------------------------------------------------
# HDF5 reading -- needs h5py, which is NOT a TPEN dependency
# --------------------------------------------------------------------------


def test_result_path_accepts_run_root_or_training_subdir(tmp_path: Path) -> None:
    nested = tmp_path / "training"
    nested.mkdir()
    (nested / "result.h5").write_bytes(b"")
    assert result_path(tmp_path) == nested / "result.h5"

    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "result.h5").write_bytes(b"")
    assert result_path(flat) == flat / "result.h5"


def test_missing_result_file_raises(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    with pytest.raises(AdapterError, match="no result.h5"):
        read_energies(tmp_path)


def test_reads_the_three_dimensional_dataset_shape(tmp_path: Path) -> None:
    """DeepQMC writes ``(steps, 1, 1)``; it must be flattened, not indexed."""

    h5py = pytest.importorskip("h5py")
    numpy = pytest.importorskip("numpy")
    run = tmp_path / "training"
    run.mkdir()
    values = numpy.asarray([-2.9, -2.91, -2.92], dtype="float32").reshape(3, 1, 1)
    with h5py.File(run / "result.h5", "w") as handle:
        handle.create_dataset(ENERGY_DATASET, data=values)

    assert read_energies(tmp_path) == pytest.approx([-2.9, -2.91, -2.92], abs=1e-6)


@pytest.mark.parametrize("shape", [(3,), (3, 1), (3, 1, 1)])
def test_read_energies_preserves_single_system_layouts(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    """Singleton molecule/state axes retain the historical flattened values."""

    h5py = pytest.importorskip("h5py")
    numpy = pytest.importorskip("numpy")
    run = tmp_path / "training"
    run.mkdir()
    values = numpy.asarray([-2.9, -2.91, -2.92], dtype="float32").reshape(shape)
    with h5py.File(run / "result.h5", "w") as handle:
        handle.create_dataset(ENERGY_DATASET, data=values)

    assert read_energies(tmp_path) == pytest.approx([-2.9, -2.91, -2.92], abs=1e-6)


@pytest.mark.parametrize("shape", [(3, 2), (3, 1, 2)])
def test_read_energies_rejects_molecule_and_state_axes(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    """Multiplicity axes must not be silently interpreted as extra steps."""

    h5py = pytest.importorskip("h5py")
    numpy = pytest.importorskip("numpy")
    run = tmp_path / "training"
    run.mkdir()
    values = numpy.arange(1, numpy.prod(shape) + 1, dtype="float32").reshape(shape)
    with h5py.File(run / "result.h5", "w") as handle:
        handle.create_dataset(ENERGY_DATASET, data=values)

    with pytest.raises(AdapterError, match="shape"):
        read_energies(tmp_path)


def test_missing_dataset_raises(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    run = tmp_path / "training"
    run.mkdir()
    with h5py.File(run / "result.h5", "w") as handle:
        handle.create_dataset("something/else", data=[1.0])

    with pytest.raises(AdapterError, match="has no 'local_energy/mean'"):
        read_energies(tmp_path)


def test_empty_dataset_raises_rather_than_returning_nothing(tmp_path: Path) -> None:
    """An absent result is an error, never a record with a null energy."""

    h5py = pytest.importorskip("h5py")
    numpy = pytest.importorskip("numpy")
    run = tmp_path / "training"
    run.mkdir()
    with h5py.File(run / "result.h5", "w") as handle:
        handle.create_dataset(ENERGY_DATASET, data=numpy.zeros((0, 1, 1), dtype="float32"))

    with pytest.raises(AdapterError, match="empty"):
        read_energies(tmp_path)


def test_build_record_end_to_end(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    numpy = pytest.importorskip("numpy")
    run = tmp_path / "training"
    run.mkdir()
    # The run must be long enough for the default estimator window: this test
    # is about the HDF5-to-record path, not about short tails, so it takes the
    # opt-in-free path and therefore has to clear statistics.MIN_TAIL_STEPS.
    # At 4000 steps it raised before the number reached any assertion below.
    # dev arrived at the same 40000 independently in #297; the count is the
    # agreement, the reshape(-1, ...) is kept so the two cannot drift apart
    # silently if the count is ever changed again.
    values = numpy.asarray(_converged(count=40000), dtype="float32").reshape(-1, 1, 1)
    with h5py.File(run / "result.h5", "w") as handle:
        handle.create_dataset(ENERGY_DATASET, data=values)
    (run / ".hydra").mkdir()
    (run / ".hydra" / "config.yaml").write_text("task:\n  seed: 17\n", encoding="utf-8")
    (tmp_path / "job.out").write_text(LOG_TEXT, encoding="utf-8")

    record = build_record(
        tmp_path,
        system_id="he_atom",
        batch_size=4096,
        ansatz="lapnet",
        log_path=tmp_path / "job.out",
        code_commit="cafe1234",
    )

    assert record.code == "deepqmc"
    assert record.ansatz == "lapnet"
    assert record.steps == 40000
    assert "last 10000 of 40000 steps" in (record.notes or "")
    assert record.gpu_model == "NVIDIA H200"
    assert record.code_commit == "cafe1234"
    assert record.seed == 17
    assert record.energy_hartree == pytest.approx(-2.9037, abs=1e-3)


def test_build_record_refuses_when_hydra_seed_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(deepqmc, "read_energies", lambda _: _converged())
    with pytest.raises(AdapterError, match="no DeepQMC Hydra config"):
        build_record(
            tmp_path,
            system_id="he_atom",
            batch_size=4096,
            ansatz="lapnet",
        )


def test_build_record_refuses_when_hydra_config_has_no_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "training" / ".hydra"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text("task:\n  steps: 200000\n", encoding="utf-8")
    monkeypatch.setattr(deepqmc, "read_energies", lambda _: _converged())

    with pytest.raises(AdapterError, match="has no task.seed"):
        build_record(
            tmp_path,
            system_id="he_atom",
            batch_size=4096,
            ansatz="lapnet",
        )


def test_build_record_refuses_a_supplied_seed_that_contradicts_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "training" / ".hydra"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text("task:\n  seed: 23\n", encoding="utf-8")
    monkeypatch.setattr(deepqmc, "read_energies", lambda _: _converged())

    with pytest.raises(AdapterError, match="supplied seed 7 contradicts run config seed 23"):
        build_record(
            tmp_path,
            system_id="he_atom",
            batch_size=4096,
            ansatz="lapnet",
            seed=7,
        )
