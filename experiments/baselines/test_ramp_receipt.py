"""Offline controls for the BL-R1 receipt, trace, and inventory mechanics."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.baselines.runbooks.ramp_receipt import (
    gate,
    inventory,
    check_qstat_poll,
    parse_ferminet_logged_config,
    record,
    validate_ferminet_trace,
)


def test_pbs_layout_records_and_gates_existing_row_directories(tmp_path):
    """Use the exact row/receipt arguments used by the PBS template."""

    root = tmp_path / "ramp"
    root.mkdir()
    rows = ["ferminet-codebase-ferminet-seed0", "ferminet-codebase-psiformer-seed0"]
    for row in rows:
        (root / row).mkdir()
        record(root / row / "receipt.json", row=row, returncode=0, producer_label="ferminet f4b1846c + logging patch test")
    assert [entry["row"] for entry in gate(root, rows)["rows"]] == rows


def test_pbs_layout_rejects_missing_and_failed_receipts(tmp_path):
    """A pre-existing row alone, or a failed child, never opens the next wave."""

    root = tmp_path / "ramp"
    root.mkdir()
    (root / "missing").mkdir()
    (root / "failed").mkdir()
    record(root / "failed" / "receipt.json", row="failed", returncode=7, producer_label="ferminet f4b1846c + logging patch test")
    with pytest.raises(RuntimeError, match="missing"):
        gate(root, ["missing", "failed"])
    with pytest.raises(RuntimeError, match="returncode"):
        gate(root, ["failed"])


def test_record_refuses_an_existing_receipt_and_missing_row(tmp_path):
    """A prior attempt is preserved and PBS must establish each row directory."""

    path = tmp_path / "a" / "receipt.json"
    with pytest.raises(FileNotFoundError):
        record(path, row="a", returncode=0, producer_label="ferminet f4b1846c + logging patch test")
    path.parent.mkdir()
    record(path, row="a", returncode=0, producer_label="ferminet f4b1846c + logging patch test")
    with pytest.raises(FileExistsError):
        record(path, row="a", returncode=0, producer_label="ferminet f4b1846c + logging patch test")


def test_producer_logged_config_rejects_real_pbs_7857872_fixture():
    """The receipt measures FermiNet's logged config, not intended overrides."""

    stderr = (Path(__file__).parent / "testdata" / "ferminet-stderr-pbs7857872-config.log").read_text()
    with pytest.raises(ValueError, match="reset_if_nan"):
        parse_ferminet_logged_config(stderr, steps=1000)


def test_producer_logged_config_accepts_same_dump_when_boolean_is_false():
    """A producer dump with the required values is accepted."""

    stderr = (Path(__file__).parent / "testdata" / "ferminet-stderr-pbs7857872-config.log").read_text()
    stderr = stderr.replace("  reset_if_nan: true", "  reset_if_nan: false")
    config = parse_ferminet_logged_config(stderr, steps=1000)
    assert config["reset_if_nan"] is False
    assert config["check_nan"] is True
    assert config["deterministic"] is True
    assert config["seed"] == 0
    assert config["batch_size"] == 4096
    assert config["iterations"] == 1000
    assert config["pretrain_iterations"] == 0


def _write_trace(path, rows, fieldnames=("step", "energy", "ewmean", "ewvar", "pmove", "walker_variance")):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_ferminet_trace_requires_pinned_schema_and_complete_values(tmp_path):
    """The producer schema permits only complete finite measurements."""

    path = tmp_path / "train_stats.csv"
    rows = [
        {"step": index, "energy": -0.5, "ewmean": -0.5, "ewvar": 0.1, "pmove": 0.5, "walker_variance": 0.2}
        for index in range(3)
    ]
    _write_trace(path, rows)
    assert validate_ferminet_trace(path, steps=3)["steps"] == 3
    _write_trace(path, rows, fieldnames=("step", "energy", "ewmean", "ewvar", "pmove"))
    with pytest.raises(ValueError, match="schema"):
        validate_ferminet_trace(path, steps=3)
    _write_trace(path, [{**row, "energy": ""} if row["step"] == 1 else row for row in rows])
    with pytest.raises(ValueError, match="missing energy"):
        validate_ferminet_trace(path, steps=3)
    _write_trace(path, rows[:-1] + [{"step": 2, "energy": -0.5}])
    with pytest.raises(ValueError, match="missing ewmean"):
        validate_ferminet_trace(path, steps=3)
    _write_trace(path, [{**row, "walker_variance": ""} if row["step"] == 1 else row for row in rows])
    with pytest.raises(ValueError, match="missing walker_variance"):
        validate_ferminet_trace(path, steps=3)
    _write_trace(path, [{**row, "walker_variance": "nan"} if row["step"] == 1 else row for row in rows])
    with pytest.raises(ValueError, match="non-finite walker_variance"):
        validate_ferminet_trace(path, steps=3)


def test_inventory_preserves_partial_account_before_failure(tmp_path):
    """Missing required artifacts are recorded rather than silently omitted."""

    (tmp_path / "present.txt").write_text("evidence", encoding="utf-8")
    output = tmp_path / "inventory.jsonl"
    with pytest.raises(RuntimeError, match="missing.txt"):
        inventory(tmp_path, output, ["present.txt", "missing.txt"])
    assert '"status": "missing"' in output.read_text(encoding="utf-8")


def test_verify_submitter_has_complete_exported_control_surface():
    """Verification submission is a whole script, not a qsub splice fragment."""

    script = (Path(__file__).parent / "runbooks" / "h-atom-r1-verify-submit.sh").read_text(encoding="utf-8")
    assert 'success-mini) INJECT_FAIL=' in script
    assert 'fail-ferminet-ferminet) INJECT_FAIL=ferminet-codebase-ferminet-seed0' in script
    assert 'export BL_R1_MODE=verify BL_R1_VERIFY_STEPS=10' in script
    assert 'QSUB_VARIABLES=RUN_ROOT,SOURCE_CHECKOUT,RUNBOOK_COMMIT,SCIENCE,PRE_FREEZE_RECEIPT,PRE_FREEZE_RECEIPT_SHA256,PATCH_BLOB,BL_R1_MODE,BL_R1_VERIFY_STEPS' in script
    assert 'PATCH_BLOB experiments/baselines/runbooks/patches/ferminet-f4b1846c-walker-variance-logging.patch' in script
    assert 'BL_R1_VERIFY_INJECT_FAIL' in script
    assert '--dry-run' in script
    assert 'poll-check --path "$query" --field state' in script
    assert 'QUEUE_POLL_LIMIT=120' in script and 'RUN_POLL_LIMIT=120' in script


def test_pbs_stages_patched_source_without_ownership_copy_and_traps_early():
    """Eagle staging must avoid ownership preservation and retain setup failures."""

    script = (Path(__file__).parent / "runbooks" / "h-atom-r1-ramp.pbs").read_text(encoding="utf-8")
    clone = 'git clone --no-hardlinks --quiet "$FN_SHARED_SRC" "$PATCHED_SRC"'
    assert clone in script
    assert 'cp -a "$FN_SHARED_SRC"' not in script
    assert script.index("trap on_exit EXIT") < script.index(clone)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("job_state = Q\nqueue = debug\n", {"state": "Q", "phase": "queue"}),
        ("job_state = R\nrun_count = 1\n", {"state": "R", "phase": "run"}),
        ("job_state = F\nrun_count = 1\nExit_status = 0\n", {"state": "F", "phase": "terminal"}),
    ],
)
def test_qstat_poll_accepts_real_queue_run_and_finished_contracts(tmp_path, text, expected):
    """Queued PBS output has no run_count; started/finished output must have one."""

    path = tmp_path / "qstat.txt"
    path.write_text(text, encoding="utf-8")
    assert check_qstat_poll(path) == expected


@pytest.mark.parametrize(
    "text",
    [
        "job_state = R\nrun_count = 2\n",
        "job_state = F\nrun_count = 2\nExit_status = 0\n",
        "job_state = X\n",
    ],
)
def test_qstat_poll_rejects_restart_and_unknown_state(tmp_path, text):
    """Restarted and unrecognized scheduler observations are stop conditions."""

    path = tmp_path / "qstat.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        check_qstat_poll(path)
