"""Offline controls for the BL-R1 receipt, trace, and inventory mechanics."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.baselines.runbooks.ramp_receipt import (
    gate,
    inventory,
    check_qstat_poll,
    run_mode_spec,
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
        parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")


def test_producer_logged_config_accepts_same_dump_when_boolean_is_false():
    """A producer dump with the required values is accepted."""

    stderr = (Path(__file__).parent / "testdata" / "ferminet-stderr-pbs7857872-config.log").read_text()
    stderr = stderr.replace("  reset_if_nan: true", "  reset_if_nan: false")
    config = parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")
    assert config["reset_if_nan"] is False
    assert config["check_nan"] is True
    assert config["deterministic"] is True
    assert config["seed"] == 0
    assert config["batch_size"] == 4096
    assert config["iterations"] == 1000
    assert config["pretrain_iterations"] == 0
    assert config["network_type"] == "ferminet"
    assert config["nucleus_count"] == 1
    assert config["nucleus_atomic_number"] == 1
    assert config["nucleus_charge"] == 1.0
    assert config["nucleus_coords"] == [0.0, 0.0, 0.0]
    assert config["electrons"] == [1, 0]


def test_full_run_modes_are_fixed_and_allowlisted():
    """Full runs expose only the two operator-approved fixed shapes."""

    ferminet = run_mode_spec("full-ferminet")
    psiformer = run_mode_spec("full-psiformer")
    assert ferminet["steps"] == 50000
    assert psiformer["steps"] == 30000
    assert [row["row"] for row in ferminet["rows"]] == [
        "ferminet-codebase-ferminet-seed0",
        "ferminet-codebase-ferminet-seed1",
        "ferminet-codebase-ferminet-seed2",
    ]
    assert [row["seed"] for row in psiformer["rows"]] == [0, 1, 2]
    assert [row["gpu"] for row in ferminet["rows"]] == [0, 1, 2]
    with pytest.raises(ValueError, match="allowlist"):
        run_mode_spec("full-ferminet-50000")


def test_producer_logged_config_checks_the_expected_row_seed():
    """A row receipt must reject a producer dump carrying another row's seed."""

    stderr = _good_logged_config()
    with pytest.raises(ValueError, match="seed"):
        parse_ferminet_logged_config(
            stderr,
            steps=1000,
            expected_network_type="ferminet",
            expected_seed=1,
        )
    config = parse_ferminet_logged_config(
        stderr,
        steps=1000,
        expected_network_type="ferminet",
        expected_seed=0,
    )
    assert config["seed"] == 0


def test_three_row_gate_requires_every_named_full_run_row(tmp_path):
    """One gate accounts for all three fixed full-run rows."""

    root = tmp_path / "full"
    root.mkdir()
    rows = [f"ferminet-codebase-ferminet-seed{seed}" for seed in range(3)]
    for row in rows:
        (root / row).mkdir()
        record(root / row / "receipt.json", row=row, returncode=0, producer_label="producer")
    assert [entry["row"] for entry in gate(root, rows)["rows"]] == rows
    (root / rows[-1] / "receipt.json").unlink()
    with pytest.raises(RuntimeError, match=rows[-1]):
        gate(root, rows)


def test_producer_logged_config_rejects_truncated_dump_before_system_record():
    """A prefix ending before the required system record is not evidence."""

    stderr = (Path(__file__).parent / "testdata" / "ferminet-stderr-pbs7857872-config.log").read_text()
    truncated = stderr.split("system:\n", 1)[0]
    with pytest.raises(ValueError, match="complete|system"):
        parse_ferminet_logged_config(truncated, steps=1000, expected_network_type="ferminet")


def test_producer_logged_config_rejects_multiple_records_and_duplicate_paths():
    """A receipt cannot merge records or silently accept duplicate required paths."""

    fixture = (Path(__file__).parent / "testdata" / "ferminet-stderr-pbs7857872-config.log").read_text()
    good = fixture.replace("  reset_if_nan: true", "  reset_if_nan: false")
    with pytest.raises(ValueError, match="single|record"):
        parse_ferminet_logged_config(
            good + "\nI1117 16:57:00.000000 1 main.py:34] System config:\n\n",
            steps=1000,
            expected_network_type="ferminet",
        )
    duplicate = good.replace("  network_type: ferminet\n", "  network_type: ferminet\n  network_type: psiformer\n")
    with pytest.raises(ValueError, match="duplicate"):
        parse_ferminet_logged_config(duplicate, steps=1000, expected_network_type="ferminet")


def _good_logged_config():
    return (Path(__file__).parent / "testdata" / "ferminet-stderr-pbs7857872-config.log").read_text().replace(
        "  reset_if_nan: true", "  reset_if_nan: false"
    )


@pytest.mark.parametrize("case", ["wrong_source", "warning", "truncated_atom"])
def test_producer_logged_config_rejects_untrusted_record_boundaries(case):
    """Only the pinned producer record may supply a complete config dump."""

    stderr = _good_logged_config()
    if case == "wrong_source":
        stderr = stderr.replace("main.py:34] System config:", "other.py:34] System config:")
    elif case == "warning":
        stderr = stderr.replace(
            "  reset_if_nan: false\n",
            "W1117 16:56:21.000000 1 worker.py:1] unrelated record\n"
            "  reset_if_nan: false\n",
        )
    else:
        stderr = stderr.split("    symbol: H\n", 1)[0]
    with pytest.raises(ValueError):
        parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")


def test_producer_logged_config_rejects_eof_truncation():
    """A config prefix without the next log record is not producer evidence."""

    stderr = _good_logged_config()
    stderr = stderr.rsplit("\nI", 1)[0] + "\n"
    with pytest.raises(ValueError, match="terminator|truncated"):
        parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")


def test_producer_logged_config_rejects_duplicate_electron_fields():
    """Repeated sequence fields cannot be concatenated into a fabricated value."""

    stderr = _good_logged_config().replace(
        "  electrons: !!python/tuple\n  - 1\n  - 0\n",
        "  electrons: !!python/tuple\n  - 1\n"
        "  electrons: !!python/tuple\n  - 0\n",
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")


def test_producer_logged_config_rejects_duplicate_nested_yaml_key():
    """Duplicate mapping keys are rejected at levels outside the receipt paths."""

    stderr = _good_logged_config().replace(
        "  check_nan: true\n", "  check_nan: true\n  check_nan: false\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")


def test_producer_logged_config_rejects_unknown_python_tag():
    """Only the explicitly supported tuple tag may appear in producer YAML."""

    stderr = _good_logged_config().replace(
        "batch_size: 4096\n", "batch_size: !!python/object:unknown.Type {}\n"
    )
    with pytest.raises(ValueError, match="tag|python"):
        parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")


def test_complete_producer_config_followed_by_warning_is_valid():
    """A complete producer record ends before a following warning record."""

    stderr = _good_logged_config() + "W1117 16:56:21.000000 1 worker.py:1] unrelated record\n"
    config = parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")
    assert config["electrons"] == [1, 0]
    assert config["reset_if_nan"] is False


def test_parser_does_not_reinterpret_atom_tag_in_scalar():
    """An Atom tag is data only when it is the producer's Atom mapping node."""

    stderr = _good_logged_config().replace(
        "  network_type: ferminet\n",
        "  network_type: '!!python/object:ferminet.utils.system.Atomferminet'\n",
    )
    with pytest.raises(ValueError, match="network_type"):
        parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")


def test_parser_requires_exact_pinned_main_py_header():
    """A similarly named source file is not the pinned FermiNet producer record."""

    stderr = _good_logged_config().replace("main.py:34] System config:", "othermain.py:34] System config:")
    with pytest.raises(ValueError, match="record"):
        parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("  electrons: {1: 99, 0: 99}\n", "electrons"),
        (
            "    coords: null\n",
            "coords",
        ),
    ],
)
def test_parser_rejects_non_sequence_electron_and_coordinate_shapes(replacement, message):
    """Mappings must not be coerced by iterating their keys as physical values."""

    stderr = _good_logged_config()
    if message == "electrons":
        stderr = stderr.replace(
            "  electrons: !!python/tuple\n  - 1\n  - 0\n", replacement
        )
    else:
        stderr = stderr.replace(
            "    coords: !!python/tuple\n    - 0.0\n    - 0.0\n    - 0.0\n", replacement
        )
    with pytest.raises(ValueError, match=message):
        parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")


@pytest.mark.parametrize("severity", ["I", "W", "E", "F"])
def test_each_log_severity_terminates_complete_config(severity):
    """Any subsequent absl severity record closes a complete producer dump."""

    body = _good_logged_config().rsplit("\nI", 1)[0] + "\n"
    stderr = body + f"{severity}1117 16:56:21.000000 1 worker.py:1] next record\n"
    config = parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")
    assert config["network_type"] == "ferminet"
    assert config["electrons"] == [1, 0]


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("  atomic_number: 1\n", "nucleus"),
        ("  charge: 1.0\n", "charge"),
        ("    - 0.0\n    - 0.0\n    - 1.0\n", "origin"),
        ("  - 2\n  - 0\n", "electrons"),
    ],
)
def test_producer_logged_config_rejects_wrong_system_identity(replacement, message):
    """The receipt checks the producer's logged H-atom identity, not intent."""

    fixture = (Path(__file__).parent / "testdata" / "ferminet-stderr-pbs7857872-config.log").read_text()
    stderr = fixture.replace("  reset_if_nan: true", "  reset_if_nan: false")
    if message == "origin":
        stderr = stderr.replace("    - 0.0\n    - 0.0\n    - 0.0\n", replacement)
    elif message == "electrons":
        stderr = stderr.replace("  - 1\n  - 0\n", replacement)
    elif message == "nucleus":
        stderr = stderr.replace("    atomic_number: 1\n", "    atomic_number: 2\n")
    else:
        stderr = stderr.replace("    charge: 1.0\n", "    charge: 2.0\n")
    with pytest.raises(ValueError, match=message):
        parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="ferminet")


def test_producer_logged_config_rejects_wrong_network_type():
    """The row's expected ansatz must match the producer's logged config."""

    stderr = (Path(__file__).parent / "testdata" / "ferminet-stderr-pbs7857872-config.log").read_text()
    stderr = stderr.replace("  reset_if_nan: true", "  reset_if_nan: false")
    with pytest.raises(ValueError, match="network_type"):
        parse_ferminet_logged_config(stderr, steps=1000, expected_network_type="psiformer")


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
    assert 'success-full-ferminet) SELECTED_RUN_MODE=full-ferminet' in script
    assert 'success-full-psiformer) SELECTED_RUN_MODE=full-psiformer' in script
    assert 'fail-full-ferminet-seed1) SELECTED_RUN_MODE=full-ferminet' in script
    assert 'export BL_R1_MODE=verify BL_R1_VERIFY_STEPS=10' in script
    assert 'export BL_R1_RUN_MODE="$SELECTED_RUN_MODE"' in script
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


def test_pbs_uses_equals_boolean_overrides_and_producer_config_receipt():
    """Boolean flags and the receipt must use the producer's actual argv/log."""

    script = (Path(__file__).parent / "runbooks" / "h-atom-r1-ramp.pbs").read_text()
    assert "--config.optim.reset_if_nan=False" in script
    assert "--config.debug.check_nan=True" in script
    assert "--config.debug.deterministic=True" in script
    assert "--config.optim.reset_if_nan False" not in script
    assert "validate-ferminet-config --path \"$root/stderr.log\"" in script
    assert "resolved-config.json" not in script


def test_pbs_uses_allowlisted_full_modes_and_row_expected_seed():
    """The PBS launcher derives rows from fixed modes and checks each seed."""

    script = (Path(__file__).parent / "runbooks" / "h-atom-r1-ramp.pbs").read_text()
    assert 'mode-rows --mode "$RUN_MODE"' in script
    assert 'mode-steps --mode "$RUN_MODE"' in script
    assert '--config.debug.seed="$seed"' in script
    assert '--expected-seed "$seed"' in script
    assert 'mode-rows --mode "$RUN_MODE"' in script


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
