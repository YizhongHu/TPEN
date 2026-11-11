"""Tests for the cluster rung helpers.

These pin two defects that reached PRODUCTION, so they are regression tests
rather than coverage.
"""

from __future__ import annotations

import importlib.util
import io
import json
import runpy
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name: str):
    """Load a rung script as a module without executing its __main__ body."""
    src = (HERE / name).read_text(encoding="utf-8")
    # These are scripts, not modules: they read sys.argv at import time. Take
    # only the definitions ABOVE the first argv use, so the tables and helpers
    # are importable without invoking the script body.
    lines = src.splitlines(keepends=True)
    cut_line = next((i for i, l in enumerate(lines) if "sys.argv" in l), len(lines))
    cut = sum(len(l) for l in lines[:cut_line])
    module = type(sys)(name)
    module.__dict__["__file__"] = str(HERE / name)
    exec(compile(src[:cut], str(HERE / name), "exec"), module.__dict__)
    return module


def test_h2_pins_its_geometry_explicitly() -> None:
    """H2 must never accept diatomic.py's default bond length.

    `diatomic.py` defaults to 0.737164 angstrom; every reference row was produced
    at 1.4 bohr. Accepting the default is SILENT and worth ~2.1e-4 Ha, about six
    times the seed standard deviation, so it cannot be caught by any structural
    check -- only by the energy.
    """

    mod = _load("rung_makeplan.py")
    flags = mod.SYSTEMS["h2"]["config_flags"]("/fn")
    assert "--config.system.bond_length" in flags
    assert flags[flags.index("--config.system.bond_length") + 1] == "1.4"
    assert flags[flags.index("--config.system.units") + 1] == "bohr"
    assert "diatomic.py" in flags[1]


@pytest.mark.parametrize("system,atom", [("he", "He"), ("li", "Li"), ("be", "Be"), ("b", "B"), ("n", "N")])
def test_atoms_take_no_geometry_override(system: str, atom: str) -> None:
    """Atoms use atom.py and must not carry a molecular geometry flag."""

    mod = _load("rung_makeplan.py")
    flags = mod.SYSTEMS[system]["config_flags"]("/fn")
    assert "atom.py" in flags[1]
    assert flags[flags.index("--config.system.atom") + 1] == atom
    assert "--config.system.bond_length" not in flags
    assert "--config.system.molecule_name" not in flags


def test_every_system_is_distinguishable_by_its_flags() -> None:
    """No two systems may emit the same command flags.

    Production job 7579539 ran H2 while labelled He. Two near-identical builders
    existed and the wrong one was selected; a single parameterised table removes
    the choice, and this asserts the table itself cannot collapse.
    """

    mod = _load("rung_makeplan.py")
    rendered = {s: mod.SYSTEMS[s]["config_flags"]("/fn") for s in mod.SYSTEMS}
    assert len(set(rendered.values())) == len(rendered), rendered


def test_energy_bands_do_not_overlap() -> None:
    """The gate's wrong-system check is only meaningful if bands are disjoint."""

    mod = _load("rung_gate.py")
    bands = mod.ENERGY_BAND
    items = sorted(bands.items(), key=lambda kv: kv[1][0])
    for (a, (alo, ahi)), (b, (blo, bhi)) in zip(items, items[1:]):
        assert ahi < blo, f"bands for {a} and {b} overlap: {(alo, ahi)} vs {(blo, bhi)}"


def test_energy_bands_cover_each_system_reference() -> None:
    """Each band must contain the value that system actually produces.

    Values observed in real runs; a band that excluded them would fail every
    correct run, which is the opposite failure and just as bad.
    """

    observed = {"he": -2.9037, "h2": -1.1744, "li": -7.4780, "be": -14.6673, "b": -24.6539, "n": -54.5893}
    mod = _load("rung_gate.py")
    for system, value in observed.items():
        lo, hi = mod.ENERGY_BAND[system]
        assert lo <= value <= hi, f"{system}: {value} outside {(lo, hi)}"


# ---------------------------------------------------------------------------
# Ansatz as an argument, and a system check that survives a short rung.
# ---------------------------------------------------------------------------


def test_every_ansatz_states_its_own_determinant_count() -> None:
    """No ansatz may inherit another's determinant count by omission.

    Both are 16 today, which is precisely why this needs a test: a shared
    constant would look correct until an ansatz with a different count is added,
    and would then be wrong silently.
    """

    mod = _load("rung_makeplan.py")
    for name, spec in mod.ANSATZES.items():
        flags = spec["network_flags"]
        assert "--config.network.determinants" in flags, name
        assert "--config.network.network_type" in flags, name
        assert flags[flags.index("--config.network.network_type") + 1] == name


def test_no_two_system_ansatz_pairs_render_identical_flags() -> None:
    """Distinct (system, ansatz) pairs must be distinguishable from argv alone.

    psiformer B was once compared against ferminet N because both lived under
    `psi-atoms-*` directories and only the recorded network_type separated them.
    """

    mod = _load("rung_makeplan.py")
    seen: dict[tuple, tuple] = {}
    for sysname, sysspec in mod.SYSTEMS.items():
        for ansname, ansspec in mod.ANSATZES.items():
            key = tuple(sysspec["config_flags"]("/fn")) + tuple(ansspec["network_flags"])
            assert key not in seen, f"{(sysname, ansname)} collides with {seen[key]}"
            seen[key] = (sysname, ansname)
    assert len(seen) == len(mod.SYSTEMS) * len(mod.ANSATZES)


def test_psiformer_flags_still_match_the_n20_production_run() -> None:
    """The committed builder must still reproduce what job 7579717 ran.

    That run is the n=20 He result of record. If the builder drifts, the result
    stops being reproducible from the repository.
    """

    mod = _load("rung_makeplan.py")
    flags = mod.ANSATZES["psiformer"]["network_flags"]
    assert flags == ("--config.network.network_type", "psiformer",
                     "--config.network.determinants", "16")


def test_flag_reader_handles_absent_and_trailing_flags() -> None:
    """`_flag` must return None rather than raising on a malformed argv.

    A flag in final position with no value would otherwise raise IndexError and
    take the whole gate down, converting a reportable failure into a crash.
    """

    mod = _load("rung_gate.py")
    argv = ["prog", "--config.system.atom", "He", "--config.network.network_type"]
    assert mod._flag(argv, "--config.system.atom") == "He"
    assert mod._flag(argv, "--config.network.network_type") is None
    assert mod._flag(argv, "--absent") is None
    assert mod._flag(" ".join(argv), "--config.system.atom") == "He"
    with pytest.raises(ValueError, match="repeated flag"):
        mod._flag(argv + ["He", "--config.system.atom", "Li"], "--config.system.atom")


def test_band_threshold_excludes_the_ramp_rungs_and_admits_production() -> None:
    """The energy band must be inapplicable at 200 and 3000 steps, and apply at 200k.

    This is the whole reason the command-based check exists: the rungs that most
    need a system check are the ones the band cannot judge.
    """

    mod = _load("rung_gate.py")
    assert 200 < mod.BAND_MIN_STEPS
    assert 3000 < mod.BAND_MIN_STEPS
    assert 200000 >= mod.BAND_MIN_STEPS


def test_system_flags_cover_both_config_styles() -> None:
    """Atoms and H2 name their system through different flags; both must be read."""

    mod = _load("rung_gate.py")
    assert "--config.system.atom" in mod.SYSTEM_FLAGS
    assert "--config.system.molecule_name" in mod.SYSTEM_FLAGS
    makeplan = _load("rung_makeplan.py")
    for sysname, spec in makeplan.SYSTEMS.items():
        flags = spec["config_flags"]("/fn")
        assert any(f in flags for f in mod.SYSTEM_FLAGS), sysname


def test_seeds_consumed_test_uses_a_trajectory_not_a_final_energy() -> None:
    """The seeds-consumed test must key on many early steps, not one final value.

    Final energies converge, so testing them for distinctness gets MORE likely to
    fire spuriously as a run improves -- an alarm governed by convergence rather
    than correctness. R2 arm lo (job 7580172) tripped it: seed-11 and seed-14 both
    printed -2.9036388 while differing from step 0 and agreeing on 1 of 2000 rows.
    """

    mod = _load("rung_gate.py")
    assert mod.TRAJECTORY_PREFIX >= 5, (
        "a short prefix reintroduces chance collisions; an unconsumed seed "
        "collides on every step, so there is no reason to test only a few"
    )


# ---------------------------------------------------------------------------
# Explicit seed selection, for re-running a row that was lost.
# ---------------------------------------------------------------------------


def _run_builder(tmp_path, *args):
    """Invoke the builder as a subprocess; return (returncode, combined output)."""
    import subprocess
    import sys as _sys

    repo = HERE.parents[2]
    proc = subprocess.run(
        [_sys.executable, str(HERE / "rung_makeplan.py"), *map(str, args)],
        capture_output=True, text=True, cwd=str(repo),
        env={"PYTHONPATH": str(repo), "PATH": "/usr/bin:/bin"},
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_seed_list_default_is_unchanged(tmp_path) -> None:
    """Omitting the seed list must build exactly what it always did.

    Asserting only "seeds are 0..N-1" would still pass if the default path had been
    rewritten, so this compares the emitted plan BYTE FOR BYTE against one built
    without the argument.
    """

    a, b = tmp_path / "a", tmp_path / "b"
    rc1, out1 = _run_builder(tmp_path, "he", "ferminet", a / "r", a / "p", 2, 4, 100, "p")
    rc2, out2 = _run_builder(tmp_path, "he", "ferminet", a / "r", b / "p", 2, 4, 100, "p", "0,1,2,3")
    assert rc1 == 0, out1
    assert rc2 == 0, out2
    assert (a / "p" / "tasks.jsonl").read_bytes() == (b / "p" / "tasks.jsonl").read_bytes()


def test_seed_list_rejects_duplicates(tmp_path) -> None:
    """Two rows with the same seed run identical trajectories.

    That is exactly what the gate's seeds-consumed fingerprint detects, so building
    such a plan would spend an allocation rediscovering something knowable up front.
    """

    rc, out = _run_builder(tmp_path, "he", "ferminet", tmp_path / "r", tmp_path / "p",
                           2, 3, 100, "p", "1,1,2")
    assert rc != 0
    assert "duplicate seed" in out, out


def test_seed_list_rejects_count_mismatch(tmp_path) -> None:
    """nrows and the seed list must agree; neither may silently truncate the other."""

    rc, out = _run_builder(tmp_path, "he", "ferminet", tmp_path / "r", tmp_path / "p",
                           2, 20, 100, "p", "1,2,3")
    assert rc != 0
    assert "nrows=20 but 3 seed" in out, out


def test_seed_list_accepts_a_single_non_zero_seed(tmp_path) -> None:
    """The case this exists for: re-run seed 5 alone after losing that row."""

    rc, out = _run_builder(tmp_path, "he", "ferminet", tmp_path / "r", tmp_path / "p",
                           4, 1, 100, "p", "5")
    assert rc == 0, out
    assert "seeds=[5]" in out, out
    import json
    rows = [json.loads(l) for l in (tmp_path / "p" / "tasks.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    argv = rows[0]["command"]
    assert argv[argv.index("--config.debug.seed") + 1] == "5"


# ---------------------------------------------------------------------------
# Direct tests of the gate decision body. These artefacts are deliberately
# short: command identity must be checked before the converged energy band.
# ---------------------------------------------------------------------------


def _gate_command(system="he", ansatz="ferminet", seed=0, steps=200):
    return [
        "python", "train.py",
        "--config.system.atom", "He" if system == "he" else "Li",
        "--config.network.network_type", ansatz,
        "--config.optim.iterations", str(steps),
        "--config.debug.seed", str(seed),
    ]


def _write_gate_fixture(tmp_path, commands, *, statuses=None, visibilities=None,
                        hosts=None, fingerprint_keys=None, steps=200):
    """Write the smallest complete set of artefacts accepted by the gate."""
    launch = tmp_path / "launch"
    launch.mkdir(parents=True)
    results = tmp_path / "results"
    records = []
    statuses = statuses or ["success"] * len(commands)
    visibilities = visibilities or [str(i) for i in range(len(commands))]
    hosts = hosts or [f"host-{i}" for i in range(len(commands))]
    fingerprint_keys = fingerprint_keys or list(range(len(commands)))
    for index, command in enumerate(commands):
        status_path = launch / f"status-{index}.json"
        status_path.write_text(json.dumps({
            "status": statuses[index],
            "inherited_visibility_value": visibilities[index],
            "placement": {"hostname": hosts[index]},
        }))
        records.append({
            "run_id": f"row-{index}",
            "status_path": str(status_path),
            "submitted_command": command,
        })
        csv = results / f"row-{index}" / "run" / "train_stats.csv"
        csv.parent.mkdir(parents=True)
        key = fingerprint_keys[index]
        lines = ["step,energy"]
        lines.extend(f"{step},{-2.9 + key * 0.00001 + step * 0.00000001}"
                     for step in range(steps))
        csv.write_text("\n".join(lines) + "\n")
    (launch / "dispatch_records.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n")


def _run_gate(mod, root, *, system="he", ansatz="ferminet", want_seeds=None):
    return mod.run_gate(root, len(list((root / "results").iterdir())), 1, 2,
                        system, ansatz, want_seeds=want_seeds)


def test_gate_decision_body_is_importable_and_reports_a_failed_status(tmp_path) -> None:
    """A bad status reaches the callable body and becomes a gate failure."""
    mod = _load("rung_gate.py")
    _write_gate_fixture(tmp_path, [_gate_command(seed=0), _gate_command(seed=1)],
                        statuses=["failed", "success"])
    result = _run_gate(mod, tmp_path)
    assert any("rows not success" in failure for failure in result.failures)


def test_gate_seeds_fail_when_duplicate_and_pass_when_distinct(tmp_path) -> None:
    mod = _load("rung_gate.py")
    commands = [_gate_command(seed=1), _gate_command(seed=1)]
    _write_gate_fixture(tmp_path, commands)
    bad = _run_gate(mod, tmp_path, want_seeds=[1, 1])
    assert any("DUPLICATE DEBUG SEEDS" in failure for failure in bad.failures)

    good_root = tmp_path / "good"
    _write_gate_fixture(good_root, [_gate_command(seed=1), _gate_command(seed=2)])
    good = _run_gate(mod, good_root, want_seeds=[1, 2])
    assert good.failures == ()
    assert good.observed_seeds == ("1", "2")


def test_gate_seeds_fail_when_expected_set_differs(tmp_path) -> None:
    mod = _load("rung_gate.py")
    _write_gate_fixture(tmp_path, [_gate_command(seed=1), _gate_command(seed=3)])
    result = _run_gate(mod, tmp_path, want_seeds=[1, 2])
    assert any("DEBUG SEEDS" in failure and "expected" in failure
               for failure in result.failures)


def test_gate_unrecorded_command_fails_and_recorded_commands_pass(tmp_path) -> None:
    mod = _load("rung_gate.py")
    _write_gate_fixture(tmp_path, [_gate_command(seed=0), []])
    bad = _run_gate(mod, tmp_path)
    assert any("no recorded command" in failure for failure in bad.failures)

    good_root = tmp_path / "good"
    _write_gate_fixture(good_root, [_gate_command(seed=0), _gate_command(seed=1)])
    good = _run_gate(mod, good_root)
    assert good.failures == ()


def test_gate_rejects_duplicate_host_visibility_pairs(tmp_path) -> None:
    mod = _load("rung_gate.py")
    commands = [_gate_command(seed=0), _gate_command(seed=1), _gate_command(seed=2)]
    _write_gate_fixture(tmp_path, commands,
                        visibilities=["0", "0", "1"],
                        hosts=["host-0", "host-0", "host-1"])
    bad = _run_gate(mod, tmp_path)
    assert any("DOUBLE-BOOKED" in failure for failure in bad.failures)

    good_root = tmp_path / "good"
    _write_gate_fixture(good_root, commands,
                        visibilities=["0", "1", "0"],
                        hosts=["host-0", "host-0", "host-1"])
    good = _run_gate(mod, good_root)
    assert good.failures == ()


def test_gate_repeated_flag_fails_and_single_flag_passes(tmp_path) -> None:
    mod = _load("rung_gate.py")
    repeated = _gate_command(seed=0)
    repeated.extend(["--config.network.network_type", "psiformer"])
    _write_gate_fixture(tmp_path, [repeated, _gate_command(seed=1)])
    bad = _run_gate(mod, tmp_path)
    assert any("repeated flag '--config.network.network_type'" in failure
               for failure in bad.failures)

    good_root = tmp_path / "good"
    _write_gate_fixture(good_root, [_gate_command(seed=0), _gate_command(seed=1)])
    good = _run_gate(mod, good_root)
    assert good.failures == ()


def test_gate_cli_replays_the_three_production_arms(tmp_path) -> None:
    """The positional CLI and its verdict tokens remain unchanged at 200 steps."""
    cases = [
        ("he", "ferminet", 0, "RUNG_GATES_PASSED"),
        ("he", "psiformer", 1, "WRONG ANSATZ IN ARGV"),
        ("h2", "ferminet", 2, "WRONG SYSTEM IN ARGV"),
    ]
    for index, (asked_system, asked_ansatz, expected_seed, token) in enumerate(cases):
        root = tmp_path / f"case-{index}"
        _write_gate_fixture(root, [_gate_command(seed=expected_seed),
                                   _gate_command(seed=expected_seed + 10)])
        old_argv = sys.argv
        output = io.StringIO()
        try:
            sys.argv = ["rung_gate.py", str(root), "2", "1", "2",
                        asked_system, asked_ansatz]
            with redirect_stdout(output):
                try:
                    runpy.run_path(str(HERE / "rung_gate.py"), run_name="__main__")
                except SystemExit as exc:
                    returncode = exc.code
                else:
                    returncode = 0
        finally:
            sys.argv = old_argv
        text = output.getvalue()
        if token == "RUNG_GATES_PASSED":
            assert returncode == 0, text
        else:
            assert returncode == 1, text
        assert token in text
