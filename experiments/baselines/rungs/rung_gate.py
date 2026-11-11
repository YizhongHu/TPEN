"""Shared artefact gate for the Polaris toolkit-dispatch ladder (L0..L3).

Usage: rung_gate.py <run_root> <expected_rows> <expected_gpus_per_row> <expected_hosts>

WHY THIS EXISTS AS A FILE: L0 failed its gate three times for reasons that were
all in the GATE, not the run. Each schema fact below cost a queue cycle to learn,
so they are written down once rather than re-derived per rung.

SCHEMA FACTS, none of which are guessable:
  * dispatch_records.jsonl entries are POINTERS, not outcomes. They carry
    `status_path` / `metadata.attempt_status_path`; the row's real status lives
    in that file. Reading record["status"] silently yields None.
  * The GPU binding evidence is `inherited_visibility_value`, NOT
    `visibility_value`. The latter is None BY DESIGN: parsl_attach.py:265 --
    Parsl's `available_accelerators` binds each HTEX worker and the task inherits
    that environment, because dispatch order must never select a GPU.
  * `placement.gpus` lists every GPU on the NODE (nvidia-smi ignores
    CUDA_VISIBLE_DEVICES), so its length is 4 regardless of binding. It cannot
    prove per-row exclusivity; `inherited_visibility_value` can.
  * verification.json exit_code is NOT a verdict: pipeline.py writes 0 whenever
    dispatch() does not raise, and a failed row is recorded, not raised.
"""
import json, sys, glob, os
from pathlib import Path

# Expected energy band per system, in hartree. A wrong-SYSTEM run passes every
# STRUCTURAL gate identically -- records, hosts, visibility, checkpoints, even
# distinct energies -- because it is a perfectly good run of the wrong thing.
# Only the energy separates them.
# This exists because production job 7579539 ran H2 while labelled He: the plan
# builder was substituted on the cluster copy but not the local one used to
# derive the production script. It was caught by an UNRELATED node fault, not by
# this check, which at the time lived only in a watch prohibition and in prose.
ENERGY_BAND = {
    "he": (-2.95, -2.85),
    "h2": (-1.22, -1.13),
    "li": (-7.52, -7.43),
    "be": (-14.72, -14.61),
    "b":  (-24.71, -24.60),
    "n":  (-54.65, -54.53),
}

#: The energy band can only judge a CONVERGED run. A 200- or 3000-step ramp rung
#: sits far above its band and would fail for the right reason at the wrong time.
#: So the band is gated on step count -- and the rungs below that threshold are
#: exactly the ones that most need a system check, which is why the command-based
#: check below exists and does not depend on convergence at all.
#: How many leading rows form a run's fingerprint for the seeds-consumed test.
#: A row whose seed was ignored reproduces another row's trajectory EXACTLY, so it
#: collides on all TRAJECTORY_PREFIX values at once. Independent rows colliding on
#: that many values by chance is not a practical concern.
TRAJECTORY_PREFIX = 10

BAND_MIN_STEPS = 50000

#: Which flag names the system, per config style. Atoms carry an element symbol;
#: H2 carries a molecule name. Reading either is enough to identify the system
#: from argv alone.
SYSTEM_FLAGS = ("--config.system.atom", "--config.system.molecule_name")


class GateResult:
    """The checks performed by :func:`run_gate` and their decision."""

    def __init__(self, failures, ran, skipped, observed_seeds):
        self.failures = failures
        self.ran = ran
        self.skipped = skipped
        self.observed_seeds = observed_seeds


def _flag(argv, name):
    """Value following `name` in argv, or None. argv may be a list or a string."""
    if isinstance(argv, str):
        argv = argv.split()
    positions = [i for i, value in enumerate(argv) if value == name]
    if len(positions) > 1:
        raise ValueError(f"repeated flag {name!r}")
    if not positions or positions[0] + 1 >= len(argv):
        return None
    return argv[positions[0] + 1]


def _read_flag(argv, name, row, failures):
    """Read one command flag and turn malformed repetition into a gate failure."""
    try:
        return _flag(argv, name)
    except ValueError as exc:
        failures.append(f"command row {row}: {exc}")
        return None


def run_gate(root, want_rows, want_gpus, want_hosts, want_system=None,
             want_ansatz=None, want_seeds=None):
    """Check one rung's recorded artefacts.

    Parameters
    ----------
    root : str or pathlib.Path
        Run root containing ``launch``, ``plan``, and ``results``.
    want_rows, want_gpus, want_hosts : int
        Structural expectations supplied by the rung's job script.
    want_system, want_ansatz : str, optional
        Expected command-line identity for the run.
    want_seeds : sequence, optional
        Expected debug seeds, when a caller is rechecking a known seed set.

    Returns
    -------
    GateResult
        The failures, checks run, checks skipped, and seeds observed in the
        submitted commands. The function also emits the production report.
    """
    root = Path(root)
    failures = []
    ran, skipped = [], []

    recs_path = root / "launch" / "dispatch_records.jsonl"
    recs = []
    if recs_path.exists():
        recs = [json.loads(line) for line in recs_path.read_text().splitlines() if line.strip()]
    else:
        failures.append("dispatch_records.jsonl MISSING (a raised completion predicate suppresses it entirely)")

    statuses, inherited, hosts, placements = [], [], set(), []
    for record in recs:
        status_path = ((record.get("metadata") or {}).get("attempt_status_path")
                       or record.get("status_path"))
        try:
            status = json.load(open(status_path))
        except Exception as exc:
            failures.append(f"attempt_status unreadable for {record.get('run_id')}: {exc!r}")
            continue
        statuses.append(status.get("status"))
        visibility = status.get("inherited_visibility_value")
        host = (status.get("placement") or {}).get("hostname")
        inherited.append(visibility)
        hosts.add(host)
        placements.append((host, visibility))

    print(f"GATE records          {len(recs)} / {want_rows}")
    print(f"GATE statuses         {statuses}")
    print(f"GATE inherited_vis    {inherited}")
    print(f"GATE host_gpu_pairs    {len(set(placements))} distinct of {len(placements)}")
    print(f"GATE hosts            {sorted(host for host in hosts if host)}")

    if len(recs) != want_rows:
        failures.append(f"record count {len(recs)} != {want_rows}")
    bad = [i for i, status in enumerate(statuses) if status != "success"]
    if bad:
        failures.append(f"rows not success at indices {bad}: {[statuses[i] for i in bad]}")

    present = [value for value in inherited if value is not None]
    # Exclusivity is per (HOST, visibility), NOT per visibility. Across N nodes the
    # same local id recurs once per node -- GPU "0" on node A is a different physical
    # device from GPU "0" on node B. Requiring globally distinct values is an
    # invariant that holds only at one node, and it produced a false DOUBLE-BOOKED
    # verdict on L2b (8 rows, 2 nodes, values ['1','0','2','3','2','0','3','1'] --
    # every one correct).
    pairs = [pair for pair in placements if pair[1] is not None and pair[0] is not None]
    if len(present) != len(recs):
        failures.append(f"inherited visibility missing for {len(recs)-len(present)} row(s)")
    elif len(set(pairs)) != len(pairs):
        dupes = sorted({pair for pair in pairs if pairs.count(pair) > 1})
        failures.append(f"DOUBLE-BOOKED: (host, visibility) pairs repeat: {dupes}")
    else:
        widths = {len(value.split(",")) for value in present}
        if widths != {want_gpus}:
            failures.append(f"gpus-per-row {sorted(widths)} != {want_gpus}")

    if len([host for host in hosts if host]) != want_hosts:
        failures.append(f"host count {len([host for host in hosts if host])} != {want_hosts}")

    energies, fingerprints = {}, {}
    for directory in sorted(glob.glob(str(root / "results" / "*"))):
        csv = Path(directory) / "run" / "train_stats.csv"
        if csv.exists():
            name = os.path.basename(directory)
            lines = csv.read_text().strip().splitlines()
            energies[name] = lines[-1].split(",")[1]
            # Skip the header, take the earliest steps: they are set by the seed's
            # initial walkers and have not yet been pulled together by optimisation.
            fingerprints[name] = tuple(line.split(",")[1]
                                       for line in lines[1:1 + TRAJECTORY_PREFIX])
    print(f"GATE energies         {energies}")
    if len(energies) != want_rows:
        failures.append(f"{len(energies)} of {want_rows} rows produced train_stats.csv")
    else:
        # SEEDS-CONSUMED TEST, on the trajectory rather than on the final energy.
        #
        # Testing final energies for distinctness looks equivalent and is not. Final
        # energies CONVERGE: as a run improves, all rows crowd toward the same value,
        # so the probability that two of them round to the same printed number RISES
        # with run quality. That check is therefore most likely to fire spuriously on
        # the best runs -- an alarm whose rate is governed by convergence, not by
        # correctness. R2 arm lo (job 7580172) tripped it at 2000 steps: seed-11 and
        # seed-14 both printed -2.9036388, while differing from step 0 onward and
        # agreeing on exactly 1 of 2000 rows. The n=20 production run 7579717 passed
        # the same check by luck.
        duplicate_fingerprints = {}
        for name, fingerprint in fingerprints.items():
            duplicate_fingerprints.setdefault(fingerprint, []).append(name)
        collided = [names for names in duplicate_fingerprints.values() if len(names) > 1]
        if collided:
            failures.append(
                f"SEEDS NOT CONSUMED: rows share their first {TRAJECTORY_PREFIX} "
                f"energies, i.e. identical trajectories: {collided}")
        # Report final-energy coincidences, but do NOT fail on them: see above.
        if len(set(energies.values())) != want_rows:
            same = {}
            for name, value in energies.items():
                same.setdefault(value, []).append(name)
            print("GATE note             final-energy coincidence (NOT a failure; "
                  f"trajectories verified distinct): "
                  f"{[group for group in same.values() if len(group) > 1]}")

    # COMMAND-BASED SYSTEM/ANSATZ/SEED CHECK.
    #
    # This reads what was SUBMITTED, and it is valid at any step count -- 200 steps
    # or 200000. It exists because the energy band, the only previous system check,
    # needs a converged run, so the short rungs had no system check at all.
    #
    # Preference order matters. `submitted_command` is what dispatch actually sent;
    # `plan/tasks.jsonl` is only what was intended. They can differ. But one failed
    # row makes dispatch raise and suppresses ALL records, so records may be absent
    # on a run whose rows were fine -- hence the fallback, which announces itself
    # rather than quietly weakening the check.
    cmd_source, cmds = None, []
    if recs:
        cmds = [record.get("submitted_command") or [] for record in recs]
        cmd_source = "dispatch_records.submitted_command"
    else:
        plan_tasks = root / "plan" / "tasks.jsonl"
        if plan_tasks.exists():
            cmds = [json.loads(line).get("command") or []
                    for line in plan_tasks.read_text().splitlines() if line.strip()]
            cmd_source = "plan/tasks.jsonl (FALLBACK: dispatch records absent or empty)"

    observed_steps = set()
    observed_seeds = []
    if not cmds:
        # Never silently skip: with no command anywhere, this check cannot run, and
        # saying nothing would read as a pass.
        failures.append("no submitted_command and no plan/tasks.jsonl -- command-based "
                        "system/ansatz check COULD NOT RUN")
    else:
        print(f"GATE command source   {cmd_source} ({len([command for command in cmds if command])} row(s))")
        got_sys, got_ans = set(), set()
        for row, argv in enumerate(cmds):
            if not argv:
                failures.append(f"command row {row} has no recorded command; cannot check argv")
                continue
            for flag in SYSTEM_FLAGS:
                value = _read_flag(argv, flag, row, failures)
                if value is not None:
                    got_sys.add(value.lower())
            value = _read_flag(argv, "--config.network.network_type", row, failures)
            if value is not None:
                got_ans.add(value.lower())
            value = _read_flag(argv, "--config.optim.iterations", row, failures)
            if value is not None:
                observed_steps.add(int(value))
            value = _read_flag(argv, "--config.debug.seed", row, failures)
            if value is not None:
                observed_seeds.append(value)
            else:
                failures.append(f"command row {row} names no --config.debug.seed; cannot confirm seed")
        print(f"GATE argv system      {sorted(got_sys)}")
        print(f"GATE argv ansatz      {sorted(got_ans)}")
        print(f"GATE argv steps       {sorted(observed_steps)}")
        if len(got_sys) > 1:
            failures.append(f"MIXED SYSTEMS in one rung: {sorted(got_sys)}")
        if len(got_ans) > 1:
            failures.append(f"MIXED ANSATZES in one rung: {sorted(got_ans)}")
        if len(observed_seeds) != len(set(observed_seeds)):
            duplicates = sorted({seed for seed in observed_seeds
                                  if observed_seeds.count(seed) > 1})
            failures.append(f"DUPLICATE DEBUG SEEDS in submitted commands: {duplicates}")
        if want_seeds is not None:
            expected_seeds = [str(seed) for seed in want_seeds]
            if sorted(observed_seeds) != sorted(expected_seeds):
                failures.append(f"DEBUG SEEDS {sorted(observed_seeds)} != expected "
                                f"{sorted(expected_seeds)}")
        if want_system:
            ran.append("argv-system")
            if got_sys and got_sys != {want_system.lower()}:
                failures.append(f"WRONG SYSTEM IN ARGV: asked for {want_system!r}, "
                                f"command says {sorted(got_sys)}")
            elif not got_sys:
                failures.append("command names no system flag; cannot confirm system from argv")
        if want_ansatz:
            ran.append("argv-ansatz")
            if got_ans and got_ans != {want_ansatz.lower()}:
                failures.append(f"WRONG ANSATZ IN ARGV: asked for {want_ansatz!r}, "
                                f"command says {sorted(got_ans)}")
            elif not got_ans:
                failures.append("command names no network_type; cannot confirm ansatz from argv")

    # WRONG-SYSTEM CHECK BY ENERGY. Production only -- see BAND_MIN_STEPS.
    band_applicable = bool(observed_steps) and min(observed_steps) >= BAND_MIN_STEPS
    if want_system and not band_applicable:
        skipped.append(
            f"energy-band (steps {sorted(observed_steps) or 'unknown'} < {BAND_MIN_STEPS}; "
            "a short rung is not converged, so the band cannot judge it)")
    if want_system and band_applicable:
        ran.append("energy-band")
    if want_system and band_applicable:
        band = ENERGY_BAND.get(want_system.lower())
        if band is None:
            failures.append(f"no energy band known for system {want_system!r}; add one rather than skipping")
        else:
            lo, hi = band
            bad = {name: value for name, value in energies.items()
                   if not (lo <= float(value) <= hi)}
            print(f"GATE system            {want_system} expects [{lo}, {hi}]")
            if bad:
                other = [system for system, (low, high) in ENERGY_BAND.items()
                         if any(low <= float(value) <= high for value in bad.values())]
                failures.append(
                    f"WRONG SYSTEM: {len(bad)} row(s) outside the {want_system} band {band}: "
                    f"{dict(list(bad.items())[:3])}"
                    + (f" -- these look like {other}" if other else "")
                )

    # State what ran and what did not. A gate that skips a check silently is
    # indistinguishable from one that passed it.
    print(f"GATE checks ran       {ran or ['(none)']}")
    if skipped:
        for skipped_check in skipped:
            print(f"GATE checks SKIPPED   {skipped_check}")

    fails = failures
    if fails:
        print("RUNG_GATES_FAILED")
        for failure in fails:
            print(f"  FAIL: {failure}")
        return GateResult(tuple(fails), tuple(ran), tuple(skipped), tuple(observed_seeds))
    print("RUNG_GATES_PASSED")
    return GateResult((), tuple(ran), tuple(skipped), tuple(observed_seeds))


root = Path(sys.argv[1])
want_rows = int(sys.argv[2])
want_gpus = int(sys.argv[3])
want_hosts = int(sys.argv[4])
want_system = sys.argv[5] if len(sys.argv) > 5 else None
want_ansatz = sys.argv[6] if len(sys.argv) > 6 else None
result = run_gate(root, want_rows, want_gpus, want_hosts, want_system, want_ansatz)
if result.failures:
    sys.exit(1)
