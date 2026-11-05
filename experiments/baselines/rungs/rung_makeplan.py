"""Build a StagePlanV2 of independent ferminet-codebase rows, one per seed.

SYSTEM AND ANSATZ ARE BOTH ARGUMENTS.

SYSTEM IS AN ARGUMENT, NOT A COPY. An earlier design kept one builder per system
(`rung_makeplan.py` for H2, `rung_makeplan_he.py` for He) differing in four
lines. Production job 7579539 then ran H2 while labelled He, because the
production script was derived from a local copy that had never received the
He substitution -- applied remotely with `sed -i` -- and the pre-submission
check invoked the He builder BY NAME, validating a code path the job never took.
A single parameterised builder removes that class of error: there is nothing to
pick wrongly.

`build_plan` gives every row the SAME argv (test_plan.py:33 asserts it), which
would run four identical seeds -- exactly the defect that silently produced
identical rows on Cannon.  The dispatch layer does support per-row argv
(`admit_tasks(argv_by_runtime=...)`: "one exact argv per entry of tasks"), so
the plan is constructed directly with per-row LogicalTaskSpec commands.

L0 tests DISPATCH, not science: 300 steps is enough to prove four rows ran
independently on four distinct GPUs with four distinct seeds.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, "/home/rhu/src/TPEN-l0-dev")
from experiments.toolkit.dispatch import StagePlanV2, LogicalTaskSpec, CompletionSpec, logical_task_id_from_parts
from experiments.toolkit.executor_inputs import ready_group, write_additive_ready_group
from experiments.toolkit.resources import ResourceSpec


#: Per-system command flags. H2 pins its geometry explicitly: `diatomic.py`
#: defaults to 0.737164 angstrom and the reference rows were produced at
#: 1.4 bohr. Accepting the default is SILENT and worth ~2.1e-4 Ha, roughly six
#: times the seed standard deviation. Atoms take no geometry override.
SYSTEMS = {
    "h2": {
        "config_flags": lambda fn: (
            "--config", f"{fn}/ferminet/configs/diatomic.py",
            "--config.system.molecule_name", "H2",
            "--config.system.units", "bohr",
            "--config.system.bond_length", "1.4",
        ),
    },
    "he": {"config_flags": lambda fn: ("--config", f"{fn}/ferminet/configs/atom.py",
                                       "--config.system.atom", "He")},
    "li": {"config_flags": lambda fn: ("--config", f"{fn}/ferminet/configs/atom.py",
                                       "--config.system.atom", "Li")},
    "be": {"config_flags": lambda fn: ("--config", f"{fn}/ferminet/configs/atom.py",
                                       "--config.system.atom", "Be")},
    "b":  {"config_flags": lambda fn: ("--config", f"{fn}/ferminet/configs/atom.py",
                                       "--config.system.atom", "B")},
    "n":  {"config_flags": lambda fn: ("--config", f"{fn}/ferminet/configs/atom.py",
                                       "--config.system.atom", "N")},
}

#: Per-ansatz network flags. ANSATZ IS AN ARGUMENT for the same reason SYSTEM is.
#: `determinants` is stated explicitly per ansatz rather than shared, so neither
#: inherits the other's value by omission. Both happen to be 16; that is a
#: coincidence of the published protocols, not a default, and a future ansatz
#: with a different count must not silently pick up 16.
#:
#: Keeping these distinct matters beyond the builder: psiformer B (0.2129) was
#: once compared against ferminet N (0.1282) because both lived under
#: `psi-atoms-*` directories and only the recorded `network_type` distinguished
#: them. The ansatz has to travel WITH the row, which is why it is also written
#: into `params` below rather than hardcoded there.
ANSATZES = {
    "psiformer": {"network_flags": ("--config.network.network_type", "psiformer",
                                    "--config.network.determinants", "16")},
    "ferminet":  {"network_flags": ("--config.network.network_type", "ferminet",
                                    "--config.network.determinants", "16")},
}

FN = "/home/rhu/src/ferminet-psiformer-60c9fab3"
PY = "/home/rhu/.venvs/ferminet-jax092-60c9fab3/bin/python"
USAGE = ("usage: rung_makeplan.py <system> <ansatz> <results_root> <plan_dir> "
         "<gpus_per_row> <nrows> <steps> <plan_id> [seeds]\n"
         "  seeds: optional comma-separated list, e.g. 5 or 3,7,11. "
         "Defaults to 0..nrows-1.")
if len(sys.argv) not in (9, 10):
    raise SystemExit(f"{USAGE}\ngot {len(sys.argv) - 1} argument(s): {sys.argv[1:]}")
# ANSATZ is REQUIRED and positional-second, deliberately. Giving it a default
# would let an older 7-argument call succeed while silently building psiformer
# rows -- the same silence that let job 7579539 run H2 under a He label. With no
# default, a stale call lands a path in argv[1] or argv[2] and dies naming it.
SYSTEM = sys.argv[1].lower()
if SYSTEM not in SYSTEMS:
    raise SystemExit(f"unknown system {SYSTEM!r}; known: {sorted(SYSTEMS)}\n{USAGE}")
ANSATZ = sys.argv[2].lower()
if ANSATZ not in ANSATZES:
    raise SystemExit(f"unknown ansatz {ANSATZ!r}; known: {sorted(ANSATZES)}\n{USAGE}")
ROOT = sys.argv[3]
PLAN_DIR = sys.argv[4]
GPUS = int(sys.argv[5])
NROWS = int(sys.argv[6])
STEPS = int(sys.argv[7])
PLAN_ID = sys.argv[8]

# Explicit seed list, for RE-RUNNING a row that was lost.
#
# `range(NROWS)` alone cannot express "just seed 5". That mattered twice: job
# 7579539 lost seed-18 to a node fault, and job 7580582 lost seed-5 to a silent
# hang at pretrain iteration 11 which then consumed the whole allocation. In both
# cases the other rows were perfectly good and the spread only needed topping up.
#
# Duplicates are rejected rather than tolerated: two rows with the same seed run
# IDENTICAL trajectories, which is precisely what the gate's seeds-consumed
# fingerprint exists to detect. Building that plan and letting the gate catch it
# later would waste an allocation to rediscover something knowable at build time.
if len(sys.argv) == 10:
    raw = [tok for tok in sys.argv[9].split(",") if tok != ""]
    SEEDS = []
    for tok in raw:
        try:
            value = int(tok)
        except ValueError:
            raise SystemExit(f"seed {tok!r} is not an integer\n{USAGE}")
        if value < 0:
            raise SystemExit(f"seed {value} is negative\n{USAGE}")
        SEEDS.append(value)
    if len(set(SEEDS)) != len(SEEDS):
        dupes = sorted({v for v in SEEDS if SEEDS.count(v) > 1})
        raise SystemExit(
            f"duplicate seed(s) {dupes}: identical seeds produce identical "
            f"trajectories, which is never a valid plan\n{USAGE}")
    if len(SEEDS) != NROWS:
        raise SystemExit(
            f"nrows={NROWS} but {len(SEEDS)} seed(s) given: {SEEDS}. "
            f"State both and keep them consistent rather than having one "
            f"silently truncate the other.\n{USAGE}")
else:
    SEEDS = list(range(NROWS))

tasks = []
for seed in SEEDS:
    run_id = f"row-seed{seed}"
    result_dir = str(Path(ROOT) / f"seed-{seed}")
    # Completion must key on a file the RUNNER ACTUALLY WRITES. The first L0
    # attempt used policy="status_completed" against a status.json that ferminet
    # never emits: all four rows succeeded, the predicate failed, dispatch raised,
    # and dispatch_records was never written -- so a fully successful run reported
    # records=0. The predicate has to match the program, not the convention.
    done_path = str(Path(result_dir) / "run" / "train_stats.csv")
    cmd = (
        PY, f"{FN}/ferminet/main.py",
        *SYSTEMS[SYSTEM]["config_flags"](FN),
        "--config.batch_size", "4096",
        "--config.optim.iterations", str(STEPS),
        *ANSATZES[ANSATZ]["network_flags"],
        "--config.debug.deterministic", "True",
        "--config.debug.seed", str(seed),
        "--config.log.save_path", f"{result_dir}/run",
    )
    tasks.append(LogicalTaskSpec(
        logical_task_id=logical_task_id_from_parts(stage="baselines", run_id=run_id, plan_id=PLAN_ID),
        stage="baselines",
        run_id=run_id,
        command=cmd,
        result_dir=result_dir,
        outputs=(done_path,),
        logs=(done_path,),
        params={"code": "ferminet", "ansatz": ANSATZ, "system": SYSTEM,
                "seed": seed, "steps": STEPS, "batch_size": 4096},
        resources=ResourceSpec(profile="cuda", device="cuda", gpus=GPUS),
        completion=CompletionSpec(policy="file_exists", output_paths=(done_path,)),
        metadata={"runtime": "ferminet",
                  "row_key": f"{SYSTEM}-{ANSATZ}-seed-{seed}"},
    ))

plan = StagePlanV2(study="baselines", stage="baselines", plan_id=PLAN_ID,
                   results_root=ROOT, tasks=tuple(tasks)).validate()
Path(PLAN_DIR).mkdir(parents=True, exist_ok=True)
plan.write(PLAN_DIR)
write_additive_ready_group(Path(PLAN_DIR) / "executor_input.json", plan, script_slot=1)
# Assert the thing build_plan would have got wrong.
cmds = {t.command for t in plan.tasks}
assert len(cmds) == NROWS, f"EXPECTED {NROWS} DISTINCT COMMANDS, got {len(cmds)}"
seeds = sorted(t.params["seed"] for t in plan.tasks)
assert seeds == sorted(SEEDS), (seeds, sorted(SEEDS))
assert {t.resources.gpus for t in plan.tasks} == {GPUS}
# Assert the ansatz reached the ARGV, not merely the params dict. A params-only
# check would pass while every row still ran the other network, which is the
# failure mode that made a psiformer directory hold ferminet rows.
want_net = ANSATZES[ANSATZ]["network_flags"][1]
for t in plan.tasks:
    argv = list(t.command)
    got = argv[argv.index("--config.network.network_type") + 1]
    assert got == want_net, f"row {t.run_id}: argv says {got!r}, wanted {want_net!r}"
    assert t.params["ansatz"] == ANSATZ, f"row {t.run_id}: params say {t.params['ansatz']!r}"
print(f"PLAN_OK system={SYSTEM} ansatz={ANSATZ} tasks={len(plan.tasks)} "
      f"distinct_commands={len(cmds)} gpus_per_row={GPUS} seeds={seeds}")
print(f"PLAN_DIR={PLAN_DIR}")
