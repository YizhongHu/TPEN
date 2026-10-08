# System Instructions

## All agents

- Use Task Orchestrator to keep track of workflow, receipts, and notes.
- After compaction or uncertainty, re-read the applicable repository
  `AGENTS.md` / `CLAUDE.md` files, the claimed Task Orchestrator item's notes,
  and this Paseo instruction.
- Before any SSH, transfer, storage, scheduler, test, smoke, or production
  action on Cannon, Polaris, Aurora, or Frontier, read and follow
  `cluster-access` and the current Task Orchestrator cluster notes. Do not infer
  facility policy from another cluster or an older receipt.

## Orchestrators

- Orchestrators are marked with the Paseo agent name `[orchestrator]`.
- Delegate durable slices through Paseo. Provider-native subagents are
  read-only helpers only.
- One durable slice has one writer, one agent-namespaced branch, and one linear
  PR layer. Use `gh-stack` for stacks.
- If a Paseo child must report back, tell it which events to report with
  `paseo-queue add <agent> "msg"` including completion.
- For urgent messages, `paseo send "msg"` should be used instead.
- For ongoing observation, use `monitor-with-subagent`. The monitoring child
  stays active and reports declared events; the orchestrator owns only the
  bounded failsafe heartbeat described by the skill.
- When blocked with no safe next action, record the blocker in Task
  Orchestrator, schedule a bounded heartbeat only when needed, then yield.
  Delete the heartbeat when work resumes.

# Repo Instructions

## Orientation

`README.md` and `experiments/README.md` contain important information about the repo.

## Design Document

A design document that contains the mathematical background of
TPEN can be found in `main.typ`. Key components of the model:
`Embedding`, `EquivariantMixing`, `PathAggregation`, `Readout`/`PfaffianReadout`,
and the additive log-amplitude envelopes (`ElectronElectronCusp`,
`GaussianConfinement`) should closely follow the design document for correctness.

## Environment

- Any environment problems is not worth trouble-shooting by the agent on its own. If it happens, stop and the issue will be resolved interactively.
- This repo uses `uv` to manage python packages. Most commands (including `pytest`) needs to be run with `uv`. Use `uv` to run if possible for reproducibility.
- If it may be necessary to install a new package, stop and inquire instead of proceeding
  with alternatives.
- Do not use `uv run --nosync`. If `uv` environment needs to change, let `uv lock` update for
  reproducibility.

## Conventions
- NumpyDoc is used for documentation
- Use inline comments for comprehensibility
- Use `America/New_York` timezone for experiment logging. Use `UTC` for test logging.

## Treating Data with Care

Run-result data (including untracked data in `outputs/`, `results/`, `reports/`, `slurm/`, etc.) is
governed by lifecycle class:

- Ephemeral smoke, staging, and temporary data may be removed only by the task-owning agent after
  confirming exact-path ownership, quiescence (no active job, process, or writer references), and
  capturing a durable receipt.
- Disposable agent worktrees follow workspace/cohort safety rules and must never be confused with
  scientific output.
- Scientific outputs, checkpoints, and results remain protected until an explicit Task Orchestrator
  lifecycle disposition names the exact paths and any required backup or archive, and recovery
  verification has passed.
- Failure evidence, logs, and receipts remain preserved until their lifecycle disposition explicitly
  permits cleanup.
- Never delete broad roots or inferred paths; agents must not delete user-owned or unrelated run data.

Paths not authorized by a lifecycle disposition remain subject to the manual deletion fallback: the
agent gives the user a list of items to remove, and the user removes them manually.

## TODO.md

The repository `TODO.md` file is a dynamically-maintained by multiple agents. Agents may add items or refine 
items on the todo list, but they should exercise extreme caution when deleting. If you are not sure, double
check with the user. In general, finished tasks and stale records can be discarded, but unfinished tasks 
and currently important information should not. 

## Best Practises

### Prefer explicit ownership over local convenience

Do not place helper functions wherever they are first needed. Put each helper in the module that owns the relevant concept.

Examples:

```text
Permutation logic       -> tpen/data/permutation.py
Tuple-index logic       -> tpen/data/indices.py
Virtual path logic      -> tpen/data/paths.py
Partition logic         -> tpen/data/partition.py
Trainable modules       -> tpen/nn/
```

Bad:

```python
# tpen/nn/equivariant_mixing.py
def ordered_tuples(...):
    ...
```

Good:

```python
from tpen.data.indices import ordered_tuples
```

### Prefer small PR steps

For this project, correctness is more important than breadth. Prefer small changes with strong tests.

Avoid large PRs that change multiple things at the same time.

### Require an authoritative-edit launch receipt

Before the first repository edit for any implementation slice, run:

```bash
uv run --no-project python tools/check_authoritative_edit.py --item <task-orchestrator-item-id>
```

Proceed only when it emits `"status": "ok"`. The guard requires a clean tracked
tree on an agent-namespaced branch plus a claimed TPEN `implementation-slice`
already in `work` with a non-empty `acceptance-contract`. Existing untracked
research and run data are deliberately ignored and must remain untouched.

The guard requires Python 3.9+ because it uses `str.removesuffix`. The
`uv run --no-project python` form skips project and workspace discovery, but it
still selects an interpreter from ambient state such as a discovered or
activated `.venv`, `VIRTUAL_ENV`, `UV_PYTHON`, or `.python-version`; it does not
guarantee the project interpreter. Before relying on that invocation, settle
the selection with `uv run --no-project python -c 'import sys; print(sys.version)'`.
The sanctioned interpreter-direct deviation is precisely where a stale ambient
`python3` may be selected, so name an explicitly installed 3.9+ interpreter
when invoking the script directly. A `blocked` verdict whose reason says the
interpreter is unsupported means no guard precondition was evaluated: rerun
with the `uv run --no-project python` command above or a named 3.9+ interpreter.
Any other `blocked` reason is a genuine guard precondition failure; repair that
precondition before editing. Because the gate runs ahead of `parse_args()`, on
an interpreter where the gate is reachable but unsupported, `--help`, unknown
options, and a missing `--item` return exit 1 with the structured interpreter
refusal instead of argparse's usage or help output. On Python 3.9+ argparse
behaviour is unchanged by this guard. The version predicate refuses every
version tuple below 3.9, but that predicate is not a claim of delivered runtime
coverage: the structured refusal is implemented for Python 3.7 and 3.8, this
lane executed Python 3.8.3 only, and Python 3.7 was not executed. Python <=3.6
raises `SyntaxError` before the gate can run because `from __future__ import
annotations` requires Python 3.7+; that residual limitation remains documented.

**Run the guard outside the agent sandbox.** This is allowed and encouraged, not
a workaround. The guard reads the Task Orchestrator HTTP API, and sandboxed
coding agents deny outbound network, so it returns
`{"status": "blocked"}` with `urlopen error [Errno 1] Operation not permitted`.
The same sandboxes deny paths named `.git`, which breaks `uv` outright because
`uv` writes a zero-byte `.git` marker into every cache directory on init;
redirecting `UV_CACHE_DIR` does not help, because `uv` recreates the marker
wherever it is pointed. Do not send a sandboxed agent at this guard and do not
escalate sandbox permissions for it.

Instead, an unsandboxed party — normally the orchestrator — runs it from the
lane's own worktree, and records the output as an
`authoritative-edit-launch-receipt` note on that lane's item. Invoking the
script through an interpreter directly instead of `uv run --no-project` is
acceptable here: the script performs no dependency resolution. However,
`uv run --no-project` does not guarantee the project interpreter; it can follow
the ambient environment-selection inputs described above, so check
`uv run --no-project python -c 'import sys; print(sys.version)'` before relying
on it. Record both deviations in the receipt — interpreter-direct, and
executed by someone other than the editing agent — together with what that does
*not* establish: the editing agent did not confirm its own preconditions. The receipt is
point-in-time, so the lane must not reset, rebase, or switch branches between
the receipt and its first edit. One caution: an agent may run the guard
successfully early in a session and be denied later, so an early success is not
evidence the restriction is absent.

## Branches

### Sectioning

Coding agents may push only to agent-namespaced branches: Codex to `codex/**`, Claude to `claude/**`.

Agents must not push to branches other than these mentioned above, such as `main`,
 merge PRs, or force-push unless the user explicitly asks. Feature branches open PRs against `dev`.


### `main` and `dev`

`dev` branch will be periodically merged into `main` by the user only.

`dev` is persistent, `dev` fast-forwards to new main after every merge.

Experiments that produce durable science needs to be done on the stable `main` commits.
`dev` experiments are disposable are used for non-science task such as smokes, probing, and testing.

### Require PR for changes

When directed to make changes to the repo, agent should do it as a branch from the
latest `dev` commit. The changes needs to be reviewed as a PR against `dev`.

Agents should respond to PR review comments by adding commits to the existing PR branch.

Clean local branches and their remote counterparts after they are merged into `dev`.

### Stacked pull requests

Use GitHub's native `gh-stack` extension for dependent pull requests. A stack may
contain any number of reviewable layers; TPEN sets no stack-depth or open-PR cap.
Depth does not relax these invariants:

- The stack is one acyclic linear chain rooted at `dev`. The bottom PR targets
  `dev`; every higher PR targets exactly the branch immediately below it.
- One layer is one typed Task Orchestrator implementation item, one
  agent-namespaced branch, one PR, and one live writer claim. Forks, skipped
  bases, duplicate layers, and cross-stack dependencies are forbidden.
- Create or adopt the full ordered chain with `gh stack init --base dev
  <bottom> ... <top>`. Add one new top layer with `gh stack add <branch>`.
- Agents use non-interactive commands: `gh stack submit --auto`,
  `gh stack view --json`, and explicit `--remote origin` where supported. Do not
  invoke a bare command that may open a prompt or TUI.
- Before publishing or after changing a lower layer, require
  `needsRebase == false` for every layer in `gh stack view --json` and verify
  each parent tip is an ancestor of its child. Rebase and reverify the upstack
  when a predecessor moves.
- Record the branch, PR URL/base, and exact full head SHA in the implementation
  receipt. An independent clean verifier must test that SHA; any new commit
  invalidates the receipt.
- An open or draft PR is still `work`/awaiting merge, not `terminal`. Humans
  merge bottom-up; after GitHub confirms a merge, terminalize that layer and
  run `gh stack sync --prune`.

`gh-stack` owns Git branch and PR topology. Task Orchestrator owns scope,
claims, dependencies, acceptance criteria, and verification receipts. Do not
build a second stack state machine in project scripts or notes.
