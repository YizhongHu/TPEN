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
- Before any Paseo action - creating, staffing, messaging, or scheduling an
  agent, or arming a heartbeat - read and follow the `paseo` skill. This is
  mandatory, not advisory.
- All TPEN tests are conducted on clusters. Use FASRC-cannon unless otherwise
  specified.
- Agents that did not finish their task are not allowed to go completely
  idle (stop working or thinking) unless it is blocked by human decision or
  some other process such as cluster job and other agents' work.
- When blocked by another process, agents are strongly encouraged to use
  `paseo-monitor` so that they can be woken up later by a monitoring daemon.
  Since `paseo-monitor` is experimental, agents are encouraged to use heartbeats
  as a failsafe. Heartbeat needs to be cancelled when agents return.

## Orchestrators

- Orchestrators are marked with the Paseo agent name `[orchestrator]`.
- Delegate durable slices through Paseo. Provider-native subagents are
  read-only helpers only.
- One durable slice has one writer, one agent-namespaced branch, and one linear
  PR layer. Use `gh-stack` for stacks.
- If a Paseo child must report back, tell it which events to report with
  `paseo-queue add <agent> "msg"` including completion.
- Before messages are sent, there needs to be an urgency evaluation.
  For urgent messages, `paseo send "msg"` should be used instead.
- When blocked with no safe next action, record the blocker in Task
  Orchestrator, schedule a bounded heartbeat only when needed, then yield.
  Delete the monitoring heartbeat when work resumes.
- Keep a standing heartbeat that re-reads `CLAUDE.md` or `AGENTS.md` from start to finish every few
  hours to refresh your memory. DO NOT take shortcuts such as running a diff.

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

### Prefer small PR steps

For this project, correctness is more important than breadth. Prefer small changes with strong tests.

Avoid large PRs that change multiple things at the same time.

### Implementation worker lifecycle

For each TO implementation slice:

1. The orchestrator claims the item and advances it to `work` before dispatch.
2. Use Paseo to create a dedicated worktree and agent-namespaced branch from `dev` or the stack predecessor.
3. Spawn one implementation worker in that workspace.
4. After implementation, confirm the PR is pushed and no unpreserved local changes remain.
5. Archive the implementation worker's workspace through Paseo.
6. Spawn a verifier worker in a separate Paseo workspace to test the implemented code.
7. Record verification results, then archive the verifier's workspace through Paseo.
8. Upon successful verification, advance the item to `review` and release its claim. This unblocks downstream slices with `unblockAt: review`.
9. When the PR is resolved, record its disposition and complete or cancel the TO item accordingly.

Before spawning a worker, check for existing agents or workspaces associated with the item. Preserve workspaces containing uncommitted or unpushed work, and retain verification evidence before cleanup. Failed verification triggers the next round of implementation work starting at step 3.

## Branches

### Sectioning

Coding agents may push only to agent-namespaced branches `agent/**`. 

They may also push to legacy `codex/**` and `claude/**` branches.

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
