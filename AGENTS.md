# TPEN project specific guidelines

## Orientation

`README.md` and `experiments/README.md` contain important information about the repo.

## Design Document

A design document that contains the mathematical background of
TPEN can be found in `main.typ`. Key components of the model:
`Embedding`, `EquivariantMixing`, `Fourier`, `Readout`, etc
should closely follow the design document for correctness.

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

## Tools
- You are strongly encouraged to autonomously spawn subagents to go faster for reading, editing, testing,
running, and debugging tasks.
- You are allowed to autonomously spawn agents for the purposes stated above.
- You are strongly encouraged to autonomously initiate slurm runs for parallizability. Keep slurm logs around
for reproducibility.
- You are allowed to autonomously submit slurm jobs for efficiency.
- Smoke runs should stay as close to the corresponding real run as possible.
  Prefer the same stage stack, launcher flags, partitions, resource defaults,
  and dependency pattern; reduce only grid size or explicitly requested scale
  controls.

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
- Use existing libraries if possible
- Vectorize with NumPy/PyTorch if possible
- If a config or file or function or class is no longer used, remove it.

Any reintroduction of `permute_tree`, `validate_tree`, `infer_particle_count`, or equivalent recursive container-probing helpers is a blocker.
These helpers erase representation semantics and are not allowed in TPEN. Particle count, permutation, comparison, and validation must come from explicit typed-object contracts (`.permute(...)`, `.compare(...)`, `.validate(...)`, explicit `n_particles`/`n_electrons` metadata), never from recursively inspecting arbitrary containers.

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

### Keep equivariance contracts executable

Values participating in equivariance checks must expose typed semantic
`.permute(...)` and `.compare(...)` contracts. Do not require arbitrary runtime
state or validation-only objects to be EquivariantState. Every equivariant
module should subclass `EquivariantMap` and implement `forward_impl`, not
`forward`.

Bad:

```python
class MyMap(nn.Module):
    def forward(self, x):
        ...
```

Good:

```python
class MyMap(EquivariantMap):
    def forward_impl(self, x):
        ...
```

`EquivariantMap.forward` owns passive trace recording and delegates to `forward_impl`; it does **not** check equivariance. Runtime equivariance checking is separate: the checkers in `tpen.equivariance.checks` (driven by the `RuntimeEquivariance` callback) plus pytest-only helpers under `tests/`. Do not override `forward` or wrap it with equivariance-check decorators, because that obscures control flow and can cause recursion.

### Separate metadata generation from model execution

Path and irrep metadata should be deterministic and cached. Model code should read metadata; it should not silently regenerate or overwrite metadata during training.

Good:

```python
paths = PathMetadata.load("tpen/cache/paths_canonical.json")
```

Avoid:

```python
# inside training or model forward
paths = generate_virtual_paths(...)
save_paths(paths)
```

Generation and saving should be explicit developer actions.

### Keep path axes explicit until correctness is established

`Interaction` should keep a visible path axis:

```text
[batch, channels, paths, indices...]
```

Do not prematurely fold paths into channels. Keeping paths explicit makes debugging, equivariance testing, and path-count checks much easier.

### Implement slow reference versions first

For mathematically delicate operations, prefer a slow, readable reference implementation before vectorizing.

Example:

```python
for path in paths:
    for K in ordered_tuples(n, path.s, distinct=True):
        ...
```

Later vectorized implementations should be tested against the slow reference:

```text
fast(x) == slow(x)
fast(pi x) == pi fast(x)
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

## Branches

### Sectioning

Coding agents may push only to agent-namespaced branches: Codex to `codex/**`, Claude to `claude/**`.

Agents must not push to branches other than these mentioned above, such as `main`,
 merge PRs, or force-push unless the user explicitly asks. Feature branches open PRs against `dev`.

`hooke` and `experiment` are retired intermediate integration branches — do not open new
PRs against them. 

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

### `main` and `dev`

**Ownership split between SpENN and SpENN-dev:** 
`SpENN/` is the production directory with experiments run. It stays on the `main` 
branch and does not commit to remote. 
`SpENN/` always tracks the lastest `main`. When `main` updates, update `SpENN`.

`SpENN-dev/` is the development directory. It stays on `dev` branch and submits PRs into `dev`.
It is also responsible for running smoke runs before full runs are run in `SpENN/`.

`dev` branch will be periodically merged into `main` by the user only.

`dev` is persistent, `dev` force-syncs with new main after every merge.

### Require PR for changes

When directed to make changes to the repo, agent should do it as a branch from the
latest `dev` commit. The changes needs to be reviewed as a PR against `dev`.

Agents should respond to PR review comments by adding commits to the existing PR branch.

Clean local branches and their remote counterparts after they are merged into `dev`.


## Config ownership

**Callbacks and loggers are config-root and owned by the `RunContext`.** They
live at the top level, *not* inside the runner block. A runner config that
declares `callbacks` or `loggers` is rejected by `run_from_config`:

```yaml
runner:
  _target_: tpen.runner.Train
  model: ${model}
  sampler: ${sampler}
  hamiltonian_terms: ${hamiltonian_terms}
  optimizer: ${optimizer}
  trainer: ${trainer}

callbacks: [...]   # config-root, RunContext-owned
loggers: [...]     # config-root, RunContext-owned
```
