# Tensor-product Permutation Equivariant Network (TPEN)

The active TPEN core scaffold is documented in the PR brief and the package
docstrings under `tpen/data`, `tpen/reps`, `tpen/nn`, and `tpen/equivariance`.

## Quick Start

Use `uv` for local environment management. Keep CPU and GPU work in separate
virtual environments so Slurm jobs never replace each other's Torch install.
Both environments resolve from this one `pyproject.toml`.

### CPU Environment

CPU work uses the default `.venv`:

```bash
uv sync --extra cpu
uv run --extra cpu python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

### GPU Environment

CUDA work uses a separate `.venv-gpu`:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-gpu
uv sync --extra cu126
uv run --extra cu126 python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

Use `cu128` or `cu130` instead if that is the CUDA Torch build you want. Keep
the `UV_PROJECT_ENVIRONMENT` setting in GPU Slurm scripts so GPU jobs do not
mutate the CPU `.venv`.

### AMD ROCm Environment (OLCF Frontier)

Frontier's MI250X GPUs use the `rocm71` extra, which resolves
`torch 2.13.0+rocm7.1` and `triton-rocm` from the PyTorch ROCm index:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-rocm
uv sync --extra rocm71
uv run --extra rocm71 python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

ROCm Torch exposes the same `torch.cuda` API as a CUDA build, so no TPEN code
path changes and `torch.cuda.is_available()` reports the MI250X.

Unlike the CUDA extras, `rocm71` pins an exact Torch version. The ROCm index
also carries old non-ROCm builds, and an unpinned `torch` resolves to 2.0.1
with aarch64-only wheels, which installs nowhere on Frontier's x86_64 nodes.
When bumping the pin, check the resolved version *and* the wheel platform, not
just that `uv lock` succeeded.

Facility selection is environment variables only; nothing facility-specific is
committed. On Frontier, provision against the system interpreter rather than a
uv-managed download:

```bash
# The exact installed ROCm minor is unverified; check `module -t avail rocm`
# on Frontier and replace X.Y.Z before provisioning.
module load rocm/X.Y.Z
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0
PYBIN="$(command -v python3)"
export UV_PYTHON="$PYBIN"
export UV_NO_MANAGED_PYTHON=1
export UV_PYTHON_DOWNLOADS=never
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/tpen-frontier-rocm71"
uv sync --extra rocm71
```

Do not add `UV_PYTHON_PREFERENCE=only-system` alongside these. On Polaris's
facility uv 0.8.23, the combination is rejected before resolution:

```text
error: the argument '--python-preference <PYTHON_PREFERENCE>' cannot be used with '--no-managed-python'
```

`UV_NO_MANAGED_PYTHON=1` already overrides the committed `python-preference`.
The same conflict was independently recorded on Cannon with uv 0.11.16, where
the wording is reversed: `error: the argument UV_NO_MANAGED_PYTHON
(environment variable) cannot be used with --python-preference`.

The Frontier profile remains documented but **has not been executed on Frontier
hardware**. The ROCm module version, Frontier's uv version, the facility
interpreter path, and the resolved `rocm71` Torch wheel have not been validated
there. Do not treat the recipe as a Frontier hardware or run-validation receipt.

### NVIDIA A100 Environment (ALCF Polaris)

There is **no presently verified, runnable TPEN bootstrap on Polaris**. Do not
run either of the historical profiles below as an operational recipe.

The former direct, absolute-interpreter profile was validated only before the
2026-08-19 Polaris/Eagle upgrade. On the changed system, that direct profile
separately failed to load Torch's CUDA runtime libraries. PBS job `7559215`
also failed while loading `conda`, reporting unavailable dependencies
`gcc-native/14.2` and `cray-hdf5-parallel/1.14.3.5`; it did not reach Python,
Torch, overlay provisioning, or TPEN tests.

That PBS result is limited: its preserved script began `#!/bin/bash`, whereas
ALCF's [published Polaris batch examples](https://docs.alcf.anl.gov/running-jobs/example-job-scripts/)
use `#!/bin/bash -l` to instantiate the login-shell environment. Job `7559215`
therefore did **not** execute the fully qualified published login-shell context,
and its failure does not establish that the documented
`module use /soft/modulefiles`, `module load conda`, and
`conda activate base` procedure fails in that context. The qualified procedure
remains unverified; no corrected job has been authorized or run.

Use the current [ALCF Polaris Python guidance](https://docs.alcf.anl.gov/polaris/data-science/python/)
and contact [ALCF Support](https://www.alcf.anl.gov/support-center) for a
supported post-upgrade base runtime. A fresh facility validation is required
before any Polaris TPEN provisioning or test run.

Once a supported Polaris base runtime is independently verified, these are
conditional invariants—not a current executable recipe:

- Polaris selects **no TPEN PyTorch extra**. ALCF owns Python, PyTorch, and
  CUDA; selecting `cu126`/`cu128`/`cu130` would shadow the facility Torch.
- The resolved facility interpreter must drive `$PYBIN -m uv` with
  `UV_NO_MANAGED_PYTHON=1`, `UV_PYTHON_DOWNLOADS=never`, and a fresh absolute
  `UV_PROJECT_ENVIRONMENT` created with `--system-site-packages`.
- Synchronization must remain `uv sync --inexact --locked`; `--inexact`
  preserves the facility stack and `--locked` prevents an implicit re-lock.
  No Torch extra or `--no-extra` flag is selected.
- Validate the resolved runtime with `import torch` after activation and again
  inside the PBS allocation before provisioning or tests. Record the resolved
  Python, Torch/CUDA, PE, lock SHA, and overlay provenance in the receipt.
- Verify interpreter selection through `pyvenv.cfg` or `sys.base_prefix`, not
  `uv python find`; invoke the overlay Python directly after synchronization,
  and never run `uv sync` concurrently in workers.

Facility paths, exports, project accounts, and Eagle run roots belong in an
approved run receipt rather than committed defaults. The pre-upgrade validation
is dated provenance only and must not be treated as a current stack definition.

Configured runs go through the single `run.py` entrypoint, which launches one
`tpen.runner.Runner` from a YAML config. The legacy Hooke pair smoke training
test config is a working example:

```bash
uv run --extra cpu python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
```

The same entrypoint is installed as a `tpen` console script, so
`uv run --extra cpu tpen --config <config>` is equivalent. `run.py` stays
supported during the transition.

Human-readable run timestamps are controlled by `run.timezone`, an IANA
timezone name. The code default is `UTC`; the Hooke smoke config uses
`America/New_York` so run IDs, `metadata.json`, `status.json`, and terminal
status boxes share the same cluster-log convention.

For a syntax-only check:

```bash
uv run --extra cpu python -m compileall tpen run.py typechecked.py
```

## Optional W&B Tracking

TPEN can optionally mirror scalar run metrics to Weights & Biases for
dashboarding and monitoring. W&B is an observability backend only; the local run
directory remains the authoritative experiment record.

Install optional W&B support:

```bash
uv sync --extra wandb
```

For interactive authentication:

```bash
wandb login
```

For non-interactive jobs:

```bash
export WANDB_API_KEY=<your-api-key>
```

Add W&B as another root-level logger:

```yaml
loggers:
  - _target_: tpen.logging.CSV
    path: ${run.dir}/metrics.csv
  - _target_: tpen.logging.JSONL
    path: ${run.dir}/metrics.jsonl
  - _target_: tpen.logging.WandB
    project: spenn-qmc
    entity: null
    mode: online
    group: hooke_pair
    tags:
      - hooke
      - vmc
```

For jobs without reliable internet, use W&B offline mode:

```bash
wandb offline
uv run --extra cpu python run.py --config experiments/hooke/configs/smoke/pair_train.yaml
wandb sync --sync-all
```

By default, TPEN does not upload checkpoints, traces, raw batches, per-sample
arrays, or full run directories to W&B. W&B receives scalar metrics and compact
config/provenance metadata; CSV/JSONL logs and local artifacts remain canonical.


## Config section types

TPEN configs use two main kinds of sections:

```text
component specs
parameter blocks
```

### Component specs

Component specs describe Python objects that Hydra should instantiate.

They usually contain `_target_`.

Examples:

```yaml
sampler:
  _target_: tpen.sampling.MetropolisSampler
  n_walkers: 16
  n_electrons: 2
  spatial_dim: 3

optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-3
```

### Parameter blocks

Parameter blocks are config-only namespaces. They are not instantiated as Python objects.

They exist to collect readable, user-facing values that component specs can reference.

The parameter blocks make the user-facing knobs easy to find:

```yaml
training:
  batch_size: 16
  n_steps: 100
  learning_rate: 1.0e-3
```

The component specs use those knobs through interpolation:

```yaml
sampler:
  n_walkers: ${training.batch_size}

optimizer:
  lr: ${training.learning_rate}

trainer:
  n_steps: ${training.n_steps}
```

## Timing Metrics

Timing instrumentation is callback-owned and logs through the same CSV/JSONL
logger path as other metrics:

```yaml
callbacks:
  - _target_: tpen.callback.RunTiming

  - _target_: tpen.callback.TrainStepTiming
    every_n_steps: 1
    rolling_window: 20
    accelerator_synchronize: false

  - _target_: tpen.callback.EvaluationTiming
    accelerator_synchronize: false

  - _target_: tpen.callback.DiagnosticTiming
    accelerator_synchronize: false
```
Device synchronization is opt-in with `accelerator_synchronize: true` for
benchmarking; it is disabled by default for normal training. The flag routes
through `tpen.accelerator.synchronize`, so it covers CUDA, ROCm, and XPU alike.

## Eager Model Invariant

TPEN model construction owns trainable state. All trainable parameters must be
registered during ``__init__`` from explicit architecture metadata such as
channel counts and maximum order. A forward pass may allocate activations whose
tuple axes depend on the runtime particle count, including zero-sized axes, but
it must not create, resize, replace, move, or cast parameters or buffers.

The sampler is never involved in model construction. There is no materialization
batch in the eager design: setup moves the model to the configured device and
dtype, the optimizer is built from the complete parameter set, and forward only
evaluates the already-constructed model.

Current tensor layouts are:

```text
Feature order m:
  [batch, channels, i1, ..., im]

Interaction order m:
  [batch, channels, paths, i1, ..., im]
```


## Checks After Changes

After code changes, run the fast syntax and test checks:

```bash
uv run python -m compileall tpen run.py typechecked.py
uv run pytest -q
```

## Runtime Type Checking

Pytest installs Typeguard instrumentation for `tpen` by default.

Run tests with Typeguard instrumentation for `tpen`:

```bash
uv run pytest -q
```

## Equivariance Checking

Equivariance checks are runtime checks on `tpen.equivariance.EquivariantMap`.
When enabled, small systems are checked against every particle permutation;
larger systems are checked against adjacent transpositions and reversal. Configs
force checks with `probability: 1.0` on the `RuntimeEquivariance` callback.

Runtime validation is a typed, per-object contract kept **separate** from
equivariance. `Feature`, `Interaction`, and `ElectronBatch` each
expose a `validate()` method (and,
where useful, `validity_metrics()`) that checks their own semantic fields; the
static contracts live in `tpen.data.validation` (`RuntimeValidatable`,
`RuntimeValidityMetrics`). This is deliberately distinct from
`tpen.data.equivariant_state.EquivariantState`, which declares only particle
permutation (`permute`) and comparison (`compare`). There is no generic
tree-validation, tree-permutation, or particle-count-inference helper:
validation, permutation, and comparison are declared by typed data objects,
never inferred by recursively probing arbitrary containers.

Exact testing strategy:

- Permutation convention and algebra:
  `tpen.data.permutation.Permutation`,
  `tpen.data.indices.permute_tuple_slots`, and
  `tests/unit/data/test_permutation.py`.
- State actions:
  `tpen.data.equivariant_state.EquivariantState`,
  `tpen.data.real.Feature`, `Interaction`, `Update`,
  `tpen.data.batch.WavefunctionOutput`, and tests in
  `tests/unit/data/test_equivariant_state.py`,
  `tests/unit/data/test_real_feature.py`,
  `tests/unit/data/test_real_interaction.py`, and
  `tests/unit/data/test_real_update.py`.
- Runtime equivariance checks:
  `tpen.equivariance.checks.FullModelEquivarianceChecker` and
  `TraceEquivarianceChecker` (driven by `tpen.callback.RuntimeEquivariance`),
  using `apply_particle_permutation` and typed `.compare(...)`. Pytest-only
  assertion helpers live under `tests/helpers/equivariance.py`, with coverage in
  `tests/unit/equivariance/test_equivariant_map.py`.
- Tensor shape checks:
  `Feature`, `Interaction`, and `Update` are dense order-indexed
  lists of tensors. Index 0 is reserved for zero-order data and must have zero
  channels; use `tpen.data.real.zero_block` to construct that sentinel.
  Validation coverage lives in
  `tests/unit/data/test_tensor_validation.py`.
- Layer-level checks:
  `tpen.nn.Updater`, `tpen.nn.PathAggregation`, and
  `tpen.nn.TPENLayer`, with forced runtime
  checks in `tests/unit/nn/test_update_equivariance.py`,
  `tests/unit/nn/test_path_aggregation_equivariance.py`, and
  `tests/unit/nn/test_tpen_layer_scaffold.py`.
- Virtual-support combinatorics:
  `tpen.reps.paths.PathMetadata`, `generate_virtual_paths`, and
  `validate_virtual_path`, with coverage in
  `tests/unit/reps/test_virtual_paths.py`.

For `n_particles <= 5`, the runtime schedule is exhaustive over all
permutations. Larger-particle tests use deterministic random inputs and random
permutations in addition to the runtime generator schedule.

The new core scaffold is direct, not a compatibility layer:

- `tpen.data`: common state names are exported at the package root for
  convenience, while helpers stay with their owner modules:
  `tpen.data.batch`, `tpen.data.real`,
  `tpen.data.partition`, `tpen.data.permutation`, and `tpen.data.indices`.
  Electron-batch geometry helpers live under `tpen.data.batch`.
- `tpen.reps`: virtual path metadata.
- `tpen.nn`: `EquivariantMixing`, `PathAggregation`,
  `ResidualUpdater`, `TPENLayer`, `TPENWaveFunction`, and readouts under
  `tpen.nn.readout`.
- `tpen.equivariance`: traceable `EquivariantMap`, passive trace recording, and
  runtime equivariance checkers (`tpen.equivariance.checks`).

## Documentation

Documentation sources live under `docs/` and use Sphinx with NumPy-style
docstrings via Numpydoc. The docs tooling is in the opt-in `docs` dependency
group, so normal installs do not include it.

Build the local HTML docs with:

```bash
uv run --extra cpu --group docs sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`, or serve them locally:

```bash
uv run --extra cpu python -m http.server --directory docs/_build/html 8000
```

## Versioning

The current Hooke integration release is `v0.5.2`; release notes live in
[`RELEASE_NOTES.md`](RELEASE_NOTES.md).

The backwards compatibility of this repository is only with respect to the behavior
of Hydra config files. Before v1.0.0, every minor version can break backwards compatibility.
v0.2.0 does not have to be able to reproduce a v0.1.0 config. But patches have to be
compatible with each other.

## Conventions

### Metrics Naming Scheme

Metric naming and logger conventions are documented in
[`tpen/metrics_naming.md`](tpen/metrics_naming.md).

### FASRC Cannon provisioning and smoke orientation

Cannon provisioning and the allocation-local pair-v1 smoke are operator
actions: supply the project account, partition, CUDA module/extra, overlay
location, and results root at run time. Do not commit those values. Inside an
existing GPU allocation, use the tracked [pair-v1 allocation runbook](experiments/hooke/tpen-pair-v1/README.md)
and its single launcher; the launcher never provisions or submits Slurm jobs.
Keep run data outside the checkout and pass the allocation's visible device
values through `--visibility-values` and `CUDA_VISIBLE_DEVICES`.

### ALCF Aurora provisioning and smoke orientation

Aurora uses the facility-provided Python/Intel overlay and PBS. Provision the
overlay and load its required runtime modules before entering the allocation;
pass the operator-supplied account, queue, filesystem, and Eagle run root on
the `qsub` command line. The [pair-v1 allocation runbook](experiments/hooke/tpen-pair-v1/README.md)
uses `--device xpu` with per-worker `ZE_AFFINITY_MASK` values. No Aurora
account, path, module version, or provisioning default is committed, and the
launcher does not invoke PBS or perform provisioning.
