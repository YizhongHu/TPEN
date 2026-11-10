# Metrics Naming and Logging Conventions

This document defines the canonical metric naming scheme for TPEN logs.

The same logical metric identity should be preserved across CSV, JSONL, and optional W&B logging.

## Goal

Metrics should be:

```text
machine-readable
human-scannable
stable across logger backends
easy to group by phase / subsystem
easy to aggregate across runs
easy to visualize in dashboard tools
```

The local run directory remains the authoritative experiment record. External dashboards such as W&B are projections of local records, not replacements for them.

---

## Canonical metric identity

A scalar metric is identified by:

```text
namespace + key
```

The canonical logical path is:

```text
<namespace>/<key>
```

Examples:

```text
train/energy
train/sampler/acceptance_rate
train/perf/step_time_sec
checks/equivariance/full_model/max_abs_error
runtime/wall_time_sec
diagnostics/energy/time_sec
```

Use:

```text
namespace:
  slash-separated ownership / phase path

key:
  underscore-separated leaf metric name
```

Examples:

| Namespace                        | Key               | Logical path                                   |
| -------------------------------- | ----------------- | ---------------------------------------------- |
| `train`                          | `energy`          | `train/energy`                                 |
| `train`                          | `energy_stderr`   | `train/energy_stderr`                          |
| `train/sampler`                  | `acceptance_rate` | `train/sampler/acceptance_rate`                |
| `train/perf`                     | `step_time_sec`   | `train/perf/step_time_sec`                     |
| `checks/data_integrity`           | `passed`          | `checks/data_integrity/passed`                  |
| `checks/equivariance/full_model` | `max_abs_error`   | `checks/equivariance/full_model/max_abs_error` |
| `runtime`                        | `wall_time_sec`   | `runtime/wall_time_sec`                        |
| `diagnostics/energy`             | `time_sec`        | `diagnostics/energy/time_sec`                  |

Do not treat `key` alone as globally unique.

---

## Namespace conventions

Namespaces describe **where the metric comes from** or **which subsystem owns it**.

Preferred namespaces:

```text
train
train/sampler
train/perf

eval
eval/sampler
eval/perf

runtime

diagnostics/energy

checks/data_integrity
checks/gradient
checks/sampler
checks/equivariance/full_model
checks/equivariance/trace
```

Phase meaning:

```text
train       optimization-time metrics from the training sampler
eval        evaluation metrics produced by the Evaluate runner through an
            Evaluator; the only place for eval/energy_error,
            eval/energy_abs_error, and eval/reference_energy
checks/*    runtime data/numerics soundness (DataIntegrity and friends),
            not model selection
```

There is **no `validation/*` phase.** No callback emits that namespace, and
`tests/integration/training/test_hooke_pair_smoke_training.py` asserts that no
emitted namespace starts with `validation`. An earlier train-end validation
callback was removed; this document described it for some time after it was
gone.

An `Evaluator`'s namespace is a **user-defined string**, not a phase the code
infers: `EvaluationContext` has no `phase` field and `Evaluator` accepts any
namespace, which `tests/unit/evaluation/test_namespace_is_user_defined.py`
pins. A run stage *called* "validation" (as in
`experiments/hooke/pair_stability_v3`) is free to log under any namespace it
configures, and today those stages configure `namespace: eval`. If a
configuration ever does pick `validation`, that is a name the config chose, not
a phase this document defines.

Two namespaces are composed rather than written literally:

```text
checks/equivariance/<checker_log_name>   one per configured checker
<evaluator or task namespace>/status     suite- and task-level status flags
```

Avoid putting hierarchy inside the key:

```text
Bad:
  namespace = train
  key = sampler.acceptance_rate

Good:
  namespace = train/sampler
  key = acceptance_rate
```

Do not mix separators unless there is a deliberate compatibility reason.

Preferred:

```text
train/sampler/acceptance_rate
checks/equivariance/full_model/max_abs_error
```

Avoid:

```text
train/sampler.acceptance_rate
checks.equivariance.full_model.max_abs_error
checks_equivariance_full_model_max_abs_error
```

---

## Key conventions

Keys describe the **leaf metric**.

Preferred key style:

```text
loss
energy
energy_variance
energy_std
energy_stderr
local_energy_n_finite
local_energy_n_total
local_energy_finite_fraction
local_energy_nonfinite_count

logabs_mean
logabs_min
logabs_max
nonfinite_logabs_fraction

grad_norm
param_norm
loss_has_grad
optimizer_step

acceptance_rate
n_walkers
burn_in
n_steps
proposal_scale
seed

position_mean_abs
position_rms
position_max_abs
radius_mean
radius_std
radius_q50
radius_q90
radius_q99
radius_max
center_of_mass_rms
electron_distance_min
electron_distance_q01
electron_distance_q05
electron_distance_mean
electron_distance_q50
electron_distance_q95
electron_distance_q99
electron_distance_max
electron_distance_n_pairs

trajectory_retained_draw_count
trajectory_discarded_draw_count
trajectory_n_walkers
trajectory_draw_stride
trajectory_sampler_burn_in
trajectory_proposal_scale
trajectory_retained_value_count
trajectory_discarded_value_count
trajectory_retained_transition_count
trajectory_discarded_transition_count
trajectory_retained_draw_acceptance_rate_mean
trajectory_discarded_draw_acceptance_rate_mean
trajectory_retained_draw_minimum_electron_nucleus_radius
trajectory_discarded_draw_minimum_electron_nucleus_radius
trajectory_intermediate_sampler_steps_observed

passed
max_abs_error
n_failed_entries

wall_time_sec
step_time_sec
```

### Uncertainty keys: `*_stderr` is IID-only

Every `*_stderr` key in this document — `energy_stderr`, `energy_term_<name>_stderr`,
and any `{prefix}_stderr` — is an **IID-only** standard error, `sigma / sqrt(N)`
over finite samples. Five call sites compute it and all agree:
`tpen.training.vmc.compute_vmc_objective`, `tpen.training.vmc.summarize_local_energy_terms`,
`tpen.physics.hamiltonian.summarize_local_energy`,
`tpen.diagnostics.energy._summarize_total_energy`, and
`tpen.evaluation.summaries.local_energy.summarize_values`.

MCMC walkers are serially correlated, so `sigma / sqrt(N)` understates the true
uncertainty by roughly `sqrt(tau_int)`. These keys are progress and diagnostic
signals. **They are not reportable error bars, and none of them is an MCSE.**

The correlation-aware quantity is produced by
`tpen.statistics.produce_trajectory_statistics` and is **not** a metric key. It is
emitted as an immutable sidecar receipt keyed on
`(stage, run_id, attempt_id, checkpoint_sha256, config_sha256, observable, evaluator_id)`,
carrying `tau_int`, `ess` and `mcse` together with a `status` of
`available`/`unresolved`/`absent`. A consumer wanting a defensible error bar reads
that receipt; it must never rescale a `*_stderr` key and call the result an MCSE,
and must never read a missing receipt as zero correlation.

Sampler snapshot geometry keys (the `position_*`/`radius_*`/`electron_distance_*`/
`center_of_mass_rms` block above) come from
`tpen.sampling.summarize_walker_geometry` and appear under whichever namespace
logged the sampler stats — `train/sampler/*` today.

Evaluation trajectories add a second typed producer,
`tpen.sampling.SamplerTrajectoryDiagnostics`. Its scalar projection uses the
`trajectory_*` keys above; `SamplerStatsSummary` applies its configured
`sampler_` prefix when logging them in the evaluation task namespace (He-v1 is
`eval/mcmc_energy`). The existing scalar `acceptance_rate` keeps its exact
meaning: the `SamplerStats` value returned by the final sampler call. It is not
replaced by a draw series. Draw-resolved values live in the versioned
`sampler_trajectory_diagnostics/v1` sidecar under the separately registered
fields `retained_draw_acceptance_rate_series` and
`discarded_draw_acceptance_rate_series`.

`trajectory_retained_draw_minimum_electron_nucleus_radius` is the raw minimum
over collector-retained walker states only. `trajectory_draw_stride` states how
many sampler steps separate those observations, and
`trajectory_intermediate_sampler_steps_observed` is always false: accepted
states between retained draws are not visible through `collect_samples`.
Therefore this metric must never be described as the minimum over every state
reached by the sampler. The discarded-draw minimum has its own key and is never
pooled with the retained minimum. Sampler-internal burn-in states are also
unobserved; `trajectory_sampler_burn_in` records their configured step count,
not their geometry.

Sampler snapshot keys are composed by `tpen.sampling.SamplerStats`, the typed
record a sampler returns from `collect_samples`; trajectory keys are composed
by `tpen.sampling.SamplerTrajectoryDiagnostics`. `SamplerStats.as_metrics`
produces the full `*/sampler` snapshot key set (named fields plus the open
geometry key set), and `SamplerStats.as_check_metrics` produces the fixed
`checks/sampler` subset `acceptance_rate`, `n_walkers`, `n_steps`, `burn_in`, to
which `SamplerHealth` adds `passed`. Draw-resolved diagnostics never enter that
fixed runtime-check subset.

Use underscores inside keys.

Avoid dots in keys.

### The live key set, by namespace

The list above is style guidance. This is the actual set the code emits today,
in emission order, with the module that spells each group. Anything not here is
not currently logged.

`train` — one record per logged iteration, from `VMCTrainer`:

```text
loss                                    | tpen.training.vmc.compute_vmc_objective
energy                                  |
energy_variance                         |
energy_std                              |
energy_stderr                           |
local_energy_n_finite                   |
local_energy_n_total                    |
local_energy_finite_fraction            |
local_energy_nonfinite_count            |

logabs_mean                             | tpen.training.vmc.summarize_logabs
logabs_min                              |
logabs_max                              |
nonfinite_logabs_fraction               |

energy_term_<name>                      | tpen.training.vmc
energy_term_<name>_variance             |   .summarize_local_energy_terms,
energy_term_<name>_std                  |   only when return_terms is enabled
energy_term_<name>_stderr               |
energy_term_<name>_n_finite             |
energy_term_<name>_n_total              |
energy_term_<name>_finite_fraction      |
energy_term_<name>_nonfinite_count      |

grad_norm                               | tpen.training.trainer (trainer-owned)
param_norm                              |
loss_has_grad                           |
optimizer_step                          |
```

`train/electron_nucleus_distance` — optional `ElectronNucleusDistanceQuantiles`
callback, one record per configured cadence:

```text
electron_<index>_nucleus_distance_min
electron_<index>_nucleus_distance_q<level>
```

The callback reads the `ElectronBatch` used by the completed iteration. For
each electron, it first takes the minimum distance over nuclei for every
configuration, then summarizes those distances across the whole training
batch. Quantile levels are supplied by the run configuration; core supplies no
default levels. Quantiles use PyTorch's `linear` interpolation, and labels use
the energy-diagnostic convention (`q001`, `q01`, `q50`, `q99`, `q999`, or a
percent label such as `q25`).

`train/sampler` — `SamplerStats.as_metrics`:

```text
acceptance_rate
n_walkers
burn_in
n_steps
proposal_scale
seed                                    only when the sampler was seeded
<geometry keys>                         the position_*/radius_*/
                                        electron_distance_*/center_of_mass_rms
                                        block, whose membership varies with
                                        n_electrons
```

`<evaluation task namespace>` — `SamplerStatsSummary`, when the generator
published `SamplerTrajectoryDiagnostics` (the configured prefix is `sampler`
by default, so these are full metric keys rather than a nested namespace):

```text
sampler_trajectory_retained_draw_count
sampler_trajectory_discarded_draw_count
sampler_trajectory_n_walkers
sampler_trajectory_draw_stride
sampler_trajectory_sampler_burn_in
sampler_trajectory_proposal_scale
sampler_trajectory_retained_value_count
sampler_trajectory_discarded_value_count
sampler_trajectory_retained_transition_count
sampler_trajectory_discarded_transition_count
sampler_trajectory_retained_draw_acceptance_rate_mean
sampler_trajectory_discarded_draw_acceptance_rate_mean       only with discarded draws
sampler_trajectory_retained_draw_minimum_electron_nucleus_radius
                                                            only with typed nuclei
sampler_trajectory_discarded_draw_minimum_electron_nucleus_radius
                                                            only with discarded draws
                                                            and typed nuclei
sampler_trajectory_intermediate_sampler_steps_observed       always false
```

The same summary continues to emit its pre-existing scalar snapshot keys such
as `sampler_acceptance_rate`. Those names and values are unchanged.

`train/perf` — `TrainStepTiming` then `TrainPhaseTiming`:

```text
step_time_sec
step_time_sec_rolling_mean
<phase>_time_sec                        see "Timing metrics" below
```

`checks/data_integrity` — `DataIntegrity`; membership depends on which state
the callback found, so this namespace's key set is genuinely conditional:

```text
local_energy_finite_count
local_energy_total_count
local_energy_nonfinite_fraction
logabs_finite_count
logabs_total_count
logabs_nonfinite_fraction
sign_invalid_fraction                   only when a sign tensor is present
output_validated
loss_is_finite
batch_validated
batch_<key>                             from ElectronBatch.validity_metrics()
passed
```

`checks/gradient` — `GradientStats`:

```text
n_grad_tensors
n_grad_elements
global_grad_norm
max_abs_grad
mean_abs_grad
nonfinite_grad_fraction
passed
```

`checks/sampler` — `SamplerStats.as_check_metrics` plus `SamplerHealth`:

```text
acceptance_rate
n_walkers
n_steps
burn_in
passed
```

`checks/equivariance/<checker_log_name>` — checker-supplied metrics plus:

```text
passed
n_comparisons
checker_class
artifact_path                           only when an artifact_dir is configured
                                        and the checker produced an artifact
```

`runtime` — `RunTiming` and `ResourceUsage`:

```text
start_time_unix
end_time_unix
wall_time_sec
failed                                  only on a failed run
peak_memory_mb
cuda_max_memory_allocated_mb            only when CUDA is available
cuda_max_memory_reserved_mb
cuda_device_count
```

`peak_memory_mb` is host process peak RSS. The three `cuda_*` fields are device
measurements and are unavailable on CPU: they are omitted, never emitted as
zero. Cost-table projections preserve that distinction with blank device-peak
cells plus `device_peak_memory_available = false`.

`<evaluator namespace>/status` and `<task namespace>/status` — `Evaluate`:

```text
suite_success                           suite level
suite_failed
task_success                            task level
task_failed
```

`<helium atlas task namespace>` — helium atlas summaries in
`tpen.evaluation.summaries.helium_atlas`:

```text
<curvature_prefix>_<window>_total_count
<curvature_prefix>_<window>_finite_count
<curvature_prefix>_<window>_nonfinite_count
<curvature_prefix>_<window>_available
<curvature_prefix>_<window>_second_derivative_mean       when available
<curvature_prefix>_<window>_second_derivative_min        when available
<curvature_prefix>_<window>_second_derivative_max        when available
<curvature_prefix>_<window>_directional_spread           when available

<tail_prefix>_total_count
<tail_prefix>_finite_count
<tail_prefix>_nonfinite_count
<tail_prefix>_direction_count
<tail_prefix>_available
<tail_prefix>_slope                                      when available
<tail_prefix>_extrema_min                                when available
<tail_prefix>_extrema_max                                when available
<tail_prefix>_sign_fraction                              when available
<tail_prefix>_outer_radius                               when available
<tail_prefix>_directional_spread                         when available

atlas_total_count
atlas_coordinate_representability_boundary_count
atlas_exact_zero_sentinel_count
atlas_computed_nonfinite_retained_count
atlas_dtype_is_float64
atlas_seed
ideal_unfloored_ee_positive_separation_domain_count      e-e atlas only
ideal_unfloored_ee_reciprocal_evaluation_undefined_count e-e atlas only
ideal_unfloored_ee_reciprocal_failure_boundary_count     e-e atlas only
executed_smoothed_ee_factor_finite_at_ideal_reciprocal_evaluation_undefined_count
                                                            e-e atlas only

helium_atlas_row_count                                   artifact writer enabled
```

The curvature prefix, nested window names, and tail prefix are explicit task
configuration. They must preserve whether a series is ideal unfloored or
executed smoothed behavior; summaries must not collapse those semantics into an
ambiguous generic prefix.

#### `train` record attribution

`grad_norm` is the norm over the parameter gradients that drove this
iteration's update, taken after clipping. `param_norm` is the L2 norm over
trainable parameters. `loss_has_grad` records whether the loss was connected to
the parameters at all; `optimizer_step` records whether an optimizer update was
actually applied. See "Step conventions" for the attribution contract that
binds these together.

---

## CSV logging

`ModuleNorms` (an optional callback, not an always-on trainer metric) writes
one namespace per configured group under `train/module_norms/<group>`. Its
`parameter_*` values are post-update model values; its `gradient_*` values are
the clipped gradients consumed by that update. `n_no_gradient_parameters` is
separate from zero gradients, and `gradient_nonfinite_count` is excluded from
the finite gradient norm reconciliation.

CSV uses long-form scalar records:

```csv
step,namespace,key,value
1,train,loss,0.6580171494985763
1,train,energy,3.658536762108973
1,train/sampler,acceptance_rate,0.8625
1,checks/data_integrity,passed,true
1,checks/equivariance/full_model,max_abs_error,0.0
```

`OptimizerState` is an optional callback, not an always-on trainer metric. It
writes one namespace per configured group under
`train/optimizer_state/<group>` with `step`, `step_min`, `step_max`, and
`step_mismatch`; `exp_avg_l2`, `exp_avg_rms`, `exp_avg_max_abs`;
`exp_avg_sq_l2`, `exp_avg_sq_rms`, `exp_avg_sq_max_abs`; and
`exp_avg_nonfinite_count`, `exp_avg_sq_nonfinite_count`, and
`n_no_optimizer_state`. The nonfinite counts cover all entries before the
finite reductions; when a moment has no finite entries, its three reductions
are `null` rather than an uninformative zero. A mismatched step is represented by `step = null` and
the explicit `step_mismatch = true` flag rather than by averaging steps.

The metric identity is:

```text
(namespace, key)
```

This format avoids sparse, very wide CSV files and supports records from trainers, callbacks, checks, diagnostics, and runtime lifecycle events.

### Required CSV columns

```text
step
namespace
key
value
```

### Optional future columns

If useful later:

```text
run_id
timestamp_unix
```

### Values

Values should be JSON-compatible scalars:

```text
int
float
bool
str
null / empty where appropriate
```

Do not duplicate the same metric under multiple spellings.

Bad:

```csv
step,namespace,key,value
1,train,sampler.acceptance_rate,0.8625
1,train/sampler,acceptance_rate,0.8625
```

Good:

```csv
step,namespace,key,value
1,train/sampler,acceptance_rate,0.8625
```

---

## JSONL logging

JSONL should preserve the same logical metric identity.

### On-disk record format

`metrics.csv` is a UTF-8 newline-delimited long-form file with the exact
header `step,namespace,key,value`. Each metric in a logged record becomes one
row; the `step` and `namespace` values are copied to every row. Values are
serialized as scalar text, with booleans written as `true` or `false` and
missing values represented by an empty field. Existing files are not rewritten
when this format is documented or when logging code changes.

`metrics.jsonl` is a UTF-8 file containing one JSON object per logged record:
`{"step": ..., "namespace": ..., "metrics": {...}}`. The `metrics` object
keeps all scalar values for that record. JSON objects are emitted with sorted
keys and strict JSON numbers (`allow_nan=False`); non-finite floats therefore
raise instead of being written as invalid JSON. The durable record has no
`event` field: typed lifecycle events are dispatched separately and are not
metric-record metadata. This schema applies to records written going forward;
existing run outputs remain unchanged.

Recommended event-oriented shape:

```json
{
  "step": 1,
  "namespace": "train",
  "metrics": {
    "loss": 0.6580171494985763,
    "energy": 3.658536762108973,
    "energy_variance": 1.463636129155399,
    "energy_stderr": 0.3024520756619343
  }
}
```

Subsystem record:

```json
{
  "step": 1,
  "namespace": "train/sampler",
  "metrics": {
    "acceptance_rate": 0.8625,
    "n_walkers": 16,
    "burn_in": 10,
    "n_steps": 5
  }
}
```

Runtime-check record:

```json
{
  "step": 1,
  "namespace": "checks/equivariance/full_model",
  "metrics": {
    "n_particles": 2,
    "n_permutations_tested": 1,
    "n_failed_permutations": 0,
    "max_abs_error": 0.0,
    "passed": true,
    "n_comparisons": 1,
    "checker_class": "FullModelEquivarianceChecker"
  }
}
```

A scalar-row JSONL format is also acceptable:

```json
{"step": 1, "namespace": "train", "key": "energy", "value": 3.658536762108973}
{"step": 1, "namespace": "train/sampler", "key": "acceptance_rate", "value": 0.8625}
```

Both forms are valid if the canonical identity remains:

```text
namespace + key
```

---

## W&B projection

W&B is a dashboard / monitoring projection of local records.

The W&B logger should consume backend-neutral records with:

```text
step
namespace
key
value
```

or:

```text
step
namespace
metrics
```

For scalar metrics, the raw W&B metric name is:

```text
<namespace>/<key>
```

Examples:

| Local namespace                  | Local key         | W&B raw metric                                 |
| -------------------------------- | ----------------- | ---------------------------------------------- |
| `train`                          | `loss`            | `train/loss`                                   |
| `train`                          | `energy`          | `train/energy`                                 |
| `train`                          | `energy_stderr`   | `train/energy_stderr`                          |
| `train/sampler`                  | `acceptance_rate` | `train/sampler/acceptance_rate`                |
| `train/perf`                     | `step_time_sec`   | `train/perf/step_time_sec`                     |
| `checks/data_integrity`           | `passed`          | `checks/data_integrity/passed`                  |
| `checks/equivariance/full_model` | `max_abs_error`   | `checks/equivariance/full_model/max_abs_error` |
| `runtime`                        | `wall_time_sec`   | `runtime/wall_time_sec`                        |
| `diagnostics/energy`             | `time_sec`        | `diagnostics/energy/time_sec`                  |

W&B should not change core metric names.

W&B may add a small, documented set of dashboard aliases and health flags. These are convenience projections, not replacements for canonical metrics.

---

## Dashboard aliases

Dashboard aliases are optional W&B-only duplicates intended to make dashboards easier to build.

If enabled, keep the alias set small.

Recommended aliases:

```text
dashboard/loss
dashboard/energy
dashboard/energy_variance
dashboard/energy_stderr
dashboard/acceptance_rate
dashboard/grad_norm
dashboard/local_energy_finite_fraction
dashboard/step_time_sec
```

Aliases should be derived from canonical metrics.

Examples:

| Canonical metric                     | Dashboard alias                          |
| ------------------------------------ | ---------------------------------------- |
| `train/loss`                         | `dashboard/loss`                         |
| `train/energy`                       | `dashboard/energy`                       |
| `train/energy_variance`              | `dashboard/energy_variance`              |
| `train/energy_stderr`                | `dashboard/energy_stderr`                |
| `train/sampler/acceptance_rate`      | `dashboard/acceptance_rate`              |
| `train/grad_norm`                    | `dashboard/grad_norm`                    |
| `train/local_energy_finite_fraction` | `dashboard/local_energy_finite_fraction` |
| `train/perf/step_time_sec`           | `dashboard/step_time_sec`                |

Do not add dozens of aliases. The canonical namespaces remain the complete metric surface.

---

## Health flags

W&B dashboards benefit from compact numeric status indicators.

If enabled, health flags should use:

```text
1.0 = OK
0.0 = failed / warning condition
```

Recommended health flags:

```text
health/numerics_ok
health/sampler_ok
health/equivariance_ok
health/run_ok
```

Examples:

```text
health/numerics_ok:
  1.0 if local-energy/logabs/batch validity checks pass
  0.0 otherwise

health/sampler_ok:
  1.0 if sampler checks pass
  0.0 otherwise

health/equivariance_ok:
  1.0 if runtime equivariance checks pass
  0.0 otherwise

health/run_ok:
  1.0 if all required health flags are OK
  0.0 otherwise
```

If a flag cannot be derived from the current record, omit it rather than guessing.

Health flags do not replace detailed metrics under `checks/...`.

---

## Step conventions

### The metric step axis is the 0-indexed trainer loop step

Three progress counters exist, and only one of them is the metric step axis:

| Counter            | What it counts                                     | Where it appears                     |
| ------------------ | -------------------------------------------------- | ------------------------------------ |
| loop step          | 0-indexed position in `for step in range(...)`      | **the metric step axis**             |
| `next_iteration`   | durable resume cursor; iterations that completed    | `trainer.json`, manifest, dir names  |
| `completed_updates`| optimizer updates that actually returned            | `trainer.json`, manifest, checkpoint cadence |

The step on a `train`, `train/sampler`, `train/perf`, or `checks/*` record is
the **0-indexed trainer loop step** — `VMCTrainer.fit`'s loop variable. Nothing
else. It is not `next_iteration` and it is not `completed_updates`.

Training metrics use it:

```text
step = 0-indexed trainer loop step
namespace = train
```

Training runtime checks use the same value:

```text
step = 0-indexed trainer loop step
namespace = checks/...
```

Because the loop advances once per *iteration* whether or not that iteration
applied an update, **on a run with vacuum skips this axis counts iterations,
not updates.** A run of `max_steps = N` produces N points on this axis and may
have applied fewer than N updates. `experiments/toolkit/cost.py` inherits this:
its `n_steps` is a count of `train/perf` records, so it counts attempted
iterations.

### `optimizer_step` is the authoritative discriminator

A reader who needs to know whether a point corresponds to a real model change
must consult `train/optimizer_step`. That is the one authoritative signal, and
the step axis alone cannot answer the question.

A skipped iteration (the zero-electron vacuum) emits the **same record shape at
the same coordinate** as any other iteration: same namespaces, same key set,
same order. That is deliberate — consistency means a consumer never branches on
whether an update happened, and record counts stay equal to iteration counts.

No new metric key was added to mark a skip, because the fact is already stated:

```text
optimizer_step = False            authoritative
loss_has_grad  = False            corroborating
grad_norm      = 0.0              corroborating
train/perf: optimizer_step_time_sec and backward_time_sec ABSENT
```

Adding a fourth spelling of one fact would violate the rule against duplicating
a metric under multiple spellings. The typed event stream says the same thing
once more, as `UpdateCompleted` versus `UpdateSkipped`, but those reach
`occurrences.jsonl` and never become a metric record.

`grad_norm = 0.0` on the skip path is a **literal, not a fabrication**:
`loss.requires_grad` is `False`, no backward runs, every `param.grad` is
`None`, and the norm over an empty gradient set is `0.0` anyway. The literal is
exactly what the computation would return. Leaving it as `0.0` rather than
`None` keeps the key set step-independent in JSONL and non-blank in CSV.

### The whole `train` record describes the pre-update model

Every key in the `train` record at loop step `k` describes the model that
produced step `k`'s samples — before that step's optimizer update. That
includes `param_norm`, which is read *before* `optimizer.step()`.

> **Discontinuity.** In runs produced before this change, `param_norm` was read
> *after* `optimizer.step()` and therefore described the post-update model,
> while every other key in the same record described the pre-update one.
> `train/param_norm` values from older runs are **not comparable** with values
> from newer runs. Every other `train` key is unaffected; `param_norm` appears
> in no dashboard alias and in no experiment collector, so the blast radius is
> limited to anyone reading the raw series.

This is a contract, not an incidental ordering: a new trainer-owned `train`
metric must be computable before the update, or it does not belong in this
record.

The `checks/*` namespaces are **not** covered by that contract, and this is a
known inconsistency rather than a claim of correctness:

| Namespace                   | What it actually observes                                     |
| --------------------------- | ------------------------------------------------------------- |
| `checks/data_integrity`     | pre-update tensors                                             |
| `checks/gradient`           | pre-update gradients, read off a post-update model object      |
| `checks/equivariance/*`     | re-runs the model, so it measures the **post-update** model    |

All three log at the same loop step as the pre-update `train/energy`.
`checks/gradient` is correct only because `optimizer.zero_grad` sits at the top
of the loop body, which is an implicit ordering contract. Where observers
should be re-pointed is a separate open question, tracked as the
observation-point contract (ADR-008); this document records the situation and
does not resolve it.

### `checks/gradient/passed` distinguishes empty from healthy

**Semantic change, no name change.** `checks/gradient` publishes the same seven
keys it always did; what `passed` *means* narrowed. (The key list earlier in this
document had omitted `n_grad_tensors`, which the callback has always published.
That was a documentation gap, corrected alongside this section, not a change.)

Every statistic in this namespace is well defined over an empty gradient set:
`global_grad_norm` is `0.0` and `nonfinite_grad_fraction` is `0.0`, so the
finiteness bound holds trivially. `GradientStats` therefore used to report
`passed = True` while observing nothing at all — measured on Cannon three times
with `fail_fast: true` configured and never firing. A silently disabled check is
worse than an absent one, because it looks like coverage.

`passed` now reads `optimizer_step` — the authoritative discriminator above — to
separate the two ways a gradient set can be empty:

```text
n_grad_tensors == 0, optimizer_step == False   passed = True    nothing to differentiate
n_grad_tensors == 0, optimizer_step == True    passed = False   observation is broken
```

The vacuum skip is the first row and stays a pass, for the same reason
`grad_norm = 0.0` there is a literal rather than a fabrication. The second row is
a real failure: an update ran, consumed gradients, and the observer saw none.

No key was added to mark the distinction. `n_grad_tensors` and
`train/optimizer_step` already state it between them and `passed` carries the
verdict, so an `observed` flag would be a third spelling of one fact — the same
reason no key was added to mark a vacuum skip. The key set stays independent of
which row applied, so JSONL/CSV columns are unchanged.

For readers of historical data: in runs produced before this change, a
`checks/gradient` record with `n_grad_tensors = 0` and
`train/optimizer_step = True` at the same step carries `passed = True` and that
value means nothing. Read the `n_grad_tensors`/`optimizer_step` pair instead.

### Cadence gates do not always fire at step 0

Steps are 0-indexed and a fresh run starts at `step = 0`, so on a fresh run the
first step satisfies `step % every_n_steps == 0` and periodic callbacks and
loggers do report immediately.

**This does not hold on `train_resume`.** The loop runs
`range(self.next_iteration, self.max_steps)`, so a resumed run's first step is
the restored cursor, not `0`. A run resumed at `next_iteration = 37` with
`every_n_steps = 10` does not report on its first step; it waits until step 40.
Do not assume a run's first emitted step is `0`, and do not assume every run
has a record at step `0`.

### Checkpoint identity uses the resume cursor, not the metric step

Checkpoint directory names use the trainer's durable resume cursor
`next_iteration`, not the 0-indexed training step. The cursor is assigned near
the end of the loop body, so it advances once per iteration that *ran to
completion*: an iteration that crashes part-way through never advances it and is
retried on resume, and the directory name is therefore exactly the step a
`train_resume` run continues from. The cursor advances whether or not that
iteration applied an optimizer update, which is what distinguishes it from
`completed_updates`, which counts optimizer updates that actually ran; the two
diverge whenever a completed iteration skips its update. A run with
`max_steps = N` writes its terminal checkpoint as the zero-padded `N` directory;
for example, `max_steps = 500` writes `checkpoints/step_000500`.

The `trainer.json` component of a checkpoint carries both counters:

```json
{"next_iteration": 500, "completed_updates": 500}
```

### Checkpoint manifest schema v2 names the cursor

`manifest.json` schema version 2 records both counters under their own names:

```json
{"schema_version": 2, "kind": "tpen.checkpoint", "next_iteration": 500, "completed_updates": 500}
```

This replaces schema v1's single `step` field, which was ambiguous by name and
in fact always held `next_iteration`. Recording the name explicitly resolves a
standing disagreement with `experiments/checkpoint-selection-options.md`, which
keys checkpoint identity on `step`: the two agree on the *value*, because
`step` there is `next_iteration`, but v1 never said so. Anything selecting
checkpoints by identity should read `next_iteration`.

Version acceptance is per restore mode, and there is no v1 -> v2 upgrade path:

| Restore mode   | Accepted `schema_version` | Accepted `kind`                       |
| -------------- | ------------------------- | ------------------------------------- |
| `model_only`   | 1, 2                      | `spenn.checkpoint` (v1), `tpen.checkpoint` (v2) |
| `train_resume` | 2                         | `tpen.checkpoint`                     |

A v1 artifact still restores weights, because `model_only` needs only
`model.pt` and the config hashes.

It cannot be resumed from — but not because the restore path needs
`manifest.completed_updates`. It does not: `train_resume` takes trainer state
from `trainer.json`, and the manifest counters only reach `RestoreReport`. The
real reason is that **a v1 manifest cannot prove which `trainer.json` key set it
carries.** B1 renamed the trainer keys from `global_step`/`completed_steps` to
`next_iteration`/`completed_updates` *without* bumping the manifest schema, so
`schema_version 1` spans both populations: written before B1 it is genuinely
unresumable, written after B1 it would resume fine. The version cannot
distinguish them, so `train_resume` refuses all v1 at the schema gate rather
than guessing, naming both the version and the mode.

Provenance in v2 records `tpen_version` (v1 recorded `spenn_version`).

**Operational note.** Any checkpoint written between the B1 merge and the C1
merge carries a v1 manifest with post-B1 trainer keys. It is materially
resumable, but `train_resume` refuses it anyway, for the reason above. Re-train,
or restore it with `model_only`. There is deliberately no migration path.

`latest.json` is deliberately minimal and unchanged: a `checkpoint_dir`
pointer, its `step` number, and a timestamp. TPEN retains every committed
checkpoint; `keep_last` remains accepted as an ignored compatibility input.
The pointer is only a resume convenience, not a retention or storage
disposition policy. Storage disposition lives outside TPEN.

### Checkpoint cadence counts completed updates

On the periodic path, the `Checkpoint` callback's `every_n_steps` counts
`completed_updates`, not the resume cursor and not the 0-based loop step:
periodic checkpoints are spaced by optimizer updates that actually ran.

Which iterations are *eligible* is decided by the typed `UpdateCompleted` event,
which the trainer emits in the one place `completed_updates` is incremented —
immediately after `optimizer.step()` returns. That yields exactly one candidate
per completed update by construction. A vacuum iteration emits `UpdateSkipped`
instead, which `Checkpoint` does not subscribe to, so no periodic checkpoint
fires for it: a run of vacuum iterations writes no periodic checkpoints at all,
rather than one per iteration. This selection is unconditional — with no
`every_n_steps` configured, the periodic path writes on every *completed
update*, not on every iteration.

Boundary and coordinate are separate, and the distinction matters across resume:

| Role                 | Value                                               |
| -------------------- | --------------------------------------------------- |
| Periodic write       | `TrainingIterationCompleted` occurrence             |
| Periodic eligibility | `UpdateCompleted` occurrence                        |
| Terminal write       | `TrainingCompleted` occurrence                      |
| Cadence coordinate   | durable `trainer.state_dict()["completed_updates"]` |
| Directory identity   | durable `trainer.state_dict()["next_iteration"]`    |

The run-local `Occurrence.count` is deliberately *not* the cadence coordinate.
Occurrence counts restart at 1 for a new `RunContext`, so a resumed run gating
on them would shift its checkpoint phase relative to an uninterrupted run.
Gating on the durable counter means a run restored at `completed_updates = N`
checkpoints at exactly the points an uninterrupted run that reached `N` would.
This is a checkpoint-local guarantee; durable cadence continuation for
occurrence-gated callbacks generally remains deferred.

The write happens at `TrainingIterationCompleted` rather than at the
`UpdateCompleted` occurrence that selects it. `UpdateCompleted` fires
immediately after `optimizer.step()` returns, which is **before** any health
check runs, so writing there would persist a model version no check has yet
accepted and would make a `fail_fast` abort leave the rejected model as the
default resume target (item `195c3ff3`). Sharing the checks' boundary means a
failed check raises out of the callback-list dispatch loop before `Checkpoint`
runs. That property is pinned by
`tests/unit/training/test_health_checkpoint_order.py`, which reads the
production callback order from
`experiments/hooke/tpen-pair-v1/configs/train.yaml` rather than restating it.

The **terminal** write is not periodic. It fires at `TrainingCompleted`, which
the runner emits once `VMCTrainer.fit` returns, and update selection
deliberately does not apply to it: a terminal checkpoint must land even when the
final iteration skipped its update, and even when the loop body never executed
at all (`max_steps = 0`, or a fully-resumed run, where `TrainerState.step` is
still `-1`) — which is exactly why the terminal moment needs its own event
rather than being read off the last iteration. When a terminal `Checkpoint` is
given an `every_n_steps`, that window keeps the resume-cursor coordinate it has
always used, not `completed_updates`.

Which of the two writes an instance performs is a semantic option — `periodic`
and `terminal`, both defaulting to true — replacing the `triggers: [step_end]`
and `triggers: [train_end]` selection configs used to spell. Under ADR-E002 a
config never names an event.

Directory identity is unaffected by cadence: a checkpoint written at
`completed_updates = 40` still lands in the directory named by its
`next_iteration`, which may be larger.

### Evaluation metrics are logged at step 0

There is no evaluation step counter. `tpen/runner/evaluate.py` hard-codes
`step=0` at all three of its logging sites — the suite status record, each
task's metrics record, and each task's status record:

```text
step = 0                                always, not "an eval step or 0"
namespace = <evaluator namespace> or <task namespace>
```

An evaluation run therefore produces a flat set of records at a single
coordinate. That is adequate because one `Evaluate` run performs one evaluation
pass; it is not a step *series*.

**Evaluation metric records deliberately carry no checkpoint identity.** They
do not record which checkpoint was restored, in any field. This is not an
oversight: nothing currently joins evaluation metrics to a training curve, so
adding an identity now would be speculative construction. `RestoreReport`
already carries the restored checkpoint's counters, and when a real consumer
appears, attaching that identity is one local change.

Evaluation's lifecycle is now typed, and the typed vocabulary is the domain's
only reporting channel: `EvaluationStarted` / `EvaluationCompleted`, the
`EvaluationTaskRun` and `ComponentRun` operations, `ComponentFailed`, and
`CheckpointRestored`, whose `RestoreReport` reaches `occurrences.jsonl`
field-wise and is where the restored checkpoint's identity is durably recorded.
All five evaluation callbacks read those occurrences, and the legacy
`evaluate_start` / `evaluate_end` / `task_*` / `<component>_*` strings are gone.
**No metric key moved**: every name in this document is unchanged, every
evaluation record is still logged at `step = 0`, and evaluation metric records
still carry no checkpoint identity.

The run's own lifecycle is typed too: `RunStarted`, `RunCompleted`, and
`RunFailed` in `tpen.run_events`, emitted only by `tpen.run.run_from_config`.
`EvaluationTiming` now writes the sole `eval/perf {failed: True}` record off
`RunFailed` rather than off the `exception` string. `ArtifactIndex` still keeps
`run_end`, the only thing that writes `diagnostics/index.json` for a suite with
no tasks, because it is a `StatefulCallback` and the run lifecycle carries no
domain state, so the dispatcher would skip a typed run-level occurrence for it.
`Status` keeps `run_start` / `run_end` / `exception` for the same reason.

**One durable change, and it is a removal of duplication rather than of a
series.** `tpen.run` emitted `run_failed` and `exception` back to back with the
same payload, and `RunTiming` and `ResourceUsage` answered both, so a FAILED run
wrote its `runtime` record twice -- with a later `end_time_unix` and a longer
`wall_time_sec` the second time. One `RunFailed` replaces both, so a failed run
now writes one `runtime` record. No metric name changes and no series
disappears.

Run-level metadata may use `step = 0`:

```csv
step,namespace,key,value
0,runtime,start_time_unix,1730000000.123
0,runtime,end_time_unix,1730000420.456
0,runtime,wall_time_sec,420.333
```

If a logger supports nullable steps, run-level metadata may use `step = null`. If not, use `0`.

---

## W&B step axes

W&B should use explicit step metrics. Do not rely only on W&B’s implicit step counter.

Suggested definitions:

```python
run.define_metric("train/*", step_metric="train/step")
run.define_metric("train/sampler/*", step_metric="train/step")
run.define_metric("train/perf/*", step_metric="train/step")

run.define_metric("eval/*", step_metric="eval/step")
run.define_metric("eval/sampler/*", step_metric="eval/step")
run.define_metric("eval/perf/*", step_metric="eval/step")

run.define_metric("checks/*", step_metric="checks/train_step")
run.define_metric("diagnostics/*", step_metric="eval/step")

run.define_metric("dashboard/*", step_metric="train/step")
run.define_metric("health/*", step_metric="train/step")
```

The W&B logger is responsible for adding appropriate step fields, for example:

```text
train/step
eval/step
checks/train_step
```

when logging records in those namespaces.

`eval/step` is registered even though evaluation records are all logged at
step `0`; it remains part of the explicit W&B projection contract.

Runtime metrics such as `runtime/wall_time_sec` may be logged once and also written to the W&B run summary.

---

## Timing metrics

Timing metrics should use seconds.

Use suffixes:

```text
_time_sec
_wall_time_sec
```

Recommended run-level timing:

```text
runtime/start_time_unix
runtime/end_time_unix
runtime/wall_time_sec
```

Recommended run-level memory (from the `ResourceUsage` callback; MiB units):

```text
runtime/peak_memory_mb
runtime/cuda_max_memory_allocated_mb
runtime/cuda_max_memory_reserved_mb
runtime/cuda_device_count
```

Recommended training timing (`step_time_sec` from `TrainStepTiming`; every
phase key from `TrainPhaseTiming` observing typed `TrainingPhase` scopes):

```text
train/perf/step_time_sec
train/perf/step_time_sec_rolling_mean
train/perf/sampling_time_sec
train/perf/batch_build_time_sec
train/perf/local_energy_time_sec
train/perf/forward_time_sec
train/perf/objective_time_sec
train/perf/backward_time_sec
train/perf/optimizer_step_time_sec
train/perf/post_step_metrics_time_sec
```

Each phase key is `f"{phase_name}_time_sec"`, where `phase_name` is a
`ClassVar` declared on the concrete `tpen.training.events.TrainingPhase`
subclass that the trainer scopes. The phase type is the single source of the
metric fragment, so the names above cannot drift from the loop:

| Phase type        | `phase_name`        | Metric key                     |
| ----------------- | ------------------- | ------------------------------ |
| `CollectSamples`  | `sampling`          | `sampling_time_sec`            |
| `BuildBatch`      | `batch_build`       | `batch_build_time_sec`         |
| `LocalEnergy`     | `local_energy`      | `local_energy_time_sec`        |
| `Forward`         | `forward`           | `forward_time_sec`             |
| `Objective`       | `objective`         | `objective_time_sec`           |
| `Backward`        | `backward`          | `backward_time_sec`            |
| `OptimizerUpdate` | `optimizer_step`    | `optimizer_step_time_sec`      |
| `Metrics`         | `post_step_metrics` | `post_step_metrics_time_sec`   |

Phase times approximately sum to at most `step_time_sec`; the difference is
unclassified loop overhead (gradient clipping, event dispatch, logging, and the
pre-update parameter-norm read). That last item used to fall inside the
`Metrics` scope: because `param_norm` is now computed before the optimizer
update, `post_step_metrics_time_sec` no longer includes it and is slightly
smaller than in older runs.
`TrainPhaseTiming` is trigger-free: it observes `Started`/`Ended` boundaries of
`TrainingPhase` scopes and reports only on a successful typed
`TrainingIterationCompleted` event. Its scalar `every_n_steps`, `start_step`,
`max_calls`, `probability`, and `seed` options configure a callback-local
occurrence cadence for those successful completions; `start_step=0` maps to
one-based occurrence `1`, and `every_n_steps=None` reports every successful
occurrence. Unconditional `Ended[TrainingIteration]` observation clears
measurements for failed and cadence-skipped iterations.

`train/perf/sampling_time_sec` at step 0 silently includes burn-in. The
`CollectSamples` scope brackets the whole `collect_samples` call, and burn-in
runs once, on the first iteration of a chain. A sampler with `burn_in = 20` and
`n_steps = 10` therefore times 30 Metropolis-Hastings transitions on iteration 0
and 10 on every later iteration, so step 0 is not comparable with the rest of
the series. This is documented rather than fixed: splitting burn-in out would
need a separate sampler phase type, and nothing consumes one today. Treat step 0
as an outlier when reading the series, or drop it.

An iteration that applied its optimizer update emits `UpdateCompleted` after
the `OptimizerUpdate` scope closes; an iteration that deliberately skipped the
update (the zero-electron vacuum) opens no `OptimizerUpdate` scope, emits
`UpdateSkipped`, and therefore logs no `optimizer_step_time_sec`.

Occurrence cadence is local to one `RunContext` and restarts for a new run or
context. Durable cadence continuation across checkpoint resume remains
deferred; A2 does not migrate other scientific cadence-bearing callbacks.

Recommended evaluation timing:

```text
eval/perf/wall_time_sec
diagnostics/energy/time_sec
```

Recommended per-task evaluation component timing (from
`EvaluationComponentTiming`, driven by the `Started`/`Ended` boundaries of the
evaluator's typed `GeneratorRun`, `CalculatorRun`, and `SummaryRun` scopes;
logged as one `eval/perf/<task_name>` record when the enclosing
`EvaluationTaskRun` scope ends, on the success and failure paths alike). Each
`<component kind>` fragment below is the `component_kind` ClassVar on the
operation type, never a literal in the callback:

```text
eval/perf/<task_name>/generator_time_sec
eval/perf/<task_name>/calculator/<calculator_name>_time_sec
eval/perf/<task_name>/summary/<summary_name>_time_sec
```

`summary/sampled_records_time_sec` measures only the explicitly scoped
`SampledRecordWriter` summary invocation. `summary/sampler_stats_time_sec`
includes the draw-diagnostics sidecar published by `SamplerStatsSummary`; both
names belong in He-v1's writer-summary subset. The complete trajectory CSV is
streamed while the generator runs, so that I/O remains inside
`generator_time_sec`; it must not be silently re-labelled as summary or writer
time. `experiments.toolkit.cost.artifact_timing_comparison` assesses that
streamed-artifact effect with interleaved artifact-off/on generator timings.
Only rows explicitly marked as measured are admitted. The projection preserves
signed `on - off` deltas (including negative values), reports min/median/spread,
and labels at least three pairs as a repeated comparable delta rather than
automatically claiming that the measured magnitude exceeds its dispersion.

`cost_by_task_rows` reconciles the existing task and component boundaries with
a signed `unattributed_time_sec`. A negative residual is retained and marks
`timing_reconciled = false`; it is never clamped. `values_per_sec` divides the
explicit discarded-plus-retained trajectory value count by
`generator_time_sec` and carries host/device/allocation provenance supplied by
the caller. A single run is throughput evidence only, never an efficiency or
speedup claim.

CSV examples:

```csv
step,namespace,key,value
0,runtime,start_time_unix,1730000000.123
0,runtime,end_time_unix,1730000420.456
0,runtime,wall_time_sec,420.333
1,train/perf,step_time_sec,0.842
1,train/perf,sampling_time_sec,0.301
1,train/perf,local_energy_time_sec,0.412
0,eval/perf,wall_time_sec,12.4
0,eval/perf/energy,generator_time_sec,3.1
0,eval/perf/energy,calculator/local_energy_time_sec,7.9
0,diagnostics/energy,time_sec,11.8
```

Timing is runtime metadata. Do not put timing inside physics/statistics helpers such as local-energy summary functions.

---

## Hamiltonian term metrics

For dict-shaped Hamiltonians, configured keys are authoritative term names.

Example:

```yaml
hamiltonian_terms:
  kinetic:
    _target_: tpen.physics.kinetic.KineticEnergy

  harmonic_trap:
    _target_: tpen.physics.potential.HarmonicTrap
    omega: ${system.omega}

  electron_electron:
    _target_: tpen.physics.potential.ElectronElectronInteraction
```

Metric names should use these configured names:

```text
train/energy_term_kinetic
train/energy_term_harmonic_trap
train/energy_term_electron_electron
```

CSV examples:

```csv
step,namespace,key,value
1,train,energy_term_kinetic,0.04717715942440409
1,train,energy_term_kinetic_variance,0.0025556148614036237
1,train,energy_term_harmonic_trap,3.4269269388366514
1,train,energy_term_electron_electron,0.18443266384791782
```

Do not generate class-name/index metric names such as:

```text
energy_term_HarmonicTrap_0
energy_term_2
```

for named Hamiltonian terms.

---

## Runtime checks

Runtime checks should live under `checks/...`.

Examples:

```text
checks/data_integrity/passed
checks/gradient/passed
checks/sampler/passed
checks/equivariance/full_model/passed
checks/equivariance/trace/passed
```

Detailed check metrics stay in the same namespace:

```text
checks/data_integrity/local_energy_nonfinite_fraction
checks/gradient/global_grad_norm
checks/sampler/acceptance_rate
checks/equivariance/full_model/max_abs_error
checks/equivariance/trace/n_failed_entries
```

### `checks/equivariance/*/n_comparisons` says how much was actually compared

**New key, added to every `checks/equivariance/<checker_log_name>` record.**
`RuntimeEquivariance` publishes it from a required field on
`EquivarianceCheckResult`, next to `passed` and for the same reason `passed` is
published there: it is a property of the result contract, not of whatever
free-form metrics a particular checker chose to report. Every checker, including
one written outside this repository, must state it.

It counts the value comparisons a checker actually performed — one per
`.compare(...)` call:

```text
FullModelEquivarianceChecker    one per permutation tested
TraceEquivarianceChecker        one per shared trace key per permutation,
                                plus one per permutation when compare_output
```

The key exists because `passed` on its own is not evidence that anything was
checked. Every verdict in this namespace is well defined over an empty
comparison set — no permutation failed, no trace key failed — so a checker that
compared nothing reports `passed = True` with nothing to contradict it. This is
the `n_grad_tensors` role under `checks/gradient`, played by a namespace that
previously had no equivalent.

`n_comparisons` is a **count, not a verdict**. Zero does not fail the check, and
`fail_fast` does not fire on it. There is a legitimate zero: a system with fewer
than two particles admits no non-identity permutation, and comparing nothing
there is correct rather than broken. Reading `passed` together with
`n_comparisons` is what distinguishes the two, which is exactly what a bare
`passed` could not express.

It is **not a second spelling of `n_permutations_tested`**, which both in-tree
checkers already publish. That key counts permutations *selected*; this one
counts comparisons *performed*. They agree for `FullModelEquivarianceChecker`,
which compares once per permutation, and come apart for
`TraceEquivarianceChecker`:

```text
n_permutations_tested = 4, n_trace_entries = 0   ->  n_comparisons = 0
```

which is a model recording no trace at all, passing while measuring nothing —
visible in the new key and in no existing one. `n_permutations_tested` keeps its
current meaning and its current spelling; nothing was renamed or removed.

For readers of historical data: records written before this change have no
`n_comparisons` key, and a `passed = True` in them does not distinguish a check
that compared many values from one that compared none. Where the checker is
`FullModelEquivarianceChecker`, `n_permutations_tested` is the closest available
substitute; for `TraceEquivarianceChecker` the product
`n_permutations_tested * n_trace_entries` reconstructs it when
`compare_output` was false.

### An empty `checkers` list is a construction error, not an empty record

`RuntimeEquivariance` raises `ValueError` when built with no checkers, so this
namespace is never *absent* on a run that configured the callback.

Every record here is emitted from inside the per-checker loop. With an empty
list the loop body never runs, nothing is logged, and the entire
`checks/equivariance/*` namespace silently disappears from the metric stream
while a config carrying `fail_fast: true` reads as though the checks were being
enforced. That is worse than a vacuous pass: a vacuous pass leaves a record whose
counts someone could notice, whereas a vacuous absence leaves nothing to notice.

Rejecting at construction was chosen over emitting a failing record at log time
because there is no log name to hang such a record on — names are derived per
checker, so a no-checker record would have to invent a namespace describing only
a misconfiguration — and because it is the only option that is loud in both
modes: a failing record would crash a `fail_fast: true` run late and would never
fire at all under `fail_fast: false`, which is the silent case being fixed.

---

## Canonical examples

Preferred:

```text
train/loss
train/energy
train/energy_variance
train/energy_stderr
train/local_energy_finite_fraction
train/logabs_mean

train/sampler/acceptance_rate
train/sampler/n_walkers

train/grad_norm
train/param_norm
train/loss_has_grad
train/optimizer_step

train/perf/step_time_sec
train/perf/sampling_time_sec

checks/data_integrity/passed
checks/gradient/global_grad_norm
checks/sampler/acceptance_rate
checks/equivariance/full_model/max_abs_error
checks/equivariance/trace/n_failed_entries

runtime/wall_time_sec

eval/energy
eval/energy_error
eval/perf/wall_time_sec

diagnostics/energy/time_sec
```

Avoid:

```text
train/sampler.acceptance_rate
sampler.acceptance_rate
energy
energy_mean
full_model_equivariance_max_abs_error
checks.equivariance.full_model.max_abs_error
runtime_wall_time_sec
```

---

## Summary

Canonical metric identity:

```text
namespace + key
```

Use:

```text
slash-separated namespace
underscore-separated key
```

CSV stores these separately:

```csv
step,namespace,key,value
1,train,energy,3.65
```

JSONL preserves the same structure:

```json
{"step": 1, "namespace": "train", "metrics": {"energy": 3.65}}
```

W&B joins them:

```text
train/energy
```

Do not maintain multiple primary names for the same metric. If a downstream tool needs flattened names, flatten only at export time.
