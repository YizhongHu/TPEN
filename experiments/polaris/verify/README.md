# Polaris CPU verification harness

This harness stages a target commit on a Polaris login node and submits the
stored PBS job for CPU-only pytest verification. It is intentionally
parameterized: the target SHA, selections, evidence root, and cell identity
are invocation inputs, not values copied into scripts.

The operator runs one command with the facility Python:

```bash
/soft/applications/conda/2025-09-25/mconda3/bin/python \
  experiments/polaris/verify/stage.py \
  --target-sha <full-sha> --arm <pytest-path-or-node-id> \
  --evidence-root /eagle/HetRxnEnergy/rhu/runs --cell-id <unique-cell>
```

Staging performs the full clone, all-ref fetch, AppleDouble scan, facility-uv
cache prime, same-binary offline locked resolve, manifest serialization, and
output-directory checks before `qsub`. The job reconstructs its venv offline
under `/local/scratch/$PBS_JOBID` and writes per-arm logs, JUnit, JSON, and a
`results.json` verdict under the cell evidence directory.

Facility-specific interpreter, uv version, uv environment policy, and the
compute-host guard are versioned once in `facility_profile.json`; staging and
the job both read that profile. The stored PBS script uses bash without a login
profile and receives only the staged run-environment path. The harness checkout
SHA and target checkout SHA are both recorded in `manifest.json`.

The deliberately failing class-9 canary is `fixtures/canary.py`. Its filename
is intentionally outside normal pytest discovery; the end-to-end command may
select it explicitly as an arm. The self-test also proves that a normal repo
collection does not collect it.

Exit status 0 means every selected arm produced parseable JUnit with at least
one testcase and no failures or errors. Nonzero means a staging/preflight
assertion or arm verdict failed. Scheduler state and job exit are not the
scientific verdict by themselves; inspect `results.json` and the JUnit files.

The harness is CPU-only. It does not launch experiments, change Hydra
configuration, edit `uv.lock`, or manage the operator's facility overlay.
