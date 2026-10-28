# Polaris CPU verification harness

This harness stages a target commit on a Polaris login node and submits the
stored PBS job for CPU-only pytest verification. It is intentionally
parameterized: the target SHA, selections, evidence root, and cell identity
are invocation inputs, not values copied into scripts.

The operator runs one command with the `facility_python` value from the
versioned `facility_profile.json` (do not duplicate that path here):

```bash
<facility-python-from-facility_profile.json> \
  experiments/polaris/verify/stage.py \
  --target-sha <full-sha> --arm <pytest-path-or-node-id> \
  --arm <another-path-or-node-id> --evidence-root <evidence-root> \
  --cell-id <unique-cell> --required-ancestor <full-sha> \
  --remote <git-remote>
```

Run this from a clean git clone of the harness. Before work and again
immediately before `qsub`, stage reads the target queue's live `qstat -Qf`
limits and full queue/job/state records from plain `qstat -f`, cross-checked
against the user's alternate `qstat -u` table. Limits are read
live and never hard-coded (measured on Polaris 2026-10-28). A per-user
`queued_jobs_threshold` counts only `Q`/`H`; a per-user `max_queued` counts
`Q`, `H`, `R`, `E`, `W`, `T`, and `B`. `max_run` and `max_run_res.*` never
refuse because PBS serializes running jobs, and overall (`o:`) limits never
refuse. Any unknown per-user/group/project limit, malformed output, or failed
qstat fails closed. A queue with no bracketed limit is unparseable and is
refused. Full queue names are required; truncated names do not count. The
full qstat text is gzip-compressed with its byte count and SHA-256 recorded;
the alternate table and admission decision are saved at each
gate. If refused, watch the queued job in that queue for `Q` to `R` (the slot
opens at start, not finish), then rerun stage once with a new cell id. Never
loop, use `qdel`, or use `qhold`; never delete evidence. Use a new cell id when
a run root already exists.

Staging performs the full clone, all-ref fetch, AppleDouble scan, facility-uv
cache prime, same-binary offline locked resolve, manifest serialization, and
output-directory checks before `qsub`. The job reconstructs its venv offline
under the profile's scratch root and writes per-arm logs, JUnit, JSON, and a
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
one passing testcase and no failures or errors. Nonzero means a staging/preflight
assertion or arm verdict failed. Scheduler state and job exit are not the
scientific verdict by themselves; inspect `results.json` and the JUnit files.

The preflight exit-code table is:

| Code | Meaning |
| --- | --- |
| 60 | staging receipt missing |
| 61 | staging receipt identity mismatch |
| 70 | login host guard failed |
| 71 | PBS nodefile host mismatch |
| 72 | target checkout SHA mismatch |
| 73 | AppleDouble files present |
| 74 | job venv Python missing |
| 75 | package versions differ from the lock |
| 76 | job venv interpreter identity mismatch |
| 77 | facility uv version mismatch |
| 78 | CPU lock resolution invalid |
| 79 | required ancestor missing |
| 80 | offline sync failed |
| 81 | copied harness identity mismatch |
| 1 | driver arm verdict failed |

The driver writes
`preflight.json`, `driver-preflight.json`, and captured stdout/stderr under the
per-job evidence directory. Watch the captured driver output at
`evidence/<PBS_JOBID>/driver.stdout.log`. PBS stdout/stderr are directed to the staged
`pbs/` directory; the detailed evidence of record is under
`evidence/<PBS_JOBID>/`. Read `preflight.json` for identity checks and
`results.json` plus its JUnit files for per-arm counts and the verdict.

The canary arm requires a target commit that contains this harness, so the
end-to-end target must be the harness branch head rather than plain `dev`.

The harness is CPU-only. It does not launch experiments, change Hydra
configuration, edit `uv.lock`, or manage the operator's facility overlay.
