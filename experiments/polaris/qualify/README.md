# Polaris GPU Stage-Q qualification launcher

This B0 layer qualifies the current real `run_stage_q` route: Q, O-composite,
Adam, seed 810001, 2048 or 4096 walkers, and the fixed 2000-update horizon
with 1000-update checkpoints. It does not alter `tpen/` or the production
He-Importance route.

The GPU environment is built exclusively from the tracked repository lock. The
qualification profile admits the tracked-lock `cu126` extra; staging uses the
pinned facility interpreter from `verify/facility_profile.json` for an online
locked prime and a separate offline locked dry sync. The job helper performs
the offline sync into `/local/scratch/<PBS_JOBID>/tpen-gpu-env`, while the
stored PBS job exports `CUDA_VISIBLE_DEVICES=0` before starting the child
Python process. No operator overlay or facility CUDA library path is used.

The allocation derives the Linux Torch version from `uv.lock`, checks the
job-local interpreter and Torch file, runs a child Torch probe, streams the
real route and qualification logs, and writes ordered checks to
`preflight.json`. `qualification.json` is the scientific verdict of record.
The measured cu126 runtime basis is Polaris PBS job 7797749: Torch
2.12.0+cu126, CUDA 12.6, A100, exact CUDA sum 524800.0, and zero matmul error.

Exit status 0 means the qualification passed; 1 means a route failure; a
qualification exit code is propagated; 2 means an unmeasured decision. Codes
60/61 cover missing or mismatched staging receipts, 64 the route entry, 70--73
and 79 identity checks, 75 package versions, 76 job environment or venv
identity, 77 uv version, 78 lock resolution, 80 offline sync, 81 copied
harness identity, 83 GPU probe, 84 runtime kind, and 85 topology.

The queue-aware PR 554 gate runs before cloning and immediately before qsub.
The stored script receives run-specific values through `RUN_ENV_PATH`; all
child-visible environment values are explicitly exported before Python starts.

Not yet implemented: rungs 1/2/3-at-20/5 need the production-controls slice
for explicit horizon/cadence, bounded stop, resume, and callback-state audit.
