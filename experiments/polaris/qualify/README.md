# Polaris GPU Stage-Q qualification launcher

This B0 layer qualifies only the current real `run_stage_q` route: Q,
O-composite, Adam, seed 810001, 2048 or 4096 walkers, and the fixed 2000-update
horizon with 1000-update checkpoints. It does not alter `tpen/` or the
He-Importance production route and does not install packages or manage the
operator overlay.

The operator copies `facility_profile.json`, fills the overlay and library
paths from an approved read-only facility profile, and runs
`gpu_stage.py` with all cell and scheduler values explicitly supplied. The
manifest is the single source for those values. Staging performs two empty
`qstat -u USER` gates, clones the target SHA, records copied-file hashes, and
submits the stored PBS script with `-V`, one node, the selected queue, walltime,
and `filesystems=home:eagle`.

The allocation receives the library profile in the stored PBS script before
the overlay Python process starts. A child Torch probe then asserts facility
Torch 2.8.0/CUDA 12.9/A100, one visible GPU, and a CUDA op. The overlay tree
is hashed before and after the payload and qualification child. The job also
records ordered per-check evidence in `preflight.json`, including target
HEAD, required ancestors, AppleDouble absence, host/nodefile membership, and
the exact `run_stage_q.py` blob identity. It invokes the exact CLI argv
recorded in `selection-argv.json`; no Hydra override or YAML edit is
possible. `qualification.json` is the atomic verdict of record and is PASS
only when every required predicate passes. Exit 0 means PASS; 1 means a
qualification or real-route failure; 2 means required evidence is incomplete
or unmeasured; 60 means a missing staging receipt; 61 means a mismatched
receipt; 64 means an invalid route entry; 70--73 and 79 are identity checks;
81 means copied harness hashes failed; 82 means the library profile failed;
83 means GPU/runtime or overlay immutability failed; 84 means the runtime kind
failed; 85 means topology failed. The submit shell must inherit the Cray
libfabric path through `-V`; the job stops at 82 if that tail is absent.

Not yet implemented: rungs 1/2/3-at-20/5 need the production-controls slice
for explicit horizon/cadence, bounded stop, resume, and callback-state audit.
