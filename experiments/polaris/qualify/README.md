# Polaris GPU Stage-Q qualification launcher

This B0 layer qualifies only the current real `run_stage_q` route: Q,
O-composite, Adam, seed 810001, 2048 or 4096 walkers, and the fixed 2000-update
horizon with 1000-update checkpoints. It does not alter `tpen/` or the
He-Importance production route and does not install packages or manage the
operator overlay.

The operator copies `facility_profile.json`, fills `overlay_python` and
`overlay_root` from an approved read-only facility overlay, and runs
`gpu_stage.py` with all cell and scheduler values explicitly supplied. The
manifest is the single source for those values. Staging performs two empty
`qstat -u USER` gates, clones the target SHA, records copied-file hashes, and
submits the stored PBS script with `-V`, one node, the selected queue, walltime,
and `filesystems=home:eagle`.

The allocation applies the library profile before importing Torch, sets one
visible GPU, asserts facility Torch 2.8.0/CUDA 12.9/A100, performs a CUDA op,
and verifies the overlay tree is unchanged. It then invokes the exact CLI
argv recorded in `selection-argv.json`; no Hydra override or YAML edit is
possible. `qualification.json` is the atomic verdict of record and is PASS
only when every required predicate passes. Exit 0 means PASS; 1 means the
real route failed; 2 means required evidence is incomplete or unmeasured; 64
means invalid staging input; 82 means the library profile failed; 83 means
GPU/runtime or overlay immutability failed; 84 means the runtime kind or
single-node topology failed.

Not yet implemented: rungs 1/2/3-at-20/5 need the production-controls slice
for explicit horizon/cadence, bounded stop, resume, and callback-state audit.
