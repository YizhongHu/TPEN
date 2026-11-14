# Stage 1 shared protocol and launch surface

This directory owns the system-neutral Stage 1 manifest and guarded Cannon
launcher. It does not collect trajectory records and does not contain the
Hooke, He, or H2 system configs; those are separate S1 slices.

Every manifest row carries the exact source SHA, the planned config SHA256,
and an external results root. Evaluation rows additionally carry the checkpoint
path and SHA256. The run's own core resolved-config identity is joined by the
collection slice; this protocol layer does not fabricate one. Evaluation
launches bind `trajectory_identity.checkpoint_file` to the same checkpoint path
as `load.path`, so the evaluator records checkpoint provenance. S1-A3 system
eval configs must define a `trajectory_identity` block with `checkpoint_file`
for this override to be accepted by Hydra.

`plan.py` writes a manifest only. `launch.py` renders a script only; submission
is an explicit operator action. The generated script requires Slurm, checks the
delivered partition, uses node-local per-job scratch, and records job identity.
