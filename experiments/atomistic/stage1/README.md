# Stage 1 shared protocol and launch surface

This directory owns the system-neutral Stage 1 manifest and guarded Cannon
launcher. It does not collect trajectory records and does not contain the
Hooke, He, or H2 system configs; those are separate S1 slices.

Every manifest row carries the exact source SHA, a real resolved-config SHA256,
and an external results root. Evaluation rows additionally carry the checkpoint
path and SHA256 plus the identity mapping passed by the collection slice to the
public v2 trajectory-record reader.

`plan.py` writes a manifest only. `launch.py` renders a script only; submission
is an explicit operator action. The generated script requires Slurm, checks the
delivered partition, uses node-local per-job scratch, and records job identity.
