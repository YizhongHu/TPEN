# Stage 1 shared protocol and launch surface

This directory owns the Stage 1 manifest, guarded Cannon launcher, and the
science-default Hooke-1e, He, and H2 train/eval configs.

Every manifest row carries the exact source SHA, the planned config SHA256, an
external results root, and the TPEN run identity (`experiment_name`, `sector`,
and `run_id`). Evaluation rows additionally carry separate absolute
`checkpoint_dir` and `checkpoint_file` fields plus the checkpoint SHA256. The
run's own core resolved-config identity is joined by the collection slice; this
protocol layer does not fabricate one. Evaluation launches bind
`load.path` to the checkpoint step directory and
`trajectory_identity.checkpoint_file` to its `model.pt`, so the evaluator
records checkpoint provenance. S1-A3 system
eval configs must define a `trajectory_identity` block with `checkpoint_file`
for this override to be accepted by Hydra.

Each evaluation config builds one `SystemDescriptor` from its instantiated
`ParticleSpec` and `hamiltonian_terms`, and passes that descriptor plus an
explicit `NaiveLocalEnergyEvaluator` to `TrajectoryMCMCGenerator`. The
descriptor content ID is computed during planning and stored as
`expected_physical_system_id` in every v3 eval row. Collection refuses a
record whose observed ID differs from that value; registry IDs are only a
cross-check, never the source of the physical hash.

Known deviations from the source studies are the shared Stage-1 defaults:
four channels/one layer, CPU device, 256 train walkers, 4096 eval walkers,
and 256 retained draws. Run-slice overrides may select approved science
budgets. The one-electron Hooke route uses the h-v2-style one-electron model
at four channels/one layer and adds the required fixed GaussianConfinement
with alpha=0.25 (omega/2); it has no electron-electron cusp. He retains the
he-v1 trainable electron-electron ranges, curvature cusp (coefficient 0.01,
range 1.0), and trainable Pfaffian readout, while reducing width and runtime
budgets. H2 follows h2-v1's two-electron cusp/readout route, with the same
width and runtime reductions. H2 retains the two-center R=1.4-bohr geometry,
and He retains the two-electron Coulomb Hamiltonian. No Hooke pair config is
included. Train checkpoint callbacks use ExplicitUpdates; launch overrides
bind each callback to exactly the manifest row's planned checkpoint steps.

`plan.py` writes a manifest only. `launch.py` renders a script only; submission
is an explicit operator action. The generated script requires Slurm, checks the
delivered partition, uses node-local per-job scratch, and records job identity.
