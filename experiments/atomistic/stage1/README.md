# Stage-1 shared atom benchmark scaffold

This directory is the common study layer for 1e-Hooke (`omega=0.5`), He, and
H2 (`R=1.4`). It records one protocol: a 1,000-update/256-walker smoke,
three independent 50,000-update/4,096-walker lineages, and four fresh
4,096-walker by 256-draw model-only evaluation streams per checkpoint.

The manifest is deterministic and contains a complete Hydra config path plus
the dotlist overrides for every row. The collector admits only TPEN's real
`sampled_eval_table.csv` and `.metadata.json` complete draw-by-walker tables.
It reports means, draw-axis correlation-aware MCSE when the blocking floor can
be met, the explicit floor condition otherwise, reference error, variance,
finite checks, and supported task values. Missing task values remain absent;
they are never interpreted as zero. Tolerances are deliberately absent until
capacity measurements exist.

Execution enters core only through the configured `tpen.run.run_from_config`
gateway. The system YAMLs are copied/adapted from the existing He-v1 and H2-v1
contracts, while Hooke's physics is owned by the sibling S1-B slice. A missing
Hooke reference key is therefore an explicit collection error. Launch renders
the guarded sbatch text directly; there is no second static template to drift.

Current limitation: the nucleus-free Hooke trajectory route is explicitly
blocked. TPEN's trajectory writer requires `Walkers.atomic_configuration`,
while `AtomicConfiguration` rejects zero nuclei; the Stage-1 test preserves
this generator/restore evidence and skips only Hooke collection. He and H2
remain runnable through the real train/eval/collect route. This PR does not
change core physics or artifact code.
