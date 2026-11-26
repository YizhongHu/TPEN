# SOAP pilot rows and offline checker

This package owns the optimizer pilot's additive `stage1-rendered-config/v1`
rows. It does not modify the Stage-1 gateway, PBS wrapper, TPEN core, or any
resume/load API.

`composer.py` starts from the committed Hooke-2e train/eval YAML and writes one
fully resolved configuration per invocation. `rows.py` binds the rendered-byte
SHA, canonical base provenance, singleton parent identity, and explicit eval
checkpoint join. `checks.py` consumes only named evidence and returns
`PASS`, `PRODUCT_FAILURE`, `HARNESS_INVALID`, or `UNMEASURED`; missing evidence
cannot pass.

The protocol inputs are intentionally explicit:

- R2 train profiles: Adam/SOAP/minSR64 at 512 walkers and 100 updates, plus a
  separate unsynchronized minSR64 20-update check.
- Target train profiles: the same three families at 512 walkers and 5,000
  updates with the declared 1,000/5,000 checkpoints.
- Frozen eval: strict `model_only`, 4,096 walkers, 256 draws, and a named
  published SOAP checkpoint-100 parent.
- Adam/SOAP alone carry `OptimizerState`; minSR carries no Adam-style state.
  Resource/timing/output evidence is read from the rendered config and named
  manifests rather than discovered recursively.

The package is exercised by the focused tests under `tests/`. TPEN pytest
execution is cluster-only; local development is limited to syntax/static
checks.
