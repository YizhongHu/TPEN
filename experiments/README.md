# TPEN experiments

This tree is deliberately separate from the `tpen/` package. Treat it as if it
were its own repository:

- Experiments reach core only through configs (Hydra `_target_` strings resolved
  by the `run.py` entrypoint), `tpen.run.run_from_config`, or the installed
  `tpen` command. The experiments/toolkit shared results reader may import
  core's public read-only inspection interface (`tpen.inspection`); nothing else
  from `tpen` may be imported.
- Tests for experiment code live under `experiments/` (next to the code they
  test), not under the repo-level `tests/` tree. Integration tests that need
  core internals, such as fixture construction or mutation hooks, belong in the
  repo-level `tests/` tree; experiment tests use core only through the
  interfaces above.
- Configs under `experiments/` reference `tpen` only through Hydra
  `_target_` strings, which the `run.py` entrypoint resolves.
