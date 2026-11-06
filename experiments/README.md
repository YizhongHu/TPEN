# TPEN experiments

This tree is deliberately separate from the `tpen/` package. Treat it as if it
were its own repository:

- Study and analysis scripts use the standard library plus small generic dependencies (PyYAML).
  Experiments reach core only through configs (Hydra `_target_` strings resolved
  by the `run.py` entrypoint), `tpen.run.run_from_config`, or the installed
  `tpen` command. The experiments/toolkit shared results reader may import
  core's public read-only inspection interface (`tpen.inspection`); nothing else
  from `tpen` may be imported.
- Tests for experiment code live under `experiments/` next to the code they test, except integration tests that need core internals, such as fixture construction or mutation hooks; those belong in the repo-level `tests/` tree. Experiment tests use core only through configs, `tpen.run.run_from_config`, or the installed `tpen` command.
- Configs under `experiments/` reference `tpen` only through Hydra
  `_target_` strings, which the `run.py` entrypoint resolves.
