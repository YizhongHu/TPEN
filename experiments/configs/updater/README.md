# Updater presets

Composable `optimizer` / `trainer.update_method` fragments for TPEN's VMC
training loop. Each file is a complete, self-contained pair of config blocks:
copy the two blocks into an experiment's `train.yaml`, or merge the file over
one, and nothing else needs to change.

They are fragments rather than Hydra config groups because `tpen.run.load_config`
loads one YAML file and applies dotlist overrides; it does not run Hydra's
defaults-list composition. So composition here is a file-level operation, and
that is deliberate: an experiment config stays a single readable file that
records exactly what ran, which is what the production-arm convention in
`experiments/atomistic/he-v1/configs/train.yaml` depends on.

| Preset | Update rule | When |
|---|---|---|
| `legacy_adam.yaml` | Adam on the score-function objective | The default. Unchanged historical behaviour |
| `sr_dense.yaml` | Stochastic reconfiguration, parameter-space `P x P` solve | Fewer parameters than samples |
| `minsr.yaml` | Stochastic reconfiguration, sample-space `B x B` solve | Fewer samples than parameters, the ordinary VMC regime |

`sr_dense.yaml` and `minsr.yaml` compute the **same update**; they differ only
in which matrix is factorized. Setting `solve_space: auto` picks the smaller
one automatically, and is the right choice unless you are specifically
pinning a route for a comparison.

## Every value resolves to an explicit object

There is no registry, no factory table, and no name-keyed dispatch: each block
names the exact class it wants with `_target_`, so a typo is an import error at
construction rather than a silent fallback to some default. `tests/unit/experiments/test_updater_presets.py`
instantiates all three and asserts the concrete types.

## SR requires plain SGD, and will refuse anything else

The SR presets pair `torch.optim.SGD` with the update method, and the method
rejects momentum, dampening, weight decay, Nesterov, `maximize`, and any
optimizer that is not SGD. That is not a limitation to work around: it is what
makes the applied step provably `-lr * preconditioned_direction`. Pointing an
SR preset at Adam raises at construction. `learning_rate` is spelled in both
blocks and the two must agree; a disagreement also raises.
