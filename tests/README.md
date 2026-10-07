# Test Layout

Unit tests live under `tests/unit/` and are grouped by package area or component:

```text
tests/unit/callback/
tests/unit/data/
tests/unit/equivariance/
tests/unit/nn/
tests/unit/physics/
tests/unit/reps/
tests/unit/sampling/
tests/unit/training/
```

Integration tests live under `tests/integration/`, grouped by workflow or domain:

```text
tests/integration/evaluation/
tests/integration/experiments/
tests/integration/training/
```

Do not add test modules under a `hooke/` integration directory. Hooke-specific
fixtures can live under `tests/integration/artifacts/hooke/`; executable tests
belong in the workflow directory they exercise.

Test-owned configs and small fixtures that reproduce integration results live
under `tests/integration/artifacts/`:

```text
tests/integration/artifacts/hooke/exact_singlet_eval.yaml
tests/integration/artifacts/hooke/exact_triplet_eval.yaml
tests/integration/artifacts/hooke/pair_train.yaml
tests/integration/artifacts/training/vmc_smoke.yaml
```

Shared pytest-only helpers live under `tests/helpers/`. Generated run outputs go
under `outputs/`.

## Verification receipts

Run cluster verification through `tools/verification_receipt.py run -- <command>`.
The receipt's `baseline_eligible=YES` means in-job provenance, an unselected
pytest invocation, stable HEAD, a clean tracked tree, known counts, and a zero
pytest exit status all passed; otherwise it records why the counts are not a
whole-suite baseline.

No test modules live directly under `tests/` -- only `README.md`, `conftest.py`,
and `__init__.py` may.
