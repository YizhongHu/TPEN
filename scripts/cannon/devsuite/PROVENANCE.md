# Devsuite template provenance

Source path: `/Users/yizhonghu/.paseo/stage1-evidence/devsuite-template/`
Import date: 2026-11-21 (America/New_York)
Last devsuite run using this template: `dev-fullsuite-6cc72747-20261122` (Task Orchestrator item `6f4c431e`)

The four imported files below were copied byte-identically. The source and
committed SHA-256 values must match:

| File | Source SHA-256 | Committed SHA-256 |
| --- | --- | --- |
| `submit_devsuite.sh` | `a36f2d271a515604b6fbd952eeb8c860ae8a9e236c062b66a9e1faca7e4ad48e` | `a36f2d271a515604b6fbd952eeb8c860ae8a9e236c062b66a9e1faca7e4ad48e` |
| `root.sbatch.in` | `82f61e8cdb753dd951b4d9237e8c2d6dda05ca4b8080e6d3fd9f52da5eebe121` | `82f61e8cdb753dd951b4d9237e8c2d6dda05ca4b8080e6d3fd9f52da5eebe121` |
| `executor.sbatch.in` | `c483e484acdd95b333871f07612fccc1989f0169fd504d0eeb77fc5ec4caab66` | `c483e484acdd95b333871f07612fccc1989f0169fd504d0eeb77fc5ec4caab66` |
| `README.md` | `8ee61cde7a6ebc23cbf6e211105526d9917d211c42625865a1b19a2c2d4d1b80` | `8ee61cde7a6ebc23cbf6e211105526d9917d211c42625865a1b19a2c2d4d1b80` |

## Known local paths in imported text

These local-path documentation lines are intentionally retained for the
byte-identical T0 import. PR-B will replace them when it renders the executor
harness from `scripts/cannon/executor-harness/` at the SHA under test:

- `README.md:29`: `HARNESS=/Users/yizhonghu/tpen-manager-receipts/executor-harness-v3-release-rc2-ca0f93b3-20261111T092513Z`
- `README.md:112`: `Template: /Users/yizhonghu/.paseo/stage1-evidence/devsuite-template/`

## PR-B intentional divergence

PR-B intentionally diverges from the imported template bytes. The launcher now
copies the executor harness from the exact-SHA staged tree at
`scripts/cannon/executor-harness/`; it no longer accepts or defaults to a
private absolute harness path. The manager-dispatched 12-hourly invocation
uses the explicit `--scheduled` flag and the `dev-fullsuite-scheduled-` folder
prefix; manual invocations do not carry that marker. The changed files are:

- `scripts/cannon/devsuite/submit_devsuite.sh`
- `scripts/cannon/devsuite/executor.sbatch.in`
- `scripts/cannon/devsuite/README.md`
- `scripts/cannon/devsuite/PROVENANCE.md`
- `scripts/cannon/tests/test_devsuite_renderer.py`
- `scripts/cannon/tests/mutants-devsuite-renderer.json`
