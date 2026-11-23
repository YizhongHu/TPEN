# Executor harness provenance

Source: `/n/netscratch/kozinsky_lab/Lab/rhu/verify/dev-fullsuite-6899e2fc-20261121T120317Z/executor-harness/`
Fetched: 2026-11-22 (source copy fetched at approximately 01:05Z)

## Imported file SHA-256

The following 12 files are the complete imported set. The table records their
CURRENT post-import SHA-256 hashes; files changed after import are identified
in the Post-import changes section below. Hashes for unchanged imported files
are copied from `REMOTE-SHA256SUMS` for the verified source copy.

```
627caaadecd877f5efe3f36a747e626d1081075257f0d5cc2c45366e9d560e4f  README.md
0b6c02b8e2ed66069a1abcfcc14c46562eb18fcc6c18e620f7cc9967d5fa1f9a  final_run.py
b20cad309068a67a1007f819b74f8d0eca4e38b9896dd7a302b2c4f1daa8e114  final_verdict2.py
f132f7706d919540e197f2a0c33ba33eede2913a1b59bb2a1d2365e8a0bc5d04  job.sbatch
0c2104dd5bc8961f20efdf4fc26664c90983063c5b1cd2dd183696a51a677c33  run_pytest.py
09bb8c7bc1cb8e3109069b7c383616005c06982b554486fd9dad1609bd7e3d81  submission-line.template
fbba9fd47dc4d1b5a541420b69d5cd3290c79277a45804465d5cfbca86a15ce0  helpers/env_audit.py
bf93fca396cfe7db8036a54661d7b75cf2cbd5089c20cfcf12de486b8aa571b0  helpers/negative_storage_checks.py
878324d30ceee5d4ead0447859d6b06461d7af9140489c94891ed255332fd3bb  helpers/path_guard.py
a77793bb816d69be2c08a13a9f8a4fcc55612c6234c05fee15687413788e8c63  helpers/reconcile_counts.py
91f11506dd0d55b18d3b42120b43f0af3f3b39814fddb782265d1cc57bdc49f0  helpers/rehearse_submit_line.py
d9ef0c0ae39fa1f9e151eb2573671ab1ac3790a304dcc68a0bb8c885edcaa903  helpers/validate_inputs.sh
```

## Excluded source files

- `executor.sbatch`: rendered by Stage 1's devsuite template; Stage 1 imports
  it separately.
- `repo.bundle`: per-run input.
- `mutants.json`: per-run input.
- `submission-line.txt`: per-run rendering with a stale reference.
- `final_run.py.pre-hv2env-20261111`: historical backup, recorded by SHA in
  the source provenance.
- `job.sbatch.pre-cpu2-20261110`: historical backup, recorded by SHA in the
  source provenance.
- `job.sbatch.pre-hv2env-20261111`: historical backup, recorded by SHA in the
  source provenance.

## Post-import changes

- `helpers/reconcile_counts.py`: imported the xfail/xpass summary-folding fix from `/Users/yizhonghu/tpen-manager-receipts/execution-lane/b1e1-verify-fbcd332a-20261121T162644Z/helpers/reconcile_counts.py`; this PR preserves that harness behavior.
- `README.md`, `job.sbatch`, and `submission-line.template`: intentionally changed locally under the TPEN root operator rule `operator-rule-job-names-20261122`; the `JOB_NAME` guard was relaxed from SHA-12 to SHA-8 and the README/template were updated to document the operator naming scheme.
