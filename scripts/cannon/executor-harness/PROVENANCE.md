# Executor harness provenance

Source: `/n/netscratch/kozinsky_lab/Lab/rhu/verify/dev-fullsuite-6899e2fc-20261121T120317Z/executor-harness/`
Fetched: 2026-11-22 (source copy fetched at approximately 01:05Z)

## Imported file SHA-256

The following 12 files are the complete imported set. Hashes are copied from
`REMOTE-SHA256SUMS` for the verified source copy.

```
d1ef0ef8ed5432fecfd30ce17bdfcf96e9dfeb2cb0b1cf324af176d47415bad6  README.md
0b6c02b8e2ed66069a1abcfcc14c46562eb18fcc6c18e620f7cc9967d5fa1f9a  final_run.py
b20cad309068a67a1007f819b74f8d0eca4e38b9896dd7a302b2c4f1daa8e114  final_verdict2.py
ca2bef917d47a33820c2122b224c44bf785b02949f41d5a39780967675347c81  job.sbatch
0c2104dd5bc8961f20efdf4fc26664c90983063c5b1cd2dd183696a51a677c33  run_pytest.py
f36cdf4ea5717e2f3cd6a5f7cbdb2ba6ecd1f67f5192a57b062e753557b83cf8  submission-line.template
fbba9fd47dc4d1b5a541420b69d5cd3290c79277a45804465d5cfbca86a15ce0  helpers/env_audit.py
bf93fca396cfe7db8036a54661d7b75cf2cbd5089c20cfcf12de486b8aa571b0  helpers/negative_storage_checks.py
878324d30ceee5d4ead0447859d6b06461d7af914048c94891ed255332fd3bb  helpers/path_guard.py
bd86f84c7f00775d047a3c5665834a33f4d2c8de4588ddbbf94d2fef1c61d02b  helpers/reconcile_counts.py
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
