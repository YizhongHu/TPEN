---
name: pr-verification
description: Independently verify the exact head of an implementation-slice PR on a cluster and record the receipt in Task Orchestrator.
---

# PR verification

A verifier that did not write the code tests one exact PR head and reports a verdict. It does not fix the code.

## Before running

- Read the item's `acceptance-contract` and `implementation-receipt`, and the PR diff against its base.
- Take the full head SHA from the PR (`gh pr view <n> --json headRefOid`). Test only that SHA.
- For a documentation-only diff whose acceptance checks are static, record independent static verification and its limits; no cluster job is needed just to fill a receipt. The job steps below apply when runtime checks are required.
- Use [test-execution](../test-execution/SKILL.md) to declare the `verify-change` affected set and its rationale. If failures are likely, start with a smaller `diagnose` stage, then complete affected verification when ready. A qualifying completed run can serve both purposes; do not repeat it just to relabel it.
- Use the `cluster-access` skill. Tests run on FASRC Cannon unless the item says otherwise, in a scheduler job on the `test` partition, never on a login node.

## In the job

- Assert `git rev-parse HEAD` equals the head SHA, and that the interpreter imports `torch` and `tpen`.
- Put the per-job checkout, uv cache and `TMPDIR` on node-local `/scratch/$USER/$SLURM_JOB_ID`, removed by an EXIT trap. Reuse the shared environments named in the Cannon storage contract; never build one per job.
- Request as many cores as the job runs processes: plain `pytest` gets one core.
- Run `pytest -q` on the declared affected set, including relevant consumers. The full suite runs once per merge batch on `dev`; a broader per-PR set needs a change-specific reason. Keep Typeguard enabled by default.
- When a deliberate `fault-check` is justified, require a green clean baseline and run `pytest` only if the edit step exited 0 and left a non-empty intended diff; purge `__pycache__` between arms; revert and assert a clean tree after each arm. Credit only the expected semantic failure. A surviving mutant is a finding; an invalid harness is not a kill.
- Keep track of any files or directories you created or generated on the cluster during the process.

## Receipt

- Write an immutable note `tip-verification-<sha8>`: verdict, full head SHA, facility, job IDs and final states, commands, test counts, each mutant's result, and log locations. Include each stage's purpose, requested versus completed scope, effective Typeguard mode, limitations and next decision. Partial or worker-only evidence cannot establish an independent verification PASS.
- Point the literal `tip-verification` note at the current head and that receipt. Before overwriting an occupied note, copy it to `tip-verification-superseded-<old sha8>`.
- Remove any temporary verification files or directories you created or generated on the cluster during the above verification process, consistent with CLAUDE.md Treating Data with Care (confirm exact-path ownership and quiescence; keep failure evidence, receipts, and any scientific or unrelated outputs).
- A new head needs a new verification, unless its git tree equals the verified head's tree; then record both SHAs and the tree hash.
- Otherwise reuse completed qualifying evidence while its scope, environment/configuration, instrumentation and independent provenance remain applicable. Record a changed condition or unresolved question before repeating or expanding a run.
- Archive the verifier workspace through Paseo once the receipt is written.
