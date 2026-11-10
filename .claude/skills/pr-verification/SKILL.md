---
name: pr-verification
description: Independently verify the exact head of an implementation-slice PR on a cluster and record the receipt in Task Orchestrator.
---

# PR verification

A verifier that did not write the code tests one exact PR head and reports a verdict. It does not fix the code.

## Before running

- Read the item's `acceptance-contract` and `implementation-receipt`, and the PR diff against its base.
- Take the full head SHA from the PR (`gh pr view <n> --json headRefOid`). Test only that SHA.
- Use the `cluster-access` skill. Tests run on FASRC Cannon unless the item says otherwise, in a scheduler job on the `test` partition, never on a login node.

## In the job

- Assert `git rev-parse HEAD` equals the head SHA, and that the interpreter imports `torch` and `tpen`.
- Put the per-job checkout, uv cache and `TMPDIR` on node-local `/scratch/$USER/$SLURM_JOB_ID`, removed by an EXIT trap. Reuse the shared environments named in the Cannon storage contract; never build one per job.
- Request as many cores as the job runs processes: plain `pytest` gets one core.
- Run `pytest -q` on the tests that cover the changed code. The full suite runs once per merge batch on `dev`, not per PR.
- Mutation arms: run `pytest` only if the edit step exited 0 and left a non-empty diff; purge `__pycache__` between arms; revert and assert a clean tree after each arm. A surviving mutant is a finding.

## Receipt

- Write an immutable note `tip-verification-<sha8>`: verdict, full head SHA, facility, job IDs and final states, commands, test counts, each mutant's result, and log locations.
- Point the literal `tip-verification` note at the current head and that receipt. Before overwriting an occupied note, copy it to `tip-verification-superseded-<old sha8>`.
- A new head needs a new verification, unless its git tree equals the verified head's tree; then record both SHAs and the tree hash.
- Archive the verifier workspace through Paseo once the receipt is written.
