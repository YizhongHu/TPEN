---
name: pr-review-loop
description: Run automated PR pre-review for a Task Orchestrator implementation slice before human review.
---

# PR review loop

Review one implementation-slice PR with relevant stack context. **Improve behavior relevant to the slice; do not maximize the number of findings, tests, or rounds.** The orchestrator owns the PR, TO item, and decisions; the reviewer independently assesses the work.

## Purpose of each decision

- **Reviewer — What could go wrong?** Identify plausible defects, regressions, or invalid assumptions, keeping the slice's design intention in mind.
- **Reviewer — What would a test establish?** Test behavior, not implementation details; include useful passing tests.
- **Orchestrator — Does it matter to this slice?** Accept, decline, or defer **each finding and test**, with a reason based on the slice's purpose and cost/benefit. For spikes, prioritize issues that inform the proof of concept or design. A deferral counts as declined *for this slice* and must identify a follow-up; it does not keep the loop open.
- **Termination — Mechanical.** Stop if the reviewer proposes nothing or the orchestrator accepts nothing. Otherwise integrate and verify accepted fixes and tests, then review again. Revisit declined findings only when new evidence changes their relevance.

## Tier gate

Read the required `review-tier`. T0: no agent review. T1: one round. T2: one full round and at most one semantic delta round; any further state-machine/wire-semantics delta requires the recorded exception from amendment-1. Mark ready only when the immutable verification receipt names the current full head; a new commit drafts the PR unless the new head's git tree equals the verified head's tree (record both SHAs and the tree hash).

## Procedure

1. **Initiate.** Once the PR is pushed and initially verified, spawn a Paseo reviewer in a dedicated workspace with the acceptance contract and PR context. Write `reviewer-initialization`.
2. **Investigate.** Reviewer examines the PR and delegates useful test writing/runs to an isolated Paseo worker on a temporary agent-namespaced branch based on the PR tip. The worker writes tests only and records results. **Do not push the temporary test branch to remote or modify the implementation branch.**
3. **Exchange.** Reviewer posts a new GitHub comment with numbered findings, proposed tests (including green ones), evidence, and local test-branch/commit identifiers; then stands down. Orchestrator posts separate decisions for every entry. Do not edit each other's comments.
4. **Act.** Orchestrator directs accepted fixes under the single-writer rule and imports accepted test commits from the test worktree (cherry-pick if Git objects are shared; patches otherwise). Push and verify the updated PR. Preserve the local test workspace until accepted changes and evidence are safely transferred; then the reviewer deletes its temporary branches and archives test workspaces via Paseo.
5. **Repeat or close.** If anything was accepted, finish integration and verification before asking the same reviewer for another round. Otherwise write `review-summary`, mark the PR ready for human review, archive the reviewer via Paseo, write `reviewer-archival`, and report to the manager. After human merge, record `pr-disposition` and complete the TO item. If abandoned, record disposition and cancel the item instead.

Use Paseo's existing workspace operations; check for existing agents/workspaces before spawning duplicates. Use `cluster-access` for cluster tests. GitHub comments are the append-only review record; TO notes hold outcomes and links. Keep the TO item in `review` until final PR disposition.
