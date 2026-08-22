---
name: backup-experiment-cannon
description: Manage backup, archive, retention, recovery proof, disposition, or lifecycle-gated cleanup for one completed Cannon experiment or run.
---

# Backup one Cannon experiment

Use this packet for one explicitly identified, completed Cannon experiment/run. Task
Orchestrator (TO) is the lifecycle control plane. This skill reads the producer's
`artifact-policy` and `artifact-inventory`; when separate work exists, it also reads
the disposition child's `disposition-contract`. It does not create schemas or gates,
silently attach traits, advance or terminalize a producer, or treat skill availability
as a substitute for TO notes or dependencies.

## Required identity and preflight

Require, before planning any mutation:

1. The producer item ID and, when applicable, the disposition child ID; one child owns
   one independently dispositioned artifact set.
2. Exact source paths, completion evidence, quiescence/no active writer evidence, and
   ownership evidence from the producer inventory. Do not infer completion from names.
3. The current canonical `cluster-access` instructions and current Cannon TO notes.
   Resolve accounts, collection IDs, virtual roots, quotas, partitions, facility
   policy, and destination paths at invocation time; never copy them into this skill.
4. An authenticated provider-native route, destination identity and capacity checks,
   and an absence/non-overwrite check for the exact destination.
5. A dry-run or rendered operation reviewed by the operator, followed immediately by
   one explicit operator approval naming the exact source and destination. Ask again
   if any path, identity, capacity, policy, or evidence changes.

Keep source data and failed or partial evidence intact. Never use login nodes for heavy
I/O. Do not handle credentials, create unattended schedules/daemons/timers, rotate
retention, mirror broad roots, retry automatically, or delete data outside an explicit
verified TO lifecycle disposition.

## Choose a disposition

- **Retain in place** when no independent backup/archive or cleanup is justified; record
  the exact retained paths in the producer summary.
- **Holystore operational backup** when direct recovery is needed and both capacity and
  the mounted destination are verified. Use a bounded provider-native copy of the exact
  roots only; include an exact manifest and checksums, verify destination absence and
  the copied manifest/checksums, and record the provider task/result. Never make a broad
  mirror or overwrite an existing completed backup.
- **HPSS durable archive** when retention or recovery warrants an append-only archive.
  From the workstation, use the authenticated local Globus CLI with the exact verified
  virtual roots and collection IDs resolved from current notes. Render and inspect the
  transfer first, submit only after the explicit approval, and record terminal task
  status plus checksum evidence. Perform and record restore proof into a new staging
  destination when the policy requires recovery proof.
- **Lifecycle-gated cleanup** only after the exact TO disposition authorizes the exact
  paths, required backup/archive is complete, and required recovery verification passes.
  Preserve source and failure evidence unless that same explicit disposition names it.

The existing `archive-experiment` helper may be used as a convenience, but this skill
must remain fully operable from TO notes and provider-native CLIs if it is absent.

## Provider procedures

For Holystore, resolve the mounted destination and provider command from current
canonical notes. Inspect only the exact source roots and destination roots; render a
bounded copy, confirm no broad-root expansion and no deletion/overwrite flags, obtain
the immediate operator approval, then copy once. Verify exact file counts or manifest,
checksums, destination identity, and capacity, retaining all command output and partial
failure evidence.

For HPSS, confirm local authenticated identity and collection access, then inspect only
the exact virtual source and destination parents. Use a rendered, checksum-verifying,
append-only Globus transfer with exact paths and current collection IDs. After approval,
submit once from the workstation, capture the task ID and terminal status, and verify
the selected manifest/checksums. A recovery drill must restore to a new staging root and
validate the manifest/checksums and a representative result or checkpoint when required.

## TO receipts and handoff

During disposition work, write the child `disposition-receipt` with producer/child IDs,
exact source and destination paths, evidence, operator approval, route, commands/settings,
identities resolved from notes, task IDs, timestamps, terminal state, checksums, and any
failure or partial evidence. During review, write `disposition-verification` with the
independent verification and restore proof (or the documented reason it is not required).
Return exact retained/backup/archive/cleanup paths and child IDs for the producer's
`artifact-disposition-summary`. Do not auto-terminalize the child or producer; TO roles,
notes, dependencies, claims, and human approval control those transitions.
