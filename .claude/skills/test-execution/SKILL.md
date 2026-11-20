---
name: test-execution
description: Choose the purpose, scope, stopping condition, and evidence for TPEN test runs; use when planning tests, diagnosing a failure, or deciding whether verification must be repeated.
---

# Purpose-specific test execution

Before a run, state the question it will answer, the selected tests, and what
result will stop the run or justify a next step. Put this briefly in the existing
TO note or runbook. Reuse that plan for unchanged stages; no extra form is needed.

These are planning and reporting labels, not new runner CLI flags. A scheduler
job may contain several explicit stages when their environment and source state
are compatible. Do not create a job for each label.

## Choose the purpose

| Purpose | Scope and stopping condition |
| --- | --- |
| `diagnose` | Smallest existing case or subset that distinguishes a stated hypothesis. Stop when that question is answered; expand only to resolve the next uncertainty. |
| `verify-change` | Complete declared affected set for the change and its relevant consumers. Stop when that set has completed with interpretable outcomes; an early failure supports diagnosis, not a verification PASS. |
| `fault-check` | Complete declared detector cases, including relevant parameters, for a named plausible defect. Establish a green clean baseline, then evaluate the deliberate fault and restore source state. |
| `batch-regression` | The declared suite for a merge batch on `dev`, including separately collected areas when relevant. Run once per batch under the existing verification policy. |
| `benchmark` | Explicit comparison question, selected workload, configurations and measurement budget. Stop after the planned comparisons; it is not a default verification stage. |

Choose affected tests from the diff, acceptance contract, owning component and
consumer boundaries. A small selection is not automatically sufficient. A broad
change may justify broad PR verification; record why. A bare root `pytest` does
not collect facility tests under `scripts/cannon/tests`.

For documentation-only edits, use appropriate static review. A purpose label
does not create a requirement to run pytest or add tests.

## Diagnose before expanding

Read [focused debugging](references/focused-debugging.md) when investigating a
failure or deciding whether a new regression test earns its cost. Start with
preserved evidence and existing tests. Make one coherent change, check the
reproducer, then complete the affected verification when ready.

Use explicit paths or node IDs with a launcher that supports them. `-x`, `--ff`
or `--lf` can help diagnosis; record their actual effect on completed scope.
An empty or unresolved selection must not silently become the whole suite.
Do not use a fail-fast shortcut to omit declared fault-detector parameters.

Repeat or expand because evidence raises a new question, an assumption changed,
or an unresolved failure needs discrimination. State that reason. A retry to
investigate intermittency needs a hypothesis and a bound; do not retry until green.

## Verification and evidence reuse

Use [pr-verification](../pr-verification/SKILL.md) for the independent PR gate.
The verifier must cover the complete affected set at the current full head,
with the required environment and instrumentation. Worker-only results do not
satisfy that independent gate.

A completed run that started as diagnosis can also provide verification evidence
if all those obligations were met. A green `-x` or `--ff` run need not be repeated
merely because of the flag. Check actual collected and completed cases: `--lf`,
filters, skips or interruption may leave required behavior unverified.

Reuse a qualifying receipt while its source, scope, environment/configuration,
instrumentation and independent provenance remain applicable. A new message is
not an invalidation event. A new head follows the existing verification rule,
including its recorded equal-tree exception. Partial evidence supports only
the behavior it actually checked.

## Deliberate faults and measurements

Choose fault checks for a specific uncertainty about a detector; there is no
automatic mutation quota. Credit a detection only after a green clean baseline,
a successful non-empty intended edit, the expected semantic failure, and clean
restoration. Setup, edit, collection or infrastructure failures are invalid
evidence for that claim, not killed mutants. Unrun arms remain unmeasured.

Read the [runner documentation](../../../scripts/cannon/README.md) before choosing
its invocation. Its current mutation path checks baseline failure after running
mutants; it does not yet enforce an early stop. Its measure mode runs a four-way
comparison, and `--full-suite` appends another run. Do not select those bundles
unless their stages answer the declared questions. A narrower plan needs a
compatible bounded runbook, not padded tests or mutants to fit the interface.

## Record the result and stop

In the existing receipt, record purpose, requested and completed scope, outcomes,
limits and the resulting next decision. Retain the required SHA, command,
environment, effective Typeguard mode, counts, logs/JUnit and available timing.
Use existing outputs; do not launch another collection job just to fill a field.
Separate a product failure from a harness failure or unmeasured behavior. Once
the question and applicable verification obligations are satisfied, stop.

Follow `cluster-access` and current TO facility notes before runtime work. TPEN
tests remain cluster-only. Keep Typeguard enabled by default; any justified
override must state the resulting evidence gap. Environment problems retain
the repository's interactive-stop rule.
