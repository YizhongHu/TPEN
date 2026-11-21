# Focused debugging and test admission

Start with the preserved traceback, JUnit, and receipt. Confirm the source
revision, environment, and scope that actually ran before requesting another
allocation. State what the next selected existing detector would settle, then
stop when that question is settled or record the next uncertainty.

Separate a broken harness or environment from a scientific or code failure.
Use an existing adequate test instead of creating a duplicate. New regression
coverage needs a missing behavioral obligation and an expectation derived
independently of the implementation under test.

Keep distinct oracles when they answer different questions: process,
provenance, RNG, side effects, and consumer behavior can remain useful even
when tests look similar. Fix the owning component, then have an independent
verifier complete the affected set.

There is no arbitrary rerun or mutation quota, and no blanket full-suite or
new-test requirement. TPEN tests run on clusters; environment problems follow
the repository's interactive-stop rule. Preserve evidence and its limits, and
do not allocate again without a stated question and stopping condition.
