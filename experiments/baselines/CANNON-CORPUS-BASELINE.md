# Cannon trunk corpus baseline

`tools/corpus_baseline.py` is a reusable, allocation-only measurement harness.
The source revision is supplied at runtime; the harness does not embed a
baseline count or node set.

Run it from a Cannon `test` allocation with a unique run root, per-arm
`UV_PROJECT_ENVIRONMENT`/`UV_CACHE_DIR`, and a full clone source. The command
must include `--source` pointing to the staged source repository and
`--revision` set to the trunk SHA being measured. The resulting
`corpus-result.json`, evidence directory, XML, logs, identity block, and
command provenance are the measurement record. Keep the raw evidence.

The harness performs collect-only and full pytest instruments without pytest
path arguments, derives its floor from the collected set in the allocation,
checks JUnit `(classname, name)` identities without requiring a `file`
attribute, records independent inner statuses, and runs a deliberate red
control in the same allocation. It asserts the allocation and exact revision,
and explicitly flushes every machine-readable result before returning. It
records, rather than asserts, tag visibility and interpreter/Torch identity.

THE FOUR NODE-ID SETS WERE DERIVED BY TWO PARTIES INDEPENDENTLY OF EACH OTHER
AND INDEPENDENTLY OF THE HARNESS'S OWN CROSS-ARM COMPARISON, WHICH COMPARED
ONLY THE COLLECTED SET. BOTH DERIVATIONS READ THE SAME PRESERVED ARTEFACTS
FROM THE SINGLE JOB 46382195: THIS IS INDEPENDENCE OF COMPUTATION, NOT OF
ACQUISITION; THERE WAS ONE ACQUISITION. A DEFECT IN THE ARTEFACTS THEMSELVES,
INCLUDING WHAT THE PLUGIN RECORDED OR JUNIT EMITTED, WOULD NOT BE DETECTED BY
THIS AGREEMENT; CLOSING THAT LIMITATION WOULD REQUIRE A SECOND INDEPENDENT RUN.

## Ordering limitation

ONE COLLECTION ORDER WAS MEASURED, UNDER TWO SELECTION AND EXECUTION ORDERS.
THE IMPORT-ORDER QUESTION IS OPEN AND UNMEASURED. CONSEQUENCE FOR CONSUMERS:
AN IMPORT-ORDER-DEPENDENT DEFECT IN TRUNK WOULD NOT BE DETECTED BY THIS
BASELINE, AND A LANE REPORTING GREEN AGAINST IT INHERITS THAT BLIND SPOT.

## R12 completeness limitation

FILE-LEVEL NARROWING: EXCLUDED BY OBSERVATION. 244 test files in the preserved
ae20c11 tree, 242 represented in the collected set, the two absent being the
harness's own deliberate-red file and the known collection-skipped module;
experiments present with 51 files, so `--ignore=experiments` is excluded. The
reviewer independently reproduced this from the IMMUTABLE TRACKED-FILE LIST -
243 tracked candidates, all accounted for - and confirmed no suffix-only
`*_test.py` files exist, which a `test_*.py` glob would have missed.

NODE-LEVEL COMPLETENESS: UNESTABLISHED. A `-k` deselecting within files while
retaining at least one node in every one of the 242 files is not excluded by a
file-level census.

NO CLAIM IS MADE ABOUT THE LIKELIHOOD OF THAT RESIDUAL.

## Quota instrumentation limitation

THE PATH-BEARING QUOTA-WRAPPER CORRECTION IS PAPER-ONLY AND UNTESTED: no
allocation re-exercised it. CONSEQUENCE FOR CONSUMERS: THE QUOTA SNAPSHOTS IN
JOB 46382195 CAPTURED A USAGE ERROR RATHER THAN QUOTA DATA - both `quota`
calls omitted the required path argument and returned rc 2 - SO NO QUOTA
HEADROOM FIGURE FROM THAT RUN IS MEANINGFUL. Only its `df` readings carry
information.
