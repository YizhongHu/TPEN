#!/usr/bin/env bash
# Cannon test-partition diagnostic runbook; submit only after operator GO.
# The caller supplies the existing edit/restore harness so this script does
# not duplicate the science-mutant source locations.
set -u -o pipefail

: "${MUTANT_APPLY:?set MUTANT_APPLY to the existing gated edit helper}"
: "${MUTANT_RESTORE:?set MUTANT_RESTORE to the existing restore helper}"

SUITE="experiments/atomistic/he-importance"
RESULTS=0

run_mutant() {
    local mutant="$1"
    local edit_rc pytest_rc restore_rc

    # Each helper invocation must create a fresh mutant copy and purge stale
    # bytecode before pytest.  A failed edit is harness-invalid, not RED.
    "$MUTANT_APPLY" "$mutant"
    edit_rc=$?
    if (( edit_rc != 0 )); then
        printf '%s edit_rc=%d result=HARNESS-INVALID\n' "$mutant" "$edit_rc"
        return 0
    fi
    find . -type d -name __pycache__ -prune -exec rm -rf {} +

    timeout 1200 python -m pytest -q -x -p no:cacheprovider \
        -o faulthandler_timeout=300 --durations=15 "$SUITE"
    pytest_rc=$?

    "$MUTANT_RESTORE" "$mutant"
    restore_rc=$?
    if (( restore_rc != 0 )); then
        printf '%s edit_rc=0 pytest_rc=%d restore_rc=%d result=HARNESS-INVALID\n' \
            "$mutant" "$pytest_rc" "$restore_rc"
        RESULTS=1
    elif (( pytest_rc == 124 )); then
        printf '%s edit_rc=0 pytest_rc=124 result=INCONCLUSIVE-TIMEOUT\n' "$mutant"
    elif (( pytest_rc == 0 )); then
        printf '%s edit_rc=0 pytest_rc=0 result=SURVIVED\n' "$mutant"
    else
        printf '%s edit_rc=0 pytest_rc=%d result=RED\n' "$mutant" "$pytest_rc"
    fi
}

# Run M3 alone first so faulthandler_timeout names any hanging test before
# spending the remaining slot on M4-M10.
run_mutant M3
for mutant in M4 M5 M6 M7 M8 M9 M10; do
    run_mutant "$mutant"
done

exit "$RESULTS"
