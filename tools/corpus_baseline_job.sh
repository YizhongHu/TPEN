#!/usr/bin/env bash
set -u
set -o pipefail

test "$#" -eq 4 || { echo 'usage: corpus_baseline_job.sh RUN_ROOT SOURCE REVISION HARNESS' >&2; exit 2; }
SCRATCH_ROOT=/scratch
test -d "$SCRATCH_ROOT" || { echo 'node-local scratch is missing' >&2; exit 70; }
MOUNT_FACTS=$(findmnt -no SOURCE,FSTYPE,OPTIONS "$SCRATCH_ROOT") || { echo 'cannot inspect scratch mount' >&2; exit 71; }
case "$MOUNT_FACTS" in
    /dev/*xfs*noquota*|/dev/*noquota*xfs*) ;;
    *) echo "scratch is not a node-local noquota xfs mount: $MOUNT_FACTS" >&2; exit 71;;
esac
SCRATCH_FREE_KIB=$(df -Pk "$SCRATCH_ROOT" | awk 'NR == 2 {print $4}')
case "$SCRATCH_FREE_KIB" in
    ''|*[!0-9]*) echo "scratch free space is not numeric: $SCRATCH_FREE_KIB" >&2; exit 72;;
esac
MIN_SCRATCH_FREE_KIB=20971520
printf 'SCRATCH_MOUNT=%s\nSCRATCH_FREE_KIB=%s\nMIN_SCRATCH_FREE_KIB=%s\n' "$MOUNT_FACTS" "$SCRATCH_FREE_KIB" "$MIN_SCRATCH_FREE_KIB"
POSITIVE_CONTROL_VALUE=0
if test "$POSITIVE_CONTROL_VALUE" -ge "$MIN_SCRATCH_FREE_KIB"; then
    echo 'scratch threshold positive control unexpectedly passed' >&2
    exit 73
fi
printf 'POSITIVE_CONTROL=threshold_can_fail_as_expected VALUE=%s\n' "$POSITIVE_CONTROL_VALUE"
set +e
false | cat >/dev/null
PIPEFAIL_CONTROL_RC=$?
set -e
test "$PIPEFAIL_CONTROL_RC" -ne 0 || { echo 'pipefail control unexpectedly passed' >&2; exit 73; }
printf 'POSITIVE_CONTROL=pipefail_detects_left_failure RC=%s\n' "$PIPEFAIL_CONTROL_RC"
test "$SCRATCH_FREE_KIB" -ge "$MIN_SCRATCH_FREE_KIB" || {
    echo "insufficient node-local scratch: ${SCRATCH_FREE_KIB}KiB < ${MIN_SCRATCH_FREE_KIB}KiB" >&2
    exit 74
}
RUN_ROOT=$1
SCRATCH_REAL=$(realpath "$SCRATCH_ROOT") || { echo 'cannot resolve scratch path' >&2; exit 75; }
run_root_is_under_scratch() {
    local candidate_real
    candidate_real=$(realpath -m "$1") || return 1
    case "$candidate_real" in
        "$SCRATCH_REAL"/*) return 0 ;;
        *) return 1 ;;
    esac
}
if run_root_is_under_scratch "$RUN_ROOT" && ! run_root_is_under_scratch "$HOME/tpen-corpus-home-control"; then
    printf 'SELF_TEST_PASS RUN_ROOT under /scratch; HOME control rejected\n'
else
    printf 'SELF_TEST_FAIL RUN_ROOT physical-path binding\n' >&2
    exit 76
fi
SOURCE=$2
REVISION=$3
HARNESS=$4
test -n "${SLURM_JOB_ID:-}" || { echo 'SLURM_JOB_ID is required' >&2; exit 90; }
UV=${UV_BIN:-$HOME/.local/bin/uv}
test -x "$UV" || { echo "uv not executable: $UV" >&2; exit 91; }
export PATH="$RUN_ROOT/venv-trunk/bin:$PATH"
export UV_CACHE_DIR="$RUN_ROOT/uv-cache-trunk"
test -n "${QUOTA_COMMANDS_JSON:-}" || { echo 'QUOTA_COMMANDS_JSON is required' >&2; exit 92; }
OUTER_LOG="$RUN_ROOT/evidence/outer-uv.log"
mkdir -p "$RUN_ROOT/evidence"
HOME_RECEIPT_DIR="$HOME/tpen-corpus-${SLURM_JOB_ID}"
copy_receipts() {
    local outer_rc=$?
    # An EXIT trap is a preservation path: errexit must never abandon it.
    set +e
    local free_kib
    local preservation_failed=0
    free_kib=$(df -Pk "$HOME" | awk 'NR == 2 {print $4}') || free_kib=UNKNOWN
    if ! mkdir -p "$HOME_RECEIPT_DIR"; then
        echo "cannot preserve receipt directory: $HOME_RECEIPT_DIR" >&2
        exit 98
    fi
    printf 'HOME_FREE_KIB=%s\nRECEIPTS_HOME_DEVIATION=declared\nOUTER_RC=%s\n' "$free_kib" "$outer_rc" >"$HOME_RECEIPT_DIR/receipt-meta.txt"
    COPY_MAX_BYTES=$((10 * 1024 * 1024))
    manifest="$HOME_RECEIPT_DIR/copyback-manifest.txt"
    {
        printf 'COPY_POLICY=keep every regular file at or below %s bytes\n' "$COPY_MAX_BYTES"
        echo 'EXCLUSIONS=top-level natural/reverse checkouts, .git, venv-*, uv-cache-* (including downloaded wheels), and files above threshold'
    } >"$manifest"
    preservation_control_source="$RUN_ROOT/evidence/preservation-control-source"
    printf 'preservation-control\n' >"$preservation_control_source"
    copyback_files() {
        local root=$1
        find "$root" -type d \( \
            -path "$root/natural" -o -path "$root/reverse" -o \
            -name 'venv-*' -o -name 'uv-cache-*' -o \
            -name .git -o -name wheels -o -name wheel \
        \) -prune -o -type f -print0
    }
    copyback_discovery_control() {
        return 73
    }
    record_excluded() {
        printf 'SKIPPED %s size=%s reason=%s\n' "$1" "$2" "$3"
    }
    probe=$(mktemp -d "${TMPDIR:-/tmp}/corpus-copyback.XXXXXX")
    mkdir -p "$probe/venv-natural/lib" "$probe/uv-cache-natural/archive-v0" \
             "$probe/natural" "$probe/natural/tpen/nn" "$probe/evidence/natural" \
             "$probe/sub/venv-natural/lib" "$probe/natural/.git/objects"
    touch "$probe/venv-natural/lib/torch.so" "$probe/uv-cache-natural/archive-v0/libtorch_cpu.so" \
          "$probe/natural/tpen/nn/readout.py" "$probe/sub/venv-natural/lib/foo.so" \
          "$probe/natural/.git/objects/abcdef" "$probe/evidence/natural/RUN_COMPLETE.marker"
    probe_files=$(copyback_files "$probe" | tr '\0' '\n')
    {
        echo 'COPYBACK_EXCLUSION_SELF_TEST=exact find -name prune expression'
        if printf '%s\n' "$probe_files" | grep -Fq "$probe/venv-natural/lib/torch.so"; then
            echo 'SELF_TEST_FAIL top-level venv-natural was reachable'
            self_test_failed=1
        else
            echo 'SELF_TEST_PASS top-level venv-natural excluded'
        fi
        if printf '%s\n' "$probe_files" | grep -Fq "$probe/uv-cache-natural/archive-v0/libtorch_cpu.so"; then
            echo 'SELF_TEST_FAIL top-level uv-cache-natural was reachable'
            self_test_failed=1
        else
            echo 'SELF_TEST_PASS top-level uv-cache-natural excluded'
        fi
        if printf '%s\n' "$probe_files" | grep -Fq "$probe/sub/venv-natural/lib/foo.so"; then
            echo 'SELF_TEST_FAIL nested venv-natural was reachable'
            self_test_failed=1
        else
            echo 'SELF_TEST_PASS nested venv-natural excluded'
        fi
        if printf '%s\n' "$probe_files" | grep -Fq "$probe/natural/tpen/nn/readout.py"; then
            echo 'SELF_TEST_FAIL checkout file was reachable'
            self_test_failed=1
        else
            echo 'SELF_TEST_PASS checkout file excluded'
        fi
        if ! printf '%s\n' "$probe_files" | grep -Fq "$probe/evidence/natural/RUN_COMPLETE.marker"; then
            echo 'SELF_TEST_FAIL evidence marker was excluded'
            self_test_failed=1
        else
            echo 'SELF_TEST_PASS evidence marker copied by default'
        fi
        self_report="$manifest.self-test"
        find "$probe" -type f \( -path "$probe/natural/*" -o -path "$probe/venv-*/*" -o -path "$probe/uv-cache-*/*" \) -print0 |
        while IFS= read -r -d '' excluded; do
            relative=$(printf '%s' "$excluded" | sed "s#^$probe/##")
            printf 'SKIPPED %s size=%s reason=excluded-known-huge-class\n' \
                "$relative" "$(stat -c '%s' "$excluded")" >>"$self_report"
        done
        if ! grep -Fq 'SKIPPED venv-natural/lib/torch.so' "$self_report"; then
            echo 'SELF_TEST_FAIL excluded file reporting was silent'
            self_test_failed=1
        else
            echo 'SELF_TEST_PASS excluded checkout/venv files produce SKIPPED manifest lines'
        fi
        discovery_control_file="$probe/discovery-control"
        discovery_control_rc=0
        copyback_discovery_control >"$discovery_control_file" || discovery_control_rc=$?
        if test "$discovery_control_rc" -eq 0; then
            echo 'SELF_TEST_FAIL discovery failure status was lost'
            self_test_failed=1
        else
            echo "SELF_TEST_PASS discovery failure status preserved rc=$discovery_control_rc"
        fi
        cat "$self_report" >>"$manifest"
        rm -f -- "$self_report"
    } >>"$manifest"
    rm -rf -- "$probe"
    if test "${self_test_failed:-0}" -ne 0; then
        printf 'SELF_TEST_FAIL=copyback exclusion/reporting control\n' >>"$manifest"
        preservation_failed=1
    fi
    forced_label_manifest="$manifest.forced-label-control"
    forced_check_rc=1
    if test "$forced_check_rc" -eq 0; then
        printf 'SELF_TEST_PASS forced-label-check\n' >"$forced_label_manifest"
    else
        printf 'SELF_TEST_FAIL forced-label-check\n' >"$forced_label_manifest"
    fi
    if grep -Fq 'SELF_TEST_FAIL forced-label-check' "$forced_label_manifest" &&
       ! grep -Fq 'SELF_TEST_PASS forced-label-check' "$forced_label_manifest"; then
        printf 'SELF_TEST_PASS label polarity control\n' >>"$manifest"
    else
        printf 'SELF_TEST_FAIL label polarity control\n' >>"$manifest"
        preservation_failed=1
    fi
    cat "$forced_label_manifest" >>"$manifest"
    rm -f -- "$forced_label_manifest"
    discovery_file=$(mktemp "${TMPDIR:-/tmp}/corpus-copyback-discovery.XXXXXX")
    discovery_rc=0
    copyback_files "$RUN_ROOT" >"$discovery_file" || discovery_rc=$?
    if test "$discovery_rc" -ne 0; then
        printf 'COPYBACK_DISCOVERY_RC=%s\nSKIPPED <copyback-discovery> reason=discovery-failed\n' "$discovery_rc" >>"$manifest"
        echo "cannot discover receipts for copy-back: rc=$discovery_rc" >&2
        preservation_failed=1
    else
        printf 'COPYBACK_DISCOVERY_RC=0\n' >>"$manifest"
    fi
    while IFS= read -r -d '' receipt; do
        relative=${receipt#"$RUN_ROOT/"}
        size=$(stat -c '%s' "$receipt") || {
            printf 'SKIPPED %s reason=stat-failed\n' "$relative" >>"$manifest"
            preservation_failed=1
            continue
        }
        record_excluded "$relative" "$size" 'excluded-known-huge-class' >>"$manifest"
    done < <(find "$RUN_ROOT" -type f \( -path "$RUN_ROOT/natural/*" -o -path "$RUN_ROOT/reverse/*" -o -path '*/venv-*/*' -o -path '*/uv-cache-*/*' \) -print0)
    while IFS= read -r -d '' receipt; do
        relative=${receipt#"$RUN_ROOT/"}
        size=$(stat -c '%s' "$receipt") || {
            printf 'SKIPPED %s reason=stat-failed\n' "$relative" >>"$manifest"
            preservation_failed=1
            continue
        }
        if test "$size" -gt "$COPY_MAX_BYTES"; then
            printf 'SKIPPED %s size=%s reason=above-threshold\n' "$relative" "$size" >>"$manifest"
            continue
        fi
        destination="$HOME_RECEIPT_DIR/$relative"
        destination_parent=$(dirname "$destination")
        mkdir -p "$destination_parent" || {
            echo "cannot preserve receipt parent: $destination_parent" >&2
            printf 'SKIPPED %s size=%s reason=parent-create-failed\n' "$relative" "$size" >>"$manifest"
            preservation_failed=1
            continue
        }
        if cp "$receipt" "$destination"; then
            printf 'COPIED %s size=%s\n' "$relative" "$size" >>"$manifest"
        else
            printf 'SKIPPED %s size=%s reason=copy-failed\n' "$relative" "$size" >>"$manifest"
            echo "cannot preserve receipt: $relative" >&2
            preservation_failed=1
        fi
    done <"$discovery_file"
    rm -f -- "$discovery_file"
    control_copy="$HOME_RECEIPT_DIR/preservation-control-copy"
    control_status="$HOME_RECEIPT_DIR/preservation-control-status.txt"
    control_forced_failure=1
    control_copied=0
    if cp "$preservation_control_source" "$control_copy"; then
        control_copied=1
    fi
    if test "$control_forced_failure" -eq 1; then
        printf 'PRESERVATION_STATUS=FAILED\n' >"$control_status"
    fi
    if test "$control_copied" -eq 1 && grep -Fq 'PRESERVATION_STATUS=FAILED' "$control_status"; then
        printf 'SELF_TEST_PASS forced self-test failure preserved evidence before failed status\n' >>"$manifest"
    else
        printf 'SELF_TEST_FAIL forced self-test failure lost evidence\n' >>"$manifest"
        preservation_failed=1
    fi
    if test "$preservation_failed" -eq 0; then
        completion_write_rc=0
        printf 'COPYBACK_COMPLETE=1\n' >>"$manifest" || completion_write_rc=$?
        if test "$completion_write_rc" -ne 0 || ! test -s "$manifest" ||
           ! grep -Fq 'COPYBACK_COMPLETE=1' "$manifest"; then
            preservation_failed=1
        fi
    fi
    if test "$preservation_failed" -ne 0; then
        printf 'COPYBACK_COMPLETE=0\n' >>"$manifest"
    fi
    if test "$preservation_failed" -ne 0; then
        printf 'PRESERVATION_STATUS=FAILED\n' >>"$manifest"
        echo 'required evidence copy-back failed' >&2
        if test "$outer_rc" -eq 0; then
            exit 98
        fi
    else
        printf 'PRESERVATION_STATUS=COMPLETE\n' >>"$manifest"
    fi
    exit "$outer_rc"
}
trap copy_receipts EXIT
printf 'BEGIN outer-uv COMMAND=%q run --no-project python %q --revision %q --run-root %q --uv %q --source %q\n' "$UV" "$HARNESS" "$REVISION" "$RUN_ROOT" "$UV" "$SOURCE" >>"$OUTER_LOG"
printf 'UV_CACHE_DIR=%q\n' "$UV_CACHE_DIR" >>"$OUTER_LOG"
sync
set +e
"$UV" run --no-project python - "$OUTER_LOG" "$QUOTA_COMMANDS_JSON" "$RUN_ROOT" "$UV_CACHE_DIR" "$UV" <<'PY'
import json, subprocess, sys, time
log, spec, *paths = sys.argv[1:]
with open(log, "a", encoding="utf-8") as stream:
    stream.write("BEGIN outer-preflight\n")
    stream.write("OBSERVED_UNIX=" + str(time.time()) + "\n")
    for command in json.loads(spec):
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        stream.write("QUOTA_COMMAND=" + json.dumps(command) + " RC=" + str(result.returncode) + "\n")
        stream.write(result.stdout); stream.write(result.stderr)
    result = subprocess.run(["df", "-P", *paths], text=True, capture_output=True, check=False)
    stream.write("DF_RC=" + str(result.returncode) + "\n" + result.stdout + result.stderr)
    stream.write("END outer-preflight RC=" + str(result.returncode) + "\n")
PY
preflight_rc=$?
set -e
printf 'OUTER_PREFLIGHT_RC=%s\n' "$preflight_rc" >>"$OUTER_LOG"
sync
set +e
"$UV" run --no-project python "$HARNESS" \
  --revision "$REVISION" \
  --run-root "$RUN_ROOT" \
  --uv "$UV" \
  --source "$SOURCE" \
  --quota-spec "$QUOTA_COMMANDS_JSON" >>"$OUTER_LOG" 2>&1
rc=$?
set -e
printf 'END outer-uv RC=%s\n' "$rc" >>"$OUTER_LOG"
sync
set +e
"$UV" run --no-project python - "$OUTER_LOG" "$QUOTA_COMMANDS_JSON" "$RUN_ROOT" "$UV_CACHE_DIR" "$UV" <<'PY'
import json, subprocess, sys, time
log, spec, *paths = sys.argv[1:]
with open(log, "a", encoding="utf-8") as stream:
    stream.write("BEGIN outer-postflight\n")
    stream.write("OBSERVED_UNIX=" + str(time.time()) + "\n")
    for command in json.loads(spec):
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        stream.write("QUOTA_COMMAND=" + json.dumps(command) + " RC=" + str(result.returncode) + "\n")
        stream.write(result.stdout); stream.write(result.stderr)
    result = subprocess.run(["df", "-P", *paths], text=True, capture_output=True, check=False)
    stream.write("DF_RC=" + str(result.returncode) + "\n" + result.stdout + result.stderr)
    stream.write("END outer-postflight RC=" + str(result.returncode) + "\n")
PY
postflight_rc=$?
set -e
printf 'OUTER_POSTFLIGHT_RC=%s\n' "$postflight_rc" >>"$OUTER_LOG"
sync
exit "$rc"
