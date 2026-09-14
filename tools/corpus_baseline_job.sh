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
    local free_kib
    free_kib=$(df -Pk "$HOME" | awk 'NR == 2 {print $4}') || free_kib=UNKNOWN
    if ! mkdir -p "$HOME_RECEIPT_DIR"; then
        echo "cannot preserve receipt directory: $HOME_RECEIPT_DIR" >&2
        return "$outer_rc"
    fi
    printf 'HOME_FREE_KIB=%s\nRECEIPTS_HOME_DEVIATION=declared\nOUTER_RC=%s\n' "$free_kib" "$outer_rc" >"$HOME_RECEIPT_DIR/receipt-meta.txt"
    COPY_MAX_BYTES=$((10 * 1024 * 1024))
    manifest="$HOME_RECEIPT_DIR/copyback-manifest.txt"
    {
        printf 'COPY_POLICY=keep every regular file at or below %s bytes\n' "$COPY_MAX_BYTES"
        echo 'EXCLUSIONS=checkout, .git, venv-*, uv-cache-* (including downloaded wheels), and files above threshold'
    } >"$manifest"
    record_excluded() {
        printf 'SKIPPED %s size=%s reason=%s\n' "$1" "$2" "$3"
    }
    probe=$(mktemp -d "${TMPDIR:-/tmp}/corpus-copyback.XXXXXX")
    mkdir -p "$probe/venv-natural/lib" "$probe/uv-cache-natural/archive-v0" \
             "$probe/natural/tpen/nn" "$probe/sub/venv-natural/lib" \
             "$probe/natural/checkout/.git/objects"
    touch "$probe/venv-natural/lib/torch.so" "$probe/uv-cache-natural/archive-v0/libtorch_cpu.so" \
          "$probe/natural/tpen/nn/readout.py" "$probe/sub/venv-natural/lib/foo.so" \
          "$probe/natural/checkout/.git/objects/abcdef"
    probe_files=$(find "$probe" -type d \( -name checkout -o -name .git -o -name 'venv-*' -o -name 'uv-cache-*' \) -prune -o -type f -print)
    {
        echo 'COPYBACK_EXCLUSION_SELF_TEST=exact find -name prune expression'
        if printf '%s\n' "$probe_files" | grep -Fq "$probe/venv-natural/lib/torch.so"; then
            echo 'SELF_TEST_FAIL top-level venv-natural was reachable'
            rm -rf -- "$probe"
            return 97
        fi
        echo 'SELF_TEST_PASS top-level venv-natural excluded'
        if printf '%s\n' "$probe_files" | grep -Fq "$probe/uv-cache-natural/archive-v0/libtorch_cpu.so"; then
            echo 'SELF_TEST_FAIL top-level uv-cache-natural was reachable'
            rm -rf -- "$probe"
            return 97
        fi
        echo 'SELF_TEST_PASS top-level uv-cache-natural excluded'
        if printf '%s\n' "$probe_files" | grep -Fq "$probe/sub/venv-natural/lib/foo.so"; then
            echo 'SELF_TEST_FAIL nested venv-natural was reachable'
            rm -rf -- "$probe"
            return 97
        fi
        echo 'SELF_TEST_PASS nested venv-natural excluded'
        if ! printf '%s\n' "$probe_files" | grep -Fq "$probe/natural/tpen/nn/readout.py"; then
            echo 'SELF_TEST_FAIL ordinary source file was excluded'
            rm -rf -- "$probe"
            return 97
        fi
        echo 'SELF_TEST_PASS ordinary source file copied by default'
        report_line=$(record_excluded 'venv-natural/lib/torch.so' 1 'self-test-known-huge-class')
        if ! printf '%s\n' "$report_line" | grep -Fq 'SKIPPED venv-natural/lib/torch.so'; then
            echo 'SELF_TEST_FAIL excluded file reporting was silent'
            rm -rf -- "$probe"
            return 97
        fi
        printf '%s\n' "$report_line"
        echo 'SELF_TEST_PASS top-level venv file produces a SKIPPED manifest line'
    } >>"$manifest"
    rm -rf -- "$probe"
    while IFS= read -r -d '' receipt; do
        relative=${receipt#"$RUN_ROOT/"}
        size=$(stat -c '%s' "$receipt") || {
            printf 'SKIPPED %s reason=stat-failed\n' "$relative" >>"$manifest"
            continue
        }
        record_excluded "$relative" "$size" 'excluded-known-huge-class' >>"$manifest"
    done < <(find "$RUN_ROOT" -type f \( -path '*/checkout/*' -o -path '*/.git/*' -o -path '*/venv-*/*' -o -path '*/uv-cache-*/*' \) -print0)
    while IFS= read -r -d '' receipt; do
        relative=${receipt#"$RUN_ROOT/"}
        size=$(stat -c '%s' "$receipt") || {
            printf 'SKIPPED %s reason=stat-failed\n' "$relative" >>"$manifest"
            continue
        }
        if test "$size" -gt "$COPY_MAX_BYTES"; then
            printf 'SKIPPED %s size=%s reason=above-threshold\n' "$relative" "$size" >>"$manifest"
            continue
        fi
        destination="$HOME_RECEIPT_DIR/$relative"
        mkdir -p "${destination%/*}" || {
            echo "cannot preserve receipt parent: ${destination%/*}" >&2
            continue
        }
        if cp "$receipt" "$destination"; then
            printf 'COPIED %s size=%s\n' "$relative" "$size" >>"$manifest"
        else
            printf 'SKIPPED %s size=%s reason=copy-failed\n' "$relative" "$size" >>"$manifest"
            echo "cannot preserve receipt: $relative" >&2
        fi
    done < <(find "$RUN_ROOT" -type d \( -name checkout -o -name .git -o -name 'venv-*' -o -name 'uv-cache-*' \) -prune -o -type f -print0)
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
