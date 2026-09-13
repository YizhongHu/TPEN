#!/usr/bin/env bash
set -u

test "$#" -eq 4 || { echo 'usage: corpus_baseline_job.sh RUN_ROOT SOURCE REVISION HARNESS' >&2; exit 2; }
RUN_ROOT=$1
SOURCE=$2
REVISION=$3
HARNESS=$4
test -n "${SLURM_JOB_ID:-}" || { echo 'SLURM_JOB_ID is required' >&2; exit 90; }
UV=${UV_BIN:-$HOME/.local/bin/uv}
test -x "$UV" || { echo "uv not executable: $UV" >&2; exit 91; }
export PATH="$RUN_ROOT/venv-trunk/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RUN_ROOT/uv-cache-trunk}"
test -n "${QUOTA_COMMANDS_JSON:-}" || { echo 'QUOTA_COMMANDS_JSON is required' >&2; exit 92; }
OUTER_LOG="$RUN_ROOT/evidence/outer-uv.log"
mkdir -p "$RUN_ROOT/evidence"
printf 'BEGIN outer-uv COMMAND=%q run --no-project python %q --revision %q --run-root %q --uv %q --source %q\n' "$UV" "$HARNESS" "$REVISION" "$RUN_ROOT" "$UV" "$SOURCE" | tee -a "$OUTER_LOG"
printf 'UV_CACHE_DIR=%q\n' "$UV_CACHE_DIR" >>"$OUTER_LOG"
set +e
python3 - "$OUTER_LOG" "$QUOTA_COMMANDS_JSON" "$RUN_ROOT" "$UV_CACHE_DIR" "$UV" <<'PY'
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
printf 'OUTER_PREFLIGHT_RC=%s\n' "$preflight_rc" | tee -a "$OUTER_LOG"
set +e
"$UV" run --no-project python "$HARNESS" \
  --revision "$REVISION" \
  --run-root "$RUN_ROOT" \
  --uv "$UV" \
  --source "$SOURCE" \
  --quota-spec "$QUOTA_COMMANDS_JSON" >>"$OUTER_LOG" 2>&1
rc=$?
set -e
printf 'END outer-uv RC=%s\n' "$rc" | tee -a "$OUTER_LOG"
set +e
python3 - "$OUTER_LOG" "$QUOTA_COMMANDS_JSON" "$RUN_ROOT" "$UV_CACHE_DIR" "$UV" <<'PY'
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
printf 'OUTER_POSTFLIGHT_RC=%s\n' "$postflight_rc" | tee -a "$OUTER_LOG"
exit "$rc"
