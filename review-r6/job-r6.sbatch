#!/bin/bash
#SBATCH --job-name=tpen-r6-review-f2006235
#SBATCH --partition=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r6-20261010T090006Z/logs/job-%j.out
#SBATCH --error=/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r6-20261010T090006Z/logs/job-%j.err

# Deliberately NO `set -e`: red mutant arms must not kill the job.
set -uo pipefail

ROOT=/n/netscratch/kozinsky_lab/Lab/rhu/vmc-if-b-review-r6-20261010T090006Z
HEAD_SHA=f2006235a81df8655fe723fe6f9309b4afd714b2
BASE_SHA=dbb486ea10744810dc95a6d3d288c47efed012fb
PRISTINE_TRAIN=a13eb986c27ddb140692018773fbf9c441c86fb30f6f818bf306ff5d6ddf48f7
PRISTINE_TRAINER=6e6723820a533d5092409df12aef1987f4f1948d090b01accd6e62c39a0e92c5

if [ -z "${SLURM_JOB_ID:-}" ]; then echo "FATAL: not inside a Slurm allocation"; exit 3; fi
echo "DATE_UTC=$(date -u +%FT%TZ) HOST=$(hostname) JOB=$SLURM_JOB_ID PARTITION_DELIVERED=${SLURM_JOB_PARTITION:-unknown}"

export PATH="$HOME/.local/bin:$PATH"
UV="$HOME/.local/bin/uv"
export UV_CACHE_DIR="$ROOT/uv-cache"
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR" "$ROOT/junit" "$ROOT/logs"

CO_HEAD="$ROOT/checkout"
CO_BASE="$ROOT/co-base"
ENV_HEAD="$ROOT/uv-env-head"
ENV_BASE="$ROOT/uv-env-base"
FOCUSED="tests/unit/training/test_update_binding.py tests/unit/training/test_update_binding_review_r1.py tests/unit/training/test_update_binding_review_r2.py tests/unit/training/test_update_binding_review_r3.py tests/unit/training/test_update_binding_review_r4.py"
INTEG="tests/integration/training/test_train_runner.py"

fail=0

assert_head() { # dir sha label
  local got
  got=$(git -C "$1" rev-parse HEAD)
  if [ "$got" != "$2" ]; then echo "FATAL: $3 HEAD=$got != $2"; exit 4; fi
  local porc
  porc=$(git -C "$1" status --porcelain | wc -l)
  echo "$3 HEAD_OK=$got PORCELAIN_LINES=$porc"
  if [ "$porc" != "0" ]; then echo "FATAL: $3 tree not clean"; exit 4; fi
}

assert_head "$CO_HEAD" "$HEAD_SHA" CO_HEAD
assert_head "$CO_BASE" "$BASE_SHA" CO_BASE
echo "IS_SHALLOW=$(git -C "$CO_HEAD" rev-parse --is-shallow-repository) TAG_COUNT=$(git -C "$CO_HEAD" tag | wc -l)"

sha_of() { sha256sum "$1" | cut -d' ' -f1; }
t=$(sha_of "$CO_HEAD/tpen/runner/train.py"); [ "$t" = "$PRISTINE_TRAIN" ] || { echo "FATAL: pristine train.py sha=$t"; exit 6; }
t=$(sha_of "$CO_HEAD/tpen/training/trainer.py"); [ "$t" = "$PRISTINE_TRAINER" ] || { echo "FATAL: pristine trainer.py sha=$t"; exit 6; }
echo "PRISTINE_SHAS_MATCH_ROUNDS_3_TO_5=yes"

echo "=== env sync (head) ==="
( cd "$CO_HEAD" && UV_PROJECT_ENVIRONMENT="$ENV_HEAD" "$UV" sync --extra cpu --locked ) || { echo "FATAL: head env sync failed"; exit 5; }
"$ENV_HEAD/bin/python" -c 'import sys, torch; assert sys.version_info >= (3,10), sys.version; assert not sys.executable.startswith("/usr/bin"), sys.executable; print("INTERP_HEAD", sys.executable, sys.version.split()[0], "torch", torch.__version__)' || { echo "FATAL: head interpreter assert failed"; exit 5; }
echo "=== env sync (base) ==="
( cd "$CO_BASE" && UV_PROJECT_ENVIRONMENT="$ENV_BASE" "$UV" sync --extra cpu --locked ) || { echo "FATAL: base env sync failed"; exit 5; }
"$ENV_BASE/bin/python" -c 'import sys, torch; assert sys.version_info >= (3,10), sys.version; assert not sys.executable.startswith("/usr/bin"), sys.executable; print("INTERP_BASE", sys.executable, sys.version.split()[0], "torch", torch.__version__)' || { echo "FATAL: base interpreter assert failed"; exit 5; }

purge_pycache() { # dir
  find "$1" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  local left
  left=$(find "$1" -name __pycache__ -type d -prune | wc -l)
  echo "PYCACHE_AFTER_PURGE=$left in $1"
}

junit_attrs() { # xmlfile
  "$ENV_HEAD/bin/python" - "$1" <<'PY'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
suite = root if root.tag == "testsuite" else root.find("testsuite")
print("JUNIT tests=%s failures=%s errors=%s skipped=%s" % (
    suite.get("tests"), suite.get("failures"), suite.get("errors"), suite.get("skipped")))
for case in suite.iter("testcase"):
    for node in case:
        if node.tag in ("failure", "error"):
            print("  %s::%s %s: %s" % (case.get("classname"), case.get("name"), node.tag.upper(), (node.get("message") or "")[:160].replace("\n", " ")))
PY
}

run_arm() { # name co env paths...
  local name=$1 co=$2 env=$3; shift 3
  echo "=== ARM $name start $(date -u +%FT%TZ) ==="
  purge_pycache "$co"
  local t0=$SECONDS rc
  ( cd "$co" && UV_PROJECT_ENVIRONMENT="$env" "$UV" run --extra cpu --locked python -m pytest -q "$@" --junitxml="$ROOT/junit/${name}-$SLURM_JOB_ID.xml" ) >"$ROOT/logs/${name}-$SLURM_JOB_ID.log" 2>&1
  rc=$?
  echo "ARM $name RC=$rc ELAPSED=$((SECONDS - t0))s"
  if [ -f "$ROOT/junit/${name}-$SLURM_JOB_ID.xml" ]; then junit_attrs "$ROOT/junit/${name}-$SLURM_JOB_ID.xml"; else echo "ARM $name: NO JUNIT PRODUCED"; fail=1; fi
}

apply_mutant() { # name
  ( cd "$CO_HEAD" && "$ENV_HEAD/bin/python" "$ROOT/apply_mutant_r6.py" "$1" )
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FATAL: mutant $1 apply rc=$rc"; fail=1; return 1; fi
  return 0
}

restore_pristine() {
  ( cd "$CO_HEAD" && git checkout -- tpen/runner/train.py tpen/training/trainer.py )
  local t1 t2 porc
  t1=$(sha_of "$CO_HEAD/tpen/runner/train.py"); t2=$(sha_of "$CO_HEAD/tpen/training/trainer.py")
  porc=$(git -C "$CO_HEAD" status --porcelain | wc -l)
  echo "RESTORE train=$t1 trainer=$t2 PORCELAIN=$porc"
  if [ "$t1" != "$PRISTINE_TRAIN" ] || [ "$t2" != "$PRISTINE_TRAINER" ] || [ "$porc" != "0" ]; then
    echo "FATAL: restore failed"; fail=1; return 1
  fi
  return 0
}

# --- green arms, pristine ---
run_arm g-focused "$CO_HEAD" "$ENV_HEAD" $FOCUSED
run_arm g-full "$CO_HEAD" "$ENV_HEAD" tests/unit
run_arm g-int-head "$CO_HEAD" "$ENV_HEAD" $INTEG
run_arm g-int-base "$CO_BASE" "$ENV_BASE" $INTEG

# --- M-RC: focused, full, integration ---
if apply_mutant M-RC; then
  run_arm m-rc-focused "$CO_HEAD" "$ENV_HEAD" $FOCUSED
  run_arm m-rc-full "$CO_HEAD" "$ENV_HEAD" tests/unit
  run_arm m-rc-int "$CO_HEAD" "$ENV_HEAD" $INTEG
fi
restore_pristine

# --- M-DB: focused ---
if apply_mutant M-DB; then
  run_arm m-db-focused "$CO_HEAD" "$ENV_HEAD" $FOCUSED
fi
restore_pristine

# --- M-A: focused ---
if apply_mutant M-A; then
  run_arm m-a-focused "$CO_HEAD" "$ENV_HEAD" $FOCUSED
fi
restore_pristine

echo "FINAL_PORCELAIN_HEAD=$(git -C "$CO_HEAD" status --porcelain | wc -l)"
echo "FINAL_PORCELAIN_BASE=$(git -C "$CO_BASE" status --porcelain | wc -l)"
echo "JOB_DONE date_utc=$(date -u +%FT%TZ) fail_flag=$fail"
exit 0
