#!/bin/bash
#SBATCH --job-name=tpen-r5-review-cbf5fcd4
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Round-5 review arms for PR 524 at head cbf5fcd4. set -u and pipefail but NO
# -e: deliberately red mutant arms must not kill the job. Fatal conditions use
# explicit exit 1.
set -uo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "FATAL: not inside a Slurm allocation; refusing to run tests" >&2
    exit 1
fi

RUN_ROOT="${RUN_ROOT:?RUN_ROOT must be exported by the submit wrapper}"
EXPECTED_HEAD="${EXPECTED_HEAD:?EXPECTED_HEAD must be exported by the submit wrapper}"
REVIEWED_SHA="cbf5fcd48a414c89690a506160df874d56f6a658"
PRISTINE_TRAIN="a13eb986c27ddb140692018773fbf9c441c86fb30f6f818bf306ff5d6ddf48f7"
PRISTINE_TRAINER="6e6723820a533d5092409df12aef1987f4f1948d090b01accd6e62c39a0e92c5"

CO="$RUN_ROOT/checkout"
export UV_PROJECT_ENVIRONMENT="$RUN_ROOT/uv-env"
export UV_CACHE_DIR="$RUN_ROOT/uv-cache"
export TMPDIR="$RUN_ROOT/tmp"
mkdir -p "$TMPDIR" "$RUN_ROOT/junit" "$RUN_ROOT/logs"
UV="$HOME/.local/bin/uv"

echo "JOB=$SLURM_JOB_ID HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-?} DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$CO" || exit 1

# --- identity block -------------------------------------------------------
HEAD="$(git rev-parse HEAD)"
echo "HEAD=$HEAD"
if [ "$HEAD" != "$EXPECTED_HEAD" ]; then
    echo "FATAL: HEAD $HEAD != expected $EXPECTED_HEAD" >&2
    exit 1
fi
PORCELAIN_LINES=$(git status --porcelain --untracked-files=no | wc -l)
echo "PORCELAIN_LINES=$PORCELAIN_LINES"
if [ "$PORCELAIN_LINES" -ne 0 ]; then
    echo "FATAL: tracked tree not clean before arms" >&2
    exit 1
fi
echo "DIFF vs reviewed head $REVIEWED_SHA (must touch no tpen/ file):"
git diff --name-only "$REVIEWED_SHA" HEAD
if git diff --name-only "$REVIEWED_SHA" HEAD | grep -q '^tpen/'; then
    echo "FATAL: production tree differs from the reviewed head" >&2
    exit 1
fi
TAG_COUNT=$(git tag | wc -l)
IS_SHALLOW=$(git rev-parse --is-shallow-repository)
echo "TAG_COUNT=$TAG_COUNT IS_SHALLOW=$IS_SHALLOW"

for f in tpen/runner/train.py tpen/training/trainer.py; do
    sha=$(sha256sum "$f" | cut -d' ' -f1)
    echo "PRISTINE $f sha256=$sha"
done
if [ "$(sha256sum tpen/runner/train.py | cut -d' ' -f1)" != "$PRISTINE_TRAIN" ]; then
    echo "FATAL: train.py pristine sha mismatch" >&2; exit 1
fi
if [ "$(sha256sum tpen/training/trainer.py | cut -d' ' -f1)" != "$PRISTINE_TRAINER" ]; then
    echo "FATAL: trainer.py pristine sha mismatch" >&2; exit 1
fi

# Interpreter assertion: uv-selected, >=3.10, non-system. First invocation
# syncs the locked set into the fresh env (prescribed, not an install).
"$UV" run --extra cpu --locked python -c 'import sys; assert sys.version_info >= (3, 10), sys.version; assert not sys.executable.startswith("/usr/bin"), sys.executable; print("INTERPRETER", sys.executable, sys.version.split()[0]); import torch; print("TORCH", torch.__version__)'
if [ $? -ne 0 ]; then
    echo "FATAL: interpreter assertion failed" >&2
    exit 1
fi

FOCUSED="tests/unit/training/test_update_binding.py tests/unit/training/test_update_binding_review_r1.py tests/unit/training/test_update_binding_review_r2.py tests/unit/training/test_update_binding_review_r3.py tests/unit/training/test_update_binding_review_r4.py tests/unit/training/test_update_binding_review_r5.py"

purge_pycache() {
    find "$CO" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
    local n
    n=$(find "$CO" -name '__pycache__' -type d | wc -l)
    echo "PYCACHE_AFTER_PURGE=$n"
    if [ "$n" -ne 0 ]; then
        echo "FATAL: __pycache__ purge did not reach zero" >&2
        exit 1
    fi
}

run_arm() {
    local name="$1"; shift
    local junit="$RUN_ROOT/junit/${name}-${SLURM_JOB_ID}.xml"
    local log="$RUN_ROOT/logs/${name}-${SLURM_JOB_ID}.log"
    purge_pycache
    local t0 t1 rc
    t0=$(date +%s)
    "$UV" run --extra cpu --locked python -m pytest "$@" -q --junitxml="$junit" >"$log" 2>&1
    rc=$?
    t1=$(date +%s)
    echo "ARM $name rc=$rc elapsed=$((t1 - t0))s junit=$junit"
    grep -o '<testsuite [^>]*' "$junit" | head -1 || echo "ARM $name: NO TESTSUITE LINE (truncated or missing junit)"
}

apply_mutant() {
    local name="$1"
    "$UV" run --extra cpu --locked python "$RUN_ROOT/apply_mutant.py" "$name"
    if [ $? -ne 0 ]; then
        echo "FATAL: mutant $name did not apply cleanly" >&2
        exit 1
    fi
}

restore_pristine() {
    git -C "$CO" checkout -- tpen/runner/train.py tpen/training/trainer.py
    local a b p
    a=$(sha256sum "$CO/tpen/runner/train.py" | cut -d' ' -f1)
    b=$(sha256sum "$CO/tpen/training/trainer.py" | cut -d' ' -f1)
    p=$(git -C "$CO" status --porcelain --untracked-files=no | wc -l)
    echo "RESTORED train.py=$a trainer.py=$b porcelain=$p"
    if [ "$a" != "$PRISTINE_TRAIN" ] || [ "$b" != "$PRISTINE_TRAINER" ] || [ "$p" -ne 0 ]; then
        echo "FATAL: restore to pristine failed" >&2
        exit 1
    fi
}

# --- arm 1+2: green -------------------------------------------------------
run_arm g-focused $FOCUSED
run_arm g-full tests/unit

# --- arm 3: M-RC focused + full -------------------------------------------
apply_mutant M-RC
run_arm m-rc-focused $FOCUSED
run_arm m-rc-full tests/unit
restore_pristine

# --- arm 4: M-DB focused, with the r4_2 failure-mechanism extraction ------
apply_mutant M-DB
run_arm m-db-focused $FOCUSED
DB_JUNIT="$RUN_ROOT/junit/m-db-focused-${SLURM_JOB_ID}.xml"
echo "M-DB r4_2 mechanism check (counts in junit):"
echo "  binding-refusal string: $(grep -c 'already bound to a different optimizer' "$DB_JUNIT" || true)"
echo "  equality-assert string: $(grep -c 'diverged from the uninterrupted run' "$DB_JUNIT" || true)"
restore_pristine

# --- arm 5: M-RS focused ---------------------------------------------------
apply_mutant M-RS
run_arm m-rs-focused $FOCUSED
restore_pristine

# --- arm 6: M-A focused ----------------------------------------------------
apply_mutant M-A
run_arm m-a-focused $FOCUSED
restore_pristine

FINAL_PORCELAIN=$(git -C "$CO" status --porcelain --untracked-files=no | wc -l)
echo "FINAL_PORCELAIN=$FINAL_PORCELAIN"
echo "ALL ARMS COMPLETE DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
