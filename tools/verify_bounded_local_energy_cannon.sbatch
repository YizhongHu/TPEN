#!/bin/bash
# Cannon CPU verification harness for bounded local-energy exact-head checks.
# The caller supplies SOURCE_SHA by editing this pinned receipt harness.
set -euo pipefail

SOURCE_SHA="${SOURCE_SHA:?set the exact tested commit SHA}"
REPO_URL="${REPO_URL:-https://github.com/YizhongHu/TPEN.git}"
UV_BIN="${HOME}/.local/bin/uv"
RUN_ROOT="${RUN_ROOT:?set a unique Netscratch run root}"
mkdir -p "${RUN_ROOT}/logs"
trap 'rc=$?; printf "%s\n" "$rc" >"$RUN_ROOT/logs/inner-exit.rc"' EXIT
exec > >(tee "${RUN_ROOT}/logs/job.log") 2>&1

record_rc() {
    local path="$1"
    local rc="$2"
    printf '%s\n' "$rc" >"$path"
}

echo "JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "HOST=$(hostname)"
echo "PARTITION=${SLURM_JOB_PARTITION:-unknown}"
echo "SOURCE_SHA=$SOURCE_SHA"
echo "RUN_ROOT=$RUN_ROOT"
echo "UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

clone_arm() {
    local arm="$1"
    local arm_root="$RUN_ROOT/$arm"
    local checkout="$arm_root/checkout"
    local env="$arm_root/venv"
    local cache="$arm_root/uv-cache"
    local logs="$arm_root/logs"
    mkdir -p "$logs"
    exec > >(tee "$logs/job.log") 2>&1

    set +e
    git clone --no-single-branch "$REPO_URL" "$checkout"
    local rc=$?
    set -e
    record_rc "$logs/clone.rc" "$rc"
    test "$rc" -eq 0

    cd "$checkout"
    set +e
    git fetch --tags origin
    rc=$?
    set -e
    record_rc "$logs/fetch-tags.rc" "$rc"
    test "$rc" -eq 0
    set +e
    git checkout --detach "$SOURCE_SHA"
    rc=$?
    set -e
    record_rc "$logs/checkout.rc" "$rc"
    test "$rc" -eq 0
    set +e
    git rev-parse HEAD >"$logs/head.txt"
    rc=$?
    set -e
    record_rc "$logs/rev-parse.rc" "$rc"
    test "$rc" -eq 0
    test "$(cat "$logs/head.txt")" = "$SOURCE_SHA"
    printf '%s\n' "$(git rev-parse --is-shallow-repository)" >"$logs/shallow.txt"
    test "$(cat "$logs/shallow.txt")" = false
    printf '%s\n' "$(git tag | wc -l)" >"$logs/tag-count.txt"
    test "$(cat "$logs/tag-count.txt")" -gt 0
    printf '%s\n' "$SOURCE_SHA" >"$logs/source.sha"

    export UV_PROJECT_ENVIRONMENT="$env"
    export UV_CACHE_DIR="$cache"
    set +e
    "$UV_BIN" lock --check
    rc=$?
    set -e
    record_rc "$logs/uv-lock-check.rc" "$rc"
    test "$rc" -eq 0

    set +e
    "$UV_BIN" sync --locked --extra cpu
    rc=$?
    set -e
    record_rc "$logs/uv-sync.rc" "$rc"
    test "$rc" -eq 0

    set +e
    "$env/bin/python" -c 'import sys, torch; print("PYTHON", sys.executable); print("VERSION", sys.version); print("TORCH", torch.__version__); print("CUDA", torch.cuda.is_available()); assert torch.__version__ == "2.12.0+cpu"; assert not torch.cuda.is_available()'
    rc=$?
    set -e
    record_rc "$logs/torch-identity.rc" "$rc"
    test "$rc" -eq 0

    export PATH="$env/bin:$PATH"
    export PYTHONPYCACHEPREFIX="$arm_root/pycache"
    mkdir -p "$PYTHONPYCACHEPREFIX"
    set +e
    "$env/bin/python" -m pytest -q \
        tests/unit/physics/test_local_energy.py \
        tests/unit/training/test_vmc_detach_equivalence.py \
        --junitxml="$logs/unit.xml"
    rc=$?
    set -e
    record_rc "$logs/pytest.rc" "$rc"
    echo "ARM=$arm PYTEST_RC=$rc UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    test "$rc" -eq 0
}

clone_arm tip
printf '%s\n' 0 >"$RUN_ROOT/logs/inner-exit.rc"
