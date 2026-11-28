#!/usr/bin/env bash
# One short Polaris debug allocation for selecting an evaluation row.
# Login nodes only submit and inspect this job; selection runs in the allocation.
#PBS -A HetRxnEnergy
#PBS -q debug
#PBS -l select=1:system=polaris:ncpus=4
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:eagle
#PBS -k doe

set -Eeuo pipefail

RESULTS_PREFIX="/eagle/HetRxnEnergy/rhu/"
RESULTS_PREFIX_CANONICAL=""
STAGE1_PATHS_SH="${STAGE1_PATHS_SH:-}"
STAGE1_PATHS_SHA256="${STAGE1_PATHS_SHA256:-}"
SOURCE_SHA="${SOURCE_SHA:-}"
RESULTS_ROOT="${RESULTS_ROOT:-}"
SYSTEM="${SYSTEM:-}"
CHECKOUT_ROOT="${CHECKOUT_ROOT:-}"
PRIME_ENV="${PRIME_ENV:-}"
PHASE2_MANIFEST="${PHASE2_MANIFEST:-}"
SELECTED_EVAL="${SELECTED_EVAL:-}"
PYBIN="${PYBIN:-/soft/applications/conda/2025-09-25/mconda3/bin/python}"

die() {
  printf 'POLARIS_EVAL_PREP FAIL %s\n' "$*" >&2
  exit 2
}

[[ -n "$STAGE1_PATHS_SH" ]] ||
  die "STAGE1_PATHS_SH is required; PBS may spool this script away from its inputs"
[[ "$STAGE1_PATHS_SH" == /* ]] ||
  die "STAGE1_PATHS_SH must be an absolute path: $STAGE1_PATHS_SH"
[[ -f "$STAGE1_PATHS_SH" && ! -L "$STAGE1_PATHS_SH" ]] ||
  die "STAGE1_PATHS_SH is not a regular file: $STAGE1_PATHS_SH"
[[ "$STAGE1_PATHS_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  die "STAGE1_PATHS_SHA256 must be a recorded lowercase SHA-256"
STAGE1_PATHS_ACTUAL_SHA256="$(sha256sum "$STAGE1_PATHS_SH")"
[[ "${STAGE1_PATHS_ACTUAL_SHA256%% *}" == "$STAGE1_PATHS_SHA256" ]] ||
  die "STAGE1_PATHS_SH SHA-256 mismatch: $STAGE1_PATHS_SH"
source "$STAGE1_PATHS_SH"

require_value() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "$name is required"
}

require_value SOURCE_SHA
require_value RESULTS_ROOT
require_value SYSTEM
require_value CHECKOUT_ROOT
require_value PRIME_ENV
require_value PHASE2_MANIFEST
require_value SELECTED_EVAL
[[ -n "${PBS_JOBID:-}" ]] || die "PBS_JOBID is required inside the eval-prep allocation"
[[ -z "${PBS_QUEUE:-}" || "$PBS_QUEUE" == debug ]] ||
  die "delivered queue is not debug: ${PBS_QUEUE}"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || die "SOURCE_SHA must be a full lowercase SHA"
[[ "$SYSTEM" == he || "$SYSTEM" == h2 ]] || die "SYSTEM must be he or h2"
RESULTS_PREFIX_CANONICAL="$(canonicalize_path "$RESULTS_PREFIX")" ||
  die "could not canonicalize results prefix: $RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$RESULTS_ROOT" RESULTS_ROOT "$RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$CHECKOUT_ROOT" CHECKOUT_ROOT "$RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$PRIME_ENV" PRIME_ENV "$RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$PHASE2_MANIFEST" PHASE2_MANIFEST "$RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$SELECTED_EVAL" SELECTED_EVAL "$RESULTS_PREFIX"
[[ -x "$PRIME_ENV/bin/python" ]] || die "prime environment interpreter is missing: $PRIME_ENV/bin/python"
[[ -f "$PHASE2_MANIFEST" ]] || die "phase-2 manifest is missing: $PHASE2_MANIFEST"
[[ "$(git -C "$CHECKOUT_ROOT" rev-parse HEAD)" == "$SOURCE_SHA" ]] ||
  die "checkout does not match SOURCE_SHA"

cd "$CHECKOUT_ROOT"
mkdir -p -- "$(dirname -- "$SELECTED_EVAL")"
"$PRIME_ENV/bin/python" -m experiments.atomistic.stage1.select_eval \
  --manifest "$PHASE2_MANIFEST" --system "$SYSTEM" --lineage 0 --checkpoint 100000 \
  --stream 0 --output "$SELECTED_EVAL"
sha256sum "$SELECTED_EVAL" > "$SELECTED_EVAL.sha256"
printf 'POLARIS_EVAL_PREP complete=PASS SOURCE_SHA=%s SELECTED_EVAL=%s\n' \
  "$SOURCE_SHA" "$SELECTED_EVAL"
