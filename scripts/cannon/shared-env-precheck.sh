#!/usr/bin/env bash
# Reuse-only precheck for the shared Holystore uv environment verify.sh
# consumes. Never builds, writes, or mutates an environment; only validates
# caller-supplied paths/names and resolves the expected env path.
# No facility absolute paths here: every path arrives by CLI.
set -Eeuo pipefail
if ! shopt -s inherit_errexit 2>/dev/null; then
  [[ "${1:-}" == --help || "${1:-}" == -h ]] || { echo "PRECHECK_FAILED inherit_errexit unavailable" >&2; exit 2; }
fi

fail() { echo "PRECHECK_FAILED $*" >&2; exit 2; }
command -v realpath >/dev/null 2>&1 || fail "realpath required for path containment checks"

usage() {
  cat >&2 <<'USAGE'
usage: shared-env-precheck.sh paths --shared-env-prefix PATH \
         --facility-run-prefix PATH --run-root PATH
       shared-env-precheck.sh resolve --shared-env-prefix PATH \
         --project NAME --extra NAME --python X.Y --lock-sha256 SHA256

"paths": refuses if --shared-env-prefix and --facility-run-prefix are equal
or either contains the other, or if --run-root is not inside
--facility-run-prefix, or if --run-root is inside --shared-env-prefix.

"resolve": computes the expected reuse-only env path
<--shared-env-prefix>/<project>-<lock-sha256[:12]>-<extra>-py<X.Y> (same
naming as build-shared-env.sh) and refuses unless it is a real directory
(not a symlink) whose canonical parent is exactly --shared-env-prefix, has a
.complete manifest recording the exact --lock-sha256, and an executable
bin/python. On success prints the resolved path to stdout.
USAGE
}

canon() { realpath -m -- "$1"; }
canon_exist() { realpath -- "$1"; }

NAME_COMPONENT_RE='^[A-Za-z0-9][A-Za-z0-9._+]*$'

MODE="${1:-}"
[[ -n "$MODE" ]] && shift || true

check_shared_env_prefix() {
  local prefix="$1"
  [[ "$prefix" == /* ]] || fail "--shared-env-prefix must be an absolute path"
  [[ -d "$prefix" ]] || fail "--shared-env-prefix must exist: $prefix"
  local canon_prefix
  canon_prefix="$(canon_exist "$prefix")"
  [[ "$canon_prefix" == "$prefix" ]] ||
    fail "--shared-env-prefix must be canonical: given=$prefix canonical=$canon_prefix"
  printf '%s' "$canon_prefix"
}

case "$MODE" in
  paths)
    SHARED_ENV_PREFIX="" FACILITY_RUN_PREFIX="" RUN_ROOT=""
    while (($#)); do
      case "$1" in
        --shared-env-prefix) SHARED_ENV_PREFIX="${2:?missing path}"; shift 2 ;;
        --facility-run-prefix) FACILITY_RUN_PREFIX="${2:?missing path}"; shift 2 ;;
        --run-root) RUN_ROOT="${2:?missing path}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown argument: $1" ;;
      esac
    done
    [[ -n "$SHARED_ENV_PREFIX" && -n "$FACILITY_RUN_PREFIX" && -n "$RUN_ROOT" ]] ||
      fail "--shared-env-prefix, --facility-run-prefix, and --run-root are all required"
    CANON_SHARED="$(check_shared_env_prefix "$SHARED_ENV_PREFIX")"
    CANON_FACILITY="$(canon "$FACILITY_RUN_PREFIX")"
    CANON_RUN_ROOT="$(canon "$RUN_ROOT")"
    if [[ "$CANON_SHARED" == "$CANON_FACILITY" ||
          "$CANON_SHARED" == "$CANON_FACILITY"/* ||
          "$CANON_FACILITY" == "$CANON_SHARED"/* ]]; then
      fail "--shared-env-prefix and --facility-run-prefix must be disjoint: shared=$CANON_SHARED facility=$CANON_FACILITY"
    fi
    [[ "$CANON_RUN_ROOT" == "$CANON_FACILITY" || "$CANON_RUN_ROOT" == "$CANON_FACILITY"/* ]] ||
      fail "--run-root must stay under --facility-run-prefix: run-root=$CANON_RUN_ROOT facility=$CANON_FACILITY"
    [[ "$CANON_RUN_ROOT" != "$CANON_SHARED" && "$CANON_RUN_ROOT" != "$CANON_SHARED"/* ]] ||
      fail "--run-root must not be under --shared-env-prefix: run-root=$CANON_RUN_ROOT shared=$CANON_SHARED"
    exit 0
    ;;
  resolve)
    SHARED_ENV_PREFIX="" PROJECT="" EXTRA="" PYTHON_VERSION="" LOCK_SHA256=""
    while (($#)); do
      case "$1" in
        --shared-env-prefix) SHARED_ENV_PREFIX="${2:?missing path}"; shift 2 ;;
        --project) PROJECT="${2:?missing name}"; shift 2 ;;
        --extra) EXTRA="${2:?missing name}"; shift 2 ;;
        --python) PYTHON_VERSION="${2:?missing version}"; shift 2 ;;
        --lock-sha256) LOCK_SHA256="${2:?missing sha256}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown argument: $1" ;;
      esac
    done
    [[ -n "$SHARED_ENV_PREFIX" && -n "$PROJECT" && -n "$EXTRA" && -n "$PYTHON_VERSION" && -n "$LOCK_SHA256" ]] ||
      fail "--shared-env-prefix, --project, --extra, --python, and --lock-sha256 are all required"
    # --project/--extra feed into the resolved target directory name; restrict
    # each to a single path component so neither can escape --shared-env-prefix.
    [[ "$PROJECT" =~ $NAME_COMPONENT_RE ]] || fail "--project must be a single path component matching $NAME_COMPONENT_RE"
    [[ "$EXTRA" =~ $NAME_COMPONENT_RE ]] || fail "--extra must be a single path component matching $NAME_COMPONENT_RE"
    [[ "$PYTHON_VERSION" =~ ^[0-9]+\.[0-9]+$ ]] || fail "--python must be X.Y"
    [[ "$LOCK_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail "--lock-sha256 must be 64 lowercase hex characters"
    CANON_SHARED="$(check_shared_env_prefix "$SHARED_ENV_PREFIX")"
    ENV_NAME="${PROJECT}-${LOCK_SHA256:0:12}-${EXTRA}-py${PYTHON_VERSION}"
    TARGET="$CANON_SHARED/$ENV_NAME"
    [[ ! -L "$TARGET" ]] ||
      fail "shared env is a symlink, refusing to reuse through it: $TARGET"
    [[ -d "$TARGET" ]] ||
      fail "shared env does not exist, build it with build-shared-env.sh first: $TARGET"
    TARGET_PARENT="$(canon "$(dirname -- "$TARGET")")"
    [[ "$TARGET_PARENT" == "$CANON_SHARED" ]] ||
      fail "shared env escapes --shared-env-prefix: $TARGET"
    MANIFEST="$TARGET/.complete"
    [[ -f "$MANIFEST" ]] ||
      fail "shared env has no .complete manifest, build it with build-shared-env.sh first: $TARGET"
    MANIFEST_LOCK_LINE="$(grep -m1 '^uv_lock_sha256=' -- "$MANIFEST" || true)"
    [[ "$MANIFEST_LOCK_LINE" =~ ^uv_lock_sha256=[0-9a-f]{64}$ ]] ||
      fail "shared env .complete manifest has no well-formed uv_lock_sha256 field: $MANIFEST"
    [[ "${MANIFEST_LOCK_LINE#uv_lock_sha256=}" == "$LOCK_SHA256" ]] ||
      fail "shared env uv.lock sha256 does not match the clone's: manifest=${MANIFEST_LOCK_LINE#uv_lock_sha256=} clone=$LOCK_SHA256 path=$TARGET"
    [[ -x "$TARGET/bin/python" ]] ||
      fail "shared env has no executable bin/python: $TARGET"
    printf '%s\n' "$TARGET"
    exit 0
    ;;
  -h|--help|"") usage; [[ "$MODE" == -h || "$MODE" == --help ]] && exit 0; exit 2 ;;
  *) fail "unknown mode: $MODE (expected 'paths' or 'resolve')" ;;
esac
