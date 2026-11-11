#!/usr/bin/env bash
# Mechanically capped builder for shared, immutable Cannon uv environments.
# No facility absolute paths here: envs-root and checkout arrive only by CLI.
set -Eeuo pipefail
if ! shopt -s inherit_errexit 2>/dev/null; then
  [[ "${1:-}" == --help || "${1:-}" == -h ]] || { echo "PRECHECK_FAILED inherit_errexit unavailable" >&2; exit 2; }
fi

ENVS_ROOT=""
CHECKOUT=""
PROJECT=""
EXTRA=""
PYTHON_VERSION=""
UV_BIN=""

usage() {
  cat >&2 <<'USAGE'
usage: build-shared-env.sh --envs-root PATH --checkout PATH --project NAME \
         --extra NAME --python X.Y --uv-bin PATH

Builds, or reuses if already complete, one immutable shared uv environment
under --envs-root, named <project>-<uv.lock sha256[:12]>-<extra>-py<X.Y>.

Refuses before any mkdir/uv call if: --envs-root is not an existing absolute
canonical path; --checkout is not a clean git work tree; the committed cap
policy at <checkout>/scripts/cannon/shared-env-policy.json is missing or does
not hold a non-negative integer MAX_SHARED_ENV_DIRS; the target exists
without a .complete manifest (partial, needs disposition); or the root's
one-level directory count (every entry, not only complete ones) is at or
above the committed cap. No CLI flag, environment variable, or default
overrides the cap; raising it requires a reviewed PR to the policy file.

Requires UV_CACHE_DIR set by the caller.
USAGE
}

fail_precheck() { echo "PRECHECK_FAILED $*" >&2; exit 2; }
fail() { echo "REFUSE $*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    --envs-root) ENVS_ROOT="${2:?missing path}"; shift 2 ;;
    --checkout) CHECKOUT="${2:?missing path}"; shift 2 ;;
    --project) PROJECT="${2:?missing name}"; shift 2 ;;
    --extra) EXTRA="${2:?missing name}"; shift 2 ;;
    --python) PYTHON_VERSION="${2:?missing version}"; shift 2 ;;
    --uv-bin) UV_BIN="${2:?missing path}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "PRECHECK_FAILED unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$ENVS_ROOT" && -n "$CHECKOUT" && -n "$PROJECT" && -n "$EXTRA" && -n "$PYTHON_VERSION" && -n "$UV_BIN" ]] ||
  fail_precheck "--envs-root, --checkout, --project, --extra, --python, and --uv-bin are all required"
[[ "$PYTHON_VERSION" =~ ^[0-9]+\.[0-9]+$ ]] || fail_precheck "--python must be X.Y"
[[ -x "$UV_BIN" && "$UV_BIN" == /* ]] || fail_precheck "--uv-bin must be an executable absolute path"
[[ -n "${UV_CACHE_DIR:-}" ]] || fail_precheck "UV_CACHE_DIR must be set by the caller"
command -v realpath >/dev/null 2>&1 || fail_precheck "realpath required for canonicalization"
command -v sha256sum >/dev/null 2>&1 || fail_precheck "sha256sum required"
command -v flock >/dev/null 2>&1 || fail_precheck "flock required"

# No override surface: the cap is read only from the committed policy file
# below. An env var or CLI flag with this name is deliberately never read.
[[ "$ENVS_ROOT" == /* ]] || fail_precheck "--envs-root must be an absolute path"
[[ -d "$ENVS_ROOT" ]] || fail_precheck "--envs-root must exist: $ENVS_ROOT"
CANON_ENVS_ROOT="$(realpath -- "$ENVS_ROOT")"
[[ "$CANON_ENVS_ROOT" == "$ENVS_ROOT" ]] ||
  fail_precheck "--envs-root must be canonical (no symlink/./..): given=$ENVS_ROOT canonical=$CANON_ENVS_ROOT"

[[ -d "$CHECKOUT" ]] || fail_precheck "--checkout must exist: $CHECKOUT"
CANON_CHECKOUT="$(realpath -- "$CHECKOUT")"
git -C "$CANON_CHECKOUT" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
  fail_precheck "--checkout is not a git work tree: $CANON_CHECKOUT"
[[ -z "$(git -C "$CANON_CHECKOUT" status --porcelain)" ]] || fail "checkout is dirty: $CANON_CHECKOUT"
HEAD_SHA="$(git -C "$CANON_CHECKOUT" rev-parse HEAD)"

POLICY_FILE="$CANON_CHECKOUT/scripts/cannon/shared-env-policy.json"
[[ -f "$POLICY_FILE" ]] || fail "policy file missing: $POLICY_FILE"
# Capture the WHOLE value token up to the next JSON delimiter (comma or
# closing brace) so "3.5", "-1", or a quoted string are rejected outright
# rather than having only their leading digits accepted.
CAP_LINE="$(grep -oE '"MAX_SHARED_ENV_DIRS"[[:space:]]*:[[:space:]]*[^,}]*' -- "$POLICY_FILE" | head -n1 || true)"
CAP_VALUE="${CAP_LINE#*:}"
CAP_VALUE="$(printf '%s' "$CAP_VALUE" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
[[ "$CAP_VALUE" =~ ^[0-9]+$ ]] || fail "policy MAX_SHARED_ENV_DIRS missing or not a non-negative integer: $POLICY_FILE"
CAP="$CAP_VALUE"

UV_LOCK="$CANON_CHECKOUT/uv.lock"
[[ -f "$UV_LOCK" ]] || fail_precheck "uv.lock missing in checkout: $UV_LOCK"
LOCK_SHA256="$(sha256sum -- "$UV_LOCK" | cut -d' ' -f1)"
LOCK_HASH12="${LOCK_SHA256:0:12}"
ENV_NAME="${PROJECT}-${LOCK_HASH12}-${EXTRA}-py${PYTHON_VERSION}"
TARGET="$CANON_ENVS_ROOT/$ENV_NAME"

LOCKFILE="$CANON_ENVS_ROOT/.build-shared-env.lock"
exec 200>"$LOCKFILE"
flock -x 200

# Count ALL one-level directory entries under the root under the lock:
# dotglob includes dot-dirs, nullglob drops the pattern if nothing matches,
# and the trailing slash matches directories only (symlinks to directories
# included). "." and ".." are never matched by a glob. No .complete filter
# and no recursion: malformed, partial, and unfinished names all count.
shopt -s nullglob dotglob
ENTRIES=("$CANON_ENVS_ROOT"/*/)
shopt -u nullglob dotglob
COUNT="${#ENTRIES[@]}"

if [[ -f "$TARGET/.complete" ]]; then
  echo "REUSE count=$COUNT path=$TARGET cap=$CAP policy=$POLICY_FILE head=$HEAD_SHA"
  exit 0
fi

if [[ -e "$TARGET" ]]; then
  fail "partial env exists without .complete, needs disposition: count=$COUNT path=$TARGET cap=$CAP policy=$POLICY_FILE head=$HEAD_SHA"
fi

if ((COUNT >= CAP)); then
  fail "cap reached: count=$COUNT path=$TARGET cap=$CAP policy=$POLICY_FILE head=$HEAD_SHA"
fi

mkdir -- "$TARGET"
(cd "$CANON_CHECKOUT" && UV_PROJECT_ENVIRONMENT="$TARGET" "$UV_BIN" sync --locked --no-install-project --extra "$EXTRA" --python "$PYTHON_VERSION")

UV_VERSION="$("$UV_BIN" --version)"
cat > "$TARGET/.complete" <<MANIFEST
head=$HEAD_SHA
uv_lock_sha256=$LOCK_SHA256
uv_version=$UV_VERSION
python_version=$PYTHON_VERSION
built_utc=$(date -u +%FT%TZ)
MANIFEST

echo "BUILT count=$COUNT path=$TARGET cap=$CAP policy=$POLICY_FILE head=$HEAD_SHA"
