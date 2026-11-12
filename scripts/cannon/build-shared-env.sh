#!/usr/bin/env bash
# Mechanically capped builder for shared, immutable Cannon uv environments.
# No facility absolute paths here: envs-root and checkout arrive only by CLI.
set -Eeuo pipefail
if ! shopt -s inherit_errexit 2>/dev/null; then
  [[ "${1:-}" == --help || "${1:-}" == -h ]] || { echo "PRECHECK_FAILED inherit_errexit unavailable" >&2; exit 2; }
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./shared-env-manifest.sh
source "$SCRIPT_DIR/shared-env-manifest.sh"

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
# --project/--extra feed into the computed target directory name. Restrict
# each to a single path component (no '/', no leading '.' so ".." is
# rejected outright) so neither can make the target escape --envs-root.
# NAME_COMPONENT_RE comes from shared-env-manifest.sh (shared with the
# reuse-only precheck).
[[ "$PROJECT" =~ $NAME_COMPONENT_RE ]] || fail_precheck "--project must be a single path component matching $NAME_COMPONENT_RE"
[[ "$EXTRA" =~ $NAME_COMPONENT_RE ]] || fail_precheck "--extra must be a single path component matching $NAME_COMPONENT_RE"
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
# Whole-document match, not a substring grep: the committed policy must be
# exactly one top-level object holding exactly one key, with a plain
# (non-leading-zero, <=3 digit) integer value. This rejects truncated
# documents, trailing garbage, duplicate/nested keys, quoted numbers,
# booleans, negatives, and floats -- not just their leading characters.
POLICY_CONTENT="$(cat -- "$POLICY_FILE")"
POLICY_RE='^[[:space:]]*\{[[:space:]]*"MAX_SHARED_ENV_DIRS"[[:space:]]*:[[:space:]]*(0|[1-9][0-9]{0,2})[[:space:]]*\}[[:space:]]*$'
[[ "$POLICY_CONTENT" =~ $POLICY_RE ]] || fail "policy MAX_SHARED_ENV_DIRS missing or not a non-negative integer: $POLICY_FILE"
CAP="${BASH_REMATCH[1]}"

UV_LOCK="$CANON_CHECKOUT/uv.lock"
[[ -f "$UV_LOCK" ]] || fail_precheck "uv.lock missing in checkout: $UV_LOCK"
LOCK_SHA256="$(sha256sum -- "$UV_LOCK" | cut -d' ' -f1)"
ENV_NAME="$(compute_shared_env_name "$PROJECT" "$LOCK_SHA256" "$EXTRA" "$PYTHON_VERSION")"
TARGET="$CANON_ENVS_ROOT/$ENV_NAME"
# manifest_is_valid (shared with the reuse-only precheck, see
# shared-env-manifest.sh) trusts a manifest for reuse only if it is exactly
# the six fields this script writes, in order, each strictly formatted, with
# the recorded lock hash matching the CURRENT checkout's uv.lock. The build
# HEAD may differ from the requesting HEAD (dependencies, not history,
# define identity); an empty, truncated (even late, even without a final
# field, even missing only a trailing newline), malformed, or mismatched
# manifest is not trusted.

LOCKFILE="$CANON_ENVS_ROOT/.build-shared-env.lock"
exec 200>"$LOCKFILE"
flock -x 200

# Defense in depth: ENV_NAME can contain no '/' (enforced by
# NAME_COMPONENT_RE above), so TARGET's parent is always CANON_ENVS_ROOT --
# but assert it explicitly under the lock in case that invariant is ever
# weakened.
TARGET_PARENT="$(realpath -m -- "$(dirname -- "$TARGET")")"
[[ "$TARGET_PARENT" == "$CANON_ENVS_ROOT" ]] || fail_precheck "computed target escapes --envs-root: $TARGET"

# Count ALL one-level directory entries under the root under the lock.
# Establish the complete intended glob state first: `set +f` and an explicit
# GLOBIGNORE reset neutralize an inherited SHELLOPTS=noglob/GLOBIGNORE;
# dotglob includes dot-dirs; nullglob drops the pattern if nothing matches;
# failglob (if inherited) is unset so an empty root does not error out; the
# trailing slash matches directories only (symlinks to directories
# included). "." and ".." are never matched by a glob. No .complete filter
# and no recursion: malformed, partial, and unfinished names all count --
# including a directory literally named '*'. Only an actually-unexpanded
# pattern (no real directory matches it) is a census failure.
set +f
unset GLOBIGNORE
shopt -u failglob nocaseglob globasciiranges 2>/dev/null || true
shopt -s nullglob dotglob
ENTRIES=("$CANON_ENVS_ROOT"/*/)
shopt -u nullglob dotglob
if ((${#ENTRIES[@]} == 1)) && [[ "${ENTRIES[0]}" == "$CANON_ENVS_ROOT/*/" ]] && [[ ! -d "$CANON_ENVS_ROOT"/'*' ]]; then
  fail_precheck "directory census failed: pathname expansion is disabled"
fi
COUNT="${#ENTRIES[@]}"

if [[ -L "$TARGET" ]]; then
  fail "target is a symlink, refusing to reuse or create through it: count=$COUNT path=$TARGET cap=$CAP policy=$POLICY_FILE head=$HEAD_SHA"
fi

if [[ -e "$TARGET" ]]; then
  if [[ -f "$TARGET/.complete" ]] && manifest_is_valid "$TARGET/.complete" "$LOCK_SHA256"; then
    echo "REUSE count=$COUNT path=$TARGET cap=$CAP policy=$POLICY_FILE head=$HEAD_SHA"
    exit 0
  fi
  fail "partial or invalid env exists without a valid .complete, needs disposition: count=$COUNT path=$TARGET cap=$CAP policy=$POLICY_FILE head=$HEAD_SHA"
fi

if ((COUNT >= CAP)); then
  fail "cap reached: count=$COUNT path=$TARGET cap=$CAP policy=$POLICY_FILE head=$HEAD_SHA"
fi

mkdir -- "$TARGET"
(cd "$CANON_CHECKOUT" && UV_PROJECT_ENVIRONMENT="$TARGET" "$UV_BIN" sync --locked --no-install-project --extra "$EXTRA" --python "$PYTHON_VERSION")

# Every manifest field is collected into a variable with a checked exit
# status before any write; a failure here must block publication (the dir
# is left partial, counted, and undeleted for disposition -- same as any
# other failed build), not be silently masked by an otherwise-successful
# heredoc write.
UV_VERSION="$("$UV_BIN" --version)" || fail "uv --version query failed, needs disposition: path=$TARGET"
[[ -n "$UV_VERSION" ]] || fail "uv --version produced no output, needs disposition: path=$TARGET"
# Record the interpreter actually built, not just the selector requested.
PY_VERSION_OUTPUT="$("$TARGET/bin/python" --version 2>&1)" || fail "interpreter version query failed, needs disposition: path=$TARGET"
PY_FULL_VERSION="${PY_VERSION_OUTPUT#Python }"
[[ -n "$PY_FULL_VERSION" ]] || fail "interpreter reported no version, needs disposition: path=$TARGET"
BUILT_UTC="$(date -u +%FT%TZ)" || fail "build timestamp query failed, needs disposition: path=$TARGET"
[[ -n "$BUILT_UTC" ]] || fail "build timestamp query produced no output, needs disposition: path=$TARGET"

TMP_MANIFEST="$TARGET/.complete.tmp.$$"
cat > "$TMP_MANIFEST" <<MANIFEST
head=$HEAD_SHA
uv_lock_sha256=$LOCK_SHA256
uv_version=$UV_VERSION
python_version=$PY_FULL_VERSION
python_request=$PYTHON_VERSION
built_utc=$BUILT_UTC
MANIFEST
# Validate the temp file with the same strict function used for reuse,
# before the rename -- a build that produced a malformed manifest must not
# be published, even though uv sync itself succeeded.
manifest_is_valid "$TMP_MANIFEST" "$LOCK_SHA256" || fail "failed to produce a well-formed completion manifest, needs disposition: path=$TARGET"
mv -f -- "$TMP_MANIFEST" "$TARGET/.complete"

echo "BUILT count=$COUNT path=$TARGET cap=$CAP policy=$POLICY_FILE head=$HEAD_SHA"
