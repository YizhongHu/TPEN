#!/usr/bin/env bash
# Generic Cannon pre-submit submission-contract linter.
# Checks a generated sbatch script and manifest BEFORE `sbatch` runs, so a
# recurring staging/argument mistake fails before an allocation is consumed.
# No facility absolute paths here: allowed roots arrive by CLI/env/manifest.
# bash 3.2-compatible (no associative arrays, no mapfile, no `-v`) so it also
# runs unmodified under the self-test's local macOS bash.
set -Eeuo pipefail

SBATCH_PATH=""
MANIFEST=""
SUBMIT_HELPER=""
SHARED_ENV_ROOT=""
HOME_ROOT="${HOME:-}"
REAL_ENV=""
DRY_ENV=""
DRY_ONLY_KEYS=""
ALLOWED_ROOTS=()

usage() {
  cat >&2 <<'USAGE'
usage: submit-lint.sh --sbatch PATH --manifest PATH --allowed-root PATH
                       [--allowed-root PATH ...] --shared-env-root PATH
                       [--submit-helper PATH] [--home PATH]
                       [--real-env PATH --dry-env PATH --dry-only-keys K1,K2]
Validates the submission contract before sbatch is invoked. Exits 0 on PASS.
USAGE
}

while (($#)); do
  case "$1" in
    --sbatch) SBATCH_PATH="${2:?missing sbatch path}"; shift 2 ;;
    --manifest) MANIFEST="${2:?missing manifest path}"; shift 2 ;;
    --allowed-root) ALLOWED_ROOTS+=("${2:?missing allowed root}"); shift 2 ;;
    --shared-env-root) SHARED_ENV_ROOT="${2:?missing shared env root}"; shift 2 ;;
    --submit-helper) SUBMIT_HELPER="${2:?missing submit helper path}"; shift 2 ;;
    --home) HOME_ROOT="${2:?missing home path}"; shift 2 ;;
    --real-env) REAL_ENV="${2:?missing real-env path}"; shift 2 ;;
    --dry-env) DRY_ENV="${2:?missing dry-env path}"; shift 2 ;;
    --dry-only-keys) DRY_ONLY_KEYS="${2:?missing dry-only keys}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "LINT_FAIL unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$SBATCH_PATH" && -n "$MANIFEST" && -n "$SHARED_ENV_ROOT" ]] || { echo "LINT_FAIL missing required arguments" >&2; usage; exit 2; }
((${#ALLOWED_ROOTS[@]} > 0)) || { echo "LINT_FAIL at least one --allowed-root required" >&2; exit 2; }
[[ -f "$SBATCH_PATH" ]] || { echo "LINT_FAIL sbatch script not found: $SBATCH_PATH" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "LINT_FAIL manifest not found: $MANIFEST" >&2; exit 2; }
command -v realpath >/dev/null 2>&1 || { echo "LINT_FAIL realpath required" >&2; exit 2; }

CRITICAL_KEYS=(HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT DRY_RUN PYTHON)
US=$'\x1f'  # unit separator; macOS/BSD awk's -F does not expand \x1f itself
FAILED=0
fail() { printf 'FAIL: %s\n' "$*"; FAILED=1; }
canon() { realpath -m -- "$1"; }

SBATCH_EXPORTS_FILE="$(mktemp)"
cleanup_tmp() { rm -f -- "$SBATCH_EXPORTS_FILE"; }
trap cleanup_tmp EXIT
grep -E '^[[:space:]]*export[[:space:]]+[A-Z_][A-Z0-9_]*=' "$SBATCH_PATH" > "$SBATCH_EXPORTS_FILE" || true

under_any_root() {
  local candidate="$1" root canon_root
  for root in "${ALLOWED_ROOTS[@]}"; do
    canon_root="$(canon "$root")"
    [[ "$candidate" == "$canon_root" || "$candidate" == "$canon_root"/* ]] && return 0
  done
  return 1
}

# Normalizes KEY=VALUE / `export KEY=VALUE` lines to `KEY<US>VALUE` (comments
# and blanks dropped, surrounding quotes stripped); later lines win on lookup.
parse_kv() {
  local file="$1" line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*#.*$ || -z "${line//[[:space:]]/}" ]] && continue
    key="${line%%=*}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key#export}"
    key="${key//[[:space:]]/}"
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    val="${line#*=}"
    val="$(printf '%s' "$val" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/")"
    printf '%s\x1f%s\n' "$key" "$val"
  done < "$file"
}

# Prints `1<US>value` if key is present (last assignment wins), else `0<US>`.
kv_get() {
  local file="$1" key="$2"
  parse_kv "$file" | awk -F "$US" -v k="$key" -v us="$US" '$1==k{val=$2; found=1} END{if(found) printf "1%s%s", us, val; else printf "0%s", us}'
}

kv_keys() { parse_kv "$1" | awk -F "$US" '{print $1}' | sort -u; }

is_in_csv() {
  local needle="$1" csv="$2" item IFS=','
  for item in $csv; do [[ "$item" == "$needle" ]] && return 0; done
  return 1
}

IFS=$'\x1f' read -r M_HEAD_SHA_HAS M_HEAD_SHA <<< "$(kv_get "$MANIFEST" HEAD_SHA)"
IFS=$'\x1f' read -r M_JOB_NAME_HAS M_JOB_NAME <<< "$(kv_get "$MANIFEST" JOB_NAME)"

# Check 1: HEAD_SHA must be exactly 40 hex characters.
[[ "$M_HEAD_SHA_HAS" == 1 && "$M_HEAD_SHA" =~ ^[0-9a-fA-F]{40}$ ]] || fail "HEAD_SHA is not a 40-hex SHA: '${M_HEAD_SHA:-<missing>}'"

# Check 2: JOB_NAME must contain the short SHA (first 12 chars).
if [[ "$M_HEAD_SHA_HAS" == 1 && "$M_HEAD_SHA" =~ ^[0-9a-fA-F]{40}$ ]]; then
  short_sha="${M_HEAD_SHA:0:12}"
  [[ "$M_JOB_NAME_HAS" == 1 && "$M_JOB_NAME" == *"$short_sha"* ]] || fail "JOB_NAME does not contain SHA[:12] ('$short_sha'): '${M_JOB_NAME:-<missing>}'"
else
  fail "JOB_NAME check skipped: no usable HEAD_SHA"
fi

# Check 3: sbatch path must be absolute, canonical (realpath -m matches the
# raw path, i.e. no unresolved '..'/'.'/symlink), and inside an allowed root.
CANON_SBATCH="$(canon "$SBATCH_PATH")"
if [[ "$SBATCH_PATH" != /* ]]; then
  fail "sbatch path is not absolute: '$SBATCH_PATH'"
elif [[ "$SBATCH_PATH" != "$CANON_SBATCH" ]]; then
  fail "sbatch path is not canonical: '$SBATCH_PATH' -> '$CANON_SBATCH'"
elif ! under_any_root "$CANON_SBATCH"; then
  fail "sbatch path escapes every declared stage root: '$CANON_SBATCH'"
fi

# Check 4: critical exports must match the manifest exactly — no stale,
# missing, or extra critical values between the sbatch script and manifest.
for key in "${CRITICAL_KEYS[@]}"; do
  IFS=$'\x1f' read -r m_has m_val <<< "$(kv_get "$MANIFEST" "$key")"
  IFS=$'\x1f' read -r s_has s_val <<< "$(kv_get "$SBATCH_EXPORTS_FILE" "$key")"
  if [[ "$m_has" == 1 && "$s_has" == 0 ]]; then
    fail "critical export '$key' missing from sbatch script (manifest has it)"
  elif [[ "$m_has" == 0 && "$s_has" == 1 ]]; then
    fail "critical export '$key' present in sbatch script but not in manifest (extra)"
  elif [[ "$m_has" == 1 && "$s_has" == 1 && "$m_val" != "$s_val" ]]; then
    fail "critical export '$key' is stale: manifest='$m_val' sbatch='$s_val'"
  fi
done

# Check 5: TASK_ROOT/RUN_ROOT must not be staged under $HOME, literally or
# after canonicalization (catches a resolved symlink into $HOME too).
if [[ -n "$HOME_ROOT" ]]; then
  CANON_HOME="$(canon "$HOME_ROOT")"
  for key in TASK_ROOT RUN_ROOT; do
    IFS=$'\x1f' read -r has val <<< "$(kv_get "$MANIFEST" "$key")"
    [[ "$has" == 1 ]] || continue
    if [[ "$val" == "$HOME_ROOT" || "$val" == "$HOME_ROOT"/* ]]; then
      fail "$key is staged under literal \$HOME: '$val'"
    else
      canon_val="$(canon "$val")"
      if [[ "$canon_val" == "$CANON_HOME" || "$canon_val" == "$CANON_HOME"/* ]]; then
        fail "$key resolves under \$HOME: '$val' -> '$canon_val'"
      fi
    fi
  done
fi

# Check 6: TASK_ROOT/RUN_ROOT must stay inside a declared allowed root
# (facility-root escape).
for key in TASK_ROOT RUN_ROOT; do
  IFS=$'\x1f' read -r has val <<< "$(kv_get "$MANIFEST" "$key")"
  [[ "$has" == 1 ]] || continue
  canon_val="$(canon "$val")"
  under_any_root "$canon_val" || fail "$key escapes every declared allowed root: '$val' -> '$canon_val'"
done

# Check 7: in a dry run, the interpreter must resolve under the declared
# shared-env root, never a system/ad-hoc python.
IFS=$'\x1f' read -r dry_has dry_run <<< "$(kv_get "$MANIFEST" DRY_RUN)"
IFS=$'\x1f' read -r py_has py_val <<< "$(kv_get "$MANIFEST" PYTHON)"
[[ "$dry_has" == 1 ]] || dry_run=0
if [[ "$dry_run" == 1 && "$py_has" == 1 ]]; then
  CANON_SHARED_ENV_ROOT="$(canon "$SHARED_ENV_ROOT")"
  canon_python="$(canon "$py_val")"
  [[ "$canon_python" == "$CANON_SHARED_ENV_ROOT" || "$canon_python" == "$CANON_SHARED_ENV_ROOT"/* ]] ||
    fail "dry-run PYTHON is not under the shared-env root: '$py_val' -> '$canon_python'"
fi

# Check 8: any ssh invocation in the submission helper must pass
# -o BatchMode=yes (case-insensitive), to avoid a 2FA prompt from a
# non-interactive submit path.
if [[ -n "$SUBMIT_HELPER" ]]; then
  if [[ ! -f "$SUBMIT_HELPER" ]]; then
    fail "submit helper not found: '$SUBMIT_HELPER'"
  else
    while IFS= read -r line; do
      shopt -s nocasematch
      if [[ "$line" != *"batchmode=yes"* ]]; then
        fail "ssh invocation without -o BatchMode=yes: '$line'"
      fi
      shopt -u nocasematch
    done < <(grep -E '(^|[^A-Za-z0-9_])ssh[[:space:]]' "$SUBMIT_HELPER" || true)
  fi
fi

# Check 9: the only env keys allowed to differ between a real and a dry run
# are the documented dry-only keys; anything else is a real-vs-dry drift.
if [[ -n "$REAL_ENV" && -n "$DRY_ENV" ]]; then
  if [[ ! -f "$REAL_ENV" ]]; then
    fail "real-env file not found: '$REAL_ENV'"
  elif [[ ! -f "$DRY_ENV" ]]; then
    fail "dry-env file not found: '$DRY_ENV'"
  else
    ALL_KEYS="$(printf '%s\n' "$(kv_keys "$REAL_ENV")" "$(kv_keys "$DRY_ENV")" | sort -u)"
    while IFS= read -r key; do
      [[ -z "$key" ]] && continue
      is_in_csv "$key" "$DRY_ONLY_KEYS" && continue
      IFS=$'\x1f' read -r rhas rval <<< "$(kv_get "$REAL_ENV" "$key")"
      IFS=$'\x1f' read -r dhas dval <<< "$(kv_get "$DRY_ENV" "$key")"
      [[ "$rhas" == 1 ]] || rval="__unset__"
      [[ "$dhas" == 1 ]] || dval="__unset__"
      [[ "$rval" == "$dval" ]] || fail "real-vs-dry export delta on undocumented key '$key': real='$rval' dry='$dval'"
    done <<< "$ALL_KEYS"
  fi
fi

if [[ "$FAILED" == 0 ]]; then
  echo "LINT_PASS"
  exit 0
fi
echo "LINT_FAIL"
exit 1
