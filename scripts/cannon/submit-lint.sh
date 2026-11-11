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
STAGING_LOCAL_ROOTS=()
STAGING_FINAL_ROOTS=()

usage() {
  cat >&2 <<'USAGE'
usage: submit-lint.sh --sbatch PATH --manifest PATH --allowed-root PATH
                       [--allowed-root PATH ...] --shared-env-root PATH
                       [--submit-helper PATH] [--home PATH]
                       [--staging-root LOCAL=FINAL ...]
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
    --staging-root)
      _staging_arg="${2:?missing staging-root mapping LOCAL=FINAL}"
      [[ "$_staging_arg" == *=* ]] || { echo "LINT_FAIL malformed --staging-root (expected LOCAL=FINAL): '$_staging_arg'" >&2; exit 2; }
      STAGING_LOCAL_ROOTS+=("${_staging_arg%%=*}")
      STAGING_FINAL_ROOTS+=("${_staging_arg#*=}")
      shift 2 ;;
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
[[ ( -n "$REAL_ENV" && -n "$DRY_ENV" ) || ( -z "$REAL_ENV" && -z "$DRY_ENV" ) ]] ||
  { echo "LINT_FAIL --real-env and --dry-env must be given together, never one alone" >&2; exit 2; }
command -v realpath >/dev/null 2>&1 || { echo "LINT_FAIL realpath required" >&2; exit 2; }
# BSD/macOS realpath lacks -m (resolve a path that may not exist); probe it
# explicitly instead of trusting that any binary named 'realpath' is GNU's.
realpath -m -- "$PWD" >/dev/null 2>&1 || {
  echo "LINT_FAIL realpath does not support -m: GNU coreutils realpath is required (on macOS: brew install coreutils, then put its gnubin directory ahead of /usr/bin on PATH)" >&2
  exit 2
}

CRITICAL_KEYS=(HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT DRY_RUN PYTHON)
US=$'\x1f'  # unit separator; macOS/BSD awk's -F does not expand \x1f itself
FAILED=0
fail() { printf 'FAIL: %s\n' "$*"; FAILED=1; }
canon() { realpath -m -- "$1"; }

under_any_root() {
  local candidate="$1" root canon_root
  for root in "${ALLOWED_ROOTS[@]}"; do
    canon_root="$(canon "$root")"
    [[ "$candidate" == "$canon_root" || "$candidate" == "$canon_root"/* ]] && return 0
  done
  return 1
}

# Maps a canonicalized path staged under a declared `--staging-root
# LOCAL=FINAL` prefix to its canonicalized final-root equivalent (realpath -m
# of FINAL + suffix: symlinks that exist locally under FINAL are resolved;
# the linter cannot see remote-only symlinks), so containment checks can
# validate a linted-in-place local staging copy as if it already sat at its
# eventual facility path (a linter run before rsync never has the final path
# to canonicalize). A path outside every declared local root is returned
# unchanged: the mapping only substitutes a matched prefix, it never widens
# or skips the containment check that runs on its result.
map_staging_to_final() {
  local candidate="$1" i n=${#STAGING_LOCAL_ROOTS[@]} local_root final_root canon_local canon_final
  for ((i = 0; i < n; i += 1)); do
    local_root="${STAGING_LOCAL_ROOTS[i]}"
    final_root="${STAGING_FINAL_ROOTS[i]}"
    canon_local="$(canon "$local_root")"
    canon_final="$(canon "$final_root")"
    if [[ "$candidate" == "$canon_local" ]]; then
      printf '%s\n' "$canon_final"
      return 0
    elif [[ "$candidate" == "$canon_local"/* ]]; then
      # Canonicalize the WHOLE mapped destination, not only its root: a
      # symlink under FINAL (FINAL/link -> $HOME) must resolve before the
      # $HOME and containment checks see the path.
      canon "$canon_final${candidate#"$canon_local"}"
      return 0
    fi
  done
  printf '%s\n' "$candidate"
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

# --- Effective job environment (input to checks 4/4b) ----------------------
# The linter never runs the sbatch script or the submit helper. It evaluates
# a deliberately NARROW, documented subset of shell syntax and fails loudly
# (LINT_FAIL naming the line) instead of inventing a value whenever a
# statement that could set a critical key falls outside that subset.
#
# 1. SEED. The job starts with the variables `sbatch --export=SPEC` hands it.
#    SPEC comes from the submit helper's own `sbatch ... --export=SPEC` line
#    when --submit-helper is given and has one (a command-line option
#    overrides a script directive, as in Slurm), else from the LAST
#    `#SBATCH --export=SPEC` directive in the script's leading comment block
#    (sbatch stops reading directives at the first command). SPEC must be
#    literal `NAME=VALUE,...` over [A-Za-z0-9_=,./:@+%-], optionally inside
#    one pair of quotes; ALL/NONE/NIL are keywords, and a bare `NAME`
#    (value propagated from the submitting shell) is UNKNOWN, not supplied.
#    Values are never taken from the manifest.
# 2. BODY, in file order (backslash continuations joined, comments dropped):
#    `export WORD...` and plain `NAME=VALUE...` statements update the
#    effective environment. Each WORD is a bare NAME (export it with its
#    CURRENT value; still absent if it never had one) or NAME=VALUE.
#    VALUE quoting is tracked per character: single quotes are literal and
#    never expanded; double quotes and unquoted text expand only $NAME,
#    ${NAME} and ${NAME:-DEFAULT} (DEFAULT literal), resolved against the
#    current effective environment; an unquoted `#` at a word start ends
#    the statement. An `export` line expands every word before assigning
#    any (bash semantics), so `export A=1 B=$A` sees A's previous value. A
#    plain assignment line followed by a command word is a one-command
#    prefix and does not change the job environment.
# 3. REJECTION. Anything else inside such a statement (command
#    substitution, backticks, other parameter forms, unquoted globs/tilde/
#    braces/operators, a quote left open at end of line) stops parsing that
#    statement. It is a LINT_FAIL when the rest of the statement mentions a
#    critical key or (outside a non-critical NAME=VALUE) any `$`/backtick;
#    otherwise only that non-critical NAME becomes UNKNOWN. `unset`,
#    `declare`, `typeset`, `readonly`, `local` lines naming a critical key
#    are a LINT_FAIL too. A critical key that ends up exported with an
#    UNKNOWN value is a LINT_FAIL.
# Not modelled (documented limitation): control flow, functions, heredoc
# bodies, `source`. A statement inside an if-block is treated as if it ran.

# Effective environment as parallel arrays (bash 3.2: no associative
# arrays). EFF_STATE is `set` (value known), `unknown` (assigned from syntax
# the linter does not evaluate) or `unset` (exported, never given a value).
EFF_NAMES=() EFF_STATE=() EFF_VALUE=() EFF_EXPORTED=()
EFF_IDX=-1

# Sets EFF_IDX to NAME's slot in the effective environment, or -1.
eff_find() {
  local i
  EFF_IDX=-1
  for ((i = 0; i < ${#EFF_NAMES[@]}; i += 1)); do
    if [[ "${EFF_NAMES[i]}" == "$1" ]]; then EFF_IDX=$i; return 0; fi
  done
  return 0
}

# eff_set NAME STATE VALUE MARK_EXPORTED(1|0). A plain assignment (0) keeps
# whatever exported flag NAME already had, as bash does.
eff_set() {
  eff_find "$1"
  if [[ "$EFF_IDX" == -1 ]]; then
    EFF_IDX=${#EFF_NAMES[@]}
    EFF_NAMES[EFF_IDX]="$1"
    EFF_EXPORTED[EFF_IDX]=0
  fi
  EFF_STATE[EFF_IDX]="$2"
  EFF_VALUE[EFF_IDX]="$3"
  if [[ "$4" == 1 ]]; then EFF_EXPORTED[EFF_IDX]=1; fi
  return 0
}

# `export NAME` with no '=': keep NAME's CURRENT value and mark it exported.
eff_export_bare() {
  eff_find "$1"
  if [[ "$EFF_IDX" == -1 ]]; then
    eff_set "$1" unset "" 1
  else
    EFF_EXPORTED[EFF_IDX]=1
  fi
  return 0
}

# Prints `1<US>value` when NAME is exported with a known value, else `0<US>`.
eff_get() {
  eff_find "$1"
  if [[ "$EFF_IDX" != -1 && "${EFF_EXPORTED[EFF_IDX]}" == 1 && "${EFF_STATE[EFF_IDX]}" == set ]]; then
    printf '1%s%s' "$US" "${EFF_VALUE[EFF_IDX]}"
  else
    printf '0%s' "$US"
  fi
}

is_critical_key() {
  local key
  for key in "${CRITICAL_KEYS[@]}"; do [[ "$1" == "$key" ]] && return 0; done
  return 1
}

# True when TEXT names any critical key as a whole word.
mentions_critical_key() {
  local key re
  for key in "${CRITICAL_KEYS[@]}"; do
    re="(^|[^A-Za-z0-9_])${key}([^A-Za-z0-9_]|\$)"
    [[ "$1" =~ $re ]] && return 0
  done
  return 1
}

# Joins backslash-continued lines into one logical line each (used for the
# submit helper; the sbatch body uses its own comment-aware join below).
join_continuations() {
  local file="$1" acc="" line
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == *\\ ]]; then
      acc+="${line%\\} "
    else
      acc+="$line"
      printf '%s\n' "$acc"
      acc=""
    fi
  done < "$file"
  [[ -n "$acc" ]] && printf '%s\n' "$acc"
  return 0
}

# Resolves one parameter reference into EXP_STATE (set|unknown) and
# EXP_VALUE. A NAME never assigned anywhere visible is UNKNOWN for a plain
# reference (it could only come from outside the job script) but takes the
# script's own DEFAULT for ${NAME:-DEFAULT}; an assigned-but-empty NAME
# also takes DEFAULT, as `:-` does in bash.
resolve_param() {
  local name="$1" has_default="$2" default="$3" state=never value=""
  eff_find "$name"
  if [[ "$EFF_IDX" != -1 ]]; then
    state="${EFF_STATE[EFF_IDX]}"
    value="${EFF_VALUE[EFF_IDX]}"
  fi
  EXP_STATE=set
  EXP_VALUE="$value"
  case "$state" in
    set) if [[ "$has_default" == 1 && -z "$value" ]]; then EXP_VALUE="$default"; fi ;;
    unknown) EXP_STATE=unknown ;;
    *) if [[ "$has_default" == 1 ]]; then EXP_VALUE="$default"; else EXP_STATE=unknown; fi ;;
  esac
  return 0
}

RE_PARAM_BRACED='^\$\{([A-Za-z_][A-Za-z0-9_]*)\}'
RE_PARAM_DEFAULT='^\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}$`"\\'"'"']*)\}'
RE_PARAM_PLAIN='^\$([A-Za-z_][A-Za-z0-9_]*)'
RE_NAME='^[A-Za-z_][A-Za-z0-9_]*$'
RE_UNSAFE_UNQUOTED='[[:space:]*?~[]'

# Expands the parameter reference at ${s:i} for scan_env_words (bash dynamic
# scoping: reads/updates the caller's s, i, quote, tok, word_state, name_ok,
# mode). Sets SCAN_ERR on any form outside the supported subset.
scan_param() {
  local rest="${s:i}" pname pdefault="" has_default=0 len k
  # Read BASH_REMATCH right away: any later =~ overwrites it.
  if [[ "$rest" =~ $RE_PARAM_BRACED ]]; then
    pname="${BASH_REMATCH[1]}"; len=${#BASH_REMATCH[0]}
  elif [[ "$rest" =~ $RE_PARAM_DEFAULT ]]; then
    pname="${BASH_REMATCH[1]}"; pdefault="${BASH_REMATCH[2]}"; has_default=1; len=${#BASH_REMATCH[0]}
    if [[ -z "$quote" && "$pdefault" =~ $RE_UNSAFE_UNQUOTED ]]; then
      SCAN_ERR="unquoted \${$pname:-...} default with whitespace or glob characters"
      return 0
    fi
  elif [[ "$rest" =~ $RE_PARAM_PLAIN ]]; then
    pname="${BASH_REMATCH[1]}"; len=${#BASH_REMATCH[0]}
  else
    SCAN_ERR="unsupported expansion '${rest:0:12}'"
    return 0
  fi
  # A plain assignment line assigns left to right, unlike `export`; refuse
  # the order-dependent case instead of modelling it.
  if [[ "$mode" == assign ]]; then
    for ((k = 0; k < ${#W_NAME[@]}; k += 1)); do
      if [[ "${W_NAME[k]}" == "$pname" ]]; then
        SCAN_ERR="'\$$pname' refers to an assignment earlier on the same line"
        return 0
      fi
    done
  fi
  resolve_param "$pname" "$has_default" "$pdefault"
  [[ "$EXP_STATE" == unknown ]] && word_state=unknown
  tok+="$EXP_VALUE"
  name_ok=0
  i=$((i + len))
  return 0
}

# Records the word scan_env_words just finished (dynamic scoping).
scan_emit() {
  local bare=0
  if [[ -z "$name" && "$name_ok" == 1 && "$tok" =~ $RE_NAME ]]; then bare=1; fi
  W_TEXT+=("$tok"); W_STATE+=("$word_state"); W_NAME+=("$name")
  W_BARE+=("$bare"); W_START+=("$start")
  return 0
}

# Splits statement text S into shell words with quote provenance, applying
# the supported expansions (see the grammar above). Fills W_TEXT (expanded
# word), W_STATE (set|unknown), W_NAME (NAME of an unquoted `NAME=` prefix,
# else empty), W_BARE (1 for a plain unquoted NAME), W_START (offset in S).
# On unsupported syntax sets SCAN_ERR, SCAN_ERR_AT (offset of the offending
# word) and SCAN_ERR_NAME (that word's NAME= prefix, if any) and stops.
scan_env_words() {
  local s="$1" mode="$2" n=${#1} i=0 c nx quote="" tok="" word_state=set
  local started=0 name_ok=1 name="" start=0
  W_TEXT=() W_STATE=() W_NAME=() W_BARE=() W_START=()
  SCAN_ERR="" SCAN_ERR_AT=0 SCAN_ERR_NAME=""
  while [[ "$i" -lt "$n" && -z "$SCAN_ERR" ]]; do
    c="${s:i:1}"
    if [[ "$quote" == "'" ]]; then
      # Single quotes: every character literal, nothing expanded.
      if [[ "$c" == "'" ]]; then quote=""; else tok+="$c"; fi
      i=$((i + 1)); continue
    fi
    if [[ "$started" == 0 && -z "$quote" ]]; then
      if [[ "$c" == [[:space:]] ]]; then i=$((i + 1)); continue; fi
      [[ "$c" == "#" ]] && break  # unquoted comment ends the statement
      started=1; start=$i; tok=""; word_state=set; name_ok=1; name=""
    fi
    if [[ "$quote" == '"' ]]; then
      case "$c" in
        '"') quote=""; i=$((i + 1)) ;;
        '\')
          nx="${s:i+1:1}"
          case "$nx" in
            '$'|'`'|'"'|'\') tok+="$nx"; i=$((i + 2)) ;;
            '') SCAN_ERR="backslash at end of line inside double quotes" ;;
            *) tok+='\'; i=$((i + 1)) ;;
          esac ;;
        '`') SCAN_ERR="command substitution (backtick)" ;;
        '$') scan_param ;;
        *) tok+="$c"; i=$((i + 1)) ;;
      esac
      continue
    fi
    case "$c" in
      [[:space:]]) scan_emit; started=0; i=$((i + 1)) ;;
      "'"|'"') quote="$c"; name_ok=0; i=$((i + 1)) ;;
      '\')
        nx="${s:i+1:1}"
        if [[ -z "$nx" ]]; then
          SCAN_ERR="backslash at end of line"
        else
          tok+="$nx"; name_ok=0; i=$((i + 2))
        fi ;;
      '$') scan_param ;;
      '`'|';'|'&'|'|'|'<'|'>'|'('|')'|'*'|'?'|'['|'~'|'{'|'}')
        SCAN_ERR="unsupported unquoted '$c'" ;;
      '=')
        if [[ "$name_ok" == 1 && -z "$name" && "$tok" =~ $RE_NAME ]]; then name="$tok"; fi
        name_ok=0; tok+="="; i=$((i + 1)) ;;
      *)
        [[ "$c" == [A-Za-z0-9_] ]] || name_ok=0
        tok+="$c"; i=$((i + 1)) ;;
    esac
  done
  if [[ -z "$SCAN_ERR" && -n "$quote" ]]; then
    SCAN_ERR="quote left open at end of line (multi-line strings unsupported)"
  fi
  if [[ -n "$SCAN_ERR" ]]; then
    if [[ "$started" == 1 ]]; then SCAN_ERR_AT=$start; SCAN_ERR_NAME="$name"; else SCAN_ERR_AT=$i; fi
  elif [[ "$started" == 1 ]]; then
    scan_emit
  fi
  return 0
}

# Applies one body statement (export / plain assignment / unset-like) to
# the effective environment; LINENO_ARG is the statement's first line.
apply_env_statement() {
  local line="$1" lineno="$2" body mode mark=0 k nw upto value rest verb
  body="${line#"${line%%[![:space:]]*}"}"
  if [[ "$body" =~ ^export([[:space:]]|$) ]]; then
    mode=export; mark=1; body="${body#export}"
  elif [[ "$body" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    mode=assign
  elif [[ "$body" =~ ^(unset|declare|typeset|readonly|local)([[:space:]]|$) ]]; then
    verb="${BASH_REMATCH[1]}"  # before mentions_critical_key's =~ clobbers it
    if mentions_critical_key "$body"; then
      fail "sbatch line $lineno: '$verb' on a critical key is outside the linter's supported syntax: '$line'"
    fi
    return 0
  else
    return 0
  fi
  scan_env_words "$body" "$mode"
  nw=${#W_TEXT[@]}
  upto=$nw
  for ((k = 0; k < nw; k += 1)); do
    if [[ "$mode" == assign && -z "${W_NAME[k]}" ]]; then
      # `NAME=VALUE cmd ...`: a one-command environment prefix, not a
      # change to the job environment, whatever follows.
      return 0
    fi
    if [[ "$mode" == export && -z "${W_NAME[k]}" && "${W_BARE[k]}" != 1 ]]; then
      # `export -f`, `export "$X"`, ...: not a NAME or NAME=VALUE word.
      if [[ -z "$SCAN_ERR" || "${W_START[k]}" -lt "$SCAN_ERR_AT" ]]; then
        SCAN_ERR="unsupported export argument '${W_TEXT[k]}'"
        SCAN_ERR_AT=${W_START[k]}; SCAN_ERR_NAME=""
      fi
      upto=$k
      break
    fi
  done
  # Words before any rejected syntax still take effect, as they would in
  # bash; for `export` they were all expanded before any is assigned.
  for ((k = 0; k < upto; k += 1)); do
    if [[ -n "${W_NAME[k]}" ]]; then
      value="${W_TEXT[k]}"
      value="${value:${#W_NAME[k]}+1}"
      eff_set "${W_NAME[k]}" "${W_STATE[k]}" "$value" "$mark"
    else
      eff_export_bare "${W_TEXT[k]}"
    fi
  done
  if [[ -n "$SCAN_ERR" ]]; then
    rest="${body:SCAN_ERR_AT}"
    if [[ -n "$SCAN_ERR_NAME" ]] && ! is_critical_key "$SCAN_ERR_NAME" && ! mentions_critical_key "$rest"; then
      # Unsupported VALUE for a non-critical NAME: only NAME is unknown.
      eff_set "$SCAN_ERR_NAME" unknown "" "$mark"
    elif [[ -z "$SCAN_ERR_NAME" ]] && ! mentions_critical_key "$rest" && [[ "$rest" != *'$'* && "$rest" != *'`'* ]]; then
      : # e.g. `export A=1; cd x`, `export -f fn`: cannot touch a critical key
    else
      if [[ -n "$SCAN_ERR_NAME" ]]; then eff_set "$SCAN_ERR_NAME" unknown "" "$mark"; fi
      fail "sbatch line $lineno: $SCAN_ERR in a statement that may set a critical export; the linter will not guess its value: '$line'"
    fi
  fi
  return 0
}

# Normalizes an --export=SPEC value (one optional pair of matching quotes)
# into SPEC_OUT, or records a LINT_FAIL naming ORIGIN and returns 1.
SPEC_SAFE_RE='^[A-Za-z0-9_=,./:@+%-]*$'
normalize_export_spec() {
  local raw="$1" origin="$2" v="$1"
  if [[ "$v" =~ ^\"(.*)\"$ ]]; then v="${BASH_REMATCH[1]}"
  elif [[ "$v" =~ ^\'(.*)\'$ ]]; then v="${BASH_REMATCH[1]}"; fi
  if [[ ! "$v" =~ $SPEC_SAFE_RE ]]; then
    fail "$origin: unsupported --export value '$raw' (supported: literal NAME=VALUE,... over [A-Za-z0-9_=,./:@+%-], optionally in one pair of quotes; shell expansions cannot be evaluated by the linter)"
    return 1
  fi
  SPEC_OUT="$v"
  return 0
}

# Seeds the effective environment from a normalized --export SPEC.
seed_from_export_spec() {
  local spec="$1" origin="$2" item key IFS=','
  for item in $spec; do
    case "$item" in
      ALL|NONE|NIL|'') continue ;;
    esac
    if [[ "$item" == *=* ]]; then
      key="${item%%=*}"
      if [[ ! "$key" =~ $RE_NAME ]]; then
        fail "$origin: malformed --export item '$item'"
        continue
      fi
      eff_set "$key" set "${item#*=}" 1
    elif [[ "$item" =~ $RE_NAME ]]; then
      # Bare NAME: sbatch copies it from the submitting shell; unknown here.
      eff_set "$item" unknown "" 1
    else
      fail "$origin: malformed --export item '$item'"
    fi
  done
  return 0
}

# Finds the --export SPEC on the submit helper's own `sbatch` invocation.
# Sets HELPER_SPEC_RAW (empty when no sbatch line in the helper passes
# --export). Several distinct specs, a space-separated `--export VALUE`, or
# a spec on a line that never names the linted job are a LINT_FAIL.
# True when LINE has a word equal to JOB_BASE or ending in /JOB_BASE.
names_job() {
  local word words=()
  # read -a splits on whitespace without pathname expansion.
  read -r -a words <<< "${1//[\"\';()]/ }"
  [[ ${#words[@]} -gt 0 ]] || return 1  # bash < 4.4 + set -u: empty "${a[@]}" is unbound
  for word in "${words[@]}"; do
    [[ "$word" == "$2" || "$word" == */"$2" ]] && return 0
  done
  return 1
}

SBATCH_CMD_RE="(^|[[:space:](\`/\"'])sbatch([[:space:]\"']|\$)"
HELPER_EXPORT_RE='--export(=?)("[^"]*"|'"'"'[^'"'"']*'"'"'|[^[:space:]"'"'"';)]*)'
find_helper_export_spec() {
  local helper="$1" logical rest spec job_base specs="" nspecs=0
  HELPER_SPEC_RAW=""
  job_base="$(basename -- "$SBATCH_PATH")"
  while IFS= read -r logical; do
    [[ "$logical" =~ $SBATCH_CMD_RE ]] || continue
    rest="${logical#*"${BASH_REMATCH[0]}"}"
    spec=""
    while [[ "$rest" =~ $HELPER_EXPORT_RE ]]; do
      if [[ -z "${BASH_REMATCH[1]}" ]]; then
        fail "submit helper: space-separated 'sbatch --export VALUE' is unsupported; use --export=VALUE: '$logical'"
        return 0
      fi
      spec="${BASH_REMATCH[2]}"  # the last --export on a command line wins
      rest="${rest#*"${BASH_REMATCH[0]}"}"
    done
    [[ -n "$spec" ]] || continue
    if ! names_job "$logical" "$job_base"; then
      fail "submit helper: sbatch --export line does not name the linted job '$job_base': '$logical'"
      continue
    fi
    if [[ $'\n'"$specs" != *$'\n'"$spec"$'\n'* ]]; then
      specs+="$spec"$'\n'
      nspecs=$((nspecs + 1))
      HELPER_SPEC_RAW="$spec"
    fi
  done < <(join_continuations "$helper" | sed -E 's/(^|[[:space:]])#.*$//')
  if [[ "$nspecs" -gt 1 ]]; then
    fail "submit helper: $nspecs different sbatch --export lists; the linter cannot tell which one submits this job"
    HELPER_SPEC_RAW=""
  fi
  return 0
}

# Seeds the effective environment once, before the first body statement:
# the helper's command-line --export overrides the script directive.
seed_effective_env() {
  local directive_spec="$1" have_directive="$2"
  if [[ -n "$SUBMIT_HELPER" && -f "$SUBMIT_HELPER" ]]; then
    find_helper_export_spec "$SUBMIT_HELPER"
  fi
  if [[ -n "$HELPER_SPEC_RAW" ]]; then
    if normalize_export_spec "$HELPER_SPEC_RAW" "submit helper sbatch --export"; then
      seed_from_export_spec "$SPEC_OUT" "submit helper sbatch --export"
    fi
  elif [[ "$have_directive" == 1 ]]; then
    if normalize_export_spec "$directive_spec" "#SBATCH --export"; then
      seed_from_export_spec "$SPEC_OUT" "#SBATCH --export"
    fi
  fi
  return 0
}

# Builds the effective job environment: seed, then every body statement in
# file order. `#SBATCH` directives count only in the leading comment block.
build_effective_env() {
  local sbatch_file="$1" line lineno=0 in_header=1 directive_spec="" have_directive=0
  local stmt="" stmt_line=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    lineno=$((lineno + 1))
    if [[ "$in_header" == 1 ]]; then
      if [[ "$line" =~ ^[[:space:]]*# || -z "${line//[[:space:]]/}" ]]; then
        if [[ "$line" =~ ^#SBATCH[[:space:]]+--export([=[:space:]]|$)(.*)$ ]]; then
          if [[ "${BASH_REMATCH[1]}" != "=" ]]; then
            fail "sbatch line $lineno: '#SBATCH --export VALUE' is unsupported; use --export=VALUE"
          else
            directive_spec="${BASH_REMATCH[2]%%[[:space:]]*}"  # last directive wins
            have_directive=1
          fi
        fi
        continue
      fi
      in_header=0
      seed_effective_env "$directive_spec" "$have_directive"
    fi
    # Join backslash continuations of a non-comment statement.
    if [[ -z "$stmt" ]]; then stmt_line=$lineno; fi
    if [[ "$line" == *\\ && ! "$line" =~ ^[[:space:]]*# ]]; then
      stmt+="${line%\\}"
      continue
    fi
    stmt+="$line"
    apply_env_statement "$stmt" "$stmt_line"
    stmt=""
  done < "$sbatch_file"
  if [[ -n "$stmt" ]]; then apply_env_statement "$stmt" "$stmt_line"; fi
  # A script with no body statement never left the header: seed it anyway.
  if [[ "$in_header" == 1 ]]; then seed_effective_env "$directive_spec" "$have_directive"; fi
  return 0
}

HELPER_SPEC_RAW=""
build_effective_env "$SBATCH_PATH"
for key in "${CRITICAL_KEYS[@]}"; do
  eff_find "$key"
  if [[ "$EFF_IDX" != -1 && "${EFF_EXPORTED[EFF_IDX]}" == 1 && "${EFF_STATE[EFF_IDX]}" == unknown ]]; then
    fail "critical export '$key' has a value the linter cannot determine (supplied from outside the job script, or set through syntax it does not evaluate)"
  fi
done

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
MAPPED_SBATCH="$(map_staging_to_final "$CANON_SBATCH")"
if [[ "$SBATCH_PATH" != /* ]]; then
  fail "sbatch path is not absolute: '$SBATCH_PATH'"
elif [[ "$SBATCH_PATH" != "$CANON_SBATCH" ]]; then
  fail "sbatch path is not canonical: '$SBATCH_PATH' -> '$CANON_SBATCH'"
elif ! under_any_root "$MAPPED_SBATCH"; then
  fail "sbatch path escapes every declared stage root: '$CANON_SBATCH' (mapped to '$MAPPED_SBATCH')"
fi

# Check 4: critical exports must match the manifest exactly — no stale,
# missing, or extra critical values between the sbatch script and manifest.
for key in "${CRITICAL_KEYS[@]}"; do
  IFS=$'\x1f' read -r m_has m_val <<< "$(kv_get "$MANIFEST" "$key")"
  IFS=$'\x1f' read -r s_has s_val <<< "$(eff_get "$key")"
  if [[ "$m_has" == 1 && "$s_has" == 0 ]]; then
    fail "critical export '$key' missing from sbatch script (manifest has it)"
  elif [[ "$m_has" == 0 && "$s_has" == 1 ]]; then
    fail "critical export '$key' present in sbatch script but not in manifest (extra)"
  elif [[ "$m_has" == 1 && "$s_has" == 1 && "$m_val" != "$s_val" ]]; then
    fail "critical export '$key' is stale: manifest='$m_val' sbatch='$s_val'"
  fi
done

# Check 4b: a critical field absent from BOTH files passes check 4 (nothing to
# compare), so require the always-required subset present and nonempty in
# both; PYTHON is required too, but only for a dry run (DRY_RUN=1).
ALWAYS_REQUIRED_KEYS=(HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT DRY_RUN)
IFS=$'\x1f' read -r M_DRY_RUN_HAS M_DRY_RUN <<< "$(kv_get "$MANIFEST" DRY_RUN)"
[[ "$M_DRY_RUN_HAS" == 1 ]] || M_DRY_RUN=0

for key in "${ALWAYS_REQUIRED_KEYS[@]}"; do
  IFS=$'\x1f' read -r m_has m_val <<< "$(kv_get "$MANIFEST" "$key")"
  IFS=$'\x1f' read -r s_has s_val <<< "$(eff_get "$key")"
  if [[ "$m_has" != 1 || -z "$m_val" || "$s_has" != 1 || -z "$s_val" ]]; then
    fail "required critical field '$key' is missing or empty (manifest and/or sbatch script)"
  fi
done

if [[ "$M_DRY_RUN" == 1 ]]; then
  IFS=$'\x1f' read -r m_has m_val <<< "$(kv_get "$MANIFEST" PYTHON)"
  IFS=$'\x1f' read -r s_has s_val <<< "$(eff_get PYTHON)"
  if [[ "$m_has" != 1 || -z "$m_val" || "$s_has" != 1 || -z "$s_val" ]]; then
    fail "required critical field 'PYTHON' is missing or empty for a dry run (DRY_RUN=1)"
  fi
fi

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
      mapped_val="$(map_staging_to_final "$canon_val")"
      if [[ "$canon_val" == "$CANON_HOME" || "$canon_val" == "$CANON_HOME"/* ||
            "$mapped_val" == "$CANON_HOME" || "$mapped_val" == "$CANON_HOME"/* ]]; then
        fail "$key resolves under \$HOME: '$val' -> '$canon_val' (mapped to '$mapped_val')"
      fi
    fi
  done
fi

# Check 6: TASK_ROOT/RUN_ROOT must stay inside a declared allowed root
# (facility-root escape). A path staged under a declared `--staging-root`
# prefix is checked at its mapped final-root equivalent, never relaxed.
for key in TASK_ROOT RUN_ROOT; do
  IFS=$'\x1f' read -r has val <<< "$(kv_get "$MANIFEST" "$key")"
  [[ "$has" == 1 ]] || continue
  canon_val="$(canon "$val")"
  mapped_val="$(map_staging_to_final "$canon_val")"
  under_any_root "$mapped_val" || fail "$key escapes every declared allowed root: '$val' -> '$canon_val' (mapped to '$mapped_val')"
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

# Check 8: any ssh/scp/rsync invocation in the submission helper must pass
# -o BatchMode=yes (case-insensitive; for rsync this means `-e "ssh -o
# BatchMode=yes"`), to avoid a 2FA prompt from a non-interactive submit path.
# Checked per LOGICAL invocation, not per physical line or whole-file
# substring: backslash continuations are joined, comments are dropped, and
# each logical line is split on `;`, `&&`, `||`, and `|` before testing which
# segments invoke one of these commands and whether that same segment
# carries the option (join_continuations is defined with the export parser
# above, which also reads the helper).

# Splits a logical line into command segments on `;`, `&&`, `||`, and `|`.
# Done with plain bash substring substitution (not `sed 's/.../\n/g'`): GNU
# sed treats `\n` in a replacement as a newline, but a BSD/POSIX sed may emit
# it as a literal backslash-n depending on build/version, which would silently
# stop splitting on some platforms. Longest operators are replaced first so a
# `||`/`&&` is never partially consumed by the single-character `|` pass.
split_logical_commands() {
  local s="$1"
  s="${s//&&/$'\n'}"
  s="${s//||/$'\n'}"
  s="${s//;/$'\n'}"
  s="${s//|/$'\n'}"
  printf '%s\n' "$s"
}

# Matches ssh/scp/rsync as a command WORD anywhere in the segment: preceded
# by start-of-segment, whitespace, `(`, a backtick, `/`, or a quote (so
# `x=$(ssh ...)`, `` `ssh ...` ``, `/usr/bin/ssh`, and quoted forms all
# count), and followed by whitespace, a quote, or end-of-segment (so
# `sshd_config`/`rsync_log` do NOT match).
SSH_LIKE_CMD_RE="(^|[[:space:](\`/\"'])(ssh|scp|rsync)([[:space:]\"']|\$)"

if [[ -n "$SUBMIT_HELPER" ]]; then
  if [[ ! -f "$SUBMIT_HELPER" ]]; then
    fail "submit helper not found: '$SUBMIT_HELPER'"
  else
    while IFS= read -r logical_line; do
      [[ -z "${logical_line//[[:space:]]/}" ]] && continue
      while IFS= read -r segment; do
        segment="${segment#"${segment%%[![:space:]]*}"}"
        [[ -z "$segment" ]] && continue
        if [[ "$segment" =~ $SSH_LIKE_CMD_RE ]]; then
          shopt -s nocasematch
          if [[ "$segment" != *"batchmode=yes"* ]]; then
            fail "ssh/scp/rsync invocation without -o BatchMode=yes: '$segment'"
          fi
          shopt -u nocasematch
        fi
      done < <(split_logical_commands "$logical_line")
    done < <(join_continuations "$SUBMIT_HELPER" | sed -E 's/(^|[[:space:]])#.*$//')
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
      # Compare presence as well as value: a literal value equal to the
      # display placeholder below must never be confused with "absent".
      if [[ "$rhas" != "$dhas" ]]; then
        fail "real-vs-dry export delta on undocumented key '$key': real='${rval:-<absent>}' dry='${dval:-<absent>}' (present on only one side)"
      elif [[ "$rhas" == 1 && "$rval" != "$dval" ]]; then
        fail "real-vs-dry export delta on undocumented key '$key': real='$rval' dry='$dval'"
      fi
    done <<< "$ALL_KEYS"
  fi
fi

if [[ "$FAILED" == 0 ]]; then
  echo "LINT_PASS"
  exit 0
fi
echo "LINT_FAIL"
exit 1
