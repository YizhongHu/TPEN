#!/usr/bin/env bash
# Shared, side-effect-free naming and manifest-validation helpers for the
# shared Cannon uv environment builder and its reuse-only consumer precheck.
# Sourced by each caller from its OWN SCRIPT_DIR (never from a target
# clone): both build-shared-env.sh and shared-env-precheck.sh must agree on
# exactly one naming scheme and one manifest contract. No facility absolute
# paths here: every path/value arrives from the caller.

NAME_COMPONENT_RE='^[A-Za-z0-9][A-Za-z0-9._+]*$'

# <project>-<lock-sha256[:12]>-<extra>-py<X.Y>, the one env-name scheme both
# the builder and the consumer compute.
compute_shared_env_name() {
  local project="$1" lock_sha256="$2" extra="$3" python_version="$4"
  printf '%s-%s-%s-py%s' "$project" "${lock_sha256:0:12}" "$extra" "$python_version"
}

# A --project-dir value is either "." (the checkout root itself) or one-or-
# more path components, each matching NAME_COMPONENT_RE (which requires a
# leading alnum and so already rejects a ".." component), joined by single
# '/' separators -- no leading '/', no trailing '/', no empty segment.
is_valid_project_dir_rel() {
  local rel="$1"
  [[ "$rel" == "." ]] && return 0
  [[ -n "$rel" ]] || return 1
  [[ "$rel" != /* ]] || return 1
  [[ "$rel" != */ ]] || return 1
  [[ "$rel" != *//* ]] || return 1
  local part
  local IFS='/'
  for part in $rel; do
    [[ "$part" =~ $NAME_COMPONENT_RE ]] || return 1
  done
  return 0
}

# A manifest is trusted for reuse only if it is the six fields
# build-shared-env.sh always wrote prior to the --project-dir slice (still
# produced by builds whose project dir is the checkout root), or those same
# six fields plus a seventh well-formed `project_dir=<REL>` line (every
# build from here on), in order, each strictly formatted, with the recorded
# lock hash matching the caller-supplied expected hash. The build HEAD may
# differ from the requesting HEAD (dependencies, not history, define
# identity); an empty, truncated (even late, even without a final field,
# even missing only a trailing newline), malformed, or mismatched manifest
# is not trusted.
manifest_is_valid() {
  local manifest="$1" expected_lock_sha256="$2"
  [[ -s "$manifest" ]] || return 1
  local -a lines
  mapfile -t lines < "$manifest"
  local n="${#lines[@]}"
  ((n == 6 || n == 7)) || return 1
  [[ "${lines[0]}" =~ ^head=[0-9a-f]{40}$ ]] || return 1
  [[ "${lines[1]}" =~ ^uv_lock_sha256=[0-9a-f]{64}$ ]] || return 1
  [[ "${lines[1]#uv_lock_sha256=}" == "$expected_lock_sha256" ]] || return 1
  [[ "${lines[2]}" =~ ^uv_version=.+$ ]] || return 1
  [[ "${lines[3]}" =~ ^python_version=[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
  [[ "${lines[4]}" =~ ^python_request=[0-9]+\.[0-9]+$ ]] || return 1
  [[ "${lines[5]}" =~ ^built_utc=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] || return 1
  if ((n == 7)); then
    [[ "${lines[6]}" =~ ^project_dir=.+$ ]] || return 1
    is_valid_project_dir_rel "${lines[6]#project_dir=}" || return 1
  fi
  return 0
}

# Root-aware containment: is canonical path "$2" equal to or nested under
# canonical path "$1"? A raw "$parent/*" glob never matches an ordinary
# absolute path when parent=="/" (that pattern starts with "//"), so the
# root case is handled explicitly rather than left to fall through.
path_contains() {
  local parent="$1" child="$2"
  if [[ "$parent" == "/" ]]; then
    [[ "$child" == /* ]]
    return
  fi
  [[ "$child" == "$parent" || "$child" == "$parent"/* ]]
}
