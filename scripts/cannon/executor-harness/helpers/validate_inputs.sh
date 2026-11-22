#!/usr/bin/env bash
# Fixed-literal storage validation. The caller may still use assert_prefix for
# descendants, but this gate prevents that helper from validating its own root.

validate_inputs() {
  local lab_root="/n/netscratch/kozinsky_lab/Lab/rhu"
  local local_fake_root="${LOCAL_FAKE_ROOT:-}"
  local path_name path_value

  [[ -n "${USER:-}" && -n "${SLURM_JOB_ID:-}" ]] || return 1
  if [[ "${DRY_RUN:-0}" == 0 ]]; then
    [[ "${NODE_TMP:-}" == "/scratch/${USER}/${SLURM_JOB_ID}" ]] || return 1
  else
    [[ "${DRY_NODE_TMP:-}" == "$lab_root" || "${DRY_NODE_TMP:-}" == "$lab_root"/* || ( "${LOCAL_FAKE:-0}" == 1 && -n "$local_fake_root" && ( "${DRY_NODE_TMP:-}" == "$local_fake_root" || "${DRY_NODE_TMP:-}" == "$local_fake_root"/* ) ) ]] || return 1
  fi

  for path_name in HARNESS_ROOT RUN_ROOT TASK_ROOT; do
    path_value="${!path_name:-}"
    [[ "$path_value" == "$lab_root" || "$path_value" == "$lab_root"/* || ( "${LOCAL_FAKE:-0}" == 1 && "${DRY_RUN:-0}" == 1 && -n "$local_fake_root" && ( "$path_value" == "$local_fake_root" || "$path_value" == "$local_fake_root"/* ) ) ]] || return 1
  done
}
