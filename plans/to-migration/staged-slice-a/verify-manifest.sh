#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
commit=c25191bfa9e825882ac0da05a4f7373e543633cd

awk -F '\t' 'NR > 1 { print $3 }' "$root/manifest.tsv" |
while IFS= read -r path; do
  case "$path" in
    claude-plugins/task-orchestrator/skills/*/SKILL.md) ;;
    *) echo "invalid manifest source: $path" >&2; exit 1 ;;
  esac
  printf '%s\n' "$commit:$path"
done
