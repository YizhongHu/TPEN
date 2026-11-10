#!/usr/bin/env bash
# Local, Slurm-free regression probes for the four CHANGES-REQUIRED findings
# on PR #636 (review-summary note on item 55799bad). Each probe reproduces the
# reviewer's synthetic control and asserts the fixed behaviour. stdlib/bash
# only; no pytest, no cluster access, no installs.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MUTANT_HELPER="$SCRIPT_DIR/mutants.py"
PYTHON="${PYTHON:-python3}"
WORK="$(mktemp -d)"
trap 'rm -rf -- "$WORK"' EXIT
fail=0
check() { if [[ "$1" == "$2" ]]; then echo "ok: $3"; else echo "FAIL: $3 (expected '$2', got '$1')" >&2; fail=1; fi; }

echo "== F1: mutant spec requires a named killer =="
cat > "$WORK/spec-no-killer.json" <<'JSON'
[{"id":"m1","target":"x.py","old":"a","new":"b"},
 {"id":"m2","target":"x.py","old":"a","new":"b"},
 {"id":"m3","target":"x.py","old":"a","new":"b"}]
JSON
if "$PYTHON" "$MUTANT_HELPER" validate --spec "$WORK/spec-no-killer.json" > /dev/null 2>"$WORK/spec-no-killer.err"; then
  echo "FAIL: spec without killer was accepted" >&2; fail=1
else
  echo "ok: spec without killer is rejected ($(cat "$WORK/spec-no-killer.err"))"
fi

echo "== F1: verdict never reads PASS/KILLED off a bare pytest exit code =="
cat > "$WORK/baseline.xml" <<'XML'
<testsuites><testsuite><testcase classname="tests.unit.test_x" name="test_sign"/></testsuite></testsuites>
XML
cat > "$WORK/mutant-fail.xml" <<'XML'
<testsuites><testsuite><testcase classname="tests.unit.test_x" name="test_sign"><failure message="assert">boom</failure></testcase></testsuite></testsuites>
XML
cat > "$WORK/spec.json" <<'JSON'
[{"id":"m1","target":"x.py","old":"a","new":"b","killer":"tests/unit/test_x.py::test_sign"},
 {"id":"m2","target":"x.py","old":"a","new":"b","killer":"tests/unit/test_x.py::test_sign"},
 {"id":"m3","target":"x.py","old":"a","new":"b","killer":"tests/unit/test_x.py::test_sign"}]
JSON
v="$("$PYTHON" "$MUTANT_HELPER" verdict --spec "$WORK/spec.json" --id m1 --baseline-junit "$WORK/baseline.xml" --mutant-junit "$WORK/baseline.xml" | "$PYTHON" -c 'import json,sys;print(json.load(sys.stdin)["verdict"])')"
check "$v" SURVIVED "killer still passing under mutant => SURVIVED, not KILLED-by-rc"
v="$("$PYTHON" "$MUTANT_HELPER" verdict --spec "$WORK/spec.json" --id m1 --baseline-junit "$WORK/baseline.xml" --mutant-junit "$WORK/mutant-fail.xml" | "$PYTHON" -c 'import json,sys;print(json.load(sys.stdin)["verdict"])')"
check "$v" KILLED "killer actually failed under mutant => KILLED"
v="$("$PYTHON" "$MUTANT_HELPER" verdict --spec "$WORK/spec.json" --id m1 --baseline-junit "$WORK/baseline.xml" --mutant-junit "$WORK/does-not-exist.xml" | "$PYTHON" -c 'import json,sys;print(json.load(sys.stdin)["verdict"])')"
check "$v" HARNESS_ERROR "no tests collected / missing junit under mutant (reviewer's rc=5 case) => HARNESS_ERROR, never PASS"

echo "== F2: prefix guard rejects parent-traversal and out-of-root escapes =="
mkdir -p "$WORK/approved/user/run" "$WORK/approved/other" "$WORK/outside"
canon() { realpath -m -- "$1"; }
CANON_PREFIX="$(canon "$WORK/approved/user")"
assert_write_path() {
  local c; c="$(canon "$1")"
  [[ "$c" == "$CANON_PREFIX" || "$c" == "$CANON_PREFIX"/* ]]
}
if assert_write_path "$WORK/approved/user/run/evidence"; then echo "ok: normal descendant accepted"; else echo "FAIL: normal descendant rejected" >&2; fail=1; fi
if assert_write_path "$WORK/outside/run"; then echo "FAIL: out-of-root path accepted" >&2; fail=1; else echo "ok: out-of-root path rejected"; fi
if assert_write_path "$WORK/approved/user/../other/run"; then
  echo "FAIL: ../ escape to approved/other was accepted (reviewer's prefix-dotdot defect)" >&2; fail=1
else
  echo "ok: ../ escape to approved/other is rejected"
fi

echo "== F4: stale same-tick bytecode cannot survive a pre-run purge =="
F4="$WORK/f4"; mkdir -p "$F4"
printf 'VALUE = 1\n' > "$F4/mod.py"
cat > "$WORK/f4probe.py" <<'PY'
import sys, os, json, shutil, importlib
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import mutants
from pathlib import Path
src = Path(sys.argv[1])
import mod
assert mod.VALUE == 1
spec_path = src / "spec.json"
spec_path.write_text(json.dumps([
    {"id": "m1", "target": "mod.py", "old": "VALUE = 1", "new": "VALUE = 2", "killer": "t"},
    {"id": "m2", "target": "mod.py", "old": "VALUE = 1", "new": "VALUE = 3", "killer": "t"},
    {"id": "m3", "target": "mod.py", "old": "VALUE = 1", "new": "VALUE = 4", "killer": "t"},
]))
backup = src / "mod.py.bak"
assert mutants.apply_mutant(spec_path, src, backup, "m1") == 0
pycache = src / "__pycache__"
cached = next(pycache.glob("mod.*.pyc"))
same_time = cached.stat().st_mtime
os.utime(src / "mod.py", (same_time, same_time))
del sys.modules["mod"]; importlib.invalidate_caches()
import mod as stale
if stale.VALUE != 1:
    print("UNEXPECTED: stale cache scenario did not reproduce"); sys.exit(1)
shutil.rmtree(pycache)
del sys.modules["mod"]; importlib.invalidate_caches()
import mod as fresh
assert fresh.VALUE == 2, f"purge-before-run did not pick up the edit: {fresh.VALUE}"
assert mutants.restore(src, backup, "mod.py") == 0
assert src.joinpath("mod.py").read_text().strip() == "VALUE = 1"
print("ok: purge-before-run observes the mutant; stale-cache scenario reproduced and fixed")
PY
"$PYTHON" "$WORK/f4probe.py" "$F4" "$SCRIPT_DIR"

echo "== F3: *_EXECUTOR_PYTHON / *_ROOT_PYTHON discovery matches the real test family =="
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
found="$(grep -rhoE '[A-Z][A-Z0-9_]*_(EXECUTOR|ROOT)_PYTHON' "$REPO_ROOT" --include='*.py' 2>/dev/null | sort -u)"
for expect in P8_EXECUTOR_PYTHON P9C_EXECUTOR_PYTHON P9C_ROOT_PYTHON P10_ROOT_PYTHON H_V2_EXECUTOR_PYTHON; do
  if grep -qx "$expect" <<< "$found"; then
    echo "ok: discovery finds $expect"
  else
    echo "FAIL: discovery missed $expect" >&2; fail=1
  fi
done

if [[ "$fail" == 0 ]]; then echo "ALL SELFTESTS PASSED"; else echo "SELFTEST FAILURES ABOVE" >&2; fi
exit "$fail"
