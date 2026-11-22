#!/usr/bin/env python3
"""Run the independent S2b-1 exact-head final matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from helpers.path_guard import assert_write_path


ROOT = Path(os.environ["RUN_ROOT"]).resolve()
WORK_ROOT = Path(os.environ["NODE_TMP"])
HARNESS_ROOT = Path(os.environ["TASK_ROOT"])
HELPERS = HARNESS_ROOT / "helpers"
RUN_PYTEST = HARNESS_ROOT / "run_pytest.py"
HEAD = os.environ["HEAD_SHA"]
NODE_TMP = Path(os.environ["NODE_TMP"])
ARMS = ("c1-py312-parsl", "c2-py314-parsl", "c3-py312-none", "c4-py314-none")
PARSL_ARMS = ARMS[:2]
REAL_CORE_IDS = (
    "tests/test_review_p5b_r1.py::test_r1_t0_real_core_module_help_prints_usage",
    "tests/test_review_p5d_r1.py::test_r1_t7_real_core_pinned_overrides_win_and_evidence_names_actual_run",
    "tests/test_review_p5d_r1.py::test_r1_t11_parent_write_trace_includes_log_opens_and_successful_members",
)


def expected_suite(killer: str) -> str:
    path = killer.split("::", 1)[0].replace("\\", "/")
    return "executor" if path == "executor/tests" or path.startswith("executor/tests/") else "root"


def load_spec() -> dict[str, object]:
    """Load and fail closed on a killer/suite boundary mismatch."""

    spec_path = Path(os.environ["MUTANT_SPEC"])
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    rows = [dict(row) for row in raw.get("mutants", [])]
    legacy_reader = raw.get("reader_mutant")
    if legacy_reader is not None and not any(row.get("id") == legacy_reader.get("id") for row in rows):
        reader = dict(legacy_reader)
        reader.setdefault("suite", "root")
        rows.append(reader)
    for row in rows:
        mutant_id = row.get("id")
        killer = row.get("killer")
        suite = row.get("suite")
        if not isinstance(mutant_id, str) or not isinstance(killer, str):
            raise ValueError("each mutant requires string id and killer")
        if suite not in {"root", "executor"}:
            raise ValueError(f"{mutant_id}: suite must be root or executor")
        required = expected_suite(killer)
        if suite != required:
            raise ValueError(f"{mutant_id}: killer {killer!r} requires suite={required}, got {suite}")
    return {**raw, "mutants": rows}


SPEC = load_spec()
MUTANT_ROWS = tuple(SPEC["mutants"])


def call(argv: list[str], *, cwd: Path, env: dict[str, str], log: Path | None = None) -> int:
    if log is None:
        return subprocess.run(argv, cwd=cwd, env=env).returncode
    assert_write_path(log)
    assert_write_path(log.parent)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        return subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode


def write_text(path: Path, text: str) -> None:
    assert_write_path(path)
    assert_write_path(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def clear_pycache(source: Path) -> int:
    """Remove only bytecode directories below the job-owned clone."""

    removed = 0
    if not source.is_dir():
        return removed
    for cache in sorted(source.rglob("__pycache__"), reverse=True):
        if cache.is_dir():
            assert_write_path(cache)
            shutil.rmtree(cache)
            removed += 1
    return removed


def compile_mutation(prefix: str, py: Path, target: Path, cwd: Path, env: dict[str, str]) -> int:
    """Compile one mutated file offline before allowing its pytest run."""

    log = ROOT / "evidence" / f"{prefix}-compile.log"
    assert_write_path(log)
    assert_write_path(target)
    # Syntax-check by file kind: py_compile only for Python; shell/PBS templates use bash -n
    # (Cannon job 59359049: py_compile on p10-smoke.pbs left M3/M4 unmeasured).
    if target.suffix == ".py":
        rc = call([str(py), "-m", "py_compile", str(target)], cwd=cwd, env=env, log=log)
    elif target.suffix in {".sh", ".pbs", ".sbatch", ".bash"}:
        rc = call(["/bin/bash", "-n", str(target)], cwd=cwd, env=env, log=log)
    else:
        write_text(log, f"no syntax checker for suffix {target.suffix!r}; skipped\n")
        rc = 0
    write_text(ROOT / "evidence" / f"{prefix}-compile.rc", f"compile_rc={rc}\n")
    cache = target.parent / "__pycache__"
    assert_write_path(cache)
    if cache.is_dir():
        for candidate in cache.glob(f"{target.stem}.*.pyc"):
            assert_write_path(candidate)
            candidate.unlink(missing_ok=True)
        if not any(cache.iterdir()):
            cache.rmdir()
    write_text(ROOT / "evidence" / f"{prefix}-compile-pycache.txt", f"exists={cache.exists()}\n")
    return rc


def baseline_killing_test(prefix: str, killer: str) -> None:
    """Record the baseline call outcome before this source is mutated."""

    payload = read_collection(ROOT / "evidence" / f"{prefix}-collect.json")
    rows = [
        row for row in payload.get("reports", [])
        if row.get("when") == "call" and row.get("nodeid") == killer
    ]
    write_text(
        ROOT / "evidence" / f"baseline-{prefix}-{killer}.json",
        json.dumps({"killing_test": killer, "rows": rows}, indent=2, sort_keys=True) + "\n",
    )


def reconcile(prefix: str, cwd: Path, env: dict[str, str], receipt: Path, junit: Path, log: Path, counts: Path, skips: Path) -> int:
    return call(
        [str(Path(env["ROOT_PYTHON"])), str(HELPERS / "reconcile_counts.py"), str(junit), str(log), str(counts), str(skips), str(receipt)],
        cwd=cwd,
        env=env,
        log=ROOT / f"{prefix}-reconcile-counts.log",
    )


def run_group(
    prefix: str,
    py: Path,
    cwd: Path,
    env: dict[str, str],
    args: list[str],
    *,
    expected_root: Path | None = None,
    allow_failure: bool = False,
) -> tuple[int, dict[str, object]]:
    evidence = ROOT / "evidence"
    receipt = evidence / f"{prefix}-collect.json"
    junit = ROOT / f"{prefix}.xml"
    log = ROOT / f"{prefix}.log"
    counts = ROOT / f"{prefix}-counts.txt"
    skips = ROOT / f"{prefix}-skip-manifest.txt"
    basetemp = NODE_TMP / f"basetemp-{prefix}"
    for output in (receipt, junit, log, counts, skips, basetemp):
        assert_write_path(output)
    invocation = [str(py), str(RUN_PYTEST), str(receipt), "--basetemp", str(basetemp), "-q", "--junitxml", str(junit), *args]
    write_text(
        evidence / f"{prefix}-selection.txt",
        "cwd=" + str(cwd) + "\n"
        + "command=" + " ".join(invocation) + "\n"
        + "PYTHONPATH=" + env.get("PYTHONPATH", "") + "\n"
        + "PATH=" + env.get("PATH", "") + "\n",
    )
    try:
        rc = call(invocation, cwd=cwd, env=env, log=log)
    except BaseException as error:
        rc = 125
        write_text(log, f"HARNESS_EXCEPTION={type(error).__name__}: {error}\n")
    detail: dict[str, object] = {"pytest_rc": rc, "cwd": str(cwd), "args": args}
    try:
        collection = json.loads(receipt.read_text(encoding="utf-8"))
        collection_errors = len(collection.get("collection", []))
        collected = len(collection.get("collected", []))
        detail.update({"collected": collected, "collection_errors": collection_errors, "rootdir": collection.get("rootdir")})
    except (OSError, ValueError, TypeError) as error:
        collection_errors = 1
        detail["collection_error"] = str(error)
    if expected_root is not None and detail.get("rootdir") != str(expected_root.resolve()):
        collection_errors += 1
        detail["rootdir_mismatch"] = True
    if collection_errors or rc not in (0, 1) or not junit.is_file():
        write_text(counts, "tests=0\npassed=0\nfailures=0\nerrors=1\nskips=0\nexpected_tests=-1\n")
        write_text(ROOT / f"{prefix}-step.rc", f"pytest={rc}\ncollection-errors={collection_errors}\nreconcile-counts=NOT-RUN\nstep=1\n")
        return 1, detail
    rec_rc = reconcile(prefix, cwd, env, receipt, junit, log, counts, skips)
    step = 0 if rc == 0 and rec_rc == 0 else 1
    write_text(ROOT / f"{prefix}-step.rc", f"pytest={rc}\ncollection-errors=0\nreconcile-counts={rec_rc}\nstep={step}\n")
    detail["reconcile_rc"] = rec_rc
    return (0 if rec_rc == 0 else 1, detail) if allow_failure else (step, detail)


def read_collection(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def check_real_core(prefix: str) -> int:
    payload = read_collection(ROOT / "evidence" / f"{prefix}-collect.json")
    collected = set(payload.get("collected", []))
    reports = [row for row in payload.get("reports", []) if row.get("when") == "call"]
    by_id = {row.get("nodeid"): row for row in reports}
    rows = [{"nodeid": nodeid, "collected": nodeid in collected, "call_outcome": by_id.get(nodeid, {}).get("outcome")} for nodeid in REAL_CORE_IDS]
    write_text(ROOT / "evidence" / f"{prefix}-real-core.json", json.dumps({"required": rows}, indent=2, sort_keys=True) + "\n")
    return 0 if all(row["collected"] and row["call_outcome"] == "passed" for row in rows) else 1


# Operator decision 2026-11-10 (root note operator-decisions-speedup-20261110, item 4): no full suite per PR.
# The mutant spec may set "root_selectors" (whole test directories of the changed modules plus their direct
# consumers) and "executor_selectors" (default: the whole executor suite). Empty root_selectors = full root suite.
ROOT_SELECTORS = [str(value) for value in SPEC.get("root_selectors", [])]
EXECUTOR_SELECTORS = [str(value) for value in SPEC.get("executor_selectors", ["tests"])]


def full_args(arm: str) -> list[str]:
    ignores = ["--ignore=tests/test_review_p7_r2.py", "--ignore=tests/test_review_p7_r4.py"] if arm.endswith("-none") else []
    return ignores + EXECUTOR_SELECTORS


def selector_for_run(row: dict[str, object]) -> str:
    killer = str(row["killer"])
    if row["suite"] == "executor" and killer.startswith("executor/"):
        return killer.removeprefix("executor/")
    return killer


def root_mutant_args(row: dict[str, object]) -> list[str]:
    selector = selector_for_run(row)
    module = selector.split("::", 1)[0]
    extras = [str(value) for value in row.get("extra_selectors", [])]
    result: list[str] = []
    # Targeted mode: a root mutant runs against the same whole directories as the root baseline.
    for value in [*ROOT_SELECTORS, module, *extras]:
        if value not in result:
            result.append(value)
    return result


def mutate(source: Path, row: dict[str, object]) -> tuple[int, int, Path, bytes]:
    path = source / str(row["target"])
    try:
        original = path.read_bytes()
    except OSError:
        return 1, 0, path, b""
    old = str(row["old"]).encode("utf-8")
    new = str(row["new"]).encode("utf-8")
    needle_count = original.count(old)
    changed = original.replace(old, new, 1)
    edit_rc = 0 if needle_count == 1 and changed != original else 1
    if edit_rc == 0:
        assert_write_path(path)
        path.write_bytes(changed)
    return edit_rc, needle_count, path, original


def main() -> int:
    if len(sys.argv) != 1:
        print("usage: final_run.py", file=sys.stderr)
        return 2
    env = dict(os.environ)
    root_source = Path(env.get("ROOT_SOURCE", WORK_ROOT / "clone-root")).resolve()
    root_py = Path(env.get("ROOT_PYTHON", WORK_ROOT / "venv-root/bin/python"))
    def arm_python(arm: str) -> Path:
        key = f"{arm.replace('-', '_')}_PYTHON"
        return Path(env.get(key, str(WORK_ROOT / f"venv-{arm}/bin/python")))

    overall = 0
    for arm in ARMS:
        source = WORK_ROOT / f"clone-{arm}"
        py = arm_python(arm)
        arm_env = {**env, "REVIEW_ARM": arm, "REVIEW_SOURCE": str(source), "EXECUTOR_PYTHON": str(py), "P8_EXECUTOR_PYTHON": str(py), "P9C_EXECUTOR_PYTHON": str(py), "P5B_REVIEW_SCIENCE_PYTHON": str(root_py), "P9C_ROOT_PYTHON": str(root_py), "P10_ROOT_PYTHON": str(root_py), "ROOT_PYTHON": str(root_py), "PYTHONPATH": f"{source}:{source / 'executor/src'}", "PATH": f"{Path(py).parent}:{env.get('PATH', '')}"}
        if py.is_file():
            rc, _ = run_group(arm, py, source / "executor", arm_env, full_args(arm), expected_root=source / "executor")
            overall |= rc
            if arm in PARSL_ARMS:
                overall |= check_real_core(arm)
        else:
            write_text(ROOT / f"{arm}-counts.txt", "tests=0\npassed=0\nfailures=0\nerrors=1\nskips=0\nexpected_tests=-1\n")
            write_text(ROOT / f"{arm}-step.rc", "pytest=125\ncollection-errors=1\nreconcile-counts=NOT-RUN\nstep=1\n")
            overall = 1

    c1_py = arm_python("c1-py312-parsl")
    # Root rows intentionally omit executor/src: importing the subproject in a
    # root process would mask project-boundary defects (for example #625 F2).
    root_env = {**env, "REVIEW_ARM": "root", "REVIEW_SOURCE": str(root_source), "ROOT_PYTHON": str(root_py), "P5B_REVIEW_SCIENCE_PYTHON": str(root_py), "P9C_ROOT_PYTHON": str(root_py), "P9C_EXECUTOR_PYTHON": str(c1_py), "P8_EXECUTOR_PYTHON": str(c1_py), "H_V2_SCIENCE_PYTHON": str(root_py), "H_V2_EXECUTOR_PYTHON": str(c1_py), "PYTHONPATH": str(root_source), "PATH": f"{Path(root_py).parent}:{env.get('PATH', '')}"}
    if root_py.is_file():
        rc, _ = run_group("root", root_py, root_source, root_env, list(ROOT_SELECTORS), expected_root=root_source)
        overall |= rc
    else:
        write_text(ROOT / "root-counts.txt", "tests=0\npassed=0\nfailures=0\nerrors=1\nskips=0\nexpected_tests=-1\n")
        write_text(ROOT / "root-step.rc", "pytest=125\ncollection-errors=1\nreconcile-counts=NOT-RUN\nstep=1\n")
        overall = 1

    for row in MUTANT_ROWS:
        mutant = str(row["id"])
        suite = str(row["suite"])
        arms = ARMS if suite == "executor" else ("root",)
        for arm in arms:
            source = WORK_ROOT / (f"clone-{arm}" if suite == "executor" else "clone-root")
            py = arm_python(arm) if suite == "executor" else root_py
            prefix = f"mutant-{mutant}-{arm}"
            target_root = source / "executor" if suite == "executor" else source
            edit_rc, needle_count, target_path, original = mutate(source, row) if source.is_dir() else (1, 0, source / "missing", b"")
            clear_pycache(source)
            if target_path.is_file() and edit_rc == 0:
                write_text(ROOT / "evidence" / f"{prefix}-mutated.py", target_path.read_text(encoding="utf-8"))
            write_text(ROOT / "evidence" / f"{prefix}-edit.rc", f"edit_rc={edit_rc}\n")
            write_text(ROOT / "evidence" / f"{prefix}-needle.txt", f"mutant={mutant}\nneedle_count={needle_count}\n")
            write_text(ROOT / "evidence" / f"{prefix}-pycache-before.txt", f"pycache_count=0\n")
            killer = selector_for_run(row)
            baseline_killing_test("root" if suite == "root" else arm, killer)
            if suite == "executor":
                row_env = {**env, "REVIEW_ARM": prefix, "REVIEW_SOURCE": str(source), "EXECUTOR_PYTHON": str(py), "P8_EXECUTOR_PYTHON": str(py), "P9C_EXECUTOR_PYTHON": str(py), "P5B_REVIEW_SCIENCE_PYTHON": str(root_py), "P9C_ROOT_PYTHON": str(root_py), "P10_ROOT_PYTHON": str(root_py), "ROOT_PYTHON": str(root_py), "PYTHONPATH": f"{source}:{source / 'executor/src'}", "PATH": f"{Path(py).parent}:{env.get('PATH', '')}"}
                args = full_args(arm)
            else:
                row_env = {**env, "REVIEW_ARM": prefix, "REVIEW_SOURCE": str(source), "ROOT_PYTHON": str(root_py), "P8_EXECUTOR_PYTHON": str(c1_py), "P9C_ROOT_PYTHON": str(root_py), "P9C_EXECUTOR_PYTHON": str(c1_py), "P5B_REVIEW_SCIENCE_PYTHON": str(root_py), "H_V2_SCIENCE_PYTHON": str(root_py), "H_V2_EXECUTOR_PYTHON": str(c1_py), "PYTHONPATH": str(source), "PATH": f"{Path(root_py).parent}:{env.get('PATH', '')}"}
                args = root_mutant_args(row)
            if edit_rc == 0 and py.is_file():
                compile_rc = compile_mutation(prefix, py, target_path, target_root, row_env)
                if compile_rc == 0:
                    run_group(prefix, py, target_root, row_env, args, expected_root=target_root, allow_failure=True)
                else:
                    write_text(ROOT / f"{prefix}-counts.txt", "tests=0\npassed=0\nfailures=0\nerrors=1\nskips=0\nexpected_tests=-1\n")
                    write_text(ROOT / f"{prefix}-step.rc", "pytest=125\ncollection-errors=1\nreconcile-counts=NOT-RUN\nstep=1\n")
            else:
                # Never run pytest after an unsuccessful edit: that would test the unmutated copy.
                write_text(ROOT / "evidence" / f"{prefix}-compile.rc", "compile_rc=125\n")
                write_text(ROOT / f"{prefix}-counts.txt", "tests=0\npassed=0\nfailures=0\nerrors=1\nskips=0\nexpected_tests=-1\n")
                write_text(ROOT / f"{prefix}-step.rc", "pytest=125\ncollection-errors=1\nreconcile-counts=NOT-RUN\nstep=1\n")
            if target_path.is_file() and edit_rc == 0:
                assert_write_path(target_path)
                target_path.write_bytes(original)
            clear_pycache(source)
    verdict_rc = 0 if root_py.is_file() else 1
    return 0 if overall == 0 and verdict_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
