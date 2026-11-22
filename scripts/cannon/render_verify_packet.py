#!/usr/bin/env python3
"""Render a checked, one-shot ``verify.sh`` packet.

The renderer deliberately has no TPEN runtime dependencies.  It stages the
facility scripts from the requested git commit, validates the optional mutant
specification, and executes the rendered shell body in a dry environment before
reporting the command an operator may submit.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
from typing import Dict, List, Optional, Sequence


SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
JOB_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TIME_RE = re.compile(r"^(?:\d{1,2}:)?\d{2}:\d{2}$")
KILLER_RE = re.compile(
    r"^(?:[A-Za-z0-9_.-]+/)*(?:test_[^/]+|[^/]+_test)\.py::"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\[.*\])?$"
)
ESCAPED_NEWLINE = chr(92) + "n"


class RenderError(ValueError):
    """Raised when a packet cannot be made safe to submit."""


def _canonical(path: Path) -> Path:
    """Resolve a path without requiring its final component to exist."""

    return path.expanduser().resolve(strict=False)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _shell_literal(value: str) -> str:
    """Return a shell literal suitable for an assignment or argument."""

    return shlex.quote(value)


def _require_absolute(name: str, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise RenderError(f"{name} must be absolute: {value!r}")
    if any(part in {".", ".."} for part in path.parts):
        raise RenderError(f"{name} must be canonical: {value!r}")
    return _canonical(path)


def _validate_sha(value: str) -> None:
    if not SHA_RE.fullmatch(value):
        raise RenderError("--sha must be exactly 40 hexadecimal characters")


def _validate_rel_path(value: str, label: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise RenderError(f"{label} must be a relative path without '..': {value!r}")


def _load_mutants(path: Path) -> List[Dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderError(f"malformed mutant spec: {path}: {exc}") from exc
    rows = payload if isinstance(payload, list) else payload.get("mutants") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RenderError("malformed mutant spec: expected a list or {mutants: list}")
    if not 3 <= len(rows) <= 6:
        raise RenderError("mutant spec must contain 3-6 mutants")
    result: List[Dict[str, str]] = []
    for row in rows:
        if isinstance(row, dict):
            result.append({key: str(row.get(key, "")) for key in ("id", "target", "old", "new", "killer")})
    if len(result) != len(rows):
        raise RenderError("malformed mutant spec: every row must be an object")
    seen = set()
    for row in result:
        if not all(row[key] for key in ("id", "target", "old", "new", "killer")):
            raise RenderError("each mutant needs id, target, old, new, and killer")
        target = PurePosixPath(row["target"])
        if target.is_absolute() or ".." in target.parts or row["id"] in seen:
            raise RenderError(f"invalid mutant: {row['id']!r}")
        seen.add(row["id"])
        if ESCAPED_NEWLINE in row["old"] or ESCAPED_NEWLINE in row["new"]:
            raise RenderError(f"mutant {row['id']!r} contains literal escaped newline text")
        killer = row["killer"]
        if not KILLER_RE.fullmatch(killer) or Path(killer.split("::", 1)[0]).is_absolute():
            raise RenderError(
                f"mutant {row['id']!r} killer must be reconstructed test_x.py::name form: {killer!r}"
            )
    return result


def _validate_with_mutant_helper(helper: Path, spec: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(helper), "validate", "--spec", str(spec)],
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if "future feature annotations is not defined" in detail:
            # Cannon login nodes still expose Python 3.6.  ``mutants.py`` is
            # validated again in the Slurm Python environment; this local
            # structural validation keeps packet rendering fail-closed when
            # the login interpreter cannot parse its future import.
            _load_mutants(spec)
            return
        raise RenderError(f"mutants.py rejected spec: {detail}")


def _stage_scripts(repo_root: Path, sha: str, stage_root: Path) -> None:
    """Extract only ``scripts/cannon`` from the exact commit using stdlib tarfile."""

    try:
        archive = subprocess.Popen(
            ["git", "-C", str(repo_root), "archive", "--format=tar", sha, "scripts/cannon"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise RenderError(f"cannot execute git archive: {exc}") from exc
    assert archive.stdout is not None
    assert archive.stderr is not None
    try:
        with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
            for member in tar:
                target = _canonical(stage_root / member.name)
                if not _inside(target, _canonical(stage_root)):
                    raise RenderError(f"git archive member escapes stage: {member.name!r}")
                tar.extract(member, path=stage_root, set_attrs=True)
    finally:
        archive.stdout.close()
    stderr = archive.stderr.read().decode(errors="replace")
    rc = archive.wait()
    if rc:
        raise RenderError(f"git archive failed for {sha}: {stderr.strip()}")


def _validate_run_root(run_root: Path, facility_prefix: Path) -> None:
    if not _inside(run_root, facility_prefix):
        raise RenderError(f"run root escapes facility prefix: {run_root}")


def _render_body(
    *,
    sha: str,
    job_name: str,
    run_root: Path,
    task_root: Path,
    facility_prefix: Path,
    repo_url: str,
    uv_bin: str,
    shared_env_prefix: str,
    tests: Sequence[str],
    mutant_spec: Optional[Path],
    account: str,
    wall_time: str,
) -> str:
    short_sha = sha[:12]
    export_values = {
        "HEAD_SHA": sha,
        "JOB_NAME": job_name,
        "TASK_ROOT": str(task_root),
        "RUN_ROOT": str(run_root),
        "DRY_RUN": "0",
        "PYTHON": "/usr/bin/python3",
        "SHARED_ENV_PREFIX": shared_env_prefix,
    }
    export_spec = ",".join(f"{key}={value}" for key, value in export_values.items())
    lines = [
        "#!/usr/bin/env bash",
        "# One-shot quick verification packet rendered by render_verify_packet.py.",
        "#SBATCH --partition=test",
        f"#SBATCH --account={account}",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=8G",
        f"#SBATCH --time={wall_time}",
        f"#SBATCH --output={run_root}/slurm-%j.out",
        f"#SBATCH --error={run_root}/slurm-%j.err",
        f"#SBATCH --export={export_spec}",
        "",
        "set -Eeuo pipefail",
        f"HEAD_SHA={_shell_literal(sha)}",
        f"JOB_NAME={_shell_literal(job_name)}",
        f"TASK_ROOT={_shell_literal(str(task_root))}",
        f"RUN_ROOT={_shell_literal(str(run_root))}",
        'DRY_RUN="${DRY_RUN:-0}"',
        'PYTHON="${PYTHON:-/usr/bin/python3}"',
        f"REPO_URL={_shell_literal(repo_url)}",
        f"FACILITY_RUN_PREFIX={_shell_literal(str(facility_prefix))}",
        f"UV_BIN={_shell_literal(uv_bin)}",
        f"SHARED_ENV_PREFIX={_shell_literal(shared_env_prefix)}",
        'WORK_ROOT="/scratch/${USER:?USER required}/${SLURM_JOB_ID:?SLURM_JOB_ID required}"',
        "",
        '[[ "${SLURM_JOB_PARTITION:-}" == test ]] || { echo "PRECHECK_FAILED wrong partition" >&2; exit 2; }',
        '[[ "${HEAD_SHA}" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "PRECHECK_FAILED bad head" >&2; exit 2; }',
        '[[ "${JOB_NAME}" == *"${HEAD_SHA:0:12}"* ]] || { echo "PRECHECK_FAILED job name missing head" >&2; exit 2; }',
        '[[ -x "${TASK_ROOT}/scripts/cannon/verify.sh" ]] || { echo "PRECHECK_FAILED verify.sh missing" >&2; exit 2; }',
        '[[ -f "${TASK_ROOT}/scripts/cannon/shared-env-policy.json" ]] || { echo "PRECHECK_FAILED sidecar missing" >&2; exit 2; }',
        '[[ "${RUN_ROOT}" == "${FACILITY_RUN_PREFIX}" || "${RUN_ROOT}" == "${FACILITY_RUN_PREFIX}"/* ]] || { echo "PRECHECK_FAILED run root escape" >&2; exit 2; }',
    ]
    if mutant_spec is not None:
        lines.append(f'[[ -f "{mutant_spec}" ]] || {{ echo "PRECHECK_FAILED mutant spec missing" >&2; exit 2; }}')
    lines.extend(
        [
            "",
            'if [[ "${DRY_RUN}" == 1 ]]; then',
            '  printf "DRY_EXECUTE PASS head=%s task_root=%s run_root=%s work_root=%s\\n" "$HEAD_SHA" "$TASK_ROOT" "$RUN_ROOT" "$WORK_ROOT"',
            "  exit 0",
            "fi",
            "",
            'mkdir -p -- "${WORK_ROOT}"',
            'cd "${TASK_ROOT}"',
            "export USER WORK_ROOT FACILITY_RUN_PREFIX SHARED_ENV_PREFIX UV_BIN RUN_ROOT REPO_URL",
            'exec "${TASK_ROOT}/scripts/cannon/verify.sh" \\',
            '  --sha "$HEAD_SHA" \\',
            '  --run-root "$RUN_ROOT"',
        ]
    )
    for test_path in tests:
        lines[-1] += " \\\n  --test " + _shell_literal(test_path)
    lines[-1] += " " + "\\"
    lines.extend([
        "  --mode quick \\",
        "  --typecheck on \\",
        "  --parallel serial",
    ])
    if mutant_spec is not None:
        lines[-1] += " \\\n  --mutants " + _shell_literal(str(mutant_spec))
    return "\n".join(lines) + "\n"


def _dry_execute(body_path: Path, task_root: Path, run_root: Path, mutant_spec: Optional[Path]) -> None:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "USER": "packet-renderer",
        "SLURM_JOB_ID": "12345",
        "SLURM_JOB_PARTITION": "test",
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["/usr/bin/env", "-i", *[f"{key}={value}" for key, value in env.items()], "/bin/bash", str(body_path)],
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        detail = (result.stdout + result.stderr).strip()
        raise RenderError(f"env -i dry-execute rejected WORK_ROOT/path expansion: {detail}")
    expected = f"task_root={task_root} run_root={run_root} work_root=/scratch/packet-renderer/12345"
    if expected not in result.stdout:
        raise RenderError("env -i dry-execute did not expand WORK_ROOT and packet paths")
    if mutant_spec is not None and not mutant_spec.is_file():
        raise RenderError("mutant spec disappeared before dry-execute")


def _validate_body(body: str, body_path: Path, task_root: Path, run_root: Path, mutant_spec: Optional[Path]) -> None:
    if r"\$" in body:
        raise RenderError("rendered body contains a literal backslash-dollar")
    if "DRY_RUN=1" in body:
        raise RenderError("rendered body leaves DRY_RUN enabled")
    result = subprocess.run(
        ["/bin/bash", "-n", str(body_path)],
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RenderError(f"bash -n rejected rendered body: {result.stderr.strip()}")
    _dry_execute(body_path, task_root, run_root, mutant_spec)


def _write_manifest(packet_dir: Path) -> Path:
    manifest = packet_dir / "packet-manifest.sha256"
    files = sorted(path for path in packet_dir.rglob("*") if path.is_file() and path != manifest)
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(packet_dir).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def render_packet(
    *,
    sha: str,
    tests: Sequence[str],
    run_root: str,
    packet_dir: str,
    repo_root: Optional[str] = None,
    mutants: Optional[str] = None,
    job_name: str = "verify",
    wall_time: str = "00:45:00",
    facility_run_prefix: Optional[str] = None,
    shared_env_prefix: Optional[str] = None,
    uv_bin: str = "/n/home10/rhu/.local/bin/uv",
    repo_url: str = "https://github.com/YizhongHu/TPEN.git",
    account: str = "kozinsky_lab",
) -> Path:
    """Render one packet and return its fresh directory."""

    _validate_sha(sha)
    if not tests:
        raise RenderError("at least one --test path is required")
    for test_path in tests:
        _validate_rel_path(test_path, "test path")
    if not TIME_RE.fullmatch(wall_time):
        raise RenderError(f"invalid --time: {wall_time!r}")
    if not JOB_NAME_RE.fullmatch(job_name):
        raise RenderError("--job-name may contain only letters, digits, '.', '_' and '-'")
    if sha[:12] not in job_name:
        job_name = f"{job_name}-{sha[:12]}"
    if not JOB_NAME_RE.fullmatch(account):
        raise RenderError("--account contains unsupported characters")
    if facility_run_prefix is None:
        raise RenderError("--facility-run-prefix is required")
    if shared_env_prefix is None:
        raise RenderError("--shared-env-prefix is required")
    facility = _require_absolute("facility run prefix", facility_run_prefix)
    run = _require_absolute("run root", run_root)
    packet = _require_absolute("packet directory", packet_dir)
    _validate_run_root(run, facility)
    shared = _require_absolute("shared env prefix", shared_env_prefix)
    if packet.exists() or packet.is_symlink():
        raise RenderError(f"refusing existing packet directory: {packet}")
    if packet == Path(packet.anchor):
        raise RenderError("refusing packet directory at filesystem root")
    packet.mkdir(parents=True)
    run.mkdir(parents=True, exist_ok=True)
    repo = _canonical(Path(repo_root)) if repo_root else _canonical(Path(__file__).parents[2])
    stage = packet / "stage"
    stage.mkdir()
    _stage_scripts(repo, sha, stage)
    staged_helper = stage / "scripts/cannon/mutants.py"
    spec_dest: Optional[Path] = None
    rows: List[Dict[str, str]] = []
    if mutants is not None:
        if not Path(mutants).is_absolute():
            raise RenderError("mutant spec path must be absolute")
        source_spec = _canonical(Path(mutants))
        rows = _load_mutants(source_spec)
        spec_dest = stage / "mutant-spec.json"
        spec_dest.write_bytes(source_spec.read_bytes())
        _validate_with_mutant_helper(staged_helper, spec_dest)
    task_root = _canonical(stage)
    body = _render_body(
        sha=sha,
        job_name=job_name,
        run_root=run,
        task_root=task_root,
        facility_prefix=facility,
        repo_url=repo_url,
        uv_bin=uv_bin,
        shared_env_prefix=shared.as_posix(),
        tests=tests,
        mutant_spec=spec_dest,
        account=account,
        wall_time=wall_time,
    )
    sbatch = packet / "verify.sbatch"
    sbatch.write_text(body, encoding="utf-8")
    sbatch.chmod(0o750)
    submission_manifest = packet / "submission.manifest"
    submission_manifest.write_text(
        "\n".join(
            [
                f"HEAD_SHA={sha}",
                f"JOB_NAME={job_name}",
                f"TASK_ROOT={task_root}",
                f"RUN_ROOT={run}",
                "DRY_RUN=0",
                "PYTHON=/usr/bin/python3",
                f"SHARED_ENV_PREFIX={shared}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _validate_body(body, sbatch, task_root, run, spec_dest)
    manifest = _write_manifest(packet)
    print(f"READY packet={packet}")
    print(f"SBATCH_COMMAND=sbatch {sbatch}")
    print('HEADROOM_GATE=squeue -u "$USER" -h -p test | wc -l')
    print(f"MANIFEST={manifest}")
    return packet


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--test", action="append", dest="tests", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--facility-run-prefix", required=True)
    parser.add_argument("--shared-env-prefix", required=True)
    parser.add_argument("--mutants")
    parser.add_argument("--repo-root")
    parser.add_argument("--job-name", default="verify")
    parser.add_argument("--time", dest="wall_time", default="00:45:00")
    parser.add_argument("--uv-bin", default="/n/home10/rhu/.local/bin/uv")
    parser.add_argument("--repo-url", default="https://github.com/YizhongHu/TPEN.git")
    parser.add_argument("--account", default="kozinsky_lab")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = _parser().parse_args(argv)
        render_packet(
            sha=args.sha,
            tests=args.tests,
            run_root=args.run_root,
            packet_dir=args.packet_dir,
            repo_root=args.repo_root,
            mutants=args.mutants,
            job_name=args.job_name,
            wall_time=args.wall_time,
            facility_run_prefix=args.facility_run_prefix,
            shared_env_prefix=args.shared_env_prefix,
            uv_bin=args.uv_bin,
            repo_url=args.repo_url,
            account=args.account,
        )
    except RenderError as exc:
        print(f"REJECTED {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
