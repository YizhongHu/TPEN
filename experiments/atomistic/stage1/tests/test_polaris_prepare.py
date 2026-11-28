"""Static contract tests for the Polaris Stage-1 preparation jobs and runbook."""

from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "polaris" / "pbs_prepare.sh"
EVAL_SCRIPT = Path(__file__).parents[1] / "polaris" / "pbs_select_eval.sh"
PATHS_SCRIPT = Path(__file__).parents[1] / "polaris" / "pbs_paths.sh"
RUNBOOK = Path(__file__).parents[1] / "runbooks" / "polaris-he-h2.md"


def test_prepare_job_is_short_debug_cpu_allocation() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "#PBS -A HetRxnEnergy" in text
    assert "#PBS -q debug" in text
    assert "#PBS -l select=1:system=polaris:ncpus=4" in text
    assert "#PBS -l walltime=00:20:00" in text
    assert "#PBS -l filesystems=home:eagle" in text


def test_prepare_job_owns_clone_uv_prime_and_row_rendering() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'git clone "$SOURCE_REPO" "$CHECKOUT_ROOT"' in text
    assert 'git -C "$CHECKOUT_ROOT" checkout --detach "$SOURCE_SHA"' in text
    assert 'rev-parse HEAD)" == "$SOURCE_SHA"' in text
    assert '"$PYBIN" -m uv lock --check --project "$CHECKOUT_ROOT"' in text
    assert '"$PYBIN" -m uv sync --project "$CHECKOUT_ROOT" --extra cu128 --locked' in text
    assert 'experiments.atomistic.stage1.plan' in text
    assert 'command_for_row' in text
    assert 'render-receipt.json' in text
    assert 'sha256sum "$WRAPPER_PATH"' in text
    assert 'derive_smoke_request' in text
    assert 'SMOKE_SELECTED_ROW' in text
    assert 'pbs_select_eval.sh.sha256' in text
    assert 'pbs_paths.sh.sha256' in text


def test_prepare_job_never_submits_or_transfers_from_the_allocation() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "qsub" not in text
    assert "scp" not in text
    assert 'RESULTS_PREFIX="/eagle/HetRxnEnergy/rhu/"' in text
    assert 'source "$STAGE1_PATHS_SH"' in text
    assert 'STAGE1_PATHS_SHA256' in text
    assert 'realpath -m -- "$1"' in PATHS_SCRIPT.read_text(encoding="utf-8")


def _run_spooled_preamble(script: Path, environ: dict[str, str]) -> subprocess.CompletedProcess[str]:
    text = script.read_text(encoding="utf-8")
    prefix = text.split("require_value SOURCE_REPO", 1)[0]
    env = {"PATH": "/usr/bin:/bin", **environ}
    return subprocess.run(
        ["bash", "-c", prefix],
        cwd=script.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_prepare_spooling_requires_and_uses_explicit_helper_path(tmp_path: Path) -> None:
    """PBS spooling must not make a sibling lookup the implicit dependency."""
    spooled = tmp_path / "pbs-spool"
    spooled.mkdir()
    spooled_script = spooled / "pbs_prepare.sh"
    spooled_script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")

    missing = _run_spooled_preamble(spooled_script, {})
    assert missing.returncode != 0
    assert "STAGE1_PATHS_SH is required" in missing.stderr

    helper = tmp_path / "inputs" / "pbs_paths.sh"
    helper.parent.mkdir()
    helper.write_text(PATHS_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
    supplied = _run_spooled_preamble(
        spooled_script,
        {"STAGE1_PATHS_SH": str(helper), "STAGE1_PATHS_SHA256": helper_sha},
    )
    assert supplied.returncode == 0, supplied.stderr


def _run_guard(path: Path, allowed_root: Path) -> subprocess.CompletedProcess[str]:
    text = SCRIPT.read_text(encoding="utf-8")
    prefix = text.split("require_value SOURCE_REPO", 1)[0]
    prefix = prefix.replace(
        'RESULTS_PREFIX="/eagle/HetRxnEnergy/rhu/"',
        f"RESULTS_PREFIX={shlex.quote(str(allowed_root))}",
    )
    command = (
        f'{prefix}\n'
        'RESULTS_PREFIX_CANONICAL="$(canonicalize_path "$RESULTS_PREFIX")"\n'
        'assert_results_path "$RESULTS_PREFIX_CANONICAL" "$1" TEST_PATH "$RESULTS_PREFIX"\n'
    )
    return subprocess.run(
        ["bash", "-c", command, "guard", str(path)],
        env={
            "PATH": "/usr/bin:/bin",
            "STAGE1_PATHS_SH": str(PATHS_SCRIPT),
            "STAGE1_PATHS_SHA256": hashlib.sha256(PATHS_SCRIPT.read_bytes()).hexdigest(),
        },
        text=True,
        capture_output=True,
        check=False,
    )


def test_prepare_path_guard_accepts_alias_and_canonical_roots_and_refuses_escapes(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "eagle" / "projects" / "HetRxnEnergy" / "rhu"
    allowed.mkdir(parents=True)
    alias = tmp_path / "eagle-alias"
    alias.symlink_to(allowed, target_is_directory=True)

    assert _run_guard(allowed / "run", allowed).returncode == 0
    assert _run_guard(alias / "run", allowed).returncode == 0
    assert _run_guard(tmp_path / "outside", allowed).returncode != 0
    assert _run_guard(allowed / ".." / "outside", allowed).returncode != 0
    assert _run_guard(allowed.parent / "rhu-other" / "run", allowed).returncode != 0


def test_prepare_uses_canonical_containment_not_lexical_prefix() -> None:
    text = PATHS_SCRIPT.read_text(encoding="utf-8")
    assert 'canonical="$(canonicalize_path "$path")"' in text
    assert '"$canonical" == "$allowed_canonical"/*' in text


def test_eval_selection_job_owns_eval_materialization() -> None:
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    assert "#PBS -q debug" in text
    assert "select_eval" in text
    assert '"$PRIME_ENV/bin/python" -m experiments.atomistic.stage1.select_eval' in text
    assert 'sha256sum "$SELECTED_EVAL"' in text
    assert 'source "$STAGE1_PATHS_SH"' in text
    assert 'STAGE1_PATHS_SHA256' in text


def test_runbook_login_blocks_do_not_execute_compute_commands() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    command_sequence = text.split("## Exact command sequence", 1)[1]
    code_blocks = re.findall(r"```bash\n(.*?)\n```", command_sequence, flags=re.DOTALL)
    assert code_blocks
    forbidden = re.compile(r"(?m)^\s*(?:\S*/)?(?:python|uv)(?:\s|$)|git\s+clone|\brender(?:\s|$)")
    assert not [block for block in code_blocks if forbidden.search(block)]
    assert not [block for block in code_blocks if re.search(r"(?m)^\s*mkdir(?:\s|$)", block)]
    assert not [block for block in code_blocks if re.search(r"(?m)^\s*sha256sum(?:\s|$)", block)]
