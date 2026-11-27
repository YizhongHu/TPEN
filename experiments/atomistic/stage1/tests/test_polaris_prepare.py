"""Static contract tests for the Polaris Stage-1 preparation job."""

from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "polaris" / "pbs_prepare.sh"


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


def test_prepare_job_never_submits_or_transfers_from_the_allocation() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "qsub" not in text
    assert "scp" not in text
    assert "RESULTS_PREFIX=\"/eagle/HetRxnEnergy/rhu/\"" in text
