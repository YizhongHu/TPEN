"""Round-11 controls for CUDA recipe binding and snapshot containment."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.atomistic.soap_pilot import p3_render as p3
from experiments.atomistic.soap_pilot.composer import (
    R2_PROFILES,
    TARGET_PROFILES,
    render_profile,
)
from experiments.atomistic.soap_pilot.p3_render import (
    P3RenderError,
    admit_row_and_command,
    finalize_eval_row,
    render_eval_template,
    render_r2_train_packet,
    render_row_packet,
)
from experiments.atomistic.soap_pilot.rows import CheckpointEvidence, make_parent_join, row_id


SOURCE = "7b88f833babcb82148c8f975a08b53970838be6a"
ROOT = Path(__file__).resolve().parents[4]
ALL_TRAIN_PROFILES = (*R2_PROFILES, *TARGET_PROFILES)


def _execution_state(root: Path, *, kind: str = "train") -> dict[str, str]:
    cache = root / "uv-cache"
    cache.mkdir(parents=True, exist_ok=True)
    return {
        "UV_CACHE_DIR": str(cache),
        "APPROVAL_ID": "soap-review-approval",
        "EXPECTED_QUEUE": "capacity" if kind == "train" else "debug",
    }


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("profile", ALL_TRAIN_PROFILES, ids=lambda profile: profile.name)
def test_all_train_profiles_render_and_admit_for_cpu_and_cuda(
    tmp_path: Path, device: str, profile,
) -> None:
    """Every independent train recipe is renderable and gateway-admissible."""

    root = tmp_path / device
    composed = render_profile(
        profile,
        output_dir=root / "inputs",
        results_root=str(root),
        source_sha=SOURCE,
        device=device,
    )
    admitted = admit_row_and_command(
        composed.row,
        python="/usr/bin/python3",
        checkout_root=str(ROOT),
        execution_state=_execution_state(root),
    )
    assert f"experiments/atomistic/stage1/{'train.py'}" in admitted.command
    assert admitted.row["device"] == device


def _finalized_eval(tmp_path: Path, *, device: str):
    root = tmp_path / device
    train = render_r2_train_packet(
        output_dir=root / "inputs",
        results_root=str(root),
        source_sha=SOURCE,
        device=device,
        profiles=(R2_PROFILES[1],),
        execution_state=_execution_state(root),
    )["r2-soap-100"]
    parent = make_parent_join(campaign_id="soap-r2", invocation_id="soap100", row=train.row)
    template = render_eval_template(
        output_dir=root / "inputs",
        results_root=str(root),
        source_sha=SOURCE,
        device=device,
    )
    checkpoint_dir = (
        root / train.row["experiment_name"] / train.row["sector"]
        / train.row["run_id"] / "checkpoints" / "step_000100"
    )
    checkpoint_dir.mkdir(parents=True)
    model = checkpoint_dir / "model.pt"
    model.write_bytes(b"soap100-model")
    complete = checkpoint_dir / "COMPLETE"
    complete.write_text("complete\n", encoding="utf-8")
    publication = CheckpointEvidence(
        100,
        "soap100-publication",
        hashlib.sha256(model.read_bytes()).hexdigest(),
        True,
        {"model.pt": str(model), "COMPLETE": str(complete)},
    )
    row = finalize_eval_row(
        template,
        train_row=train.row,
        publication=publication,
        parent=parent,
        checkpoint_dir=checkpoint_dir,
    )
    return root, row


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_finalized_eval_renders_and_admits_for_cpu_and_cuda(tmp_path: Path, device: str) -> None:
    """The finalized eval recipe remains admissible for both row devices."""

    root, row = _finalized_eval(tmp_path, device=device)
    admitted = admit_row_and_command(
        row,
        python="/usr/bin/python3",
        checkout_root=str(ROOT),
        execution_state=_execution_state(root, kind="eval"),
    )
    assert "experiments/atomistic/stage1/eval.py" in admitted.command
    assert admitted.row["device"] == device


def test_probe_device_mismatch_refuses_against_validated_cuda_row(tmp_path: Path) -> None:
    """A known allocator probe cannot silently target another device."""

    composed = render_profile(
        R2_PROFILES[1],
        output_dir=tmp_path / "inputs",
        results_root=str(tmp_path),
        source_sha=SOURCE,
        device="cuda",
    )
    config = deepcopy(composed.rendered.values)
    resource = next(callback for callback in config["callbacks"] if callback["_target_"] == "tpen.callback.ResourceUsage")
    resource["allocator_probe"]["device"] = "cpu"
    with pytest.raises(P3RenderError, match="allocator probe device|row device"):
        p3._audit_row_recipe(composed.row, config)


def test_unrelated_callback_mutation_refuses_with_atomic_callback_comparison(tmp_path: Path) -> None:
    """Only the recognized probe leaf is device-bound; other callback drift refuses."""

    composed = render_profile(
        R2_PROFILES[1],
        output_dir=tmp_path / "inputs",
        results_root=str(tmp_path),
        source_sha=SOURCE,
        device="cuda",
    )
    config = deepcopy(composed.rendered.values)
    timing = next(callback for callback in config["callbacks"] if callback["_target_"] == "tpen.callback.RunTiming")
    timing["accelerator_synchronize"] = False
    with pytest.raises(P3RenderError, match="unenumerated recipe diff|callbacks"):
        p3._audit_row_recipe(composed.row, config)


def test_snapshot_rejects_inputs_symlink_to_sibling_without_writing_there(tmp_path: Path) -> None:
    """An escaping inputs ancestor is refused before the first snapshot write."""

    source = render_profile(
        R2_PROFILES[0],
        output_dir=tmp_path / "source" / "inputs",
        results_root=str(tmp_path / "source"),
        source_sha=SOURCE,
        device="cuda",
    )
    root = tmp_path / "results"
    root.mkdir()
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    (root / "inputs").symlink_to(sibling, target_is_directory=True)
    row = dict(source.row)
    row["results_root"] = str(root)
    row["row_id"] = row_id(row)
    with pytest.raises(P3RenderError, match="inputs directory escapes|snapshot destination escapes"):
        render_row_packet(
            row,
            python="/usr/bin/python3",
            checkout_root=str(ROOT),
            execution_state=_execution_state(root),
        )
    assert list(sibling.iterdir()) == []


def test_snapshot_contained_positive_case_keeps_content_and_hash_binding(tmp_path: Path) -> None:
    """A normal inputs directory still receives the canonical row snapshot."""

    root = tmp_path / "results"
    composed = render_profile(
        R2_PROFILES[0],
        output_dir=root / "inputs",
        results_root=str(root),
        source_sha=SOURCE,
        device="cpu",
    )
    packet = render_row_packet(
        composed.row,
        python="/usr/bin/python3",
        checkout_root=str(ROOT),
        execution_state=_execution_state(root),
    )
    snapshot = Path(packet["env"]["SELECTED_ROW"])
    assert snapshot.resolve().is_relative_to(root.resolve())
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == packet["selected_row_sha256"]


def test_wrapper_selected_row_and_approval_checks_run_for_genuine_cuda_row(tmp_path: Path) -> None:
    """Invoke the unchanged wrapper validation functions in a write-free subprocess."""

    root = tmp_path / "results"
    composed = render_r2_train_packet(
        output_dir=root / "inputs",
        results_root=str(root),
        source_sha=SOURCE,
        device="cuda",
        profiles=(R2_PROFILES[1],),
        execution_state=_execution_state(root),
    )["r2-soap-100"]
    packet = render_row_packet(
        composed.row,
        python="/usr/bin/python3",
        checkout_root=str(ROOT),
        execution_state=_execution_state(root),
    )
    wrapper = ROOT / "experiments/atomistic/stage1/polaris/pbs_stage1.sh"
    wrapper_text = wrapper.read_text(encoding="utf-8")
    assert wrapper_text.endswith('main "$@"\n')
    functions = tmp_path / "pbs-functions.sh"
    functions.write_text(wrapper_text.removesuffix('main "$@"\n'), encoding="utf-8")
    env = {
        **dict(__import__("os").environ),
        "MODE": "train",
        "SOURCE_SHA": SOURCE,
        "RESULTS_ROOT": str(root),
        "SELECTED_ROW": packet["env"]["SELECTED_ROW"],
        "APPROVAL_ID": "soap-review-approval",
        "PYBIN": sys.executable,
        "USER": "p3-review",
    }
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; validate_selected_row_or_die "$SELECTED_ROW" train; require_approval_or_die', "bash", str(functions)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
