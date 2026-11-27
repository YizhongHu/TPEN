"""Independent literal-recipe and concrete-binding delta probes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from experiments.atomistic.soap_pilot import composer, p3_render as p3
from experiments.atomistic.soap_pilot.approved_recipes import APPROVED_RECIPES
from experiments.atomistic.stage1.launch import command_for_row

SOURCE = '7b88f833babcb82148c8f975a08b53970838be6a'
ROOT = Path(__file__).resolve().parents[4]


def _execution_state(tmp_path: Path, *, kind: str = "train") -> dict[str, str]:
    cache = tmp_path / "uv-cache"
    cache.mkdir(exist_ok=True)
    return {
        "UV_CACHE_DIR": str(cache),
        "APPROVAL_ID": "soap-review-approval",
        "EXPECTED_QUEUE": "capacity" if kind == "train" else "debug",
    }


def _caller_packet(row, *, state: dict[str, str], python: str) -> dict[str, object]:
    """Build a concrete packet as a caller, not by copying P3's result."""

    root = Path(row["results_root"]).resolve()
    run_dir = root / row["experiment_name"] / row["sector"] / row["run_id"]
    snapshot = root / "inputs" / f"row-{row['row_id']}.json"
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(payload)
    return {
        "env": {
            "MODE": row["kind"],
            "SOURCE_SHA": row["source_sha"],
            "CHECKOUT_ROOT": str(ROOT.resolve()),
            "UV_CACHE_DIR": state["UV_CACHE_DIR"],
            "RESULTS_ROOT": str(root),
            "SELECTED_ROW": str(snapshot),
            "APPROVAL_ID": state["APPROVAL_ID"],
            "EXPECTED_QUEUE": state["EXPECTED_QUEUE"],
        },
        "argv": command_for_row(row, python=python, checkout_root=str(ROOT)),
        "selected_row_sha256": hashlib.sha256(payload).hexdigest(),
        "output": {
            "run_dir": str(run_dir),
            "inputs": str(root / "inputs"),
            "checkpoint_dir": str(run_dir / "checkpoints"),
            "log": str(root / "logs" / f"row-{row['row_id']}-${{PBS_JOBID}}.log"),
            "status": str(root / "logs" / f"row-{row['row_id']}-${{PBS_JOBID}}.status"),
        },
    }

@pytest.mark.parametrize('profile', (*composer.R2_PROFILES, *composer.TARGET_PROFILES), ids=lambda p: p.name)
def test_literal_train_matches_composer(profile):
    config, _ = composer._materialize_train(profile, results_root='/review/results', run_id='review-run', source_sha=SOURCE, device='cpu')
    actual = p3._normalise_recipe(config, results_root='/review/results', run_id='review-run')
    assert p3.recipe_diff(APPROVED_RECIPES[profile.name]['config'], actual) == {}


def test_literal_eval_matches_composer():
    config, _ = composer._materialize_eval(results_root='/review/results', run_id='review-run', checkpoint_dir='/review/parent/checkpoints/step_000100', checkpoint_file='/review/parent/checkpoints/step_000100/model.pt', source_sha=SOURCE, device='cpu')
    actual = p3._normalise_recipe(config, results_root='/review/results', run_id='review-run', parent_run_dir='/review/parent')
    assert p3.recipe_diff(APPROVED_RECIPES['eval-soap-100']['config'], actual) == {}


@pytest.mark.parametrize('field', ['row_file', 'env', 'argv', 'destination', 'placeholder'])
def test_concrete_binding_handoff_refuses_foreign_state(tmp_path, field):
    """A supplied packet must match independent row-bound expectations."""
    train = p3.render_r2_train_packet(output_dir=tmp_path/'inputs', results_root=str(tmp_path), source_sha=SOURCE, device='cpu')['r2-soap-100']
    state = _execution_state(tmp_path)
    admitted = p3.admit_row_and_command(
        train.row, python='/usr/bin/python3', checkout_root=str(ROOT),
        execution_state=state,
    )
    supplied = deepcopy(admitted.packet)
    if field == 'row_file':
        supplied['env']['SELECTED_ROW'] = str(tmp_path / 'inputs' / 'foreign-row.json')
    elif field == 'env':
        supplied['env']['SOURCE_SHA'] = '0' * 40
    elif field == 'argv':
        supplied['argv'] = '/usr/bin/python3 /foreign/driver.py --unapproved'
    elif field == 'destination':
        supplied['output']['run_dir'] = str(tmp_path/'foreign')
    else:
        supplied['env']['UV_CACHE_DIR'] = '<UV_CACHE_DIR>'
    with pytest.raises(p3.P3RenderError, match='binding|packet|argv|env|output'):
        p3.admit_row_and_command(
            train.row, python='/usr/bin/python3', checkout_root=str(ROOT),
            packet=supplied, execution_state=state,
        )


def test_concrete_binding_handoff_accepts_valid_supplied_packet(tmp_path):
    """A concrete caller packet with real PBS values is admitted."""
    train = p3.render_r2_train_packet(output_dir=tmp_path/'inputs', results_root=str(tmp_path), source_sha=SOURCE, device='cpu')['r2-soap-100']
    state = _execution_state(tmp_path)
    supplied = _caller_packet(train.row, state=state, python='/usr/bin/python3')
    admitted = p3.admit_row_and_command(
        train.row, python='/usr/bin/python3', checkout_root=str(ROOT), packet=supplied,
        execution_state=state,
    )
    assert admitted.packet == supplied


def test_packet_audit_uses_literal_mode_expectation_not_rendered_packet(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A rendered train packet cannot redefine the literal mode policy."""

    train = p3.render_r2_train_packet(
        output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"
    )["r2-soap-100"]
    rendered = p3.render_row_packet(
        train.row, python="/usr/bin/python3", checkout_root=str(ROOT),
        execution_state=_execution_state(tmp_path),
    )
    monkeypatch.setitem(p3.PACKET_EXPECTATIONS["env"]["MODE_BY_KIND"], "train", "foreign-mode")

    with pytest.raises(p3.P3RenderError, match="packet|argv|recipe|enumerated"):
        p3.admit_row_and_command(
            train.row,
            python="/usr/bin/python3",
            checkout_root=str(ROOT),
            packet=rendered, execution_state=_execution_state(tmp_path),
        )


def test_admission_without_packet_audits_the_rendered_packet(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """The implicit render path is audited even when callers omit ``packet``."""

    train = p3.render_r2_train_packet(
        output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"
    )["r2-soap-100"]
    state = _execution_state(tmp_path)
    original_render = p3.render_row_packet

    def drifted_render(*args, **kwargs):
        packet = original_render(*args, **kwargs)
        packet["env"]["EXPECTED_QUEUE"] = "foreign-queue"
        return packet

    monkeypatch.setattr(p3, "render_row_packet", drifted_render)

    with pytest.raises(p3.P3RenderError, match="packet|EXPECTED_QUEUE|recipe|enumerated"):
        p3.admit_row_and_command(
            train.row, python="/usr/bin/python3", checkout_root=str(ROOT),
            execution_state=state,
        )
