"""Independent literal-recipe and concrete-binding delta probes."""
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.atomistic.soap_pilot import composer, p3_render as p3
from experiments.atomistic.soap_pilot.approved_recipes import APPROVED_RECIPES

SOURCE = '7b88f833babcb82148c8f975a08b53970838be6a'
ROOT = Path(__file__).resolve().parents[4]

@pytest.mark.parametrize('profile', (*composer.R2_PROFILES, *composer.TARGET_PROFILES), ids=lambda p: p.name)
def test_literal_train_matches_composer(profile):
    config, _ = composer._materialize_train(profile, results_root='/review/results', run_id='review-run', source_sha=SOURCE, device='cpu')
    actual = p3._normalise_recipe(config, results_root='/review/results', run_id='review-run')
    assert p3.recipe_diff(APPROVED_RECIPES[profile.name]['config'], actual) == {}


def test_literal_eval_matches_composer():
    config, _ = composer._materialize_eval(results_root='/review/results', run_id='review-run', checkpoint_dir='/review/parent/checkpoints/step_000100', checkpoint_file='/review/parent/checkpoints/step_000100/model.pt', source_sha=SOURCE, device='cpu')
    actual = p3._normalise_recipe(config, results_root='/review/results', run_id='review-run', parent_run_dir='/review/parent')
    assert p3.recipe_diff(APPROVED_RECIPES['eval-soap-100']['config'], actual) == {}


@pytest.mark.parametrize('field', ['env', 'argv', 'destination'])
def test_concrete_binding_handoff_refuses_foreign_state(tmp_path, monkeypatch, field):
    """Inject supplied state at the only current concrete-binding handoff.

    There is no public supplied-packet argument. This boundary probe demonstrates
    missing validation; it does not claim a public packet-input API exists.
    """
    train = p3.render_r2_train_packet(output_dir=tmp_path/'inputs', results_root=str(tmp_path), source_sha=SOURCE, device='cpu')['r2-soap-100']
    command = p3.command_for_row(train.row, python='/usr/bin/python3', checkout_root=str(ROOT))
    supplied = deepcopy(p3._runner_bindings(train.row, command=command))
    if field == 'env':
        supplied['env']['SOURCE_SHA'] = '0' * 40
    elif field == 'argv':
        supplied['argv'] = '/usr/bin/python3 /foreign/driver.py --unapproved'
    else:
        supplied['output']['run_dir'] = str(tmp_path/'foreign')
    monkeypatch.setattr(p3, '_runner_bindings', lambda row, *, command: supplied)
    with pytest.raises(p3.P3RenderError, match='binding|packet|argv|env|output'):
        p3.admit_row_and_command(train.row, python='/usr/bin/python3', checkout_root=str(ROOT))
