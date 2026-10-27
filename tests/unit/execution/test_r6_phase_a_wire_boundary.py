"""Round-6 Phase-A A3: decoder owns malformed CheckpointEvidence failures."""

from pathlib import Path
from types import SimpleNamespace

import pytest


def _saved_evidence(tmp_path: Path):
    import torch
    from omegaconf import OmegaConf
    from tpen.checkpoint import TrainResume, save_checkpoint
    from tpen.checkpoint.reference import CheckpointRef
    from tpen.execution.recovery import CheckpointEvidence

    root = tmp_path / "run"
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(state_dict=lambda: {"next_iteration": 1, "completed_updates": 1})
    sampler = SimpleNamespace(mcmc_state_dict=lambda: {"walker": 1})
    context = SimpleNamespace(
        cfg=OmegaConf.create({"study": {"name": "r6"}, "trainer": {"max_steps": 4}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id="r6-wire", extra={}),
        run_dir=root,
    )
    checkpoint = save_checkpoint(
        output_dir=root / "checkpoints",
        next_iteration=1,
        completed_updates=1,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
        payload=TrainResume(),
        context=context,
    )
    ref = CheckpointRef.from_directory(checkpoint)
    return CheckpointEvidence(ref=ref, science_input_digest="c" * 64, target_iterations=4)


@pytest.mark.parametrize("value", [None, [], "not-a-mapping", 7], ids=["none", "list", "string", "scalar"])
def test_r6_checkpoint_evidence_bad_wire_values_raise_decoder_value_error(value):
    from tpen.execution.recovery import CheckpointEvidence

    with pytest.raises(ValueError):
        CheckpointEvidence.from_mapping(value)


def test_r6_checkpoint_evidence_valid_round_trip_and_malformed_mapping(tmp_path):
    from tpen.execution.recovery import CheckpointEvidence

    evidence = _saved_evidence(tmp_path)
    assert CheckpointEvidence.from_mapping(evidence.to_dict()) == evidence
    malformed = evidence.to_dict()
    malformed.pop("science_input_digest")
    with pytest.raises(ValueError):
        CheckpointEvidence.from_mapping(malformed)
