"""Round-5 review probe for the one-bound-update-method lifecycle (PR 524).

Round 4's R4-2 pin (`test_r4_2_a_runner_driven_adam_resume_is_bitwise_identical`)
aligns its resume-side ``RunContext.cfg`` with the save-side cfg via
``_aligned_context``, because the restore hash gate compares ``context.cfg``
against the checkpoint manifest. The round-5 question named for attack: does
that alignment MASK a restore that should have been refused?

This control answers it empirically rather than by reading: it drives the SAME
runner resume path with an UNALIGNED context (the empty cfg that
`make_run_context` ships) and requires the hash gate's refusal. Green here
proves the gate is live on exactly the path the R4-2 pin exercises -- so the
pin's alignment is a load-bearing fixture that satisfies the gate with matching
content, not a fixture that disabled it.

It is also, deliberately, a second-order pin on the runner itself: the refusal
can only arrive if `Train.run` actually calls the restore machinery, so a
mutant that silently skips the ``train_resume`` branch turns this test red
(DID NOT RAISE) alongside the R4-2 pin.
"""

from __future__ import annotations

import pytest
import torch

from tpen.runner import Train
from tpen.training.trainer import VMCTrainer
from tests.helpers.hooke_models import (
    build_tiny_hamiltonian_terms,
    build_tiny_spenn,
)
from tests.helpers.run_context import make_run_context
from tests.unit.training.test_update_binding import (
    _fit,
    _FixedSampler,
    _save_resume_checkpoint,
)

LEARNING_RATE = 0.01


def test_r5_the_hash_gate_is_live_on_the_runner_resume_path(tmp_path) -> None:
    """An unaligned resume context is refused at the hash gate, through `Train.run`.

    The checkpoint is written through the direct trainer path (the same
    fixtures the binding tests use), so this test exercises the runner only on
    the RESUME side -- the side whose cfg alignment is under question. The
    expected message is the gate's own "current config is missing model for
    restore": `make_run_context` ships an empty cfg, `checkpoint_hashes`
    returns ``None`` for every component of an empty cfg, and `_verify_hash`
    refuses a ``None`` current hash before any mutable load.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sampler = _FixedSampler()
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    _fit(trainer, model, optimizer, sampler=sampler)
    checkpoint = _save_resume_checkpoint(
        tmp_path, model=model, optimizer=optimizer, trainer=trainer, sampler=sampler
    )

    torch.manual_seed(7)
    with pytest.raises(ValueError, match="current config is missing model for restore"):
        Train(
            model=build_tiny_spenn(),
            sampler=_FixedSampler(),
            hamiltonian_terms=build_tiny_hamiltonian_terms(),
            optimizer=lambda params: torch.optim.Adam(params, lr=LEARNING_RATE),
            trainer=VMCTrainer(max_steps=2, log_every_n_steps=1),
            load={"mode": "train_resume", "path": str(checkpoint)},
        ).run(make_run_context(tmp_path / "unaligned-resume"))
