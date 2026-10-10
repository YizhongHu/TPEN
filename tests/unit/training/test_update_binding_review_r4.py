"""Round-4 review finding R4-2: the runner-driven restore-carrier axis.

The reviewer MEASURED this gap rather than inferring it. Its M4 mutant hands
`restore_checkpoint_with_events` a FRESH factory build while `fit` keeps the
run's real carrier -- precisely the lifecycle class this slice exists to
remove, state restored into an object the loop never uses -- and **no test in
`tests/unit` detected the PAYLOAD**, while a runtime witness proved the mutated
path had executed. (Round 4's M4-full arm did measure `failures=1`: its own
witness instrument, an `open(..., "a")` probe, tripped the durable-append
census. Round 4 carried that qualifier and an earlier version of this
docstring dropped it, saying "every test passed" -- corrected in round 5,
R5-2.) The property was not unprotected in the repository; it was
unprotected in the suite that every verification receipt in this PR ran at the
time. `tests/integration/training/test_train_runner.py` does catch this
payload -- measured rather than inferred, after this docstring first asserted
it unmeasured: round 6 ran that file green at both the head and the stack
base, and red under this exact mutant, failing on the diverged-parameters
assert in `test_resume_reproduces_the_uninterrupted_run_bitwise`.

Why the existing coverage missed it, from the reviewer's analysis:

- `test_r1_4_*` pins the construction COUNT on the resumed runner path, not
  the identity of the carrier that restore is handed.
- The exactness resume tests enumerated here -- `test_adam_moments_*`,
  `test_r1_3_*`, `test_the_default_adapter_survives_restore_as_one_instance`
  -- call `restore_checkpoint` DIRECTLY, bypassing `Train.run`, so the
  runner's own wiring of restore to the carrier is never exercised. The
  enumeration is closed and describes the coverage this finding was made
  against: `test_r4_2_*` below goes through `Train.run`, and is the exception
  that closing the gap created.
- Block-NG's runner-driven bitwise resume rides a stateless SGD carrier, so a
  lost optimizer state dict is numerically invisible there.

Adam is the discriminating choice: its moments ARE the state that a swapped
carrier silently loses, so exact trajectory equality through the real runner
is what makes the swap observable.
"""

from __future__ import annotations

import torch

from tpen.runner import Train
from tpen.training.trainer import VMCTrainer
from tests.helpers.hooke_models import (
    build_tiny_hamiltonian_terms,
    build_tiny_spenn,
)
from tests.helpers.run_context import make_run_context
from tests.unit.training.test_update_binding import (
    _checkpoint_context,
    _FixedSampler,
    _save_resume_checkpoint,
)

LEARNING_RATE = 0.01


def _aligned_context(tmp_path, name):
    """A real `RunContext` whose cfg matches the one the checkpoint was saved under.

    `make_run_context` ships an empty cfg, and restore hashes `context.cfg`
    against the manifest, so an unaligned cfg is refused at the hash gate
    before the behaviour under test is ever reached. `RunContext` is a plain
    dataclass, so the two sides are aligned post-construction rather than by
    hand-building a second context -- the same approach round 1's probe used.
    """

    context = make_run_context(tmp_path / name)
    context.cfg = _checkpoint_context(tmp_path).cfg
    context.source_cfg = context.cfg
    return context


def test_r4_2_a_runner_driven_adam_resume_is_bitwise_identical(tmp_path) -> None:
    """Four steps through `Train.run` equal two, a checkpoint, and two more.

    Both arms go through the REAL runner, which is the whole point: the
    existing exactness tests call `restore_checkpoint` directly and therefore
    cannot see how `Train.run` wires restore to the carrier. Exact equality,
    not a tolerance -- a carrier swap loses Adam's moments outright, and a
    tolerance is precisely what would let that pass.
    """

    terms = build_tiny_hamiltonian_terms()

    def adam_factory(params):
        return torch.optim.Adam(params, lr=LEARNING_RATE)

    # --- Arm A: four steps, uninterrupted, through the runner -------------
    torch.manual_seed(0)
    straight_model = build_tiny_spenn()
    Train(
        model=straight_model,
        sampler=_FixedSampler(),
        hamiltonian_terms=terms,
        optimizer=adam_factory,
        trainer=VMCTrainer(max_steps=4, log_every_n_steps=1),
    ).run(_aligned_context(tmp_path, "straight"))
    expected = [p.detach().clone() for p in straight_model.parameters()]

    # Non-vacuity: two models that never moved would compare equal and pin
    # nothing at all.
    torch.manual_seed(0)
    untrained = [p.detach().clone() for p in build_tiny_spenn().parameters()]
    assert any(
        not torch.equal(before, after)
        for before, after in zip(untrained, expected, strict=True)
    ), "the uninterrupted arm did not move any parameter"

    # --- Arm B: two steps, checkpoint, then resume through the runner -----
    torch.manual_seed(0)
    model = build_tiny_spenn()
    sampler = _FixedSampler()
    trainer = VMCTrainer(max_steps=2, log_every_n_steps=1)
    built = []

    def capturing_factory(params):
        built.append(torch.optim.Adam(params, lr=LEARNING_RATE))
        return built[-1]

    Train(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=capturing_factory,
        trainer=trainer,
    ).run(_aligned_context(tmp_path, "first-half"))
    assert trainer.next_iteration == 2
    assert len(built) == 1, "the runner must build exactly one carrier per run"

    checkpoint = _save_resume_checkpoint(
        tmp_path, model=model, optimizer=built[0], trainer=trainer, sampler=sampler
    )

    # A genuinely fresh run: different seed, new objects. Everything it knows
    # about the first two steps has to arrive through the checkpoint, and that
    # includes Adam's moments.
    torch.manual_seed(999)
    resumed_model = build_tiny_spenn()
    Train(
        model=resumed_model,
        sampler=_FixedSampler(),
        hamiltonian_terms=terms,
        optimizer=adam_factory,
        trainer=VMCTrainer(max_steps=4, log_every_n_steps=1),
        load={"mode": "train_resume", "path": str(checkpoint)},
    ).run(_aligned_context(tmp_path, "resumed"))

    for reference, resumed in zip(expected, resumed_model.parameters(), strict=True):
        assert torch.equal(reference, resumed.detach()), (
            "the runner-driven resume diverged from the uninterrupted run; a "
            "carrier reaching restore that the loop does not then step would "
            "lose Adam's moments exactly this way"
        )
