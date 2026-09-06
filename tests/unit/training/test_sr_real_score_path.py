"""SR consumes what `TPENWaveFunction` actually EMITS, not a synthetic stand-in.

Every other test of this engine builds its `ScoreUpdateInput` from hand-made
score blocks, which is right for pinning the algebra but blind to one thing: a
mismatch between what the engine consumes and what the wavefunction really
produces.  Layout order, sample-shape flattening, dtype, detachment, and the
sign convention are all agreements between two modules, and a synthetic input
satisfies the consumer's half of every one of them by construction.

So these tests drive the real provider,
:meth:`TPENWaveFunction.evaluate_materialized_parameter_score_request`, and feed
its output straight into the update method.  No trainer is involved -- the
trainer's own integration is a later slice -- so this establishes that the seam
is consumable at THIS layer, where a mismatch is cheap to find.

It found one: at two electrons the seam refused the whole request, because 16
of `build_tiny_spenn`'s 39 parameters have no autograd path into ``logabs`` at
an even electron count and ``allow_unused=False`` could not tell that apart from
a parameter no code path consumes.  That is fixed at the seam, and this file now
drives the TWO-ELECTRON fixture directly -- the shape of the helium target --
rather than the three-electron stand-in it used while the blocker stood.  See
`test_the_pair_model_scores_exact_zeros_for_its_inactive_parameters`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.helpers.hooke_models import (
    INACTIVE_PAIR_PARAMETERS,
    build_tiny_spenn,
    tiny_pair_batch,
)
from tests.helpers.score_reachability import disconnected_parameter_names
from tests.helpers.sr_dense_oracle import energy_gradient, sr_direction
from tpen.data.batch import ElectronBatch, ParameterScoreForwardPacket
from tpen.nn import MaterializedParameterScoreRequest
from tpen.training.qgt import DampingPolicy
from tpen.training.score_geometry import ScoreConventions
from tpen.training.sr import SRPolicy, StochasticReconfigurationUpdate
from tpen.training.update import ModelParameterBinding, ScoreUpdateInput
from tpen.training.vmc import compute_vmc_objective

LEARNING_RATE = 1.0e-3
SOLVE_TOLERANCE = 1.0e-9


def _pair_model():
    """Build the real two-electron smoke fixture -- the helium-shaped model.

    Promoted from the three-electron stand-in this file used while the score
    seam refused even electron counts. Its 16 structurally inactive parameters
    are now served exact-zero score columns, so SR consumes it unchanged, and
    the tests below therefore exercise the particle count the target programme
    actually has.
    """

    torch.manual_seed(0)
    return build_tiny_spenn()


def _batch(n_walkers: int = 6) -> ElectronBatch:
    """Return a flat-sample-shape TWO-electron batch from the owning helper."""

    return tiny_pair_batch(n_walkers)


def _odd_batch(n_walkers: int = 6, *, seed: int = 5) -> ElectronBatch:
    """Return a flat-sample-shape three-electron batch for the parity check."""

    generator = torch.Generator().manual_seed(seed)
    return ElectronBatch(
        positions=torch.randn(n_walkers, 3, 3, generator=generator, dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0, 1.0]] * n_walkers, dtype=torch.float64),
    )


def _flat_gradients(model) -> np.ndarray:
    """Flatten ``.grad`` over the model's parameters, zero-filling ``None``.

    A parameter with no path into ``logabs`` gets no ``.grad`` from
    ``backward()`` at all, and its true gradient is exactly zero, so the
    zero-fill is the value rather than a convenience.
    """

    return np.concatenate(
        [
            (
                np.zeros(parameter.numel())
                if parameter.grad is None
                else parameter.grad.detach().numpy().reshape(-1)
            )
            for parameter in model.parameters()
        ]
    )


def _energies(n_walkers: int, *, seed: int = 17) -> torch.Tensor:
    """Deterministic finite local energies.

    These are synthetic on purpose. The seam under test here is the SCORE path;
    where the energies come from is the Hamiltonian's business and would only
    add an unrelated failure source.
    """

    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_walkers, generator=generator, dtype=torch.float64)


def _emit(model, batch: ElectronBatch, *, chunk_size: int | None = None):
    """Ask the real provider for raw parameter-score blocks."""

    packet = model.evaluate_materialized_parameter_score_request(
        request=MaterializedParameterScoreRequest(chunk_size=chunk_size),
        batch=batch,
    )
    assert isinstance(packet, ParameterScoreForwardPacket)
    return packet


def _independent_flatten(packet) -> np.ndarray:
    """Flatten emitted blocks to ``[B, P]`` WITHOUT the module under test.

    `flatten_parameter_score_blocks` is the subject here, so using it would
    make the oracle comparison circular. Re-deriving the column order from the
    layout is the point: if the engine's ordering ever diverges from the
    layout's own slot order, this disagrees.
    """

    scores = packet.parameter_scores
    n_samples = int(np.prod(scores.sample_shape)) if scores.sample_shape else 1
    columns = [
        block.detach().numpy().reshape(n_samples, slot.numel)
        for slot, block in zip(scores.layout.slots, scores.blocks, strict=True)
    ]
    return np.hstack(columns)


def _method(model, *, solve_space: str = "parameter", relative: float = 1.0e-2):
    """Build an SR method over the model's live parameters."""

    parameters = tuple(model.parameters())
    policy = SRPolicy(
        solve_space=solve_space,
        damping=DampingPolicy(absolute=0.0, relative=relative, minimum=1.0e-12),
        learning_rate=LEARNING_RATE,
    )
    return StochasticReconfigurationUpdate(
        torch.optim.SGD(parameters, lr=LEARNING_RATE),
        model_parameters=ModelParameterBinding(parameters=parameters),
        policy=policy,
        conventions=ScoreConventions(solve_dtype=torch.float64),
    )


def _score_input(model, batch, packet, energies, *, step: int = 0) -> ScoreUpdateInput:
    """Assemble the typed input from a REAL emitted packet.

    The batch is passed explicitly rather than stashed at module scope: these
    tests must not depend on execution order, and pytest-xdist would make a
    shared global genuinely wrong rather than merely untidy.
    """

    return ScoreUpdateInput(
        batch=batch,
        wavefunction=packet.output,
        local_energy=energies,
        step=step,
        parameter_scores=packet.parameter_scores,
        parameter_binding=model.parameter_binding,
    )


def test_the_emitted_blocks_are_the_uncentered_scores_the_engine_assumes() -> None:
    """The provider emits ``d log|psi| / d theta`` per sample, raw and uncentered.

    The engine's entire geometry rests on that convention. Recomputing the same
    quantity with an independent per-sample autograd loop checks the emitted
    payload against its stated meaning rather than against the engine.

    SCOPE LIMIT, stated because the reference is not independent everywhere.
    For the 16 parameters in `INACTIVE_PAIR_PARAMETERS` this reference applies
    the SAME zero-substitution rule as the seam, so for those columns the
    comparison is vacuous by construction. Those columns are witnessed instead
    by `test_the_pair_model_scores_exact_zeros_for_its_inactive_parameters`,
    which checks them against an autograd reachability probe and a literal name
    set. The 23 remaining columns are genuinely independently derived here.
    """

    model = _pair_model()
    batch = _batch()
    packet = _emit(model, batch)
    emitted = _independent_flatten(packet)

    parameters = model.parameter_binding.parameters
    with torch.enable_grad():
        logabs = model(batch).logabs
        rows = []
        for index in range(int(logabs.numel())):
            grads = torch.autograd.grad(
                logabs.reshape(-1)[index],
                parameters,
                retain_graph=index + 1 < int(logabs.numel()),
                allow_unused=True,
            )
            rows.append(
                np.concatenate(
                    [
                        (
                            np.zeros(parameter.numel())
                            if gradient is None
                            else gradient.detach().numpy().reshape(-1)
                        )
                        for parameter, gradient in zip(parameters, grads, strict=True)
                    ]
                )
            )
    reference = np.vstack(rows)

    assert emitted.shape == reference.shape
    np.testing.assert_allclose(emitted, reference, rtol=1.0e-12, atol=1.0e-12)
    # Raw means UNCENTERED: the column means are generally nonzero, and the
    # engine is the thing that centers them.
    assert np.abs(emitted.mean(axis=0)).max() > 0.0
    # Detached: a live graph here would leak into the optimizer's parameters.
    assert not any(block.requires_grad for block in packet.parameter_scores.blocks)


def test_sr_consumes_a_real_emitted_packet_and_matches_the_oracle() -> None:
    """End to end at the engine layer: real emission in, oracle-checked step out.

    This is the check that a synthetic `ScoreUpdateInput` cannot make. It uses
    the model's own `parameter_binding` and the provider's own blocks, so a
    disagreement in layout order, sample flattening, dtype, or sign convention
    between the two modules shows up here.
    """

    model = _pair_model()
    batch = _batch()
    packet = _emit(model, batch)
    energies = _energies(int(packet.output.logabs.numel()))
    before = [p.detach().clone() for p in model.parameters()]

    method = _method(model)
    result = method.update(_score_input(model, batch, packet, energies))

    assert result.applied is True
    expected = sr_direction(
        _independent_flatten(packet),
        energies.numpy(),
        absolute=0.0,
        relative=1.0e-2,
    )
    displacement = np.concatenate(
        [
            (b - p.detach()).numpy().reshape(-1)
            for b, p in zip(before, model.parameters(), strict=True)
        ]
    )
    np.testing.assert_allclose(
        displacement,
        LEARNING_RATE * expected.direction,
        rtol=SOLVE_TOLERANCE,
        atol=SOLVE_TOLERANCE,
    )
    assert result.grad_norm == pytest.approx(
        float(np.linalg.norm(expected.gradient)), rel=1.0e-9
    )


def test_minsr_agrees_with_dense_sr_on_real_emitted_scores() -> None:
    """The two routes agree on real emissions, not only on synthetic matrices."""

    batch = _batch()
    results = {}
    for space in ("parameter", "sample"):
        model = _pair_model()
        packet = _emit(model, batch)
        energies = _energies(int(packet.output.logabs.numel()))
        method = _method(model, solve_space=space)
        method.update(_score_input(model, batch, packet, energies))
        results[space] = np.concatenate(
            [p.detach().numpy().reshape(-1) for p in model.parameters()]
        )
        assert method.last_telemetry.diagnostics.space == space

    np.testing.assert_allclose(
        results["parameter"], results["sample"], rtol=1.0e-8, atol=1.0e-8
    )


def test_euclidean_limit_on_real_scores_matches_the_real_objective_gradient() -> None:
    """With damping dominant, the step aligns with the model's own VMC gradient.

    Both sides now come from the same real model: the scores from the provider,
    the reference from autograd through `compute_vmc_objective`. That closes the
    convention loop end to end rather than assuming the emitted sign.
    """

    model = _pair_model()
    batch = _batch()
    packet = _emit(model, batch)
    energies = _energies(int(packet.output.logabs.numel()))

    with torch.enable_grad():
        logabs = model(batch).logabs
        compute_vmc_objective(logabs, energies).loss.backward()
    reference = _flat_gradients(model)
    for parameter in model.parameters():
        parameter.grad = None

    method = _method(model, relative=1.0e10)
    method.update(_score_input(model, batch, packet, energies))
    direction = _flat_gradients(model)

    np.testing.assert_allclose(
        direction / np.linalg.norm(direction),
        reference / np.linalg.norm(reference),
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    # And the oracle agrees with autograd on the emitted scores, so the
    # reference itself is not taken on trust.
    np.testing.assert_allclose(
        energy_gradient(_independent_flatten(packet), energies.numpy()),
        reference,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_chunked_emission_is_consumable_and_gives_the_same_step() -> None:
    """The provider has two implementations; the engine must consume both.

    `chunk_size=None` materializes one gradient per sample, a nonzero
    `chunk_size` uses batched VJPs. They are different code paths and only one
    of them is exercised by default.
    """

    batch = _batch()
    steps = []
    for chunk_size in (None, 2):
        model = _pair_model()
        packet = _emit(model, batch, chunk_size=chunk_size)
        energies = _energies(int(packet.output.logabs.numel()))
        method = _method(model)
        assert method.update(_score_input(model, batch, packet, energies)).applied is True
        steps.append(
            np.concatenate([p.detach().numpy().reshape(-1) for p in model.parameters()])
        )

    np.testing.assert_allclose(steps[0], steps[1], rtol=1.0e-9, atol=1.0e-9)


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_the_pair_model_scores_exact_zeros_for_its_inactive_parameters(
    chunk_size: int | None,
) -> None:
    """WAS A PINNED BLOCKER; NOW ASSERTS THE PROPERTY THAT REPLACED IT.

    This test used to require ``pytest.raises(RuntimeError)``: the seam passed
    ``allow_unused=False`` and refused the entire request because 16 of this
    model's 39 parameters have no autograd path into ``logabs`` at an even
    electron count. Helium is two electrons, so that refusal blocked SR for the
    target programme outright.

    The score of a parameter with no path into ``logabs`` is analytically
    exactly zero, so the seam now substitutes zeros. Three separate things are
    checked here, because the interesting failure mode is a seam that zeroes too
    much rather than one that raises:

    1. Every parameter in the literal `INACTIVE_PAIR_PARAMETERS` set has an
       exactly-zero block -- counted nonzeros, not a tolerance, since any
       nonzero entry would be wrong rather than imprecise.
    2. An INDEPENDENT autograd reachability probe agrees that exactly that set
       is unreachable, so the set is pinned by something other than the seam.
    3. Some other block is nonzero, which a seam that zeroed everything fails.

    ``chunk_size=2`` over four samples splits the batch, so the batched-VJP
    route is exercised across a chunk boundary rather than in one shot.
    """

    model = _pair_model()
    batch = _batch(4)
    packet = _emit(model, batch, chunk_size=chunk_size)
    scores = packet.parameter_scores

    names_by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    blocks_by_name = {
        names_by_identity[id(parameter)]: block
        for parameter, block in zip(
            model.parameter_binding.parameters, scores.blocks, strict=True
        )
    }

    for name in sorted(INACTIVE_PAIR_PARAMETERS):
        block = blocks_by_name[name]
        assert int(torch.count_nonzero(block)) == 0, f"{name} scored nonzero"

    assert disconnected_parameter_names(model, batch) == INACTIVE_PAIR_PARAMETERS
    assert any(
        int(torch.count_nonzero(block)) > 0
        for name, block in blocks_by_name.items()
        if name not in INACTIVE_PAIR_PARAMETERS
    ), "every active block came back zero"


@pytest.mark.parametrize("solve_space", ["parameter", "sample"])
def test_sr_leaves_the_inactive_pair_parameters_bitwise_unchanged(solve_space: str) -> None:
    """A zero score column must produce a zero update, not a small one.

    Exact-zero scores make the corresponding QGT rows and gradient entries zero,
    so the solve should return exactly zero for those coordinates and the
    optimizer should write nothing. ``rtol=atol=0.0`` is the point: a 1e-18 drift
    would mean the dead coordinates are being driven by damping or solver noise,
    which over a long run is a random walk in a direction the physics does not
    define.

    Run through both solve spaces, because dense SR and sample-space minSR reach
    the same update by different linear algebra and only one is the default.
    """

    model = _pair_model()
    batch = _batch()
    packet = _emit(model, batch)
    energies = _energies(int(packet.output.logabs.numel()))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    method = _method(model, solve_space=solve_space)
    assert method.update(_score_input(model, batch, packet, energies)).applied is True

    for name, parameter in model.named_parameters():
        if name in INACTIVE_PAIR_PARAMETERS:
            torch.testing.assert_close(parameter.detach(), before[name], rtol=0.0, atol=0.0)
    moved = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    ]
    assert moved, "the update applied but moved nothing, so the check above is vacuous"


def test_the_inactive_set_is_a_parity_property_not_a_two_electron_one() -> None:
    """The disconnection follows the ELECTRON COUNT'S PARITY, not the number 2.

    The original report described these weights as tensor paths that "carry
    nothing at two electrons". That gloss is wrong and would license a fix keyed
    to ``n == 2``. What actually happens is that the order-1 OUTPUT subtree
    reaches ``logabs`` only through the odd-electron Pfaffian padding block, so
    it is disconnected whenever there is no padding -- at every even count, in a
    one-layer model.

    ONE model construction, TWO electron counts. Two different models would
    leave any difference attributable to the models; varying only the batch
    makes the electron count the sole difference, which is the claim. The odd
    arm is also why the strict CI guard in
    `tests/unit/nn/test_tpen_wavefunction_parameter_scores.py` sits at odd n:
    there, and only there, an unreachable parameter is unambiguously a defect.
    """

    assert disconnected_parameter_names(_pair_model(), _batch()) == INACTIVE_PAIR_PARAMETERS
    assert disconnected_parameter_names(_pair_model(), _odd_batch()) == frozenset()
