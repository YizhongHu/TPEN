"""The ``update_envelope`` seam works on the REAL path, not just in scaffolds.

An unused seam has an unverified contract. Before the A8 Gaussian arm is built on
``TPENLayer(update_envelope=...)``, that seam has to be shown carrying a real
:class:`~tpen.data.real.Update` and a real
:class:`~tpen.nn.context.TPENForwardContext` through a live forward -- not
because the signature looks wrong, but because a signature is a promise and
nothing had ever collected on it.

What was already there, and why it was not enough
-------------------------------------------------
``tests/unit/nn/test_tpen_layer_scaffold.py`` does exercise the seam, including
once with ``GaussianCoordinateEnvelope``. But those tests build the layer from
toy stubs -- ``TwoPathMixing``, ``SumPathAggregation``, a hand-written
``Feature`` -- and one of them passes ``TPENForwardContext(batch=object())``.
They establish that ``TPENLayer`` calls what it is given, in the declared order.
They cannot establish that the production stack produces an ``Update`` the
envelope can consume, or that a real batch reaches it.

So this module instantiates the ACTUAL control configuration through Hydra and
runs a real forward. Same emitting path the study will run.

The three things a call-recording test would miss
-------------------------------------------------
1. That the recorded argument is an ``Update`` and not something else that
   happens to have blocks.
2. That the context's batch is the one that was passed in, rather than a
   default or a stale object.
3. That the envelope's RETURN VALUE reaches the output. A seam that is faithfully
   called and whose result is then discarded would satisfy every "was it
   called?" assertion while doing nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hydra")

from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data.batch import ElectronBatch
from tpen.data.real import Update
from tpen.nn import GaussianCoordinateEnvelope
from tpen.nn.context import TPENForwardContext

CONTROL_CONFIG = Path("experiments/atomistic/he-importance/configs/train.yaml")


def _control_model():
    """Instantiate the control config's model exactly as a run would.

    Two details that a first attempt got wrong, recorded because both are the
    kind of thing that makes a live-path test report a false finding:

    ``TPENWaveFunction`` does NOT expose ``.layers``. Constructed layers are
    wrapped into a ``TPENStack`` and live at ``model.stack.layers`` -- see
    ``tpen/nn/tpen_wave_function.py``, "the layers always live in
    ``self.stack``". Reaching for ``model.layers`` raises ``AttributeError``
    from ``nn.Module.__getattr__``, which looks exactly like "the seam is not
    wired up" rather than "you used the wrong accessor".

    ``instantiate(cfg.model)`` builds float32 parameters. The config's
    ``runtime.dtype: float64`` is applied by the RUNNER, not by model
    construction, so a float64 batch against a freshly instantiated model dies
    with ``mat1 and mat2 must have the same dtype``. The dtype is read from the
    config here rather than hardcoded, so the test follows the config if the
    study ever changes precision.
    """

    cfg = OmegaConf.load(CONTROL_CONFIG)
    OmegaConf.resolve(cfg)
    dtype = getattr(torch, str(cfg.runtime.dtype))
    return instantiate(cfg.model).to(dtype), dtype


def _control_layer(model):
    """Return the single TPEN layer, via the stack that actually owns it."""

    return model.stack.layers[0]


def _helium_batch(dtype, n_walkers: int = 4):
    """A real helium batch: two electrons, one nucleus of charge 2 at the origin."""

    generator = torch.Generator().manual_seed(0)
    positions = torch.randn(n_walkers, 2, 3, generator=generator, dtype=dtype)
    return ElectronBatch(
        positions=positions,
        nuclear_positions=torch.zeros(1, 3, dtype=dtype),
        nuclear_charges=torch.tensor([2.0], dtype=dtype),
        spins=torch.tensor([[1, -1]] * n_walkers, dtype=dtype),
    )


class _SpyEnvelope(GaussianCoordinateEnvelope):
    """The real envelope, recording exactly what the seam handed it."""

    def __init__(self, records: list, **kwargs) -> None:
        super().__init__(**kwargs)
        self._records = records

    def forward_impl(self, features, context):
        self._records.append(
            {
                "type": type(features),
                "is_update": isinstance(features, Update),
                "context_type": type(context),
                "batch_positions": getattr(getattr(context, "batch", None), "positions", None),
                "n_blocks": len(features.blocks),
            }
        )
        return super().forward_impl(features, context)


class TestTheControlModelBuildsAndRuns:
    def test_the_control_config_instantiates(self) -> None:
        """If this fails, nothing else in the module means anything."""

        model, dtype = _control_model()
        assert model is not None
        assert len(model.stack.layers) == 1

    def test_a_real_forward_produces_finite_output(self) -> None:
        model, dtype = _control_model()
        output = model(_helium_batch(dtype))
        assert torch.isfinite(output.logabs).all()


class TestTheSeamCarriesWhatItPromises:
    def test_the_envelope_is_reached_by_a_real_forward(self) -> None:
        records: list = []
        model, dtype = _control_model()
        batch = _helium_batch(dtype)
        _control_layer(model).update_envelope = _SpyEnvelope(records, sigma=1.0)

        model(batch)

        assert len(records) == 1, "the update_envelope seam was not reached by a real forward"

    def test_the_seam_hands_over_a_real_update(self) -> None:
        """Not a Feature, not a bare tensor -- an Update with populated blocks."""

        records: list = []
        model, dtype = _control_model()
        _control_layer(model).update_envelope = _SpyEnvelope(records, sigma=1.0)
        model(_helium_batch(dtype))

        record = records[0]
        assert record["is_update"], f"seam handed over {record['type']}, not an Update"
        assert record["n_blocks"] >= 2

    def test_the_seam_hands_over_the_real_batch(self) -> None:
        """The context must carry the batch that was passed to the model."""

        records: list = []
        model, dtype = _control_model()
        batch = _helium_batch(dtype)
        _control_layer(model).update_envelope = _SpyEnvelope(records, sigma=1.0)
        model(batch)

        seen = records[0]["batch_positions"]
        assert records[0]["context_type"] is TPENForwardContext
        assert seen is not None, "context carried no batch positions"
        assert torch.equal(seen, batch.positions)


class TestTheEnvelopeActuallyChangesTheOutput:
    """A seam that is called and then discarded would pass every test above."""

    def test_installing_the_gate_changes_the_log_amplitude(self) -> None:
        """Same weights, same batch, envelope the only difference.

        The envelope is attached to an ALREADY BUILT model rather than
        constructing two models, because two constructions would differ by
        random initialization and the comparison would prove nothing.
        """

        model, dtype = _control_model()
        batch = _helium_batch(dtype)

        without = model(batch).logabs.detach().clone()
        _control_layer(model).update_envelope = GaussianCoordinateEnvelope(sigma=1.0)
        with_gate = model(batch).logabs.detach().clone()

        assert not torch.allclose(without, with_gate), (
            "the update_envelope result never reached the output; the seam is called "
            "but its return value is discarded"
        )

    def test_a_wide_gate_moves_the_output_less_than_a_narrow_one(self) -> None:
        """The direction of the effect is right, not merely that there is one.

        A wide Gaussian is closer to 1 everywhere, so it should perturb the
        ungated result less. Without this, any bug that merely scrambled the
        update would satisfy the test above.
        """

        model, dtype = _control_model()
        batch = _helium_batch(dtype)
        without = model(batch).logabs.detach().clone()

        _control_layer(model).update_envelope = GaussianCoordinateEnvelope(sigma=100.0)
        wide = (model(batch).logabs.detach() - without).abs().sum()

        _control_layer(model).update_envelope = GaussianCoordinateEnvelope(sigma=0.25)
        narrow = (model(batch).logabs.detach() - without).abs().sum()

        assert wide < narrow

    def test_gradients_reach_the_model_through_the_gate(self) -> None:
        """The gate sits inside the wavefunction, so training must see through it."""

        model, dtype = _control_model()
        _control_layer(model).update_envelope = GaussianCoordinateEnvelope(sigma=1.0)
        model(_helium_batch(dtype)).logabs.sum().backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads, "no parameter received a gradient through the gated layer"
        assert any(g.abs().sum() > 0 for g in grads)
