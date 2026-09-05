"""Tests for the top-level neural-network namespace."""

from __future__ import annotations

import tpen.nn as spenn_nn
from tpen.nn.activation import (
    ChannelActivationAxes,
    ChannelPreservingMLPActivation,
    OrderMLPLayout,
    OrderMLPSpec,
)
from tpen.nn.coordinate_envelopes import GaussianCoordinateEnvelope, GaussianDecayGate
from tpen.nn.initialization import SeededLinear, TorchInitializer
from tpen.nn.tpen_stack import TPENStack
from tpen.nn.update import ResidualUpdater


def test_spenn_nn_namespace_keeps_baseline_surface() -> None:
    assert spenn_nn.ChannelActivationAxes is ChannelActivationAxes
    assert spenn_nn.ChannelPreservingMLPActivation is ChannelPreservingMLPActivation
    assert spenn_nn.GaussianCoordinateEnvelope is GaussianCoordinateEnvelope
    assert spenn_nn.GaussianDecayGate is GaussianDecayGate
    assert spenn_nn.ResidualUpdater is ResidualUpdater
    assert spenn_nn.SeededLinear is SeededLinear
    assert spenn_nn.TorchInitializer is TorchInitializer
    assert spenn_nn.TPENStack is TPENStack
    assert spenn_nn.OrderMLPLayout is OrderMLPLayout
    assert spenn_nn.OrderMLPSpec is OrderMLPSpec
    assert not hasattr(spenn_nn, "ActivationByType")
    assert not hasattr(spenn_nn, "ActivationByIrrep")
    assert not hasattr(spenn_nn, "ChannelMappedUpdater")
    assert not hasattr(spenn_nn, "NormGatedUpdater")


def test_replace_updater_is_now_public() -> None:
    """``ReplaceUpdater`` was ADMITTED, and this pin was updated deliberately.

    It used to be asserted absent here, alongside ``NormGatedUpdater`` and
    ``ChannelMappedUpdater``, as one of three experimental updaters kept off
    the public surface. It is no longer experimental: the helium-importance
    study's A8 coordinate admits exactly three feature updates -- ``x + u``,
    ``u``, and ``x + g(R) u`` -- and ``u`` is this class. A config cannot name
    it without the export.

    Nothing about its behaviour changed, only its status. The other two
    assertions above are untouched and must stay: ``NormGatedUpdater`` in
    particular is NOT an A8 arm and must not become reachable as a substitute
    for the Gaussian one, which is a coordinate gate rather than a norm gate.

    Split into its own test rather than flipped in place so that the change is
    visible in the diff as an addition with a reason, instead of a deleted
    line.
    """

    from tpen.nn.update import ReplaceUpdater

    assert spenn_nn.ReplaceUpdater is ReplaceUpdater
    assert "ReplaceUpdater" in spenn_nn.__all__
