"""Tests for public data-structure contracts."""

from __future__ import annotations

import pytest
import torch

import tpen.data.real as real
from tpen.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from tpen.data.indices import (
    flatten_tuple_indices,
    ordered_tuples,
    ordered_tuple_tensor,
    permute_particle_axis,
    permute_tuple_slots,
    select_tuple_tensor,
    tuple_particle_inputs,
)
from tpen.data.permutation import Permutation


def test_real_submodule_defines_public_tensor_state_surface() -> None:
    assert hasattr(real, "Feature")
    assert hasattr(real, "Interaction")
    assert hasattr(real, "Update")
    assert hasattr(real, "zero_block")


def test_electron_batch_accepts_higher_rank_sample_shape() -> None:
    positions = torch.zeros(2, 3, 4, 5)
    nuclear_positions = torch.zeros(2, 3, 7, 5)
    nuclear_charges = torch.ones(2, 3, 7)

    batch = ElectronBatch(
        positions=positions,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        aux={"tag": "multi"},
    )

    assert batch.validate() is batch
    assert batch.sample_shape == (2, 3)
    assert batch.n_configurations == 6
    assert batch.batch_size == 6
    assert batch.n_electrons == 4
    assert batch.spatial_dim == 5


def test_electron_batch_rejects_mismatched_nuclear_counts() -> None:
    with pytest.raises(ValueError, match="n_nuclei"):
        ElectronBatch(
            positions=torch.zeros(2, 4, 5),
            nuclear_positions=torch.zeros(7, 5),
            nuclear_charges=torch.ones(6),
        )


def test_electron_batch_flatten_samples_preserves_metadata() -> None:
    particle_features = torch.arange(2 * 3 * 4 * 2, dtype=torch.float64).reshape(2, 3, 4, 2)
    batch = ElectronBatch(
        positions=torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5),
        nuclear_positions=torch.zeros(2, 3, 7, 5),
        nuclear_charges=torch.ones(2, 3, 7),
        aux={"tag": "multi", "particle_features": particle_features},
    )

    flat = batch.flatten_samples()

    assert flat.positions.shape == (6, 4, 5)
    assert flat.nuclear_positions is not None and flat.nuclear_positions.shape == (6, 7, 5)
    assert flat.nuclear_charges is not None and flat.nuclear_charges.shape == (6, 7)
    assert flat.aux["tag"] == "multi"
    torch.testing.assert_close(flat.aux["particle_features"], particle_features.reshape(6, 4, 2))


def test_electron_batch_permute_uses_data_particle_axis_helper() -> None:
    positions = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    spins = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], dtype=torch.float64)
    particle_features = torch.arange(2 * 3 * 2, dtype=torch.float64).reshape(2, 3, 2)
    batch = ElectronBatch(positions=positions, spins=spins, aux={"particle_features": particle_features})
    permutation = Permutation((2, 0, 1))

    assert batch.validate() is batch
    permuted = batch.permute(permutation)

    torch.testing.assert_close(permuted.positions, permute_particle_axis(positions, permutation, axis=-2))
    assert permuted.spins is not None
    torch.testing.assert_close(permuted.spins, permute_particle_axis(spins, permutation, axis=-1))
    torch.testing.assert_close(
        permuted.aux["particle_features"],
        permute_particle_axis(particle_features, permutation, axis=-2),
    )


def test_spin_tensors_are_validated_and_preserved() -> None:
    positions = torch.zeros(2, 3, 4, 5)
    spins = torch.tensor([[[1.0, -1.0, 1.0, -1.0]] * 3] * 2)
    batch = ElectronBatch(positions=positions, spins=spins)

    assert batch.spins is not None
    assert batch.spins.shape == (2, 3, 4)
    flat = batch.flatten_samples()
    assert flat.spins is not None
    assert flat.spins.shape == (6, 4)
    assert flat.to(dtype=torch.float32).spins is not None
    assert flat.to(dtype=torch.float32).spins.dtype == torch.float32

    walkers = Walkers(positions=torch.zeros(6, 4, 5), spins=flat.spins)
    assert walkers.validate() is walkers
    assert walkers.spins is not None
    assert torch.equal(walkers.spins, flat.spins)

    with pytest.raises(ValueError, match="exactly"):
        ElectronBatch(positions=torch.zeros(2, 4, 3), spins=torch.zeros(2, 4))
    with pytest.raises(ValueError, match="shape"):
        Walkers(positions=torch.zeros(2, 4, 3), spins=torch.ones(2, 3))


@pytest.mark.parametrize(
    ("nuclear_positions", "nuclear_charges"),
    [
        (torch.zeros(1, 3), None),
        (None, torch.ones(1)),
    ],
)
def test_walkers_reject_partial_nuclear_context(nuclear_positions, nuclear_charges) -> None:
    with pytest.raises(ValueError, match="provided together"):
        Walkers(
            positions=torch.zeros(2, 2, 3),
            nuclear_positions=nuclear_positions,
            nuclear_charges=nuclear_charges,
        )


def test_wavefunction_output_accepts_exact_nodes_and_sample_shapes() -> None:
    logabs = torch.tensor([[0.0, -torch.inf], [-3.0, -4.0]])
    sign = torch.tensor([[1.0, 0.0], [-1.0, 1.0]])
    output = WavefunctionOutput(logabs=logabs, sign=sign)

    assert output.validate(sample_shape=(2, 2)) is output
    assert output.validate(batch_size=4) is output
    WavefunctionOutput(logabs=torch.zeros(3), sign=torch.ones(3)).validate(batch_size=3)

    with pytest.raises(ValueError, match="sample shape"):
        output.validate(sample_shape=(4,))
    with pytest.raises(ValueError, match="batch size"):
        output.validate(batch_size=3)


def test_wavefunction_output_rejects_inconsistent_exact_nodes() -> None:
    with pytest.raises(ValueError, match="exact zeros"):
        WavefunctionOutput(logabs=torch.tensor([0.0]), sign=torch.tensor([0.0]))
    with pytest.raises(ValueError, match="exact zeros"):
        WavefunctionOutput(logabs=torch.tensor([-torch.inf]), sign=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="phase"):
        WavefunctionOutput(logabs=torch.zeros(2), sign=torch.ones(2), phase=torch.zeros(3))


def test_ordered_tuple_helper_generalizes_across_orders() -> None:
    assert ordered_tuples(4, 4)[0] == (0, 1, 2, 3)
    assert len(ordered_tuples(4, 4)) == 24
    assert (0, 0, 0) in ordered_tuples(2, 3, distinct=False)
    assert len(ordered_tuples(2, 3, distinct=False)) == 8


def test_tuple_tensor_helpers_preserve_data_owned_index_conventions() -> None:
    tuples = ordered_tuple_tensor(3, 2)

    assert tuples.tolist() == [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
    torch.testing.assert_close(select_tuple_tensor(tuples, (1,)), tuples[:, 1:])
    torch.testing.assert_close(flatten_tuple_indices(tuples, 3), torch.tensor([1, 2, 3, 5, 6, 7]))


def test_tuple_particle_inputs_builds_raw_ordered_particle_vectors() -> None:
    particles = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    tuples = tuple_particle_inputs(particles, order=2)

    assert tuples.shape == (1, 3, 3, 4)
    torch.testing.assert_close(tuples[0, 0, 2], torch.tensor([1.0, 2.0, 5.0, 6.0]))
    torch.testing.assert_close(tuples[0, 2, 0], torch.tensor([5.0, 6.0, 1.0, 2.0]))


def test_tuple_slot_permutation_reorders_tuple_axes_without_particle_action() -> None:
    tensor = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)

    permuted = permute_tuple_slots(tensor, Permutation((1, 2, 0)), axis_start=0, order=3)

    torch.testing.assert_close(permuted[0, 1, 2], tensor[1, 2, 0])
