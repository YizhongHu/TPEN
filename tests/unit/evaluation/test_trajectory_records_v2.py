"""Contracts for the opt-in ``trajectory_records/v2`` writer/reader.

Companion to ``test_trajectory_records.py`` (v1, unchanged by this file).
These tests build a real one-electron Hooke `SystemDescriptor` -- no nuclear
resource -- so "Hooke records with no fabricated nuclear resource" is proved
by construction, not asserted against a stub.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.batch.walkers import Walkers
from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.evaluation.bundle import EvaluationBundle, GeneratedConfigurations, LocalEnergyValues
from tpen.evaluation.calculators import LocalEnergyCalculator
from tpen.evaluation.generators import TrajectoryMCMCGenerator
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.trajectory_records import (
    TRAJECTORY_RECORD_METADATA_KEYS_V2,
    TRAJECTORY_RECORD_SCHEMA,
    TRAJECTORY_RECORD_SCHEMA_V2,
    TamperedTrajectoryRecordError,
    TrajectoryRecordArtifactV2,
    TrajectoryRecordBatch,
    TrajectoryRecordArtifact,
    TrajectoryRecordStreamWriter,
    inspect_trajectory_record,
)
from tpen.physics.hamiltonian import LocalEnergyResult, NaiveLocalEnergyEvaluator
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.operators import OperatorId
from tpen.physics.potential import HarmonicTrap
from tpen.physics.receipt import ExecutionReceipt, encode_ordered_contribution_sequence
from tpen.physics.system import ParticleSpec, build_system_descriptor
from tpen.sampling.metropolis import MetropolisSampler
from tpen.statistics.trajectory import ObservableTrajectory
from tests.unit.evaluation.test_trajectory_record_inspection import _write_v1_artifact

_TERMS = {"kinetic": KineticEnergy(), "trap": HarmonicTrap(omega=0.5)}
_PARTICLE = ParticleSpec(n_electrons=1, n_spin_up=1, n_spin_down=0, spatial_dim=3)


def _system():
    return build_system_descriptor(_PARTICLE, _TERMS, units="atomic_units")


def _receipt(system):
    return NaiveLocalEnergyEvaluator().describe_execution_receipt(_TERMS, system.system_id())


def _v2_artifact(tmp_path: Path, values: torch.Tensor, *, name: str = "trajectory.csv") -> TrajectoryRecordArtifactV2:
    values = values.to(torch.float64)
    n_draws, n_walkers = values.shape
    positions = torch.zeros(n_draws, n_walkers, 1, 3, dtype=torch.float64)
    system = _system()
    receipt = _receipt(system)

    def record(draw: int) -> TrajectoryRecordBatch:
        return TrajectoryRecordBatch(
            draw_index=torch.full((n_walkers,), draw, dtype=torch.int64),
            walker_index=torch.arange(n_walkers, dtype=torch.int64),
            positions=positions[draw],
            local_energy=values[draw],
            term_energies={"kinetic": values[draw] * 0.5, "trap": values[draw] * 0.5},
            logabs=torch.full_like(values[draw], -2.0),
            sign=torch.ones(n_walkers, dtype=torch.float64),
            finite_mask=torch.isfinite(values[draw]),
        )

    writer = TrajectoryRecordStreamWriter(
        tmp_path / name,
        observable="local_energy",
        n_draws=n_draws,
        n_walkers=n_walkers,
        term_names=("kinetic", "trap"),
        system_descriptor=system,
        execution_receipt=receipt,
        first_draw=record(0),
    )
    for draw in range(1, n_draws):
        writer.append(record(draw))
    return writer.finalize(
        ObservableTrajectory(observable="local_energy", values=values, draw_stride=1, burn_in_draws=0)
    )


def _read_metadata(artifact: TrajectoryRecordArtifactV2) -> dict:
    return json.loads(artifact.metadata_path.read_text(encoding="utf-8"))


def _rewrite_metadata(artifact: TrajectoryRecordArtifactV2, metadata: object) -> None:
    artifact.metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")


FINITE = torch.tensor([[1.0, 2.0], [3.0, 4.0]])


def test_freshly_written_v2_artifact_validates(tmp_path: Path) -> None:
    artifact = _v2_artifact(tmp_path, FINITE)
    artifact.validate()
    assert artifact.system_descriptor.resources == {}, "one-electron Hooke must record no nuclear resource"


def test_v2_sidecar_key_set_equals_declared_constant(tmp_path: Path) -> None:
    artifact = _v2_artifact(tmp_path, FINITE)
    assert len(set(TRAJECTORY_RECORD_METADATA_KEYS_V2)) == len(TRAJECTORY_RECORD_METADATA_KEYS_V2)
    assert set(_read_metadata(artifact)) == set(TRAJECTORY_RECORD_METADATA_KEYS_V2)


def test_v1_and_v2_cannot_both_be_selected(tmp_path: Path) -> None:
    system = _system()
    receipt = _receipt(system)
    draw = TrajectoryRecordBatch(
        draw_index=torch.zeros(1, dtype=torch.int64),
        walker_index=torch.zeros(1, dtype=torch.int64),
        positions=torch.zeros(1, 1, 3, dtype=torch.float64),
        local_energy=torch.tensor([1.0]),
        term_energies={"kinetic": torch.tensor([0.5]), "trap": torch.tensor([0.5])},
        logabs=torch.tensor([-1.0]),
        sign=torch.tensor([1.0]),
        finite_mask=torch.tensor([True]),
    )
    with pytest.raises(ValueError, match="cannot select both"):
        TrajectoryRecordStreamWriter(
            tmp_path / "bad.csv",
            observable="local_energy",
            n_draws=1,
            n_walkers=1,
            term_names=("kinetic", "trap"),
            atomic_configuration=AtomicConfiguration(
                positions=torch.zeros(1, 3, dtype=torch.float64), charges=torch.tensor([1.0])
            ),
            system_descriptor=system,
            execution_receipt=receipt,
            first_draw=draw,
        )


def test_v2_is_selected_only_explicitly_and_default_remains_v1(tmp_path: Path) -> None:
    draw = TrajectoryRecordBatch(
        draw_index=torch.zeros(1, dtype=torch.int64),
        walker_index=torch.zeros(1, dtype=torch.int64),
        positions=torch.zeros(1, 1, 3, dtype=torch.float64),
        local_energy=torch.tensor([1.0]),
        term_energies={"kinetic": torch.tensor([0.5]), "trap": torch.tensor([0.5])},
        logabs=torch.tensor([-1.0]),
        sign=torch.tensor([1.0]),
        finite_mask=torch.tensor([True]),
    )
    artifact = TrajectoryRecordStreamWriter(
        tmp_path / "default.csv",
        observable="local_energy",
        n_draws=1,
        n_walkers=1,
        term_names=("kinetic", "trap"),
        atomic_configuration=AtomicConfiguration(
            positions=torch.zeros(1, 3, dtype=torch.float64),
            charges=torch.ones(1, dtype=torch.float64),
        ),
        first_draw=draw,
    ).finalize(
        ObservableTrajectory(
            observable="local_energy",
            values=torch.tensor([[1.0]], dtype=torch.float64),
            draw_stride=1,
            burn_in_draws=0,
        )
    )
    assert isinstance(artifact, TrajectoryRecordArtifact)
    metadata = artifact.metadata_path.read_text(encoding="utf-8")
    assert f'"schema": "{TRAJECTORY_RECORD_SCHEMA}"' in metadata
    assert TRAJECTORY_RECORD_SCHEMA_V2 not in metadata


def test_inspect_decodes_v2_and_reports_available_physical_context(tmp_path: Path) -> None:
    system = _system()
    artifact = _v2_artifact(tmp_path, FINITE)
    inspection = inspect_trajectory_record(artifact.metadata_path)
    assert inspection.record_schema == TRAJECTORY_RECORD_SCHEMA_V2
    assert inspection.atomic_configuration_id is None
    assert inspection.physical_context.available is True
    assert inspection.physical_context.system_id == system.system_id()
    assert inspection.observable_values_digest.available is True
    assert inspection.observable_values_digest.content_id == artifact.observable_values_content_id
    metadata = _read_metadata(artifact)
    metadata["observable_values_content_id"] = "0" * 64
    _rewrite_metadata(artifact, metadata)
    with pytest.raises(TamperedTrajectoryRecordError, match="observable_values_content_id"):
        inspect_trajectory_record(artifact.metadata_path)


@pytest.mark.parametrize("key", ["system_descriptor", "system_descriptor_id"])
def test_swapped_system_descriptor_is_rejected(tmp_path: Path, key: str) -> None:
    """Tamper control (contract 2): a swapped system descriptor is caught."""

    artifact = _v2_artifact(tmp_path, FINITE)
    other = build_system_descriptor(
        ParticleSpec(n_electrons=1, n_spin_up=1, n_spin_down=0, spatial_dim=3),
        {"kinetic": KineticEnergy(), "trap": HarmonicTrap(omega=0.75)},
        units="atomic_units",
    )
    metadata = _read_metadata(artifact)
    if key == "system_descriptor":
        from tpen.evaluation.trajectory_records import _serialize_system_descriptor

        metadata[key] = _serialize_system_descriptor(other)
    else:
        metadata[key] = other.system_id()
    _rewrite_metadata(artifact, metadata)
    with pytest.raises(TamperedTrajectoryRecordError):
        inspect_trajectory_record(artifact.metadata_path)


def test_swapped_execution_receipt_is_rejected(tmp_path: Path) -> None:
    """Tamper control (contract 2): a swapped receipt is caught by the cross-binding check."""

    artifact = _v2_artifact(tmp_path, FINITE)
    other_system = build_system_descriptor(
        ParticleSpec(n_electrons=1, n_spin_up=1, n_spin_down=0, spatial_dim=3),
        {"kinetic": KineticEnergy(), "trap": HarmonicTrap(omega=0.75)},
        units="atomic_units",
    )
    from tpen.evaluation.trajectory_records import _serialize_execution_receipt

    other_receipt = _receipt(other_system)
    metadata = _read_metadata(artifact)
    metadata["execution_receipt"] = _serialize_execution_receipt(other_receipt)
    metadata["execution_receipt_method_id"] = other_receipt.method_id()
    _rewrite_metadata(artifact, metadata)
    with pytest.raises(TamperedTrajectoryRecordError, match="system_id"):
        inspect_trajectory_record(artifact.metadata_path)


def test_edited_csv_is_rejected(tmp_path: Path) -> None:
    artifact = _v2_artifact(tmp_path, FINITE)
    with artifact.path.open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(TamperedTrajectoryRecordError, match="csv_sha256"):
        inspect_trajectory_record(artifact.metadata_path)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_fused_column_manifest_must_cover_exactly_the_declared_operators(
    tmp_path: Path, mutation: str
) -> None:
    """A fused contribution's manifest must still claim every operator it covers, once."""

    system = _system()
    receipt = _receipt(system)
    bad_manifest = dict(receipt.manifest)
    if mutation == "missing":
        (one_operator, _) = next(iter(bad_manifest.items()))
        del bad_manifest[one_operator]
    else:
        bad_manifest[OperatorId("test.extra", "undeclared")] = ("fused",)

    short_receipt = ExecutionReceipt(
        system_id=receipt.system_id,
        implementation=receipt.implementation,
        version=receipt.version,
        numerical_policy=receipt.numerical_policy,
        manifest=bad_manifest,
    )
    artifact = TrajectoryRecordArtifactV2(
        path=tmp_path / "x.csv",
        metadata_path=tmp_path / "x.metadata.json",
        observable="local_energy",
        n_draws=1,
        n_walkers=1,
        draw_stride=1,
        burn_in_draws=0,
        n_electrons=1,
        spatial_dim=3,
        term_names=("kinetic", "trap"),
        row_count=1,
        finite_count=1,
        mean=1.0,
        variance=0.0,
        observable_values_content_id="0" * 64,
        csv_sha256="0" * 64,
        byte_count=1,
        system_descriptor=system,
        execution_receipt=short_receipt,
        final_draw=TrajectoryRecordBatch(
            draw_index=torch.zeros(1, dtype=torch.int64),
            walker_index=torch.zeros(1, dtype=torch.int64),
            positions=torch.zeros(1, 1, 3, dtype=torch.float64),
            local_energy=torch.tensor([1.0]),
            term_energies={"kinetic": torch.tensor([0.5]), "trap": torch.tensor([0.5])},
            logabs=torch.tensor([-1.0]),
            sign=torch.tensor([1.0]),
            finite_mask=torch.tensor([True]),
        ),
    )
    with pytest.raises(ValueError, match="manifest does not cover exactly"):
        artifact.validate(check_files=False)


def test_v2_reader_rejects_a_missing_resource_payload(tmp_path: Path) -> None:
    artifact = _v2_artifact(tmp_path, FINITE)
    metadata = _read_metadata(artifact)
    metadata["system_descriptor"]["terms"][0]["resource_id"] = "missing"
    metadata["system_descriptor"]["resources"] = {}
    _rewrite_metadata(artifact, metadata)
    with pytest.raises(
        TamperedTrajectoryRecordError, match="resource_id and resource must be provided together"
    ):
        inspect_trajectory_record(artifact.metadata_path)


def test_v2_reader_propagates_mixed_nonfinite_rows_from_real_writer(tmp_path: Path) -> None:
    artifact = _v2_artifact(
        tmp_path,
        torch.tensor([[float("inf"), float("-inf")], [1.0, 2.0]]),
    )
    inspection = inspect_trajectory_record(artifact.metadata_path)
    assert math.isnan(inspection.mean)
    assert math.isnan(inspection.variance)
    assert inspection.observable_values_digest.available is True
    assert inspection.observable_values_digest.content_id == artifact.observable_values_content_id

    # The public reader shares this non-finite reduction contract with the
    # historical v1 CSV path; keep the named survivor killer on that path too.
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    v1_artifact = _write_v1_artifact(v1_dir, torch.tensor([[math.inf, -math.inf], [1.0, 2.0]]))
    v1_inspection = inspect_trajectory_record(v1_artifact.metadata_path)
    assert math.isnan(v1_inspection.mean)
    assert math.isnan(v1_inspection.variance)


class _GaussianModel(torch.nn.Module):
    """``log|psi| = -0.5 * alpha * |r|^2``: real, twice-differentiable in positions."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        radius_squared = batch.positions.square().sum(dim=(1, 2))
        logabs = -0.5 * self.alpha * radius_squared
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def _hooke_context(tmp_path: Path) -> EvaluationContext:
    return EvaluationContext(
        namespace="eval",
        artifact_level="records",
        task_failure_policy="fail_fast",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=11,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )


def test_hooke_one_electron_records_v2_with_no_nuclear_resource(tmp_path: Path) -> None:
    """The real-generator end-to-end case that motivated the whole redesign.

    One electron, a harmonic trap, no nucleus -- the legacy v1 path cannot
    record this (it hard-requires `Walkers.atomic_configuration`); the v2
    path must, and the reader must decode what it wrote.
    """

    system = _system()
    sampler = MetropolisSampler(
        n_walkers=4,
        burn_in=2,
        n_steps=2,
        n_electrons=1,
        spatial_dim=3,
        seed=11,
        dtype=torch.float64,
    )
    assert sampler.atomic_configuration is None, "the sampler must carry no nuclear context"

    generator = TrajectoryMCMCGenerator(
        sampler=sampler,
        hamiltonian_terms=_TERMS,
        n_draws=3,
        discard_draws=1,
        chunk_size=2,
        system=system,
    )
    generated = generator.generate(model=_GaussianModel(), context=_hooke_context(tmp_path))

    records = generated.trajectory_records
    assert isinstance(records, TrajectoryRecordArtifactV2)
    assert records.system_descriptor.resources == {}
    assert records.execution_receipt.system_id == system.system_id()
    records.validate(check_files=True)

    inspection = inspect_trajectory_record(records.metadata_path)
    assert inspection.record_schema == TRAJECTORY_RECORD_SCHEMA_V2
    assert inspection.physical_context.system_id == system.system_id()
    assert inspection.observable_values_digest.available is True
    assert inspection.observable_values_digest.content_id == records.observable_values_content_id

    calculated = LocalEnergyCalculator(hamiltonian_terms=_TERMS, return_terms=True).calculate(
        model=_GaussianModel(),
        bundle=EvaluationBundle(generated=generated),
        context=_hooke_context(tmp_path),
    )
    assert isinstance(calculated.local_energy, LocalEnergyValues)
    assert calculated.local_energy.receipt == records.execution_receipt


def test_calculator_rejects_reuse_under_a_different_requested_method(tmp_path: Path) -> None:
    """Contract 5: requested method X but producer receipt Y is rejected, no naive fallback."""

    artifact = _v2_artifact(tmp_path, FINITE)

    class _OtherEvaluator(NaiveLocalEnergyEvaluator):
        def describe_execution_receipt(self, terms, system_id):
            receipt = super().describe_execution_receipt(terms, system_id)
            return replace(receipt, implementation="not." + receipt.implementation)

    generated = GeneratedConfigurations(
        batch=ElectronBatch(positions=artifact.final_draw.positions[:1]),
        metadata={},
        trajectory_records=artifact,
    )
    calculator = LocalEnergyCalculator(hamiltonian_terms=_TERMS, evaluator=_OtherEvaluator())
    with pytest.raises(ValueError, match="different execution method"):
        calculator.calculate(
            model=_GaussianModel(),
            bundle=EvaluationBundle(generated=generated),
            context=_hooke_context(tmp_path),
        )


def test_calculator_reuses_a_valid_fused_v2_contribution(tmp_path: Path) -> None:
    artifact = _v2_artifact(tmp_path, FINITE)
    receipt = _receipt(_system())
    fused_name = "kinetic_plus_trap"
    fused_receipt = replace(
        receipt,
        manifest={operator_id: (fused_name,) for operator_id in receipt.manifest},
        numerical_policy={
            **receipt.numerical_policy,
            "contribution_order": encode_ordered_contribution_sequence((fused_name,)),
        },
    )
    fused = replace(
        artifact,
        term_names=(fused_name,),
        execution_receipt=fused_receipt,
        final_draw=replace(
            artifact.final_draw,
            term_energies={fused_name: artifact.final_draw.local_energy},
        ),
    )
    generated = GeneratedConfigurations(
        batch=ElectronBatch(positions=fused.final_draw.positions[:1]),
        metadata={},
        trajectory_records=fused,
    )
    calculator = LocalEnergyCalculator(hamiltonian_terms=_TERMS, evaluator=NaiveLocalEnergyEvaluator())
    calculated = calculator.calculate(
        model=_GaussianModel(),
        bundle=EvaluationBundle(generated=generated),
        context=_hooke_context(tmp_path),
    )
    assert calculated.local_energy is not None


class _OrderedContributionEvaluator:
    def make_context(self, wavefunction: object, batch: ElectronBatch) -> ElectronBatch:
        del wavefunction
        return batch

    def evaluate(self, terms, context, *, return_terms: bool = False):
        del terms, context
        values = {
            "a": torch.tensor([1.0e16], dtype=torch.float64),
            "b": torch.tensor([-1.0e16], dtype=torch.float64),
            "c": torch.tensor([1.0], dtype=torch.float64),
        }
        result = LocalEnergyResult(
            total=torch.tensor([1.0], dtype=torch.float64),
            terms=values,
            wavefunction_output=WavefunctionOutput(
                logabs=torch.tensor([0.0], dtype=torch.float64),
                sign=torch.tensor([1.0], dtype=torch.float64),
            ),
            receipt=ExecutionReceipt(
                system_id="ordered-system",
                implementation="ordered-test",
                version=1,
                numerical_policy={
                    "contribution_order": encode_ordered_contribution_sequence(("a", "b", "c"))
                },
                manifest={
                    OperatorId("test", "a"): ("a",),
                    OperatorId("test", "c"): ("c",),
                    OperatorId("test", "b"): ("b",),
                },
            ),
        )
        return result if return_terms else result.total


def test_generator_reconstructs_total_in_receipt_contribution_order() -> None:
    generator = TrajectoryMCMCGenerator(
        sampler=object(),
        hamiltonian_terms=_TERMS,
        n_draws=1,
        evaluator=_OrderedContributionEvaluator(),
    )
    result = generator._local_energy_per_walker(
        model=_GaussianModel(),
        walkers=Walkers(positions=torch.zeros(1, 1, 3, dtype=torch.float64)),
    )
    assert torch.equal(result.total, torch.tensor([1.0], dtype=torch.float64))
