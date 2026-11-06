"""Core-test harness: exact Hooke singlet/triplet raw `[draw, walker]` capture.

Per note amplitude-risk-design-20261106 ("Stage 2 ... Exact Hooke singlet and
triplet"): `tpen.evaluation.generators.trajectory.TrajectoryMCMCGenerator`
refuses records without `Walkers.atomic_configuration`
(`generators/trajectory.py:168-173`), and the exact Hooke samplers are
nucleus-free. This harness calls the SAME core collector the real route uses,
`tpen.sampling.trajectory.collect_observable_trajectory_with_diagnostics`, and
the same per-draw local-energy evaluation
(`tpen.evaluation.calculators.local_energy.evaluate_local_energy_in_chunks`),
without TrajectoryRecordBatch/TrajectoryRecordStreamWriter (which require
atomic_configuration). This is a test artifact, not a trajectory_records/v1
science artifact: it writes raw NumPy partitions directly.

Not a core callback or config target; invoked only from a spike driver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from tpen.evaluation.calculators.local_energy import evaluate_local_energy_in_chunks
from tpen.physics.hamiltonian import LocalEnergyResult, normalize_hamiltonian_terms
from tpen.sampling.trajectory import collect_observable_trajectory_with_diagnostics


def run_hooke_exact_trajectory(
    *,
    model: torch.nn.Module,
    sampler: object,
    hamiltonian_terms: Any,
    n_draws: int,
    observable_name: str,
    seed: int,
    chunk_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Collect ``n_draws`` retained draws of the exact Hooke state, raw.

    Returns
    -------
    dict of np.ndarray
        ``draw_index``, ``walker_index`` (both ``[n_draws * n_walkers]``,
        row-major draw-major order matching the real route's convention),
        ``positions`` ``[n_draws, n_walkers, n_electrons, spatial_dim]``,
        ``logabs``, ``sign``, ``local_energy`` (``[n_draws, n_walkers]``), and
        one ``term/<name>`` array per configured Hamiltonian term.
    """

    terms = normalize_hamiltonian_terms(hamiltonian_terms)
    term_names = tuple(terms)

    draws_logabs: list[np.ndarray] = []
    draws_sign: list[np.ndarray] = []
    draws_local_energy: list[np.ndarray] = []
    draws_positions: list[np.ndarray] = []
    draws_terms: dict[str, list[np.ndarray]] = {name: [] for name in term_names}

    def observable(walkers) -> torch.Tensor:
        batch = walkers.make_batch().flatten_samples()
        result = evaluate_local_energy_in_chunks(
            terms, model, batch, return_terms=True, chunk_size=chunk_size
        )
        if not isinstance(result, LocalEnergyResult):
            raise TypeError("Hooke exact harness local-energy evaluation must return LocalEnergyResult")
        output = result.wavefunction_output
        if output is None:
            raise ValueError(
                "Hooke exact harness requires the captured wavefunction output used by "
                "the kinetic term; a fallback model pass is forbidden"
            )
        draws_logabs.append(output.logabs.detach().to("cpu", dtype=torch.float64).numpy().copy())
        draws_sign.append(output.sign.detach().to("cpu", dtype=torch.float64).numpy().copy())
        draws_local_energy.append(result.total.detach().to("cpu", dtype=torch.float64).numpy().copy())
        draws_positions.append(walkers.positions.detach().to("cpu", dtype=torch.float64).numpy().copy())
        for name in term_names:
            draws_terms[name].append(
                result.terms[name].detach().to("cpu", dtype=torch.float64).numpy().copy()
            )
        return result.total

    collect_observable_trajectory_with_diagnostics(
        sampler,
        model,
        observable,
        observable_name=observable_name,
        n_draws=n_draws,
        discard_draws=0,
        reset=True,
        verify_model_unchanged=True,
    )

    if len(draws_logabs) != n_draws:
        raise RuntimeError(
            f"Hooke exact harness collected {len(draws_logabs)} draws, expected {n_draws}"
        )
    n_walkers = draws_logabs[0].shape[0]
    draw_index = np.repeat(np.arange(n_draws, dtype=np.int64), n_walkers)
    walker_index = np.tile(np.arange(n_walkers, dtype=np.int64), n_draws)

    result: dict[str, np.ndarray] = {
        "draw_index": draw_index,
        "walker_index": walker_index,
        "positions": np.stack(draws_positions, axis=0),
        "logabs": np.stack(draws_logabs, axis=0),
        "sign": np.stack(draws_sign, axis=0),
        "local_energy": np.stack(draws_local_energy, axis=0),
        "seed": np.array([seed], dtype=np.int64),
    }
    for name in term_names:
        result[f"term/{name}"] = np.stack(draws_terms[name], axis=0)
    return result


def save_hooke_raw(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


__all__ = ["run_hooke_exact_trajectory", "save_hooke_raw"]
