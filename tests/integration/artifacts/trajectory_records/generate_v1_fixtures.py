"""Regenerate the immutable v1 trajectory-record compatibility fixtures.

Run this script from the checkout whose commit should be pinned as each
fixture's generating commit (see each fixture's ``PROVENANCE.json``, written
by this script). It uses only the frozen ``trajectory_records/v1`` writer
(``tpen.evaluation.trajectory_records.TrajectoryRecordStreamWriter``), never
the new historical reader, so the checked-in bytes are independent evidence
that the reader decodes genuine writer output rather than a format the
reader itself invented.

``tpen/evaluation/trajectory_records.py`` is unmodified by S1-F (frozen v1
writer), so this script additionally records whether the running checkout's
blob of that file is byte-identical to ``origin/main``'s current blob, giving
each fixture a verifiable link to the frozen-main writer even when generated
from a stacked feature branch whose other files differ from ``main``.

Usage (inside a Cannon ``test``-partition allocation)::

    uv run --extra cpu python \\
        tests/integration/artifacts/trajectory_records/generate_v1_fixtures.py <output_dir>

The script is deterministic: identical inputs always produce identical
output bytes, so re-running it is itself a provenance check (acceptance
contract item 4 on TO item S1-F).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.evaluation.trajectory_records import (
    TrajectoryRecordBatch,
    TrajectoryRecordStreamWriter,
)
from tpen.statistics.trajectory import ObservableTrajectory

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WRITER_PATH = "tpen/evaluation/trajectory_records.py"

#: Small, deterministic draw/walker grid shared by both fixtures (few draws
#: by few walkers, per the S1-F acceptance contract).
N_DRAWS = 2
N_WALKERS = 3


def _git(*args: str) -> str | None:
    """Return trimmed git output, or ``None`` if the ref/command is unavailable."""

    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _write_fixture(
    out_dir: Path,
    *,
    name: str,
    n_electrons: int,
    nuclear_charge: float,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    atoms = AtomicConfiguration(
        positions=torch.zeros(1, 3, dtype=torch.float64),
        charges=torch.tensor([nuclear_charge], dtype=torch.float64),
    )
    # Deterministic, hand-chosen values -- not a physical trajectory. Row
    # (draw, walker) gets local energy `draw * 10 + walker + 1`, so values
    # are easy to eyeball in the checked-in CSV.
    values = torch.tensor(
        [[draw * 10 + walker + 1 for walker in range(N_WALKERS)] for draw in range(N_DRAWS)],
        dtype=torch.float64,
    )
    positions = torch.zeros(N_DRAWS, N_WALKERS, n_electrons, 3, dtype=torch.float64)
    for draw in range(N_DRAWS):
        for walker in range(N_WALKERS):
            for electron in range(n_electrons):
                positions[draw, walker, electron, 0] = 0.1 * (draw + 1) + 0.01 * (walker + 1) + 0.001 * electron

    def record(draw: int) -> TrajectoryRecordBatch:
        row_values = values[draw]
        return TrajectoryRecordBatch(
            draw_index=torch.full((N_WALKERS,), draw, dtype=torch.int64),
            walker_index=torch.arange(N_WALKERS, dtype=torch.int64),
            positions=positions[draw],
            local_energy=row_values,
            term_energies={"kinetic": row_values},
            logabs=torch.full_like(row_values, -1.5),
            sign=torch.ones(N_WALKERS, dtype=torch.float64),
            finite_mask=torch.isfinite(row_values),
        )

    writer = TrajectoryRecordStreamWriter(
        out_dir / "sampled_eval_table.csv",
        observable="local_energy",
        n_draws=N_DRAWS,
        n_walkers=N_WALKERS,
        term_names=("kinetic",),
        atomic_configuration=atoms,
        first_draw=record(0),
    )
    for draw in range(1, N_DRAWS):
        writer.append(record(draw))
    artifact = writer.finalize(
        ObservableTrajectory(
            observable="local_energy",
            values=values,
            draw_stride=1,
            burn_in_draws=0,
        )
    )
    artifact.validate()
    print(f"wrote {name}: {artifact.path} ({artifact.csv_sha256})")
    return {
        "fixture": name,
        "n_electrons": n_electrons,
        "nuclear_charge": nuclear_charge,
        "n_draws": N_DRAWS,
        "n_walkers": N_WALKERS,
        "csv_sha256": artifact.csv_sha256,
        "byte_count": artifact.byte_count,
        "atomic_configuration_id": atoms.content_id(),
    }


def _write_provenance(out_dir: Path, fixture_payload: dict[str, object]) -> None:
    generating_commit = _git("rev-parse", "HEAD")
    writer_blob = _git("rev-parse", f"HEAD:{_WRITER_PATH}")
    main_commit = _git("rev-parse", "origin/main")
    main_blob = _git("rev-parse", f"origin/main:{_WRITER_PATH}") if main_commit else None
    provenance = {
        "generated_by": "tests/integration/artifacts/trajectory_records/generate_v1_fixtures.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generating_commit": generating_commit,
        "writer_module_path": _WRITER_PATH,
        "writer_module_blob_sha1": writer_blob,
        "origin_main_commit": main_commit,
        "origin_main_writer_module_blob_sha1": main_blob,
        "writer_blob_matches_origin_main": (
            None if writer_blob is None or main_blob is None else writer_blob == main_blob
        ),
        **fixture_payload,
    }
    (out_dir / "PROVENANCE.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    base = Path(argv[1])
    for out_dir, name, n_electrons, nuclear_charge in (
        (base / "h_like_frozen_main", "h_like_frozen_main", 1, 1.0),
        (base / "he_like_frozen_main", "he_like_frozen_main", 2, 2.0),
    ):
        payload = _write_fixture(out_dir, name=name, n_electrons=n_electrons, nuclear_charge=nuclear_charge)
        _write_provenance(out_dir, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
