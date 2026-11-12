"""Regenerate the immutable v1 trajectory-record compatibility fixtures.

Run this script from the checkout whose commit should be pinned as each
fixture's generating commit (see each fixture's ``PROVENANCE.json``, written
by this script). It uses only the frozen ``trajectory_records/v1`` writer
(``tpen.evaluation.trajectory_records.TrajectoryRecordStreamWriter``), never
the new historical reader, so the checked-in bytes are independent evidence
that the reader decodes genuine writer output rather than a format the
reader itself invented.

Unlike comparing whole-file git blobs (which disagrees once a later slice,
e.g. S1-F's own reader, appends unrelated code to this same module), this
script hashes the *executed* writer classes' source directly via
``inspect.getsource`` on the live imported objects, and compares that digest
to the same classes' source as they existed at the fixed, named
``_FROZEN_WRITER_COMMIT`` -- so the provenance attests to code this process
actually ran, not an inferred historical blob.

Usage (inside a Cannon ``test``-partition allocation, exactly this
invocation -- also recorded verbatim as each fixture's
``generation_command``)::

    uv run --extra cpu python \\
        tests/integration/artifacts/trajectory_records/generate_v1_fixtures.py <output_dir>

The script is deterministic: identical inputs always produce identical
output bytes, so re-running it from a clean checkout of the recorded
``generating_commit`` is itself a provenance check (acceptance contract item
4 on TO item S1-F) -- a replay that reproduces the committed CSV, metadata,
and provenance fields byte-for-byte is positive evidence; a replay that does
not must regenerate the checked-in fixtures rather than paper over the
mismatch.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

import tpen.evaluation.trajectory_records as trajectory_records_module
from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.evaluation.trajectory_records import (
    TrajectoryRecordBatch,
    TrajectoryRecordStreamWriter,
)
from tpen.statistics.trajectory import ObservableTrajectory

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WRITER_PATH = "tpen/evaluation/trajectory_records.py"
_GENERATOR_PATH = "tests/integration/artifacts/trajectory_records/generate_v1_fixtures.py"
_GENERATION_COMMAND = (
    "uv run --extra cpu python "
    "tests/integration/artifacts/trajectory_records/generate_v1_fixtures.py <output_dir>"
)

#: Fixed, named reference commit for the frozen v1 writer classes, predating
#: any S1-F reader code in this same module. Deliberately NOT inferred from
#: `git log` at generation time (a moving "history[1]" pointer silently
#: tracks whichever commit is currently second-to-last, which drifted once
#: this file kept gaining new, unrelated commits -- the exact gap review r1
#: flagged).
_FROZEN_WRITER_COMMIT = "2630ea7b30dcb23db5df6aae3be1b55c5e7778cd"
_FROZEN_WRITER_SYMBOLS = ("TrajectoryRecordBatch", "TrajectoryRecordStreamWriter")

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


def _extract_symbol_sources(file_text: str, names: tuple[str, ...]) -> dict[str, str]:
    """Return the exact source text of each top-level class/function in `names`."""

    tree = ast.parse(file_text)
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            segment = ast.get_source_segment(file_text, node)
            if segment is not None:
                found[node.name] = segment
    return found


def _symbols_source_sha256(sources: dict[str, str], names: tuple[str, ...]) -> str | None:
    if set(names) - set(sources):
        return None
    digest = hashlib.sha256()
    for name in names:
        digest.update(sources[name].encode("utf-8"))
    return digest.hexdigest()


def _executed_writer_source_sha256() -> str:
    """Hash the source of the writer classes this process actually imported and ran.

    Uses `inspect.getsource` on the live objects `_write_fixture` calls, so
    the digest attests to code that executed in this process -- not a blob
    picked out of git history.
    """

    sources = {name: inspect.getsource(getattr(trajectory_records_module, name)) for name in _FROZEN_WRITER_SYMBOLS}
    digest = _symbols_source_sha256(sources, _FROZEN_WRITER_SYMBOLS)
    assert digest is not None  # the live module always defines both symbols
    return digest


def _frozen_commit_writer_source_sha256(commit: str) -> str | None:
    """Hash the same writer classes' source as they existed at `commit`."""

    blob = _git("show", f"{commit}:{_WRITER_PATH}")
    if blob is None:
        return None
    sources = _extract_symbol_sources(blob, _FROZEN_WRITER_SYMBOLS)
    return _symbols_source_sha256(sources, _FROZEN_WRITER_SYMBOLS)


def _config_hash(*, n_draws: int, n_walkers: int, n_electrons: int, nuclear_charge: float) -> str:
    """Hash the deterministic generation parameters and running environment.

    Lets a replay detect environment drift (Python/torch version) alongside
    byte-for-byte output comparison, rather than assuming two runs are
    equivalent just because their output happens to match.
    """

    config = {
        "n_draws": n_draws,
        "n_walkers": n_walkers,
        "n_electrons": n_electrons,
        "nuclear_charge": nuclear_charge,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _write_provenance(out_dir: Path, fixture_payload: dict[str, object]) -> None:
    generating_commit = _git("rev-parse", "HEAD")
    generator_source_sha256 = hashlib.sha256(
        (_REPO_ROOT / _GENERATOR_PATH).read_bytes()
    ).hexdigest()
    executed_writer_source_sha256 = _executed_writer_source_sha256()
    frozen_commit_writer_source_sha256 = _frozen_commit_writer_source_sha256(_FROZEN_WRITER_COMMIT)
    writer_matches_frozen_commit = (
        None
        if frozen_commit_writer_source_sha256 is None
        else executed_writer_source_sha256 == frozen_commit_writer_source_sha256
    )
    metadata_path = out_dir / "sampled_eval_table.metadata.json"
    metadata_sha256 = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    provenance = {
        "generated_by": _GENERATOR_PATH,
        "generator_source_sha256": generator_source_sha256,
        "generation_command": _GENERATION_COMMAND,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generating_commit": generating_commit,
        "config_hash": _config_hash(
            n_draws=int(fixture_payload["n_draws"]),
            n_walkers=int(fixture_payload["n_walkers"]),
            n_electrons=int(fixture_payload["n_electrons"]),
            nuclear_charge=float(fixture_payload["nuclear_charge"]),
        ),
        "writer_module_path": _WRITER_PATH,
        "frozen_writer_commit": _FROZEN_WRITER_COMMIT,
        "executed_writer_source_sha256": executed_writer_source_sha256,
        "frozen_commit_writer_source_sha256": frozen_commit_writer_source_sha256,
        "writer_matches_frozen_commit": writer_matches_frozen_commit,
        "metadata_sha256": metadata_sha256,
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
