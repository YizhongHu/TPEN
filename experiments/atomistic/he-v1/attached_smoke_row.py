"""Select one He TRAIN row for the attached-route smoke.

Minimal by design: this fixture exists only to select the approved smoke row,
and it stays as small as possible so the smoke stays informative.  It has no
retries, fallbacks, defaults that substitute for missing inputs, exception
swallowing, output rewriting, or logic that could mask or substitute for a
route failure.  Every error from :func:`prepare_he_train` propagates
unchanged with a non-zero exit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.toolkit.study_imports import sibling

_attached = sibling(__file__, "attached")
prepare_he_train = _attached.prepare_he_train


def _absolute(value: str | Path, name: str) -> Path:
    """Require an explicitly absolute launch-packet path."""

    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def prepare_smoke_row(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    results_root: str | Path,
    source_config: str | Path,
    row_id: str,
) -> _attached.PreparedHeTrain:
    """Prepare exactly one explicitly selected He TRAIN row.

    Parameters
    ----------
    manifest_path, output_dir, repo_root, results_root, source_config : str or pathlib.Path
        Absolute paths for the plan manifest, handoff directory, checkout,
        result root, and source configuration.
    row_id : str
        The one plan row to select.

    Returns
    -------
    _attached.PreparedHeTrain
        The value returned by :func:`prepare_he_train`.

    Raises
    ------
    ValueError
        If a launch-packet path is not absolute.
    FileNotFoundError
        If the manifest path does not identify a readable file.
    """

    manifest_file = _absolute(manifest_path, "manifest_path")
    output_path = _absolute(output_dir, "output_dir")
    repo_path = _absolute(repo_root, "repo_root")
    results_path = _absolute(results_root, "results_root")
    source_path = _absolute(source_config, "source_config")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    return prepare_he_train(
        manifest,
        output_dir=output_path,
        repo_root=repo_path,
        results_root=results_path,
        source_config=source_path,
        enable_attached=True,
        enable_reduced_callbacks=True,
        checkpoint_every_n=25,
        row_ids=(row_id,),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit smoke launch packet."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--row-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare the selected row from an explicit launch packet."""

    args = _parse_args(argv)
    prepare_smoke_row(
        args.manifest,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        results_root=args.results_root,
        source_config=args.source_config,
        row_id=args.row_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
