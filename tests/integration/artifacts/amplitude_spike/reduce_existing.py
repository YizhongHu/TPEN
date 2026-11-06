"""Stage-1 reduction CLI: apply the amplitude-risk analysis to an ALREADY-WRITTEN raw table.

Used on Cannon to reduce retained records named by note
amplitude-risk-design-20261106 ("Stage 1, reduce preserved data before
generating new data"): H-v2 part-(e)/exact-canary ``sampled_eval_table.csv``
receipts, He ``source-inventory.csv``-named tables, and trained-Hooke
Holystore archive tables. Column names are NOT assumed identical across these
sources, so every column is named explicitly on the command line rather than
hard-coded; inspect the actual header first (the design's own instruction:
"Find ... resolve from job scripts/logs ... do not guess a raw-file path").

Caps rows read per the design's bounds (first 64 complete draws of 4096
walkers for He tables: 1,048,576 rows; 65536 rows/run for trained Hooke) via
``--max-rows``; this script does not decide the bound, the caller does, from
the design note and the actual file's declared draw/walker grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402


def _load_csv(path: Path, *, draw_col: str, walker_col: str, logabs_col: str, sign_col: str, max_rows: int | None) -> dict[str, np.ndarray]:
    draws: list[int] = []
    walkers: list[int] = []
    logabs: list[float] = []
    sign: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            draws.append(int(float(row[draw_col])))
            walkers.append(int(float(row[walker_col])))
            logabs.append(float(row[logabs_col]))
            sign.append(float(row[sign_col]))
    if not draws:
        raise ValueError(f"no rows read from {path} (missing or empty, not zero fraction)")
    return {
        "draw_index": np.array(draws, dtype=np.int64),
        "walker_index": np.array(walkers, dtype=np.int64),
        "logabs": np.array(logabs, dtype=np.float64),
        "sign": np.array(sign, dtype=np.float64),
    }


def _reduce(table: dict[str, np.ndarray]) -> dict:
    n_draws = int(table["draw_index"].max()) + 1
    draws = []
    walker_lists = []
    ell_lists = []
    for d in range(n_draws):
        mask = table["draw_index"] == d
        if not mask.any():
            continue
        ell = table["logabs"][mask]
        sign = table["sign"][mask]
        walkers = table["walker_index"][mask]
        draws.append(analysis.reduce_draw(ell, sign))
        walker_lists.append(walkers)
        ell_lists.append(ell)
    agg = analysis.aggregate_stream(draws, walker_lists, ell_lists)
    boot = analysis.block_bootstrap_fraction(draws, walker_lists, ell_lists)
    return {
        "n_draws_present": len(draws),
        "aggregate": agg,
        "bootstrap": {str(tau): v for tau, v in boot.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="CSV path to reduce")
    parser.add_argument("--draw-col", default="draw_index")
    parser.add_argument("--walker-col", default="walker_index")
    parser.add_argument("--logabs-col", default="logabs")
    parser.add_argument("--sign-col", default="sign")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--label", required=True, help="Arm label recorded in the receipt")
    parser.add_argument("--out", required=True, help="Output JSON receipt path")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_file():
        payload = {"label": args.label, "path": str(path), "status": "MISSING"}
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return

    table = _load_csv(
        path,
        draw_col=args.draw_col,
        walker_col=args.walker_col,
        logabs_col=args.logabs_col,
        sign_col=args.sign_col,
        max_rows=args.max_rows,
    )
    reduction = _reduce(table)
    payload = {
        "label": args.label,
        "path": str(path),
        "status": "OK",
        "row_count": int(table["logabs"].size),
        **reduction,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, default=str))


if __name__ == "__main__":
    main()
