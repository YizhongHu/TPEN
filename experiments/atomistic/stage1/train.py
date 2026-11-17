"""Thin Stage-1 training gateway."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Direct script execution puts only this directory on sys.path.  Add the
# checkout root before importing the source-owned experiments package.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tpen.run import run_from_config
from experiments.atomistic.stage1.protocol import resolve_config_path, validate_gateway_binding


def main() -> int:
    """Run one configured training job through TPEN's public gateway."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--kind", choices=("smoke", "train"), required=True)
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    validate_gateway_binding(binding=args.binding, config=str(config_path),
                             overrides=args.overrides, kind=args.kind)
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-1 config path does not exist: {config_path}")
    # Config loading remains owned by the installed tpen entrypoint; this
    # wrapper is intentionally only a gateway and does not duplicate Runner.
    from tpen.run import load_config
    cfg = load_config(str(config_path), args.overrides)
    return run_from_config(cfg, config_path=str(config_path), command="stage1 train")


if __name__ == "__main__":
    raise SystemExit(main())
