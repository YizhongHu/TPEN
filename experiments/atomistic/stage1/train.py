"""Thin Stage-1 training gateway."""

from __future__ import annotations

import argparse

from tpen.run import run_from_config
from experiments.atomistic.stage1.protocol import validate_gateway_binding


def main() -> int:
    """Run one configured training job through TPEN's public gateway."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--kind", choices=("smoke", "train"), required=True)
    args = parser.parse_args()
    validate_gateway_binding(binding=args.binding, config=args.config,
                             overrides=args.overrides, kind=args.kind)
    # Config loading remains owned by the installed tpen entrypoint; this
    # wrapper is intentionally only a gateway and does not duplicate Runner.
    from tpen.run import load_config
    cfg = load_config(args.config, args.overrides)
    return run_from_config(cfg, config_path=args.config, command="stage1 train")


if __name__ == "__main__":
    raise SystemExit(main())
