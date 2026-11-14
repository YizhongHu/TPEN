"""Thin Stage-1 training gateway."""

from __future__ import annotations

import argparse

from tpen.run import run_from_config


def main() -> int:
    """Run one configured training job through TPEN's public gateway."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    # Config loading remains owned by the installed tpen entrypoint; this
    # wrapper is intentionally only a gateway and does not duplicate Runner.
    from tpen.run import load_config
    cfg = load_config(args.config, args.overrides)
    return run_from_config(cfg, config_path=args.config, command="stage1 train")


if __name__ == "__main__":
    raise SystemExit(main())
