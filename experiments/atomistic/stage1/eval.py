"""Thin Stage-1 strict model-only evaluation gateway."""

from __future__ import annotations

import argparse

from tpen.run import run_from_config


def main() -> int:
    """Run evaluation through the public gateway with model-only restore."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    from tpen.run import load_config
    cfg = load_config(args.config, args.overrides)
    # The config/launcher supplies the strict model-only restore policy; this
    # driver does not import checkpoint internals or silently alter identity.
    return run_from_config(cfg, config_path=args.config, command="stage1 eval")


if __name__ == "__main__":
    raise SystemExit(main())
