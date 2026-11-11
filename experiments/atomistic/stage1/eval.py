"""Thin Stage-1 model-only evaluation gateway."""

from __future__ import annotations

import argparse


def main() -> int:
    """Invoke the configured TPEN evaluator; restore strictness belongs in YAML."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, extra = parser.parse_known_args()
    from tpen.run import load_config, run_from_config

    cfg = load_config(args.config, extra)
    return int(run_from_config(cfg, config_path=args.config, command="stage1 eval"))


if __name__ == "__main__":
    raise SystemExit(main())
