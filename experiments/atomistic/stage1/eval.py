"""Thin Stage-1 strict model-only evaluation gateway."""

from __future__ import annotations

import argparse

from tpen.run import run_from_config
from experiments.atomistic.stage1.protocol import validate_gateway_binding


def main() -> int:
    """Run evaluation through the public gateway with model-only restore."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--kind", choices=("eval",), required=True)
    parser.add_argument("--expected-resolved-config-identity", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    args = parser.parse_args()
    validate_gateway_binding(
        binding=args.binding, overrides=args.overrides, kind=args.kind,
        expected_identity=args.expected_resolved_config_identity,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    from tpen.run import load_config
    cfg = load_config(args.config, args.overrides)
    # The config/launcher supplies the strict model-only restore policy; this
    # driver does not import checkpoint internals or silently alter identity.
    return run_from_config(cfg, config_path=args.config, command="stage1 eval")


if __name__ == "__main__":
    raise SystemExit(main())
