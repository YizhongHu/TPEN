"""Thin Stage-1 strict model-only evaluation gateway."""

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
    """Run evaluation through the public gateway with model-only restore."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--kind", choices=("eval",), required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    validate_gateway_binding(
        binding=args.binding, config=str(config_path), overrides=args.overrides, kind=args.kind,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-1 config path does not exist: {config_path}")
    from tpen.run import load_config
    cfg = load_config(str(config_path), args.overrides)
    # The config/launcher supplies the strict model-only restore policy; this
    # driver does not import checkpoint internals or silently alter identity.
    return run_from_config(cfg, config_path=str(config_path), command="stage1 eval")


if __name__ == "__main__":
    raise SystemExit(main())
