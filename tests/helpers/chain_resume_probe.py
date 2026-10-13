"""Tiny subprocess probe used by the chain-resume adapter tests.

The probe is deliberately independent of TPEN constructors.  It emits the
same named channels that the E5 adapter observes, with switches that model a
single disabled limb, an observability suppression, and the two mutual-masking
mutants.  The adapter still launches it as a fresh OS process and records its
real PID; this helper is not a scientific fixture or a replacement for E5.
"""

from __future__ import annotations

import argparse
import json
import random


CHANNELS = {
    "model": ("model_logabs", "first_model_evaluation"),
    "optimizer": ("optimizer_update", "first_optimizer_update"),
    "trainer_cursor": ("trainer_cursor", "first_resumed_step"),
    "sampler_mcmc": ("sampler_proposal", "first_sampler_proposal"),
    "process_rng": ("process_random", "first_process_random"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume-step", type=int, default=2)
    parser.add_argument("--disable-limb", action="append", default=[])
    parser.add_argument("--suppress-channel")
    parser.add_argument("--omit-channel")
    return parser


def main() -> int:
    """Emit deterministic first-event records for one fresh process."""

    args = _parser().parse_args()
    rng = random.Random(args.seed)
    disabled = set(args.disable_limb)
    for step in range(args.resume_step + 1, args.resume_step + 3):
        for index, (limb, (channel, event_kind)) in enumerate(CHANNELS.items()):
            # Every value is derived from a draw made at this step.  The
            # trainer cursor is included as a named observation of the same
            # resumed event, not as an inferred final counter.
            draw = rng.random()
            value = round(draw + index * 0.03125 + step * 0.0078125, 15)
            if limb in disabled:
                value = round(value + 0.125, 15)
            if channel == args.omit_channel:
                continue
            if channel == args.suppress_channel:
                # The limb stays live and consumes entropy; only its emitted
                # channel is suppressed to a constant observable.
                value = 0.0
            print(
                json.dumps(
                    {
                        "record": "channel_event",
                        "channel": channel,
                        "step": step,
                        "event_kind": event_kind,
                        "value": value,
                        "random_dependent": True,
                    },
                    sort_keys=True,
                )
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
