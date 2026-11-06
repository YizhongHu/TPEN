"""Public command-line entrypoint for dispatching one executor input."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .composition import CLIRefusal, admit, dispatch


class _Parser(argparse.ArgumentParser):
    """Argument parser that converts usage errors into the public envelope."""

    def error(self, message: str) -> None:
        raise CLIRefusal("usage", message)


def build_parser() -> argparse.ArgumentParser:
    """Build the public executor CLI parser."""

    parser = _Parser(prog="tpen-executor", description="Dispatch one executor_input.json ReadyGroup")
    commands = parser.add_subparsers(dest="command", required=True)
    dispatch_parser = commands.add_parser("dispatch", help="admit and dispatch one ReadyGroup")
    dispatch_parser.add_argument("--input", required=True, metavar="FILE", help="ReadyGroup JSON")
    dispatch_parser.add_argument("--store", required=True, metavar="DIR", help="fresh output directory")
    dispatch_parser.add_argument("--allocation", required=True, metavar="FILE", help="ConfiguredAllocation JSON")
    dispatch_parser.add_argument("--worker-config", required=True, metavar="FILE", help="executor worker YAML")
    dispatch_parser.add_argument("--runs", required=True, metavar="FILE", help="configured-runs sidecar JSON")
    dispatch_parser.add_argument("--dispatcher", choices=("local", "parsl"), default="local")
    dispatch_parser.add_argument("--timeout", type=float, default=None, metavar="SECONDS")
    return parser


def _error_payload(error: Exception, code: str | None = None) -> bytes:
    """Encode one CLI error envelope for stderr."""

    if isinstance(error, CLIRefusal):
        payload = error.to_dict()
    else:
        payload = {"schema": "tpen-executor/cli-error-v1", "code": code or "dispatch-error", "reason": str(error)}
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _operational_exit(manifest) -> int:
    """Map manifest member outcomes to the operational CLI status."""

    return 0 if manifest.attempts and all(
        attempt.outcome_receipt is not None and attempt.outcome_receipt.status == "completed"
        for attempt in manifest.attempts
    ) else 3


def main(argv: Sequence[str] | None = None) -> int:
    """Run the executor CLI and return its documented process exit code.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments after the executable name. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` for completed operational dispatch, ``2`` for admission or
        usage refusal, ``3`` for an operationally incomplete dispatch, ``4``
        for an unexpected dispatch error, and ``130`` for cancellation.

    Raises
    ------
    CLIRefusal
        Converted to the typed stderr envelope and exit code ``2``.
    """

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except CLIRefusal as error:
        sys.stderr.buffer.write(_error_payload(error))
        return 2
    except SystemExit as exit_signal:
        return int(exit_signal.code)

    if args.command != "dispatch":
        sys.stderr.buffer.write(_error_payload(CLIRefusal("usage", "a dispatch command is required")))
        return 2
    try:
        plan = admit(
            args.input,
            args.store,
            args.allocation,
            args.worker_config,
            args.runs,
            dispatcher_name=args.dispatcher,
            timeout=args.timeout,
        )
    except CLIRefusal as error:
        sys.stderr.buffer.write(_error_payload(error))
        return 2
    try:
        manifest = dispatch(plan)
    except KeyboardInterrupt as error:
        sys.stderr.buffer.write(_error_payload(error, "cancelled"))
        return 130
    except Exception as error:
        sys.stderr.buffer.write(_error_payload(error))
        return 4
    sys.stdout.buffer.write(manifest.encode() + b"\n")
    return _operational_exit(manifest)


__all__ = ["build_parser", "main"]
