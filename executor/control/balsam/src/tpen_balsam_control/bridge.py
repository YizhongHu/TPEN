"""Process bridge for the isolated Balsam control project.

The bridge reads one S3 wire request from standard input and writes one S3
wire result to standard output.  Balsam is imported only after the validated
client path is installed in the environment; no SDK object crosses the wire.
"""

from __future__ import annotations

import os
import sys

from tpen_executor.lifecycle.balsam_client import select_balsam_client_path
from tpen_executor.lifecycle.balsam_wire import (
    BalsamWireRequest,
    BalsamWireResult,
    decode_request,
    encode_result,
)

from .application import BalsamControlAmbiguous, BalsamControlRefusal, InjectedSDKOperations


def dispatch(request: BalsamWireRequest, operations: InjectedSDKOperations) -> BalsamWireResult:
    """Apply one of the five closed wire operations.

    Parameters
    ----------
    request : BalsamWireRequest
        Strictly decoded executor request.
    operations : InjectedSDKOperations
        Finite adapter for the pinned SDK models.

    Returns
    -------
    BalsamWireResult
        Primitive acknowledgement, refusal, or ambiguous-effect result.
    """

    try:
        if request.operation == "register":
            backend_id = operations.register(
                str(request.group_id), request.group_digest or "", request.deployment_digest or ""
            )
            from tpen_executor.contracts import BackendJobId

            return BalsamWireResult.ack(request, backend_job_id=BackendJobId(backend_id))
        if request.operation == "observe":
            state = operations.observe(str(request.group_id), str(request.backend_job_id))
            return BalsamWireResult.ack(request, backend_job_id=request.backend_job_id, state=state)
        if request.operation == "submit":
            batch_id = operations.submit(
                str(request.group_id), str(request.backend_job_id), request.batch_digest or ""
            )
            from tpen_executor.contracts import BatchJobId

            return BalsamWireResult.ack(
                request, backend_job_id=request.backend_job_id, batch_job_id=BatchJobId(batch_id)
            )
        if request.operation == "rearm":
            state = operations.rearm(
                str(request.group_id), str(request.backend_job_id), request.recovery_digest or ""
            )
            return BalsamWireResult.ack(request, backend_job_id=request.backend_job_id, state=state)
        if request.operation == "readback":
            state = operations.readback(str(request.group_id), str(request.backend_job_id))
            return BalsamWireResult.ack(request, backend_job_id=request.backend_job_id, state=state)
    except BalsamControlRefusal as error:
        return BalsamWireResult.refused(request, str(error))
    except BalsamControlAmbiguous as error:
        return BalsamWireResult.ambiguous(request, str(error))
    except (TypeError, ValueError) as error:
        return BalsamWireResult.refused(request, f"invalid SDK response: {error}")
    except Exception as error:  # External SDK failures are unknown-effect failures.
        return BalsamWireResult.ambiguous(request, f"SDK operation failed: {error}")
    raise AssertionError(f"unhandled wire operation: {request.operation}")


def load_sdk_operations(client_path: str) -> InjectedSDKOperations:
    """Load the pinned SDK adapter after validating the client path.

    The concrete SDK adapter is intentionally a later deployment concern; the
    bridge's process contract is testable with an injected finite adapter.
    """

    os.environ["BALSAM_CLIENT_PATH"] = str(select_balsam_client_path(client_path))
    from .sdk_adapter import DocumentedSDKOperations

    return DocumentedSDKOperations.from_imported_sdk()


def main() -> int:
    """Run one bounded stdin/stdout bridge transaction."""

    try:
        client_path = select_balsam_client_path(environ=os.environ)
        request = decode_request(sys.stdin.buffer.read())
        operations = load_sdk_operations(str(client_path))
        sys.stdout.buffer.write(encode_result(dispatch(request, operations)))
        sys.stdout.buffer.flush()
        return 0
    except Exception as error:
        print(f"bridge failed: {error}", file=sys.stderr)
        return 64


if __name__ == "__main__":  # pragma: no cover - exercised by process tests.
    raise SystemExit(main())
