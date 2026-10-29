"""Explicitly selected E2E canary; filename keeps normal repository collection green."""


def test_deliberate_failure() -> None:
    """Fail only when the operator explicitly selects this path."""
    assert False, "intentional harness verdict canary"
