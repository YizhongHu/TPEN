"""Strict and counting targets for the independent P4 review probes."""

CONSTRUCTIONS = 0


def reset() -> None:
    """Reset the process-local construction counter."""

    global CONSTRUCTIONS
    CONSTRUCTIONS = 0


def _count() -> None:
    global CONSTRUCTIONS
    CONSTRUCTIONS += 1


class StrictAllocation:
    """Target with no kwargs, exposing accidental environment leakage."""

    def __init__(self, label):
        _count()
        self.label = label


class Environment:
    """Environment target accepting ordinary options."""

    def __init__(self, python="x", **kw):
        _count()
        self.python = python
        self.options = kw
        for key, value in kw.items():
            setattr(self, key, value)


class Dispatcher:
    """Dispatcher target used by the plan-section-9 shape probe."""

    def __init__(self, worker=None, **kw):
        _count()
        self.worker = worker
        self.options = kw


class Runner:
    """Runner target used inside the nested worker marker."""

    def __init__(self, command="r", **kw):
        _count()
        self.command = command
        self.options = kw


class Hijacked:
    """Sentinel proving that an injected target is never constructed."""

    def __init__(self):
        _count()

