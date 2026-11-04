"""Counting targets used to prove staged composition does not instantiate."""

CONSTRUCTIONS = 0


def reset() -> None:
    """Reset the process-local construction counter."""

    global CONSTRUCTIONS
    CONSTRUCTIONS = 0


class CountedAllocation:
    """Allocation target whose construction is observable by tests."""

    def __init__(self, worker=None, **kwargs):
        global CONSTRUCTIONS
        CONSTRUCTIONS += 1
        self.worker = worker
        self.options = kwargs


class CountedWorker:
    """Worker target whose construction is observable by tests."""

    def __init__(self, command="worker", **kwargs):
        global CONSTRUCTIONS
        CONSTRUCTIONS += 1
        self.command = command
        self.options = kwargs
