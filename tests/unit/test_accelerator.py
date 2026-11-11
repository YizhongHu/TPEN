"""Contract tests for accelerator backend resolution.

These defend the property that makes the module exist: no call site may name a
concrete backend, because on a non-CUDA build ``torch.cuda`` still exists and
reports unavailable rather than raising, so a hardcoded CUDA path degrades to
CPU silently.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tpen.accelerator import (  # noqa: E402
    AcceleratorIdentity,
    AcceleratorKind,
    AllocatorUnavailable,
    TorchAllocatorPeakProbe,
    canonical_device,
    current_accelerator_type,
    device_module,
    seed_all,
    synchronize,
)


def test_allocator_probe_uses_the_configured_device_and_owns_one_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tpen.accelerator as accelerator

    calls: list[tuple[str, object]] = []

    class Backend:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            calls.append(("current_device", None))
            return 0

        @staticmethod
        def get_device_properties(index: int):
            calls.append(("properties", index))
            return type("Properties", (), {"uuid": "GPU-1"})()

        @staticmethod
        def reset_peak_memory_stats(device: object) -> None:
            calls.append(("reset", device))

        @staticmethod
        def max_memory_allocated(device: object) -> int:
            calls.append(("allocated", device))
            return 3 * 1024 * 1024

        @staticmethod
        def max_memory_reserved(device: object) -> int:
            calls.append(("reserved", device))
            return 8 * 1024 * 1024

        @staticmethod
        def device_count() -> int:
            calls.append(("count", None))
            return 4

    class FakeTorch:
        device = staticmethod(torch.device)
        version = type("Version", (), {"hip": None})()

    monkeypatch.setattr(accelerator, "_torch", lambda feature: FakeTorch)
    monkeypatch.setattr(accelerator, "device_module", lambda *args, **kwargs: Backend)
    probe = TorchAllocatorPeakProbe(torch.device("cuda:1"))

    identity = probe.reset()
    usage = probe.read()

    assert identity.kind is AcceleratorKind.CUDA
    assert identity.index == 1
    assert identity.uuid == "GPU-1"
    assert usage.allocated_mb == 3.0
    assert usage.reserved_mb == 8.0
    assert usage.device_count == 4
    assert calls.count(("reset", torch.device("cuda:1"))) == 1
    assert calls.count(("allocated", torch.device("cuda:1"))) == 1
    assert calls.count(("reserved", torch.device("cuda:1"))) == 1


def test_allocator_probe_coerces_torch_uuid_objects_to_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tpen.accelerator as accelerator

    class FakeUUID:
        def __str__(self) -> str:
            return "97933565-57cb-fec7-c442-188c7b300fd3"

        def __repr__(self) -> str:
            return "<fake torch uuid object>"

    class Backend:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(index: int):
            assert index == 1
            return type("Properties", (), {"uuid": FakeUUID()})()

    class FakeTorch:
        device = staticmethod(torch.device)
        version = type("Version", (), {"hip": None})()

    monkeypatch.setattr(accelerator, "_torch", lambda feature: FakeTorch)
    monkeypatch.setattr(accelerator, "device_module", lambda *args, **kwargs: Backend)

    identity = TorchAllocatorPeakProbe("cuda:1")._identity()

    assert identity.uuid == "97933565-57cb-fec7-c442-188c7b300fd3"
    assert isinstance(identity.uuid, str)


def test_allocator_probe_distinguishes_rocm_and_types_unavailable_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tpen.accelerator as accelerator

    class FakeTorch:
        device = staticmethod(torch.device)
        version = type("Version", (), {"hip": "6.2"})()

    class Backend:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setattr(accelerator, "_torch", lambda feature: FakeTorch)
    monkeypatch.setattr(accelerator, "device_module", lambda *args, **kwargs: Backend)
    usage = TorchAllocatorPeakProbe("cuda:3").read()

    assert usage.identity.kind is AcceleratorKind.ROCM
    assert usage.identity.index == 3
    assert isinstance(usage.allocated_mb, AllocatorUnavailable)
    assert isinstance(usage.reserved_mb, AllocatorUnavailable)


def test_allocator_probe_pins_an_indexless_configured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tpen.accelerator as accelerator

    current_devices = iter((0, 1))
    calls: list[tuple[str, object]] = []

    class Backend:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            index = next(current_devices)
            calls.append(("current_device", index))
            return index

        @staticmethod
        def reset_peak_memory_stats(device: object) -> None:
            calls.append(("reset", device))

        @staticmethod
        def max_memory_allocated(device: object) -> int:
            calls.append(("allocated", device))
            return 3 * 1024 * 1024

        @staticmethod
        def max_memory_reserved(device: object) -> int:
            calls.append(("reserved", device))
            return 5 * 1024 * 1024

        @staticmethod
        def device_count() -> int:
            return 2

    class FakeTorch:
        device = staticmethod(torch.device)
        version = type("Version", (), {"hip": None})()

    monkeypatch.setattr(accelerator, "_torch", lambda feature: FakeTorch)
    monkeypatch.setattr(accelerator, "device_module", lambda *args, **kwargs: Backend)

    probe = TorchAllocatorPeakProbe("cuda")
    reset_identity = probe.reset()
    usage = probe.read()

    expected_device = torch.device("cuda:0")
    assert reset_identity == AcceleratorIdentity(
        kind=AcceleratorKind.CUDA,
        index=0,
        uuid=None,
    )
    assert usage.identity.index == 0
    assert usage.allocated_mb == 3.0
    assert usage.reserved_mb == 5.0
    assert calls == [
        ("current_device", 0),
        ("reset", expected_device),
        ("allocated", expected_device),
        ("reserved", expected_device),
    ]


def test_allocator_probe_reset_reports_unavailable_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tpen.accelerator as accelerator

    class Backend:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        device = staticmethod(torch.device)
        version = type("Version", (), {"hip": None})()

    monkeypatch.setattr(accelerator, "_torch", lambda feature: FakeTorch)
    monkeypatch.setattr(accelerator, "device_module", lambda *args, **kwargs: Backend)

    result = TorchAllocatorPeakProbe("cuda:0").reset()

    assert isinstance(result, AllocatorUnavailable)


def test_device_module_resolves_backend_per_device_type() -> None:
    # The whole point: one call maps any device type to its own backend module,
    # so no caller has to branch on cuda/xpu/cpu.
    assert device_module("cpu") is torch.cpu
    assert device_module("cuda") is torch.cuda
    assert device_module("xpu") is torch.xpu


def test_device_module_accepts_torch_device_and_indexed_device() -> None:
    assert device_module(torch.device("cpu")) is torch.cpu
    assert device_module("cuda:3") is torch.cuda


def test_device_module_none_follows_the_active_accelerator() -> None:
    # None must agree with the reported accelerator type, whatever this host is.
    assert device_module() is torch.get_device_module(current_accelerator_type())


def test_current_accelerator_type_is_a_real_device_type() -> None:
    device_type = current_accelerator_type()
    assert device_type in {"cpu", "cuda", "xpu", "mps", "hpu"}
    # Must be usable as a device type, not a decorated string.
    assert torch.device(device_type).type == device_type


def test_canonical_device_leaves_cpu_index_free() -> None:
    # Tensors report CPU index-free, so adding an index would break comparison.
    assert canonical_device("cpu") == torch.device("cpu")
    assert canonical_device(torch.device("cpu")) == torch.device("cpu")


def test_canonical_device_preserves_an_explicit_index() -> None:
    assert canonical_device("cuda:2") == torch.device("cuda", 2)
    assert canonical_device("xpu:1") == torch.device("xpu", 1)


def test_canonical_device_is_idempotent() -> None:
    once = canonical_device("cpu")
    assert canonical_device(once) == once


def test_canonical_device_passes_through_a_device_without_an_accelerator_module() -> None:
    # `meta` is a valid device type with no registered module, so there is no
    # current device to resolve. It must pass through rather than raising, or
    # MetropolisSampler's fail-loud device-mismatch check would surface a torch
    # internal error instead of its own message.
    assert canonical_device("meta") == torch.device("meta")
    assert canonical_device(torch.device("meta")) == torch.device("meta")


def test_canonical_device_of_unavailable_accelerator_stays_index_free() -> None:
    # Guards the CPU-only/CI path: an index must never be invented for a device
    # that is not present, because that would fabricate a comparison mismatch.
    absent = "xpu" if not torch.xpu.is_available() else "cuda"
    if torch.get_device_module(absent).is_available():
        pytest.skip("needs an accelerator that is absent on this host")
    assert canonical_device(absent) == torch.device(absent)


def test_canonical_device_indexes_an_available_accelerator() -> None:
    device_type = current_accelerator_type()
    module = torch.get_device_module(device_type)
    if device_type == "cpu" or not module.is_available():
        pytest.skip("needs a live accelerator")
    resolved = canonical_device(device_type)
    assert resolved.type == device_type
    assert resolved.index == module.current_device()


def test_canonical_device_rejects_an_unknown_device_type() -> None:
    with pytest.raises(RuntimeError):
        canonical_device("definitely_not_a_backend")


def test_synchronize_is_safe_without_an_accelerator() -> None:
    # Must not raise on a CPU-only host; timing callbacks call this per event.
    synchronize("cpu")
    synchronize()


def test_seed_all_is_reproducible_and_never_raises() -> None:
    # Run-level global accelerator seeding must work on every backend, including
    # hosts where the accelerator module exposes no seeding at all.
    seed_all(1234)
    seed_all(1234, feature="test seeding")

    device_type = current_accelerator_type()
    module = torch.get_device_module(device_type)
    if device_type == "cpu" or not module.is_available():
        pytest.skip("needs a live accelerator to observe seeded state")
    seed_all(4321)
    first = torch.randn(8, device=device_type)
    seed_all(4321)
    second = torch.randn(8, device=device_type)
    assert torch.equal(first, second)


def test_seed_all_does_not_disturb_component_generators() -> None:
    # ADR-013: run-level global seeding must not reach a component-owned
    # generator, or a resumed Markov chain would silently change.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    expected = torch.randn(4, generator=generator, dtype=torch.float64)

    generator.manual_seed(7)
    seed_all(999999)
    observed = torch.randn(4, generator=generator, dtype=torch.float64)
    assert torch.equal(expected, observed)


def test_device_event_timer_uses_backend_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elapsed device time comes from events, never a host clock."""

    import tpen.accelerator as accelerator

    class Event:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing
            self.recorded = False

        def record(self) -> None:
            self.recorded = True

        def elapsed_time(self, other: object) -> float:
            assert self.recorded
            assert isinstance(other, Event) and other.recorded
            return 250.0

    class Backend:

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            return None
    Backend.Event = Event

    monkeypatch.setattr(accelerator, "device_module", lambda *args, **kwargs: Backend)
    timer = accelerator.device_event_timer()
    timer.start()
    assert timer.stop() == 0.25


def test_device_event_timer_rejects_unavailable_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in device timing must not silently fall back to host duration."""

    import tpen.accelerator as accelerator

    class Backend:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setattr(accelerator, "device_module", lambda *args, **kwargs: Backend)
    with pytest.raises(RuntimeError, match="available accelerator"):
        accelerator.device_event_timer()
