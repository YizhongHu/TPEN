"""Unit tests for the O6 explicit, bounded ``NvidiaSmiDeviceProbe``."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import threading
import time
from typing import Any

import pytest
from typeguard import TypeCheckError, suppress_type_checks

from tpen.accelerator import (
    AcceleratorIdentity,
    AcceleratorKind,
    DEFAULT_NVIDIA_SMI_EXECUTABLE,
    DEFAULT_NVIDIA_SMI_MAX_OUTPUT_BYTES,
    DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS,
    MAX_NVIDIA_SMI_TIMEOUT_SECONDS,
    NVIDIA_DEVICE_READING_STATUSES,
    NvidiaDeviceCommandResult,
    NvidiaDeviceReading,
    NvidiaDeviceUnavailable,
    NvidiaSmiDeviceProbe,
    canonicalize_nvidia_device_uuid,
    parse_nvidia_smi_query_output,
)


_RAW_UUID = "97933565-57cb-fec7-c442-188c7b300fd3"
_CANONICAL_UUID = f"GPU-{_RAW_UUID}"
_OTHER_UUID = "GPU-11111111-2222-3333-4444-555555555555"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_executable_python_script(path: Path, body: str) -> Path:
    impl = path.with_suffix(".impl.py")
    impl.write_text(body, encoding="utf-8")
    path.write_text(
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(impl))} \"$@\"\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IRUSR)
    return path


def test_constructor_is_pure_and_enforces_hard_ten_second_timeout_cap() -> None:
    """Constructor validates strictly without launching subprocesses or scanning disk."""

    identity = AcceleratorIdentity(AcceleratorKind.CUDA, 0, _RAW_UUID)
    probe = NvidiaSmiDeviceProbe(
        identity=identity,
        executable="/definitely/nonexistent/nvidia-smi-path",
        timeout_seconds=MAX_NVIDIA_SMI_TIMEOUT_SECONDS,
        max_output_bytes=4096,
    )
    assert probe.identity == identity
    assert probe.executable == "/definitely/nonexistent/nvidia-smi-path"
    assert probe.timeout_seconds == 10.0
    assert probe.max_output_bytes == 4096
    assert DEFAULT_NVIDIA_SMI_EXECUTABLE == "nvidia-smi"
    assert DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS == 5.0
    assert MAX_NVIDIA_SMI_TIMEOUT_SECONDS == 10.0
    assert DEFAULT_NVIDIA_SMI_MAX_OUTPUT_BYTES == 65536
    assert NVIDIA_DEVICE_READING_STATUSES == frozenset(
        {"ok", "partial", "unavailable", "error"}
    )

    for bad_timeout in (0, -1.0, 10.001, 15.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="timeout_seconds"):
            NvidiaSmiDeviceProbe(identity=identity, timeout_seconds=bad_timeout)
    with pytest.raises(TypeError, match="timeout_seconds"):
        NvidiaSmiDeviceProbe(identity=identity, timeout_seconds=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_output_bytes"):
        NvidiaSmiDeviceProbe(identity=identity, max_output_bytes=0)
    with pytest.raises(TypeError, match="max_output_bytes"):
        NvidiaSmiDeviceProbe(identity=identity, max_output_bytes=True)  # type: ignore[arg-type]

    for bad_exe in ("", "   ", "nvidia-smi\n--help", "nvidia\x00smi"):
        with pytest.raises(ValueError, match="executable"):
            NvidiaSmiDeviceProbe(identity=identity, executable=bad_exe)
    # Under Typeguard instrumentation (--typecheck on), function-entry checks
    # raise TypeCheckError before entering __init__; suppress_type_checks()
    # below verifies __init__'s own runtime TypeError guards as well.
    with pytest.raises((TypeError, TypeCheckError), match="executable"):
        NvidiaSmiDeviceProbe(identity=identity, executable=123)  # type: ignore[arg-type]
    with pytest.raises((TypeError, TypeCheckError), match="identity"):
        NvidiaSmiDeviceProbe(identity="cuda:0")  # type: ignore[arg-type]
    with pytest.raises((TypeError, TypeCheckError), match="env"):
        NvidiaSmiDeviceProbe(identity=identity, env={"KEY": 1})  # type: ignore[dict-item]
    with suppress_type_checks():
        with pytest.raises(TypeError, match="executable"):
            NvidiaSmiDeviceProbe(identity=identity, executable=123)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="identity"):
            NvidiaSmiDeviceProbe(identity="cuda:0")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="env"):
            NvidiaSmiDeviceProbe(identity=identity, env={"KEY": 1})  # type: ignore[dict-item]


def test_canonicalize_nvidia_device_uuid_normalizes_and_blocks_mig_or_injection() -> None:
    """Valid UUIDs normalize to ``GPU-<lower>``; MIG and injection tokens fail closed."""

    assert canonicalize_nvidia_device_uuid(_RAW_UUID) == _CANONICAL_UUID
    assert canonicalize_nvidia_device_uuid(_RAW_UUID.upper()) == _CANONICAL_UUID
    assert canonicalize_nvidia_device_uuid(f"gpu-{_RAW_UUID.upper()}") == _CANONICAL_UUID
    assert canonicalize_nvidia_device_uuid(_CANONICAL_UUID) == _CANONICAL_UUID

    for mig_token in (
        f"MIG-{_CANONICAL_UUID}/0/0",
        "mig-12345678-1234-1234-1234-1234567890ab",
    ):
        res = canonicalize_nvidia_device_uuid(mig_token)
        assert isinstance(res, NvidiaDeviceUnavailable)
        assert res.status == "unavailable"
        assert res.reason_code == "mig_subdevice_ambiguous"

    for invalid in (
        None,
        123,
        "",
        "MIG-",
        "-i",
        "--help",
        "GPU-0",
        f" {_CANONICAL_UUID}",
        f"{_CANONICAL_UUID} ",
        f"{_CANONICAL_UUID},0",
        f"{_CANONICAL_UUID}\n",
        f"{_CANONICAL_UUID}\r",
        "GPU-zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz",
    ):
        res = canonicalize_nvidia_device_uuid(invalid)
        assert isinstance(res, NvidiaDeviceUnavailable)
        assert res.status == "error"
        assert res.reason_code == "invalid_device_uuid"


def test_selected_identity_binding_refuses_missing_non_cuda_invalid_and_mig_without_calling_command() -> None:
    """Probe never falls back to GPU 0 or launches a command when identity binding fails."""

    invoked: list[list[str]] = []

    def _forbid_runner(argv: list[str], **kwargs: Any) -> NvidiaDeviceCommandResult:
        del kwargs
        invoked.append(list(argv))
        raise AssertionError("command_runner must not be called")

    cases: list[tuple[AcceleratorIdentity | None, dict[str, str] | None, str, str, bool]] = [
        (None, None, "unavailable", "no_selected_identity", False),
        (
            AcceleratorIdentity(AcceleratorKind.CPU, None, None),
            None,
            "unavailable",
            "non_cuda_accelerator",
            False,
        ),
        (
            AcceleratorIdentity(AcceleratorKind.ROCM, 0, _RAW_UUID),
            None,
            "unavailable",
            "non_cuda_accelerator",
            False,
        ),
        (
            AcceleratorIdentity(AcceleratorKind.OTHER, 0, _RAW_UUID),
            None,
            "unavailable",
            "non_cuda_accelerator",
            False,
        ),
        (
            AcceleratorIdentity(AcceleratorKind.CUDA, 0, None),
            None,
            "unavailable",
            "missing_device_uuid",
            False,
        ),
        (
            AcceleratorIdentity(AcceleratorKind.CUDA, 0, ""),
            None,
            "unavailable",
            "missing_device_uuid",
            False,
        ),
        (
            AcceleratorIdentity(AcceleratorKind.CUDA, 0, f"MIG-{_CANONICAL_UUID}/0/0"),
            None,
            "unavailable",
            "mig_subdevice_ambiguous",
            True,
        ),
        (
            AcceleratorIdentity(AcceleratorKind.CUDA, 0, "--query-gpu=name"),
            None,
            "error",
            "invalid_device_uuid",
            False,
        ),
        (
            AcceleratorIdentity(AcceleratorKind.CUDA, 0, _RAW_UUID),
            {"CUDA_VISIBLE_DEVICES": f"MIG-{_CANONICAL_UUID}/0/0,{_OTHER_UUID}"},
            "unavailable",
            "mig_subdevice_ambiguous",
            True,
        ),
        (
            AcceleratorIdentity(AcceleratorKind.CUDA, None, _RAW_UUID),
            {"CUDA_VISIBLE_DEVICES": f"{_CANONICAL_UUID},MIG-{_OTHER_UUID}/0/0"},
            "unavailable",
            "mig_subdevice_ambiguous",
            True,
        ),
    ]

    for identity, env, exp_status, exp_code, exp_mig in cases:
        probe = NvidiaSmiDeviceProbe(
            identity=identity,
            command_runner=_forbid_runner,
            env=env,
        )
        reading = probe.read()
        assert reading.status == exp_status
        assert reading.reason_code == exp_code
        assert reading.mig_ambiguous is exp_mig
        assert reading.shared_device_not_exclusive is True
        assert isinstance(reading.framebuffer_used_bytes, NvidiaDeviceUnavailable)
        assert isinstance(reading.framebuffer_total_bytes, NvidiaDeviceUnavailable)
        assert isinstance(reading.busy_fraction, NvidiaDeviceUnavailable)

    assert invoked == []


def test_cuda_visible_devices_allows_non_mig_selected_index_and_uses_exact_argv() -> None:
    """When selected CUDA index is a full GPU alongside a sibling MIG entry, query proceeds."""

    seen_argv: list[list[str]] = []
    identity = AcceleratorIdentity(AcceleratorKind.CUDA, 1, _RAW_UUID)

    def _runner(argv: list[str], **kwargs: Any) -> NvidiaDeviceCommandResult:
        assert kwargs["timeout_seconds"] == 3.5
        assert kwargs["max_output_bytes"] == 8192
        seen_argv.append(list(argv))
        return NvidiaDeviceCommandResult(
            returncode=0,
            stdout=(
                f"{_CANONICAL_UUID}, 2026/11/24 12:30:00.123, "
                "1024, 40960, 37, Disabled\n"
            ),
        )

    probe = NvidiaSmiDeviceProbe(
        identity=identity,
        executable="nvidia-smi",
        timeout_seconds=3.5,
        max_output_bytes=8192,
        command_runner=_runner,
        env={"CUDA_VISIBLE_DEVICES": f"MIG-{_OTHER_UUID}/0/0,{_CANONICAL_UUID}"},
    )
    reading = probe.read()
    assert seen_argv == [
        [
            "nvidia-smi",
            "-i",
            _CANONICAL_UUID,
            "--query-gpu=uuid,timestamp,memory.used,memory.total,utilization.gpu,mig.mode.current",
            "--format=csv,noheader,nounits",
        ]
    ]
    assert reading.status == "ok"
    assert reading.framebuffer_used_bytes == 1024 * 1024 * 1024
    assert reading.framebuffer_total_bytes == 40960 * 1024 * 1024
    assert reading.busy_fraction == pytest.approx(0.37)
    assert reading.vendor_timestamp == "2026/11/24 12:30:00.123"
    assert reading.observed_uuid == _CANONICAL_UUID
    assert reading.selected_identity == identity
    assert reading.mig_mode == "disabled"
    assert reading.mig_ambiguous is False
    assert reading.shared_device_not_exclusive is True


def test_parser_preserves_zero_readings_and_non_mig_not_supported_gpus() -> None:
    """Legitimate 0 MiB / 0% stay ok, and non-MIG GPUs with [Not Supported] succeed."""

    reading = parse_nvidia_smi_query_output(
        f"\n{_RAW_UUID.upper()}, 2026/11/24 12:31:00.000, 0, 81920, 0, [Not Supported]\r\n",
        expected_uuid=_RAW_UUID,
    )
    assert reading.status == "ok"
    assert reading.framebuffer_used_bytes == 0
    assert reading.framebuffer_total_bytes == 81920 * 1024 * 1024
    assert reading.busy_fraction == 0.0
    assert reading.mig_mode == "not_supported"
    assert reading.mig_ambiguous is False
    assert reading.vendor_timestamp == "2026/11/24 12:31:00.000"


def test_parser_rejects_identity_mismatch_mig_enabled_and_mig_insufficient_permissions() -> None:
    """UUID mismatch, MIG enabled, and ambiguous MIG permissions fail closed."""

    mismatch = parse_nvidia_smi_query_output(
        f"{_OTHER_UUID}, 2026/11/24 12:32:00.000, 100, 40960, 15, Disabled\n",
        expected_uuid=_CANONICAL_UUID,
    )
    assert mismatch.status == "error"
    assert mismatch.reason_code == "identity_mismatch"
    assert mismatch.observed_uuid == _OTHER_UUID
    assert isinstance(mismatch.framebuffer_used_bytes, NvidiaDeviceUnavailable)
    assert isinstance(mismatch.framebuffer_total_bytes, NvidiaDeviceUnavailable)
    assert isinstance(mismatch.busy_fraction, NvidiaDeviceUnavailable)

    mig_enabled = parse_nvidia_smi_query_output(
        f"{_CANONICAL_UUID}, 2026/11/24 12:32:01.000, 100, 40960, 15, Enabled\n",
        expected_uuid=_CANONICAL_UUID,
    )
    assert mig_enabled.status == "unavailable"
    assert mig_enabled.reason_code == "mig_subdevice_ambiguous"
    assert mig_enabled.mig_mode == "enabled"
    assert mig_enabled.mig_ambiguous is True
    assert mig_enabled.vendor_timestamp == "2026/11/24 12:32:01.000"
    assert isinstance(mig_enabled.framebuffer_used_bytes, NvidiaDeviceUnavailable)
    assert mig_enabled.framebuffer_used_bytes.reason_code == "mig_subdevice_ambiguous"
    assert isinstance(mig_enabled.framebuffer_total_bytes, NvidiaDeviceUnavailable)
    assert isinstance(mig_enabled.busy_fraction, NvidiaDeviceUnavailable)

    mig_perm = parse_nvidia_smi_query_output(
        f"{_CANONICAL_UUID}, 2026/11/24 12:32:02.000, 100, 40960, 15, [Insufficient Permissions]\n",
        expected_uuid=_CANONICAL_UUID,
    )
    assert mig_perm.status == "unavailable"
    assert mig_perm.reason_code == "mig_status_unavailable"
    assert mig_perm.mig_mode == "insufficient_permissions"
    assert mig_perm.mig_ambiguous is True
    assert isinstance(mig_perm.framebuffer_used_bytes, NvidiaDeviceUnavailable)
    assert isinstance(mig_perm.busy_fraction, NvidiaDeviceUnavailable)


def test_parser_handles_per_field_unavailable_partial_and_malformed_outputs() -> None:
    """Per-field [N/A]/[Not Supported]/[Insufficient Permissions] yield typed partial readings."""

    partial = parse_nvidia_smi_query_output(
        f"{_CANONICAL_UUID}, [N/A], 512, 40960, [Not Supported], Disabled\n",
        expected_uuid=_CANONICAL_UUID,
    )
    assert partial.status == "partial"
    assert partial.reason_code == "metric_not_supported"
    assert partial.vendor_timestamp is None
    assert partial.framebuffer_used_bytes == 512 * 1024 * 1024
    assert partial.framebuffer_total_bytes == 40960 * 1024 * 1024
    assert isinstance(partial.busy_fraction, NvidiaDeviceUnavailable)
    assert partial.busy_fraction.status == "unavailable"
    assert partial.busy_fraction.reason_code == "metric_not_supported"

    all_unavail = parse_nvidia_smi_query_output(
        f"{_CANONICAL_UUID}, 2026/11/24 12:33:00.000, [N/A], [Not Supported], [Insufficient Permissions], Disabled\n",
        expected_uuid=_CANONICAL_UUID,
    )
    assert all_unavail.status == "unavailable"
    assert all_unavail.reason_code == "metric_not_available"
    assert isinstance(all_unavail.framebuffer_used_bytes, NvidiaDeviceUnavailable)
    assert isinstance(all_unavail.framebuffer_total_bytes, NvidiaDeviceUnavailable)
    assert isinstance(all_unavail.busy_fraction, NvidiaDeviceUnavailable)
    assert all_unavail.busy_fraction.reason_code == "permission_denied"

    for bad_row in (
        "",
        "   \n\n",
        f"{_CANONICAL_UUID}, 2026/11/24 12:00:00.000, 10, 20, 5\n",
        f"{_CANONICAL_UUID}, 2026/11/24 12:00:00.000, 10, 20, 5, Disabled, extra\n",
        (
            f"{_CANONICAL_UUID}, 2026/11/24 12:00:00.000, 10, 20, 5, Disabled\n"
            f"{_CANONICAL_UUID}, 2026/11/24 12:00:01.000, 11, 20, 6, Disabled\n"
        ),
        f"{_CANONICAL_UUID}, 2026/11/24 12:00:00.000, 10, 20, 5, UnknownMode\n",
    ):
        err = parse_nvidia_smi_query_output(bad_row, expected_uuid=_CANONICAL_UUID)
        assert err.status == "error"
        assert err.reason_code == "malformed_output"

    for bad_busy in ("-1", "101", "NaN", "inf", "abc", "1_0"):
        res = parse_nvidia_smi_query_output(
            f"{_CANONICAL_UUID}, 2026/11/24 12:00:00.000, 100, 40960, {bad_busy}, Disabled\n",
            expected_uuid=_CANONICAL_UUID,
        )
        assert res.status == "partial"
        assert res.reason_code == "malformed_output"
        assert isinstance(res.busy_fraction, NvidiaDeviceUnavailable)
        assert res.busy_fraction.status == "error"

    neg_mem = parse_nvidia_smi_query_output(
        f"{_CANONICAL_UUID}, 2026/11/24 12:00:00.000, -5, -1, 150, Disabled\n",
        expected_uuid=_CANONICAL_UUID,
    )
    assert neg_mem.status == "error"
    assert neg_mem.reason_code == "malformed_output"


def test_nonzero_exit_codes_classify_permission_not_found_and_generic_error() -> None:
    """Non-zero exit statuses map to typed errors without any fallback query."""

    identity = AcceleratorIdentity(AcceleratorKind.CUDA, 0, _RAW_UUID)
    cases = [
        (1, "Insufficient Permissions to query GPU", "permission_denied"),
        (6, "No devices were found matching UUID", "device_not_found"),
        (2, "NVML driver/library version mismatch", "command_failed"),
    ]
    for code, stderr, expected_code in cases:
        probe = NvidiaSmiDeviceProbe(
            identity=identity,
            command_runner=lambda argv, _c=code, _e=stderr, **kw: NvidiaDeviceCommandResult(
                returncode=_c, stdout="", stderr=_e
            ),
        )
        reading = probe.read()
        assert reading.status == "error"
        assert reading.reason_code == expected_code


def test_real_bounded_subprocess_unavailable_ok_timeout_cancel_and_output_limit(
    tmp_path: Path,
) -> None:
    """Real supervised subprocess handles missing exe, ok CSV, timeout, cancel, and output cap."""

    identity = AcceleratorIdentity(AcceleratorKind.CUDA, 0, _RAW_UUID)

    # 1. Unavailable executable (explicit path and empty PATH)
    missing_path_reading = NvidiaSmiDeviceProbe(
        identity=identity,
        executable=tmp_path / "does-not-exist-nvidia-smi",
    ).read()
    assert missing_path_reading.status == "unavailable"
    assert missing_path_reading.reason_code == "executable_unavailable"

    empty_path_reading = NvidiaSmiDeviceProbe(
        identity=identity,
        executable="definitely-not-installed-nvidia-smi-o6",
        env={"PATH": ""},
    ).read()
    assert empty_path_reading.status == "unavailable"
    assert empty_path_reading.reason_code == "executable_unavailable"

    # 2. Healthy real script
    ok_script = _write_executable_python_script(
        tmp_path / "fake_nvidia_smi_ok.py",
        (
            "import sys\n"
            f"assert '-i' in sys.argv and {_CANONICAL_UUID!r} in sys.argv\n"
            f"print('{_CANONICAL_UUID}, 2026/11/24 12:40:00.500, 2048, 81920, 64, Disabled')\n"
        ),
    )
    ok_reading = NvidiaSmiDeviceProbe(
        identity=identity,
        executable=ok_script,
        timeout_seconds=5.0,
    ).read()
    assert ok_reading.status == "ok"
    assert ok_reading.framebuffer_used_bytes == 2048 * 1024 * 1024
    assert ok_reading.framebuffer_total_bytes == 81920 * 1024 * 1024
    assert ok_reading.busy_fraction == pytest.approx(0.64)
    assert ok_reading.vendor_timestamp == "2026/11/24 12:40:00.500"

    # 3. Real timeout kills sleeping child cleanly without zombie processes
    pid_file = tmp_path / "sleep_child.pid"
    sleep_script = _write_executable_python_script(
        tmp_path / "fake_nvidia_smi_sleep.py",
        (
            "import os, time\n"
            f"with open({str(pid_file)!r}, 'w', encoding='utf-8') as f:\n"
            "    f.write(str(os.getpid()))\n"
            "    f.flush()\n"
            "time.sleep(30.0)\n"
        ),
    )
    t0 = time.monotonic()
    timeout_reading = NvidiaSmiDeviceProbe(
        identity=identity,
        executable=sleep_script,
        timeout_seconds=0.4,
    ).read()
    elapsed = time.monotonic() - t0
    assert timeout_reading.status == "error"
    assert timeout_reading.reason_code == "read_timeout"
    assert elapsed < 3.0
    if pid_file.exists():
        child_pid = int(pid_file.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)

    # 4. Early cancel_event terminates in-flight child promptly well before timeout
    cancel_event = threading.Event()
    timer = threading.Timer(0.15, cancel_event.set)
    timer.start()
    try:
        t0 = time.monotonic()
        cancel_reading = NvidiaSmiDeviceProbe(
            identity=identity,
            executable=sleep_script,
            timeout_seconds=5.0,
        ).read(cancel_event=cancel_event)
        cancel_elapsed = time.monotonic() - t0
    finally:
        timer.cancel()
    assert cancel_reading.status == "unavailable"
    assert cancel_reading.reason_code == "read_cancelled"
    assert cancel_elapsed < 2.0

    # 5. Live output-limit truncation kills noisy child with input_limit_exceeded
    noisy_script = _write_executable_python_script(
        tmp_path / "fake_nvidia_smi_noisy.py",
        (
            "import sys\n"
            "chunk = 'X' * 4096\n"
            "for _ in range(64):\n"
            "    sys.stdout.write(chunk)\n"
            "    sys.stdout.flush()\n"
        ),
    )
    limit_reading = NvidiaSmiDeviceProbe(
        identity=identity,
        executable=noisy_script,
        timeout_seconds=5.0,
        max_output_bytes=1024,
    ).read()
    assert limit_reading.status == "error"
    assert limit_reading.reason_code == "input_limit_exceeded"


def test_nvidia_device_reading_enforces_status_invariants() -> None:
    """``NvidiaDeviceReading`` rejects contradictory status/field combinations."""

    unavail = NvidiaDeviceUnavailable(reason="missing", reason_code="metric_not_available")

    with pytest.raises(ValueError, match="status='ok'"):
        NvidiaDeviceReading(
            status="ok",
            framebuffer_used_bytes=0,
            framebuffer_total_bytes=1024,
            busy_fraction=unavail,
        )
    with pytest.raises(ValueError, match="mig_ambiguous"):
        NvidiaDeviceReading(
            status="ok",
            framebuffer_used_bytes=0,
            framebuffer_total_bytes=1024,
            busy_fraction=0.5,
            mig_ambiguous=True,
        )
    with pytest.raises(ValueError, match="status='partial'"):
        NvidiaDeviceReading(
            status="partial",
            framebuffer_used_bytes=0,
            framebuffer_total_bytes=1024,
            busy_fraction=unavail,
            reason_code=None,
        )
    with pytest.raises(ValueError, match="status='unavailable'"):
        NvidiaDeviceReading(
            status="unavailable",
            framebuffer_used_bytes=0,
            framebuffer_total_bytes=unavail,
            busy_fraction=unavail,
            reason_code="x",
        )
    with pytest.raises(ValueError, match="busy_fraction"):
        NvidiaDeviceReading(
            status="ok",
            framebuffer_used_bytes=0,
            framebuffer_total_bytes=1024,
            busy_fraction=1.5,
        )


def test_nvidia_smi_and_accelerator_imports_are_torch_and_psutil_free() -> None:
    """Importing ``tpen._nvidia_smi`` and ``tpen.accelerator`` never pulls in torch/psutil."""

    code = (
        "import sys\n"
        "import tpen._nvidia_smi\n"
        "import tpen.accelerator\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "assert 'psutil' not in sys.modules, 'psutil was imported'\n"
        "p = tpen.accelerator.NvidiaSmiDeviceProbe()\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'psutil' not in sys.modules\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
