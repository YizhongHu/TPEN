"""Facility-aware GPU strata for the He-cutover smoke."""

from __future__ import annotations

from dataclasses import dataclass

# Siblings are loaded study-scoped, not by bare import: experiments/ has several
# same-named modules and the first study loaded would otherwise own the bare name
# for every study after it. See experiments/toolkit/study_imports.py.
import importlib.util as _tpen_importlib
import sys as _tpen_sys
from pathlib import Path as _TpenPath

if "_tpen_study_imports" not in _tpen_sys.modules:
    _tpen_spec = _tpen_importlib.spec_from_file_location(
        "_tpen_study_imports",
        _TpenPath(__file__).resolve().parents[3] / "experiments" / "toolkit" / "study_imports.py",
    )
    _tpen_module = _tpen_importlib.module_from_spec(_tpen_spec)
    _tpen_sys.modules["_tpen_study_imports"] = _tpen_module
    _tpen_spec.loader.exec_module(_tpen_module)
sibling = _tpen_sys.modules["_tpen_study_imports"].sibling



@dataclass(frozen=True)
class PolarisStratum:
    name: str = "a100_40gb"
    required_device_substring: str = "a100-sxm4-40gb"
    queue: str = "debug"
    wall_limit_min: int = 60


POLARIS = PolarisStratum()
POLARIS_SCALING = PolarisStratum(queue="debug-scaling")
POLARIS_CAPACITY = PolarisStratum(queue="capacity", wall_limit_min=168 * 60)


def validate_placement(*, facility: str, partition: str, stratum: str, timeout_min: int):
    """Validate a Cannon or Polaris placement without weakening either policy."""

    if facility == "cannon":
        hev1 = sibling(__file__, "hev1")

        return hev1.strata.validate_canary_gpu_placement(
            partition=partition, stratum_name=stratum, timeout_min=timeout_min
        )
    if facility not in {"polaris", "polaris_scaling"}:
        raise ValueError(f"unknown facility {facility!r}")
    placements = {item.queue: item for item in (POLARIS, POLARIS_SCALING, POLARIS_CAPACITY)}
    placement = placements.get(partition)
    if placement is None or stratum != placement.name:
        raise ValueError(
            "Polaris requires debug or capacity on a100_40gb; debug-scaling is also supported"
        )
    if timeout_min <= 0 or timeout_min > placement.wall_limit_min:
        raise ValueError(
            f"Polaris {partition} wall time must be in 1..{placement.wall_limit_min} minutes"
        )
    return placement


def check_delivered_device(*, facility: str, stratum: str, delivered: str | None) -> None:
    """Fail unless the allocation delivered the requested device stratum."""

    if facility == "cannon":
        hev1 = sibling(__file__, "hev1")

        hev1.strata.check_delivered_device(stratum_name=stratum, delivered=delivered)
        return
    if stratum != POLARIS.name or POLARIS.required_device_substring not in str(delivered).lower():
        raise RuntimeError(
            f"delivered device {delivered!r} does not match Polaris {POLARIS.name!r}; "
            f"required substring {POLARIS.required_device_substring!r}"
        )
