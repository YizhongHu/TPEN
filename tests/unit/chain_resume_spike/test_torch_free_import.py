"""The shared fixture must import WITHOUT torch, and that is enforced by a test.

WHY THIS IS A TEST AND NOT A CONVENTION. The torch-free property is what lets
R1/R2/R3 exercise continuation, retry classification, generation preservation
and lost-acknowledgement recovery on a facility worker with no GPU and no torch
import -- decoupling backend-mechanics testing from facility runtime
qualification, which the plan flags as the largest schedule risk in the
five-lane program. If a shared module ever acquires a torch import, that
property dies SILENTLY and three candidate lanes inherit a facility dependency
nobody chose. Measured in the previous round: an ``import torch`` inserted into
shared ``fixture.py`` behind a sentinel left all 88 unit tests green.

WHY A SUBPROCESS. An in-process check is vacuous exactly when it matters. The
native arm (``test_native_reference_arm.py``) imports torch into the same pytest
process, so ``"torch" in sys.modules`` inside a live session reports the native
arm's import, not the shared package's. That is the same failure class as the
in-process resume this lane forbids: state surviving in memory makes the check
pass, or fail, for a reason unrelated to the thing under test.

``native_probe`` is EXEMT and is expected to import torch -- it is the native
arm. Its torch imports are function-local, matching the
``tpen/checkpoint/rng.py`` 100/150/222/271 precedent, so importing the shared
surface never drags torch in transitively.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Modules R1/R2/R3 touch. Every one must import torch-free.
SHARED_MODULES = (
    "tests.helpers.chain_resume_spike.fixture",
    "tests.helpers.chain_resume_spike.entrypoint",
    "tests.helpers.chain_resume_spike.faults",
    "tests.helpers.chain_resume_spike.parity",
    "tests.helpers.chain_resume_spike.restore_limbs",
    "tests.helpers.chain_resume_spike.receipt",
    "tests.helpers.chain_resume_spike.identity",
    "tests.helpers.chain_resume_spike.outcomes",
)

#: The native arm. Exempt: it is SUPPOSED to reach torch, but only when called.
NATIVE_MODULE = "tests.helpers.chain_resume_spike.native_probe"

_PROBE = textwrap.dedent(
    """
    import importlib, json, sys
    target = sys.argv[1]
    importlib.import_module(target)
    print(json.dumps({
        "module": target,
        "torch_loaded": "torch" in sys.modules,
        "loaded_count": len(sys.modules),
    }))
    """
)


def _import_in_clean_process(module: str, *, extra_path: Path | None = None) -> dict:
    """Import ``module`` in a fresh interpreter and report whether torch loaded.

    A fresh process is the only way to get a genuinely clean ``sys.modules``.
    ``sys.executable`` is absolute; a bare interpreter name is forbidden across
    this package.
    """

    environment = None
    if extra_path is not None:
        import os

        environment = dict(os.environ)
        environment["PYTHONPATH"] = (
            str(extra_path) + os.pathsep + environment.get("PYTHONPATH", "")
        )

    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, module],
        cwd=str(_REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"importing {module} failed (rc={completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_detector_fires_on_a_module_that_imports_torch(tmp_path) -> None:
    """INSTRUMENT CHECK, run before the real result is believed.

    Uses a STUB torch on the probe's path, so this control works identically on
    a host where torch is installed and on one where it is not -- this
    workstation has no torch, and a control that depended on the real package
    would silently stop being a control here.

    Without this, "no shared module loaded torch" is equally consistent with a
    probe that can never observe a torch import at all.
    """

    (tmp_path / "torch.py").write_text("VERSION = 'stub'\n", encoding="utf-8")
    (tmp_path / "cr_r0_poisoned.py").write_text("import torch\n", encoding="utf-8")

    result = _import_in_clean_process("cr_r0_poisoned", extra_path=tmp_path)
    assert result["torch_loaded"] is True, (
        "the probe did not notice a module importing torch, so its clean "
        "results below establish nothing"
    )


def test_the_detector_does_not_fire_on_a_module_that_leaves_torch_alone(
    tmp_path,
) -> None:
    """The other direction: a probe that always reported torch would also be useless."""

    (tmp_path / "torch.py").write_text("VERSION = 'stub'\n", encoding="utf-8")
    (tmp_path / "cr_r0_clean.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = _import_in_clean_process("cr_r0_clean", extra_path=tmp_path)
    assert result["torch_loaded"] is False


@pytest.mark.parametrize("module", SHARED_MODULES, ids=lambda name: name.rsplit(".", 1)[-1])
def test_a_shared_module_imports_without_pulling_in_torch(module: str) -> None:
    """The clause itself, module by module so a regression names the file."""

    result = _import_in_clean_process(module)
    assert result["torch_loaded"] is False, (
        f"{module} pulled torch into sys.modules on import. The shared fixture "
        "must run on a worker with no torch installed; this change would give "
        "R1/R2/R3 a facility dependency nobody chose."
    )


def test_importing_every_shared_module_together_stays_torch_free() -> None:
    """One process importing all of them, in case a pair only interacts jointly.

    Per-module nodes cannot see an import that only happens when two modules are
    loaded together -- a conditional import keyed on another module being
    present, for instance.

    THE POPULATION IS ``SHARED_MODULES``, A HAND-MAINTAINED LIST, not a
    computed set of this package's modules. "Every shared module" in the node
    name means every member of that list; a shared module absent from it is
    outside the claim and this node cannot report it. Bounded by the round-5
    claims sweep.
    """

    joint = "; ".join(f"import {name}" for name in SHARED_MODULES)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; {joint}; print('torch' in sys.modules)",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False", completed.stdout


def test_the_native_probe_keeps_its_torch_imports_function_local() -> None:
    """``native_probe`` is exempt, but must not drag torch in AT IMPORT.

    The exemption is that it may USE torch, not that importing it may cost a
    torch import. If it imported torch at module scope, anything importing the
    package's native surface -- including a future shared module that reached
    for one helper from it -- would acquire the dependency transitively, and the
    per-module assertions above would not catch it.
    """

    result = _import_in_clean_process(NATIVE_MODULE)
    assert result["torch_loaded"] is False, (
        "native_probe imports torch at module scope; move it inside the "
        "functions that need it, as tpen/checkpoint/rng.py does"
    )
