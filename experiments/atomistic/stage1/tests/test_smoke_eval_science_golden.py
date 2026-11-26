"""Exact-base byte goldens for unchanged science manifests."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA,
    HOOKE2_ACT50_A27,
    HOOKE2_ACT50_SIGMOID_PROFILE,
    HOOKE_A37,
    HOOKE_SCIENCE_PROFILE,
    STAGE1_PROTOCOL,
    bind_evaluation_manifest,
    build_manifest,
    config_identities,
    manifest_hash,
    planned_physical_system_ids,
)


ROOT = Path(__file__).parents[1]
REPO_ROOT = ROOT.parents[2]
CONFIG_ROOT = ROOT / "configs"


def _science_snapshot() -> bytes:
    """Serialize native and ACT50 phase-1/phase-2 science rows."""

    snapshots: dict[str, object] = {}
    for name, system, profile, roster in (
        ("native", "hooke1", HOOKE_SCIENCE_PROFILE, HOOKE_A37),
        ("act50", "hooke2", HOOKE2_ACT50_SIGMOID_PROFILE, HOOKE2_ACT50_A27),
    ):
        config_profile = None if profile == HOOKE_SCIENCE_PROFILE else profile
        checkpoints = list(range(
            0,
            50_001 if config_profile is not None else STAGE1_PROTOCOL.lineage_updates + 1,
            1_000,
        ))
        phase1 = build_manifest(
            protocol=STAGE1_PROTOCOL,
            source_sha="a" * 40,
            config_sha256=config_identities(
                config_root=CONFIG_ROOT, systems=(system,), science_profile=config_profile,
            ),
            checkpoints=checkpoints,
            lineage_seeds=[11, 12, 13], checkpoint_sha256={},
            results_root="/fixture/stage1-golden", systems=(system,), phase=1,
            science_profile=profile,
            evaluation_selection="A27" if config_profile is not None else "A37",
            include_smoke=False, auto_overlay=True,
        )
        phase1_hash = manifest_hash(phase1)
        evidence = {
            "schema": CHECKPOINT_EVIDENCE_SCHEMA,
            "phase1_manifest_hash": phase1_hash,
            "phase1_train_row_hashes": {
                f"{row['system']}:{row['lineage']}": manifest_hash([row])
                for row in phase1
            },
            "checkpoints": {
                f"{system}:{lineage}:{checkpoint}": "c" * 64
                for lineage in range(STAGE1_PROTOCOL.lineages)
                for checkpoint in roster
            },
        }
        phase2 = bind_evaluation_manifest(
            phase1_rows=phase1, checkpoints=list(roster), checkpoint_evidence=evidence,
            expected_physical_system_id=planned_physical_system_ids(),
            source_sha="a" * 40,
            config_sha256=config_identities(
                config_root=CONFIG_ROOT, systems=(system,), science_profile=config_profile,
            ),
            protocol=STAGE1_PROTOCOL, phase1_manifest_hash=phase1_hash,
        )
        snapshots[name] = {"phase1": phase1, "phase2": phase1 + phase2}
    return (json.dumps(snapshots, indent=2, sort_keys=True) + "\n").encode()


_BASE_SCRIPT = r'''
import json
import sys
from pathlib import Path
source_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source_root))
import tpen
import experiments.atomistic.stage1.protocol as protocol_module
for module in (protocol_module, tpen):
    module_path = Path(module.__file__).resolve()
    if source_root not in module_path.parents:
        raise AssertionError(
            f"{module.__name__} imported outside archive: {module_path}"
        )
from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA, HOOKE2_ACT50_A27,
    HOOKE2_ACT50_SIGMOID_PROFILE, HOOKE_A37, HOOKE_SCIENCE_PROFILE,
    STAGE1_PROTOCOL, bind_evaluation_manifest, build_manifest,
    config_identities, manifest_hash, planned_physical_system_ids,
)
root = Path("experiments/atomistic/stage1")
config_root = root / "configs"
out = {}
for name, system, profile, roster in (
    ("native", "hooke1", HOOKE_SCIENCE_PROFILE, HOOKE_A37),
    ("act50", "hooke2", HOOKE2_ACT50_SIGMOID_PROFILE, HOOKE2_ACT50_A27),
):
    config_profile = None if profile == HOOKE_SCIENCE_PROFILE else profile
    checkpoints = list(range(0, 50_001 if config_profile is not None else STAGE1_PROTOCOL.lineage_updates + 1, 1_000))
    phase1 = build_manifest(
        protocol=STAGE1_PROTOCOL, source_sha="a" * 40,
        config_sha256=config_identities(config_root=config_root, systems=(system,), science_profile=config_profile),
        checkpoints=checkpoints, lineage_seeds=[11, 12, 13], checkpoint_sha256={},
        results_root="/fixture/stage1-golden", systems=(system,), phase=1,
        science_profile=profile, evaluation_selection="A27" if config_profile is not None else "A37",
        include_smoke=False, auto_overlay=True,
    )
    phase1_hash = manifest_hash(phase1)
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {f"{row['system']}:{row['lineage']}": manifest_hash([row]) for row in phase1},
        "checkpoints": {f"{system}:{lineage}:{checkpoint}": "c" * 64 for lineage in range(STAGE1_PROTOCOL.lineages) for checkpoint in roster},
    }
    phase2 = bind_evaluation_manifest(
        phase1_rows=phase1, checkpoints=list(roster), checkpoint_evidence=evidence,
        expected_physical_system_id=planned_physical_system_ids(), source_sha="a" * 40,
        config_sha256=config_identities(config_root=config_root, systems=(system,), science_profile=config_profile),
        protocol=STAGE1_PROTOCOL, phase1_manifest_hash=phase1_hash,
    )
    out[name] = {"phase1": phase1, "phase2": phase1 + phase2}
print(json.dumps(out, indent=2, sort_keys=True) + "\n", end="")
'''


def _base_snapshot(*, hostile_pythonpath: Path | None = None) -> bytes:
    """Run the same serializer from an isolated exact-merge-base checkout."""

    base = subprocess.check_output(
        ["git", "merge-base", "HEAD", "origin/dev"], cwd=REPO_ROOT, text=True,
    ).strip()
    archive = subprocess.check_output(["git", "archive", base], cwd=REPO_ROOT)
    checkout = REPO_ROOT / ".pytest-base-science-golden"
    checkout.mkdir()
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar.getmembers():
                target = (checkout / member.name).resolve()
                if checkout.resolve() not in target.parents:
                    raise AssertionError("git archive escaped the fixture checkout")
                tar.extract(member, checkout)
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PYTHON")
        }
        if hostile_pythonpath is not None:
            env["PYTHONPATH"] = str(hostile_pythonpath)
        return subprocess.check_output(
            [sys.executable, "-c", _BASE_SCRIPT, str(checkout.resolve())],
            cwd=checkout.resolve(), env=env, text=False,
        )
    finally:
        for path in sorted(checkout.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        checkout.rmdir()


def test_native_and_act50_science_rows_are_byte_identical_to_merge_base() -> None:
    """The smoke change must not perturb either science manifest family."""

    assert _science_snapshot() == _base_snapshot()


def test_merge_base_serializer_rejects_head_pythonpath_self_comparison() -> None:
    """A hostile checkout path cannot redirect the exact-base serializer."""

    assert _science_snapshot() == _base_snapshot(hostile_pythonpath=REPO_ROOT)

