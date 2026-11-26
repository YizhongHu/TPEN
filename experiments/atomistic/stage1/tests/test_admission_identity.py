"""Independent LR-2 review probes for admission and identity stability."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1 import eval as stage1_eval
from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.launch import _binding_token
from experiments.atomistic.stage1.protocol import (
    HOOKE_SCIENCE_PROFILE,
    SYSTEMS,
    Protocol,
    STAGE1_PROTOCOL,
    build_manifest,
    config_identities,
    manifest_hash,
    overlay_path_for_system,
    overlay_sha256,
    planned_physical_system_ids,
    validate_gateway_binding,
)


ROOT = Path(__file__).parents[1]
CONFIG_ROOT = ROOT / "configs"
REPO_ROOT = ROOT.parents[2]
BASE_SHA = "8240a55549159a0bacfba4be92a162205f2453a0"


def _snapshot_semantics(checkout: Path) -> dict[str, object]:
    """Run the same row/config builder in an isolated checkout."""

    script = r'''
import json
from pathlib import Path
from omegaconf import OmegaConf
from experiments.atomistic.stage1 import eval as stage1_eval
from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA, HE_H2_EVAL_CHECKPOINTS, HOOKE_A37,
    HOOKE_SCIENCE_PROFILE, QHO_Q21,
    SYSTEMS, Protocol, STAGE1_PROTOCOL,
    bind_evaluation_manifest, build_manifest, config_identities,
    overlay_path_for_system, overlay_sha256, planned_physical_system_ids,
    manifest_hash,
)
root = Path("experiments/atomistic/stage1").resolve()
config_root = root / "configs"
def rows(system, phase, science=False):
    protocol = STAGE1_PROTOCOL
    overlay = root / overlay_path_for_system(system)
    if phase == 1:
        checkpoints = list(range(0, STAGE1_PROTOCOL.lineage_updates + 1,
                                 STAGE1_PROTOCOL.retention_cadence))
    elif science:
        checkpoints = list(QHO_Q21 if system == "qho2e" else HOOKE_A37)
    else:
        checkpoints = list(HE_H2_EVAL_CHECKPOINTS)
    if phase == 2 and not science:
        phase1 = rows(system, 1, science=False)
        phase1_hash = manifest_hash(phase1)
        evidence = {
            "schema": CHECKPOINT_EVIDENCE_SCHEMA,
            "phase1_manifest_hash": phase1_hash,
            "phase1_train_row_hashes": {
                f"{system}:{row['lineage']}": manifest_hash([row])
                for row in phase1 if row["kind"] == "train"
            },
            "checkpoints": {
                f"{system}:{lineage}:{checkpoint}": "b" * 64
                for lineage in range(STAGE1_PROTOCOL.lineages)
                for checkpoint in checkpoints
            },
        }
        return phase1 + bind_evaluation_manifest(
            phase1_rows=phase1, checkpoints=checkpoints,
            checkpoint_evidence=evidence,
            expected_physical_system_id=planned_physical_system_ids(),
            source_sha="a" * 40,
            config_sha256=config_identities(config_root=config_root, systems=(system,)),
            protocol=STAGE1_PROTOCOL, phase1_manifest_hash=phase1_hash,
            device="cpu",
        )
    return build_manifest(
        protocol=protocol, source_sha="a" * 40,
        config_sha256=config_identities(config_root=config_root, systems=(system,)),
        checkpoints=checkpoints, lineage_seeds=[11, 12, 13],
        checkpoint_sha256={(system, lineage, checkpoint): "b" * 64
                           for lineage in range(3) for checkpoint in checkpoints},
        results_root="/external/stage1-review-r1", systems=(system,),
        phase=phase, expected_physical_system_id=planned_physical_system_ids()
        if phase == 2 else None,
        science_profile=HOOKE_SCIENCE_PROFILE if science else None,
        evaluation_selection=("Q21" if system == "qho2e" else "A37") if science else None,
        overlay=str(overlay), overlay_sha256=overlay_sha256(overlay), device="cpu")
def fields(row, cfg):
    return {"row": dict(row), "config": OmegaConf.to_container(cfg, resolve=True)}
out = {"systems": {}}
for system in SYSTEMS:
    science = system in {"hooke1", "hooke2", "qho2e"}
    phase1 = rows(system, 1, science=science)
    train = next(row for row in phase1 if row["kind"] == "train")
    phase2 = rows(system, 2, science=science)
    evaluation = next(row for row in phase2 if row["kind"] == "eval")
    out["systems"][system] = {
        "phase1": phase1,
        "phase1_hash": manifest_hash(phase1),
        "phase2": phase2,
        "train": fields(train, stage1_train.build_train_config(train)),
        "eval": fields(evaluation, stage1_eval.build_eval_config(evaluation)),
    }
fixture = json.loads((root / "tests/fixtures/canonical-manifest-v3.json").read_text())
canonical_rows = build_manifest(
    protocol=STAGE1_PROTOCOL, source_sha=fixture["source_sha"],
    config_sha256=fixture["config_sha256"], checkpoints=fixture["checkpoints"],
    lineage_seeds=fixture["lineage_seeds"], checkpoint_sha256={},
    results_root=fixture["results_root"], systems=tuple(fixture["systems"]),
    phase=int(fixture["phase"]),
)
canonical_doc = {"schema": "stage1/v3", "phase": int(fixture["phase"]),
                 "config_identity_rule": "stage1-config-v1",
                 "manifest_hash": manifest_hash(canonical_rows), "rows": canonical_rows}
out["canonical"] = {"bytes": len((json.dumps(canonical_doc, indent=2) + "\n").encode()),
                    "sha256": __import__("hashlib").sha256(
                        (json.dumps(canonical_doc, indent=2) + "\n").encode()).hexdigest()}
print(json.dumps(out, sort_keys=True))
'''
    return json.loads(subprocess.check_output([sys.executable, "-c", script], cwd=checkout))


def _phase1_rows(*, system: str, science: bool = False) -> list[dict[str, object]]:
    """Build rows from the production builders and real config files."""

    protocol = STAGE1_PROTOCOL if science else Protocol(1, 2, 1, 2, 3, 4, 2, 2, 1)
    overlay_file = ROOT / overlay_path_for_system(system)
    return build_manifest(
        protocol=protocol,
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=CONFIG_ROOT, systems=(system,)),
        checkpoints=(
            list(range(0, STAGE1_PROTOCOL.lineage_updates + 1,
                       STAGE1_PROTOCOL.retention_cadence))
            if science else [0, 1]
        ),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/external/stage1-review-r1",
        systems=(system,),
        phase=1,
        science_profile=HOOKE_SCIENCE_PROFILE if science else None,
        evaluation_selection=("Q21" if system == "qho2e" else "A37") if science else None,
        overlay=str(overlay_file),
        overlay_sha256=overlay_sha256(overlay_file),
        device="cpu",
    )


def _phase2_eval_row(
    system: str, *, checkpoint_root: Path | None = None,
) -> dict[str, object]:
    """Build one real eval row with a checkpoint binding."""

    rows = build_manifest(
        protocol=Protocol(1, 2, 1, 2, 3, 4, 2, 2, 1),
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=CONFIG_ROOT, systems=(system,)),
        checkpoints=[0, 1],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={(system, lineage, checkpoint): "b" * 64
                           for lineage in range(3) for checkpoint in [0, 1]},
        results_root="/external/stage1-review-r1",
        systems=(system,),
        phase=2,
        expected_physical_system_id=planned_physical_system_ids(),
        device="cpu",
    )
    row = next(row for row in rows if row["kind"] == "eval")
    if checkpoint_root is not None:
        checkpoint_dir = checkpoint_root / system / "step_000000"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "model.pt"
        payload = b"matching checkpoint bytes"
        checkpoint_file.write_bytes(payload)
        row["checkpoint_dir"] = str(checkpoint_dir.resolve())
        row["checkpoint_file"] = str(checkpoint_file.resolve())
        row["checkpoint_sha256"] = hashlib.sha256(payload).hexdigest()
    return row


@pytest.mark.parametrize("system", SYSTEMS)
def test_every_real_stage1_train_config_is_identity_bound(system: str) -> None:
    """Every committed train config is admitted with its actual file digest."""

    expected = config_identities(config_root=CONFIG_ROOT, systems=(system,))[system]
    rows = _phase1_rows(system=system)
    assert len(rows) == 4
    assert [row["kind"] for row in rows].count("smoke") == 1
    assert [row["kind"] for row in rows].count("train") == 3
    assert {row["config_sha256"] for row in rows} == {expected}
    for row in rows:
        validate_gateway_binding(binding=_binding_token(row))


@pytest.mark.parametrize("system", SYSTEMS)
def test_real_hooke_overlay_roster_is_admitted(system: str) -> None:
    """Every current system's declared diagnostics overlay is admitted."""

    rows = _phase1_rows(system=system, science=system in {"hooke1", "hooke2", "qho2e"})
    train_rows = [row for row in rows if row["kind"] == "train"]
    assert len(rows) == 4 and len(train_rows) == 3
    assert {row["overlay"] for row in train_rows} == {overlay_path_for_system(system)}
    overlay_file = ROOT / overlay_path_for_system(system)
    assert {row["overlay_sha256"] for row in train_rows} == {overlay_sha256(overlay_file)}
    for row in train_rows:
        validate_gateway_binding(binding=_binding_token(row))


@pytest.mark.parametrize("system", SYSTEMS)
def test_real_train_and_eval_builders_compose_roster_rows(
    system: str, tmp_path: Path,
) -> None:
    """Both real builders consume the admitted train/eval row populations."""

    train = next(row for row in _phase1_rows(system=system) if row["kind"] == "train")
    train_cfg = stage1_train.build_train_config(train)
    assert OmegaConf.select(train_cfg, "run.run_id") == train["run_id"]
    eval_row = _phase2_eval_row(system, checkpoint_root=tmp_path)
    validate_gateway_binding(binding=_binding_token(eval_row))
    eval_cfg = stage1_eval.build_eval_config(eval_row)
    assert OmegaConf.select(eval_cfg, "load.mode") == "model_only"
    assert OmegaConf.select(eval_cfg, "load.strict") is True


def test_canonical_manifest_identity_is_byte_stable() -> None:
    """The committed canonical manifest remains exactly reproducible."""

    from experiments.atomistic.stage1.tests.test_step_zero import _canonical_manifest_bytes

    fixture = Path(__file__).with_name("fixtures") / "canonical-manifest-v3.json"
    actual = _canonical_manifest_bytes()
    metadata = __import__("json").loads(fixture.read_text(encoding="utf-8"))
    assert len(actual) == metadata["manifest_bytes"]
    assert hashlib.sha256(actual).hexdigest() == metadata["manifest_sha256"]


def test_v3_rows_keep_identity_and_compose_after_v4_planner_addition() -> None:
    """The v4 planner path does not invalidate existing v3 train rows."""

    legacy_rows = _phase1_rows(system="he")
    legacy_train = [row for row in legacy_rows if row["kind"] == "train"]
    assert len(legacy_rows) == 4

    # The planner's new production shape is the old train population with the
    # compatibility smoke row omitted.  Compare the identity-bearing fields,
    # not merely the row count, so adding the v4 path cannot silently alter a
    # pre-existing v3 row hash, row id, or config identity.
    v4_train = build_manifest(
        protocol=Protocol(1, 2, 1, 2, 3, 4, 2, 2, 1),
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=CONFIG_ROOT, systems=("he",)),
        checkpoints=[0, 1],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/external/stage1-review-r1",
        systems=("he",),
        phase=1,
        device="cpu",
        include_smoke=False,
        auto_overlay=True,
    )
    assert v4_train == legacy_train

    expected_legacy_identity = {
        "smoke": (
            "64ab378e3465b621286feaa8ab9a30346dcf8c7f53de0c7db7ccdecc344a06ee",
            "47eed69f5ef0658662accbe834d1dfbd4b3f90d08a4a720f9195c84be714a523",
        ),
        "train-lineage-0": (
            "309ba3050eaf252c89127817419d52bd4cb966be0fad5536ad62388a2449deed",
            "5912e1969658dbaa7e990d1fb2472369d6d738c3f71b992106f28fe1d488eb77",
        ),
        "train-lineage-1": (
            "c0f9675275fb0d91512b615b21b4bb2d3696431f5afe1c7e1188c9cf2688a779",
            "e86707988cf15ab56cbdc32dd25165a50b1debb3511e6ac8f21523e3547cac58",
        ),
        "train-lineage-2": (
            "7fa73f0362f35d52c0473f32496c607d67d6475d57c7052b72e3cb0b2b4f3f62",
            "d4b55a26670ae2853ffaf8d12e71df725d43b34dea720058ef036ecc83179502",
        ),
    }
    for row in legacy_rows:
        assert (row["row_id"], manifest_hash([row])) == expected_legacy_identity[row["run_id"]]
        assert row["config_identity_rule"] == "stage1-config-v1"
        assert row["config_sha256"] == config_identities(
            config_root=CONFIG_ROOT, systems=("he",),
        )["he"]
        validate_gateway_binding(binding=_binding_token(row))
        if row["kind"] == "train":
            cfg = stage1_train.build_train_config(row)
            assert OmegaConf.select(cfg, "run.run_id") == row["run_id"]


@pytest.mark.parametrize("system", SYSTEMS)
def test_base_and_pr_config_identity_inputs_match(system: str, tmp_path: Path) -> None:
    """The PR preserves each base config identity and composed input population."""

    base_root = tmp_path / "base"
    base_root.mkdir()
    for name in ("train.yaml", "eval.yaml"):
        target = base_root / system / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output([
            "git", "show", f"{BASE_SHA}:"
            f"experiments/atomistic/stage1/configs/{system}/{name}",
        ]))
    assert config_identities(config_root=base_root, systems=(system,))[system] == (
        config_identities(config_root=CONFIG_ROOT, systems=(system,))[system]
    )


def test_base_and_pr_built_rows_and_composed_configs_match(tmp_path: Path) -> None:
    """Compare actual rows/configs; fixture digests are historical inputs, not admission."""

    archive = subprocess.check_output(["git", "archive", BASE_SHA], cwd=REPO_ROOT)
    base_root = tmp_path / "base-checkout"
    base_root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        source.extractall(base_root)
    base_snapshot = _snapshot_semantics(base_root)
    pr_snapshot = _snapshot_semantics(REPO_ROOT)
    assert base_snapshot == pr_snapshot
    fixture = json.loads((ROOT / "tests/fixtures/canonical-manifest-v3.json").read_text())
    assert pr_snapshot["canonical"] == {
        "bytes": fixture["manifest_bytes"],
        "sha256": fixture["manifest_sha256"],
    }


@pytest.mark.parametrize("gateway", ["train", "eval"])
def test_valid_binding_control_reaches_core_spy(
    monkeypatch: pytest.MonkeyPatch, gateway: str, tmp_path: Path,
) -> None:
    """A matching binding reaches only the substituted core call."""

    row = (next(row for row in _phase1_rows(system="he") if row["kind"] == "train")
           if gateway == "train" else _phase2_eval_row("he", checkpoint_root=tmp_path))
    row["results_root"] = str(tmp_path / "results")
    module = stage1_train if gateway == "train" else stage1_eval
    reached: list[object] = []

    def core_spy(*args: object, **kwargs: object) -> int:
        reached.append((args, kwargs))
        return 0

    monkeypatch.setattr(module, "run_from_config", core_spy)
    monkeypatch.setattr(sys, "argv", [f"stage1-{gateway}", "--binding", _binding_token(row)])
    assert module.main() == 0
    assert len(reached) == 1


@pytest.mark.parametrize("gateway", ["train", "eval"])
def test_gateway_rejects_cross_kind_binding_before_core(
    monkeypatch: pytest.MonkeyPatch, gateway: str, tmp_path: Path,
) -> None:
    """A valid row for the other gateway must never reach core construction."""

    train_row = next(row for row in _phase1_rows(system="he") if row["kind"] == "train")
    eval_row = _phase2_eval_row("he", checkpoint_root=tmp_path)
    row = eval_row if gateway == "train" else train_row
    row["results_root"] = str(tmp_path / "results")
    module = stage1_train if gateway == "train" else stage1_eval
    reached: list[object] = []

    def core_spy(*args: object, **kwargs: object) -> int:
        reached.append((args, kwargs))
        return 0

    monkeypatch.setattr(module, "run_from_config", core_spy)
    monkeypatch.setattr(sys, "argv", [f"stage1-{gateway}", "--binding", _binding_token(row)])
    with pytest.raises(Exception):
        module.main()
    assert not reached, f"{gateway} accepted a {row['kind']} binding into core"


def test_eval_binding_rejects_checkpoint_byte_tamper_before_core(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A changed checkpoint file must fail admission before the core spy."""

    row = _phase2_eval_row("he")
    row["results_root"] = str(tmp_path / "results")
    checkpoint_dir = tmp_path / "checkpoints" / "step_000000"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_file = checkpoint_dir / "model.pt"
    original = b"canonical checkpoint bytes"
    checkpoint_file.write_bytes(original)
    row["checkpoint_dir"] = str(checkpoint_dir.resolve())
    row["checkpoint_file"] = str(checkpoint_file.resolve())
    row["checkpoint_sha256"] = hashlib.sha256(original).hexdigest()
    checkpoint_file.write_bytes(b"tampered checkpoint bytes")
    reached: list[object] = []

    def core_spy(*args: object, **kwargs: object) -> int:
        reached.append((args, kwargs))
        return 0

    monkeypatch.setattr(stage1_eval, "run_from_config", core_spy)
    monkeypatch.setattr(sys, "argv", ["stage1-eval", "--binding", _binding_token(row)])
    with pytest.raises(Exception, match="checkpoint|digest|sha"):
        stage1_eval.main()
    assert not reached
