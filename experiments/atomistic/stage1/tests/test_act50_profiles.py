"""Behavioural contracts for the finite Hooke-2e ACT50 profiles."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.atomistic.stage1.protocol import (
    HOOKE2_ACT50_NOLAYER_PROFILE,
    HOOKE2_ACT50_PROFILES,
    HOOKE2_ACT50_SIGMOID_PROFILE,
    HOOKE2_ACT50_TANH_PROFILE,
    STAGE1_PROTOCOL,
    ProtocolError,
    build_manifest,
    config_identities,
    derive_smoke_request,
)
from experiments.atomistic.stage1.train import build_train_config
from experiments.toolkit.job_names import stage1_row_job_name


CONFIG_ROOT = Path(__file__).parents[1] / "configs"
ACT50_JOB_VARIANTS = {"sigmoid": "sg", "tanh": "th", "nolayer": "nl"}


def _snapshot(root: Path) -> tuple[bool, tuple[tuple[str, bool, bytes], ...]]:
    """Capture an output tree before and after a refusal boundary."""

    if not root.exists():
        return False, ()
    entries = []
    for path in sorted(root.rglob("*")):
        is_file = path.is_file()
        entries.append((str(path.relative_to(root)), is_file, path.read_bytes() if is_file else b""))
    return True, tuple(entries)


def _rows(profile: str) -> list[dict[str, object]]:
    identities = config_identities(
        config_root=CONFIG_ROOT, systems=("hooke2",), science_profile=profile,
    )
    return build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=identities,
        checkpoints=list(range(0, 50_001, 1_000)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/external/act50-test",
        systems=("hooke2",),
        phase=1,
        device="cuda",
        science_profile=profile,
        include_smoke=False,
        auto_overlay=True,
    )


@pytest.mark.parametrize("profile", HOOKE2_ACT50_PROFILES)
def test_act50_manifest_is_finite_50k_and_a27(profile: str) -> None:
    rows = _rows(profile)
    assert len(rows) == 3
    assert {row["science_profile"] for row in rows} == {profile}
    assert {row["updates"] for row in rows} == {50_000}
    assert {len(row["checkpoint_steps"]) for row in rows} == {51}
    assert {row["evaluation_selection"] for row in rows} == {"A27"}
    expected_variant = profile.removeprefix("hooke2-").removesuffix("50k-v1")
    assert {row["variant"] for row in rows} == {expected_variant}
    assert stage1_row_job_name(rows[0]) == (
        f"hooke2-train-{ACT50_JOB_VARIANTS[expected_variant]}-l0-s1-aaaaaaaa"
    )


@pytest.mark.parametrize("profile", HOOKE2_ACT50_PROFILES)
def test_act50_compatibility_smoke_has_variant_bound_job_name(profile: str) -> None:
    identities = config_identities(
        config_root=CONFIG_ROOT, systems=("hooke2",), science_profile=profile,
    )
    rows = build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=identities,
        checkpoints=list(range(0, 50_001, 1_000)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/external/act50-test",
        systems=("hooke2",),
        phase=1,
        science_profile=profile,
        include_smoke=True,
        auto_overlay=True,
    )
    smoke = next(row for row in rows if row["kind"] == "smoke")
    variant = str(smoke["variant"])
    assert smoke["science_profile"] == profile
    assert stage1_row_job_name(smoke) == (
        f"hooke2-smoke-{ACT50_JOB_VARIANTS[variant]}-s1-aaaaaaaa"
    )


def test_act50_job_names_are_bounded_and_unique_across_row_kinds() -> None:
    names: list[str] = []
    for profile in HOOKE2_ACT50_PROFILES:
        variant = profile.removeprefix("hooke2-").removesuffix("50k-v1")
        rows = [
            {"kind": kind, "system": "hooke2", "variant": variant,
             "lineage": 0, "checkpoint": 50_000, "stream": 0,
             "source_sha": "a" * 40}
            for kind in ("smoke", "train", "eval", "collect")
        ]
        names.extend(stage1_row_job_name(row) for row in rows)
    assert all(len(name) <= 40 for name in names)
    assert len(names) == len(set(names))


def test_act50_profile_hashes_are_distinct_and_native_hash_is_unchanged() -> None:
    native = config_identities(config_root=CONFIG_ROOT, systems=("hooke2",))["hooke2"]
    sigmoid = config_identities(
        config_root=CONFIG_ROOT, systems=("hooke2",),
        science_profile=HOOKE2_ACT50_SIGMOID_PROFILE,
    )["hooke2"]
    tanh = config_identities(
        config_root=CONFIG_ROOT, systems=("hooke2",),
        science_profile=HOOKE2_ACT50_TANH_PROFILE,
    )["hooke2"]
    assert native != sigmoid != tanh
    assert native != tanh
    nolayer = config_identities(
        config_root=CONFIG_ROOT, systems=("hooke2",),
        science_profile=HOOKE2_ACT50_NOLAYER_PROFILE,
    )["hooke2"]
    assert len({native, sigmoid, tanh, nolayer}) == 4


def test_act50_profiles_reuse_the_same_initial_seed_schedule() -> None:
    sigmoid = _rows(HOOKE2_ACT50_SIGMOID_PROFILE)
    tanh = _rows(HOOKE2_ACT50_TANH_PROFILE)
    assert [(row["seed"], row["sampler_seed"]) for row in sigmoid] == [
        (row["seed"], row["sampler_seed"]) for row in tanh
    ]


@pytest.mark.parametrize("profile", HOOKE2_ACT50_PROFILES)
def test_act50_smoke_derivation_preserves_profile_and_stops_at_1000(profile: str) -> None:
    train = _rows(profile)[0]
    smoke = derive_smoke_request(
        train, results_root="/external/act50-smoke", stop_at=1_000,
    )
    assert smoke["science_profile"] == profile
    assert smoke["variant"] == train["variant"]
    assert smoke["execution"] == {"stop_at": 1_000}
    assert smoke["updates"] == train["updates"] == 50_000
    assert smoke["config_sha256"] == train["config_sha256"]


@pytest.mark.parametrize(
    ("profile", "expected_activation"),
    [
        (HOOKE2_ACT50_SIGMOID_PROFILE, "torch.nn.Sigmoid"),
        (HOOKE2_ACT50_TANH_PROFILE, "torch.nn.Tanh"),
    ],
)
def test_layered_profiles_set_embedding_and_layer_activations(
    profile: str, expected_activation: str,
) -> None:
    cfg = build_train_config(_rows(profile)[0])
    assert cfg.model.embedding.activation._target_ == expected_activation
    assert cfg.model.layers[0].mixing.activation._target_ == expected_activation
    assert cfg.model.layers[0].path_aggregation.activation._target_ == expected_activation
    assert cfg.trainer.max_steps == 50_000


@pytest.mark.parametrize("profile", HOOKE2_ACT50_PROFILES)
def test_eval_profiles_resolve_the_same_model_variant(
    profile: str, tmp_path: Path,
) -> None:
    from experiments.atomistic.stage1.eval import build_eval_config

    row = dict(_rows(profile)[0])
    checkpoint_dir = (tmp_path / "checkpoint").resolve()
    checkpoint_dir.mkdir()
    row.update({
        "kind": "eval",
        "config": "configs/hooke2/eval.yaml",
        "checkpoint": 0,
        "stream": 0,
        "draws": 256,
        "run_id": "eval-profile",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": str((checkpoint_dir / "model.pt").resolve()),
    })
    cfg = build_eval_config(row)
    if profile == HOOKE2_ACT50_NOLAYER_PROFILE:
        assert list(cfg.model.layers) == []
    else:
        expected = "torch.nn.Sigmoid" if profile == HOOKE2_ACT50_SIGMOID_PROFILE else "torch.nn.Tanh"
        assert cfg.model.embedding.activation._target_ == expected
        assert cfg.model.layers[0].mixing.activation._target_ == expected
        assert cfg.model.layers[0].path_aggregation.activation._target_ == expected


def test_nolayer_profile_removes_only_layer_diagnostic_groups() -> None:
    cfg = build_train_config(_rows(HOOKE2_ACT50_NOLAYER_PROFILE)[0])
    assert list(cfg.model.layers) == []
    assert cfg.trainer.max_steps == 50_000
    for callback in cfg.callbacks:
        if callback["_target_"] in {"tpen.callback.ModuleNorms", "tpen.callback.OptimizerState"}:
            assert set(callback.groups) == {"embedding", "readout", "factors"}


def test_nolayer_profile_forward_grad_local_energy_and_singlet_swap(
    tmp_path: Path,
) -> None:
    from tpen.checkpoint import restore_checkpoint, save_checkpoint
    from tpen.data.batch import ElectronBatch
    from tpen.data.permutation import Permutation
    from tpen.physics.hamiltonian import local_energy

    row = _rows(HOOKE2_ACT50_NOLAYER_PROFILE)[0]
    cfg = build_train_config(row)
    model = instantiate(cfg.model).to(dtype=torch.float64)
    positions = torch.tensor(
        [[[0.3, -0.2, 0.8], [-0.1, 0.5, 0.3]],
         [[1.0, 0.0, 0.5], [0.0, -0.5, 1.5]]],
        dtype=torch.float64,
    )
    batch = ElectronBatch(
        positions=positions,
        spins=torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float64),
    )
    output = model(batch)
    assert torch.isfinite(output.logabs).all()
    assert torch.isfinite(output.sign).all()
    output.logabs.square().mean().backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    terms = instantiate(cfg.hamiltonian_terms)
    energy = local_energy(terms, model, batch)
    assert energy.shape == batch.positions.shape[:-2]
    assert torch.isfinite(energy).all()
    # Hooke-2e has n_up=1,n_down=1: this exchanges opposite-spin labels, so
    # the singlet spatial wavefunction has the expected +1 sign ratio.
    swapped = model(batch.permute(Permutation((1, 0))))
    assert torch.equal(output.sign * swapped.sign, torch.ones_like(output.sign))

    context = type("Context", (), {
        "cfg": cfg,
        "run_dir": tmp_path,
        "metadata": type("Metadata", (), {"device": "cpu", "dtype": "float64"})(),
    })()
    checkpoint = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=1,
        completed_updates=1,
        model=model,
        context=context,
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )
    restored = instantiate(cfg.model).to(dtype=torch.float64)
    report = restore_checkpoint(
        load={"mode": "model_only", "path": str(checkpoint)},
        model=restored,
        context=context,
        strict=True,
    )
    assert report.loaded_model
    torch.testing.assert_close(restored(batch).logabs, output.logabs)


def test_act50_refuses_non_a27_phase2_roster() -> None:
    identities = config_identities(
        config_root=CONFIG_ROOT, systems=("hooke2",),
        science_profile=HOOKE2_ACT50_SIGMOID_PROFILE,
    )
    with pytest.raises(ProtocolError, match="A27"):
        build_manifest(
            protocol=STAGE1_PROTOCOL,
            source_sha="a" * 40,
            config_sha256=identities,
            checkpoints=[0, 50_000],
            lineage_seeds=[11, 12, 13],
            checkpoint_sha256={},
            results_root="/external/act50-test",
            systems=("hooke2",),
            phase=2,
            device="cuda",
            science_profile=HOOKE2_ACT50_SIGMOID_PROFILE,
            include_smoke=False,
            auto_overlay=True,
        )


@pytest.mark.parametrize(
    "source_profile,target_profile",
    (
        (HOOKE2_ACT50_SIGMOID_PROFILE, HOOKE2_ACT50_TANH_PROFILE),
        (HOOKE2_ACT50_TANH_PROFILE, HOOKE2_ACT50_SIGMOID_PROFILE),
        (HOOKE2_ACT50_SIGMOID_PROFILE, HOOKE2_ACT50_NOLAYER_PROFILE),
        (HOOKE2_ACT50_NOLAYER_PROFILE, HOOKE2_ACT50_SIGMOID_PROFILE),
    ),
)
def test_act50_eval_refuses_cross_variant_checkpoint_before_writes(
    source_profile: str, target_profile: str, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.atomistic.stage1 import eval as stage1_eval
    from tpen.checkpoint import save_checkpoint

    source_row = _rows(source_profile)[0]
    source_cfg = build_train_config(source_row)
    source_model = instantiate(source_cfg.model).to(dtype=torch.float64)
    checkpoint_root = tmp_path / "source-checkpoints"
    source_context = SimpleNamespace(
        cfg=source_cfg,
        run_dir=tmp_path / "source-run",
        metadata=SimpleNamespace(device="cpu", dtype="float64"),
    )
    checkpoint_dir = save_checkpoint(
        output_dir=checkpoint_root,
        next_iteration=1,
        completed_updates=1,
        model=source_model,
        context=source_context,
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )
    row = dict(_rows(target_profile)[0])
    results_root = tmp_path / "eval-results"
    row.update({
        "kind": "eval",
        "config": str(CONFIG_ROOT / "hooke2" / "eval.yaml"),
        "results_root": str(results_root),
        "experiment_name": "act50_eval",
        "sector": "atomistic",
        "run_id": f"cross-{source_profile[:6]}-{target_profile[:6]}",
        "checkpoint": 50_000,
        "stream": 0,
        "draws": 1,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": str(checkpoint_dir / "model.pt"),
    })
    before = _snapshot(results_root)
    monkeypatch.setattr(stage1_eval, "validate_gateway_binding", lambda **_: row)
    monkeypatch.setattr(
        stage1_eval, "run_from_config", lambda *_args, **_kwargs: pytest.fail("eval ran")
    )
    monkeypatch.setattr(sys, "argv", ["eval.py", "--binding", "unused"])
    with pytest.raises(stage1_eval.EvalRunRefusal, match="different model profile"):
        stage1_eval.main()
    assert _snapshot(results_root) == before
