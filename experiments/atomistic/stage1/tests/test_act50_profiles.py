"""Behavioural contracts for the finite Hooke-2e ACT50 profiles."""

from __future__ import annotations

from pathlib import Path

import pytest
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
        f"hooke2-train-{expected_variant}-l0-s1-aaaaaaaa"
    )


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
    assert smoke["updates"] == 1_000
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


def test_act50_eval_refuses_cross_variant_checkpoint_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.atomistic.stage1 import eval as stage1_eval
    import tpen.checkpoint

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        tpen.checkpoint, "read_manifest",
        lambda *_args, **_kwargs: type("Manifest", (), {
            "hashes": {"model_config": "checkpoint-profile"},
        })(),
    )
    monkeypatch.setattr(
        tpen.checkpoint, "checkpoint_hashes",
        lambda _cfg: {"model_config": "eval-profile"},
    )
    row = {
        "science_profile": HOOKE2_ACT50_SIGMOID_PROFILE,
        "checkpoint_dir": str(checkpoint_dir),
    }
    with pytest.raises(stage1_eval.EvalRunRefusal, match="different model profile"):
        stage1_eval._refuse_cross_variant_checkpoint(OmegaConf.create({}), row)
