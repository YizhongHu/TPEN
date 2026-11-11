"""Contract tests for the system-neutral Stage-1 manifest."""

from pathlib import Path

from omegaconf import OmegaConf

from experiments.atomistic.stage1.protocol import Protocol, ProtocolError, build_manifest, manifest_hash
from experiments.atomistic.stage1.launch import command_for_row

HERE = Path(__file__).parent


def test_manifest_has_three_systems_and_four_fresh_streams() -> None:
    rows = build_manifest(Protocol(), checkpoints=[1000], seeds=[1, 2, 3], results_root="/results/stage1")
    assert len([row for row in rows if row["kind"] == "smoke"]) == 3
    assert len([row for row in rows if row["kind"] == "train"]) == 9
    assert len([row for row in rows if row["kind"] == "eval"]) == 36
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert manifest_hash(rows) == manifest_hash(rows)


def test_protocol_rejects_collapsed_independence() -> None:
    try:
        Protocol(lineages=2).validate()
    except ProtocolError as exc:
        assert "three lineages" in str(exc)
    else:
        raise AssertionError("two lineages must be rejected")


def test_manifest_rows_are_runnable_configs_with_explicit_overrides() -> None:
    rows = build_manifest(Protocol(), checkpoints=[1000], seeds=[1, 2, 3], results_root="/results/stage1")
    for row in rows:
        command = command_for_row(row, python="/opt/python", driver="train.py")
        assert row["config"].startswith("configs/")
        assert command[:4] == ["/opt/python", "train.py", "--config", row["config"]]
        assert row["overrides"]


def test_every_manifest_row_resolves_in_struct_mode() -> None:
    """Apply every launch override so missing paths cannot hide until Slurm."""

    protocol = Protocol()
    rows = build_manifest(protocol, checkpoints=[50_000], seeds=[73001, 73002, 73003],
                          results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1-results")
    for row in rows:
        cfg = OmegaConf.load(HERE / row["config"])
        OmegaConf.set_struct(cfg, True)
        for key, value in row["overrides"].items():
            OmegaConf.update(cfg, key, value, merge=False)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        if row["kind"] == "smoke":
            assert resolved["trainer"]["max_steps"] == protocol.smoke_updates
            assert resolved["sampler"]["n_walkers"] == protocol.smoke_walkers
        elif row["kind"] == "train":
            assert resolved["trainer"]["max_steps"] == protocol.lineage_updates
            assert resolved["sampler"]["n_walkers"] == protocol.lineage_walkers
        else:
            assert resolved["evaluation_sampler"]["n_walkers"] == protocol.eval_walkers
            assert resolved["evaluation_tasks"]["mcmc_energy"]["generator"]["n_draws"] == protocol.eval_draws
            assert resolved["load"]["path"].endswith("step_050000")
            assert resolved["trajectory_identity"]["checkpoint_file"].endswith("model.pt")
