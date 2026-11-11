"""Operational artifact routing with a constant science config (S3, item af595b39).

``tpen.artifacts.ArtifactDestination`` is a closed, launcher-supplied binding
from the logical ``run.dir`` every allocation of one scientific cell shares to
the physical leaf one allocation has exclusively claimed. ``prepare_run_context``
(and ``run_from_config``) accept it as an operational parameter, outside
``cfg``: the recorded ``run.dir`` -- and therefore the resolved science config C
-- stays byte-identical across allocations, while ``ArtifactManager`` and the
closed set of callback/logger sinks HI bakes with concrete ``${run.dir}``
paths before this function ever runs are rebound to the physical leaf.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.artifacts import ArtifactDestination, ArtifactRoutingError
from tpen.run import prepare_run_context


def _routed_cfg(logical_root: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "experiment": {"name": "routing", "sector": "unit", "run_name": "routing_unit"},
            "run": {"root": str(logical_root), "run_id": "fixed-run-id", "dir": None},
            "runtime": {"device": "cpu", "dtype": "float64"},
            "callbacks": [
                {"_target_": "tpen.callback.snapshot.ConfigSnapshot", "output_path": "${run.dir}/snapshot.yaml"},
                {"_target_": "tpen.callback.metadata.Metadata", "output_path": "${run.dir}/metadata.json"},
            ],
            "loggers": [
                {"_target_": "tpen.logging.csv.CSV", "path": "${run.dir}/metrics.csv"},
                {"_target_": "tpen.logging.jsonl.JSONL", "path": "${run.dir}/metrics.jsonl"},
            ],
        }
    )


def test_without_a_destination_sinks_stay_under_the_logical_run_dir(tmp_path: Path) -> None:
    cfg = _routed_cfg(tmp_path / "logical")

    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    logical_run_dir = context.artifact_manager.logical_run_dir
    assert context.run_dir == logical_run_dir
    snapshot, metadata_cb = context.callbacks
    csv_logger, jsonl_logger = context.loggers
    assert snapshot.output_path == logical_run_dir / "snapshot.yaml"
    assert metadata_cb.output_path == logical_run_dir / "metadata.json"
    assert csv_logger.path == logical_run_dir / "metrics.csv"
    assert jsonl_logger.path == logical_run_dir / "metrics.jsonl"


def test_destination_routes_every_known_sink_beneath_the_claimed_physical_leaf(
    tmp_path: Path,
) -> None:
    logical_root = tmp_path / "logical"
    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    cfg = _routed_cfg(logical_root)
    destination = ArtifactDestination(physical_root=physical_root)

    context = prepare_run_context(
        cfg, config_path="test.yaml", command="pytest", artifact_destination=destination
    )

    logical_run_dir = context.artifact_manager.logical_run_dir
    # The claimed leaf, not the logical path, is where everything physically lands.
    assert context.run_dir == physical_root
    assert context.run_dir != logical_run_dir
    snapshot, metadata_cb = context.callbacks
    csv_logger, jsonl_logger = context.loggers
    for path in (
        context.run_dir,
        snapshot.output_path,
        metadata_cb.output_path,
        csv_logger.path,
        jsonl_logger.path,
    ):
        assert physical_root in path.parents or path == physical_root

    # C itself -- including the recorded run.dir -- stays the untouched logical value.
    assert context.cfg.run.dir == str(logical_run_dir)
    assert context.metadata.run_dir == str(logical_run_dir)

    routing_receipt = __import__("json").loads((physical_root / "artifact_routing.json").read_text())
    assert routing_receipt["logical_run_dir"] == str(logical_run_dir)
    assert routing_receipt["destination_physical_root"] == str(physical_root)
    assert routing_receipt["routed_sinks"]["tpen.callback.snapshot.ConfigSnapshot.output_path"] == str(
        snapshot.output_path
    )


def test_resolved_config_is_byte_identical_across_two_allocation_destinations(
    tmp_path: Path,
) -> None:
    """Two allocations of the same cell must record the same C -- mutant (3)."""

    logical_root = tmp_path / "logical"
    first_physical = tmp_path / "allocation-1"
    second_physical = tmp_path / "allocation-2"
    first_physical.mkdir()
    second_physical.mkdir()

    first = prepare_run_context(
        _routed_cfg(logical_root),
        config_path="test.yaml",
        command="pytest",
        artifact_destination=ArtifactDestination(physical_root=first_physical),
    )
    second = prepare_run_context(
        _routed_cfg(logical_root),
        config_path="test.yaml",
        command="pytest",
        artifact_destination=ArtifactDestination(physical_root=second_physical),
    )

    assert OmegaConf.to_yaml(first.cfg, resolve=True) == OmegaConf.to_yaml(second.cfg, resolve=True)
    assert first.run_dir != second.run_dir


def test_predecessor_leaf_is_untouched_by_a_later_allocations_destination(tmp_path: Path) -> None:
    logical_root = tmp_path / "logical"
    predecessor_physical = tmp_path / "allocation-1"
    predecessor_physical.mkdir()
    prepare_run_context(
        _routed_cfg(logical_root),
        config_path="test.yaml",
        command="pytest",
        artifact_destination=ArtifactDestination(physical_root=predecessor_physical),
    )
    before = sorted(p.relative_to(predecessor_physical) for p in predecessor_physical.rglob("*"))
    before_bytes = {
        str(p): (predecessor_physical / p).read_bytes()
        for p in before
        if (predecessor_physical / p).is_file()
    }

    successor_physical = tmp_path / "allocation-2"
    successor_physical.mkdir()
    prepare_run_context(
        _routed_cfg(logical_root),
        config_path="test.yaml",
        command="pytest",
        artifact_destination=ArtifactDestination(physical_root=successor_physical),
    )

    after = sorted(p.relative_to(predecessor_physical) for p in predecessor_physical.rglob("*"))
    after_bytes = {
        str(p): (predecessor_physical / p).read_bytes()
        for p in after
        if (predecessor_physical / p).is_file()
    }
    assert after == before
    assert after_bytes == before_bytes


# ---------------------------------------------------------------------------
# ArtifactDestination.bind refusals
# ---------------------------------------------------------------------------


def test_bind_refuses_a_path_outside_the_logical_run_dir(tmp_path: Path) -> None:
    """Mutant (4): an unknown sink is refused, never silently guessed at."""

    destination = ArtifactDestination(physical_root=tmp_path / "physical")
    logical_run_dir = tmp_path / "logical" / "run"
    elsewhere = tmp_path / "elsewhere" / "file.json"

    with pytest.raises(ArtifactRoutingError):
        destination.bind(logical_run_dir, elsewhere)


def test_bind_refuses_an_escape_through_a_symlink(tmp_path: Path) -> None:
    """Mutant (5): a path-traversal/symlink escape is refused, not followed.

    A symlink under the LOGICAL tree is already caught by the first check:
    ``Path.resolve()`` follows it before ``relative_to`` ever runs, so a sink
    path through it simply does not fall under ``logical_root`` (the "unknown
    destination" refusal). The second, independent check exists for a symlink
    planted inside the PHYSICAL destination itself -- for example a residue
    from a prior occupant of the claimed leaf -- where the sink path computed
    from the clean logical tree is innocent, but joining it onto the physical
    root and resolving it walks through that planted symlink and lands
    outside the claimed root.
    """

    logical_run_dir = tmp_path / "logical" / "run"
    physical_root = tmp_path / "physical"
    physical_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (physical_root / "checkpoints").symlink_to(outside)
    destination = ArtifactDestination(physical_root=physical_root)

    with pytest.raises(ArtifactRoutingError):
        destination.bind(logical_run_dir, logical_run_dir / "checkpoints" / "payload.json")


def test_bind_maps_a_nested_sink_to_the_same_relative_physical_path(tmp_path: Path) -> None:
    logical_run_dir = tmp_path / "logical" / "run"
    destination = ArtifactDestination(physical_root=tmp_path / "physical")

    physical = destination.bind(logical_run_dir, logical_run_dir / "checkpoints" / "step_000003")

    assert physical == tmp_path / "physical" / "checkpoints" / "step_000003"


def test_destination_requires_an_absolute_physical_root(tmp_path: Path) -> None:
    with pytest.raises(ArtifactRoutingError):
        ArtifactDestination(physical_root=Path("relative/leaf"))


# ---------------------------------------------------------------------------
# Mutant (1): a root-only overlay must not be mistaken for sink routing.
# ---------------------------------------------------------------------------


def test_changing_run_root_alone_does_not_route_an_already_baked_sink(tmp_path: Path) -> None:
    """HI bakes ``${run.dir}``-derived sink paths before ``run_from_config`` runs.

    Changing only ``run.root`` (today's worker overlay path, or any future
    regression that regresses to it) recomputes the logical run directory --
    it does not, by itself, claim or create a physical destination for that
    new logical directory, and it never redirects a different allocation's
    already-claimed physical leaf. This pins the negative: routing requires
    the explicit ``artifact_destination`` binding, not a config-side overlay.
    """

    cfg = _routed_cfg(tmp_path / "root-a")
    context_a = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    overlaid_cfg = _routed_cfg(tmp_path / "root-b")
    context_b = prepare_run_context(overlaid_cfg, config_path="test.yaml", command="pytest")

    # Both runs physically land at their own freshly recomputed logical
    # directories -- overlaying run.root changes what C says, which is exactly
    # the byte-identity problem this slice avoids by never touching run.root.
    assert context_a.run_dir != context_b.run_dir
    assert context_a.cfg.run.dir != context_b.cfg.run.dir
