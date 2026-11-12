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
from tpen.logging.csv import CSV
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
    """Two allocations of the same cell must record the same C -- mutant (3).

    Checked against the ADMITTED C itself, not only against each other: the
    only change `prepare_run_context` is entitled to make to the supplied
    config is writing the one admitted `run.dir` value back in -- everything
    else must reach both contexts byte-for-byte unchanged (review-r2 F).
    """

    logical_root = tmp_path / "logical"
    first_physical = tmp_path / "allocation-1"
    second_physical = tmp_path / "allocation-2"
    first_physical.mkdir()
    second_physical.mkdir()
    supplied_cfg = _routed_cfg(logical_root)

    first = prepare_run_context(
        supplied_cfg,
        config_path="test.yaml",
        command="pytest",
        artifact_destination=ArtifactDestination(physical_root=first_physical),
    )
    second = prepare_run_context(
        supplied_cfg,
        config_path="test.yaml",
        command="pytest",
        artifact_destination=ArtifactDestination(physical_root=second_physical),
    )

    from tpen.artifacts import resolve_run_clock
    from tpen.checkpoint.hashing import populate_resolved_config_defaults

    run_clock = resolve_run_clock(supplied_cfg)
    expected = OmegaConf.create(
        populate_resolved_config_defaults(supplied_cfg, timezone=run_clock.timezone)
    )
    OmegaConf.update(
        expected, "run.dir", str(first.artifact_manager.logical_run_dir), merge=False, force_add=True
    )
    expected_yaml = OmegaConf.to_yaml(expected, resolve=True)
    assert OmegaConf.to_yaml(first.cfg, resolve=True) == expected_yaml
    assert OmegaConf.to_yaml(second.cfg, resolve=True) == expected_yaml
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

    This bakes ONE config's sinks against ``root-a`` first -- exactly as HI's
    eager ``${run.dir}`` resolution does, by writing the concrete, no-longer-
    interpolated ``run.dir`` into the same config object -- then overlays
    only ``run.root`` on that SAME, already-baked config and reruns
    preparation. Changing ``run.root`` recomputes the LOGICAL run directory
    cfg.run.dir records, but the sinks' already-baked, literal output paths
    stay pointing at the stale ``root-a`` location: overlaying ``run.root``
    is not a routing mechanism, it is exactly the byte-divergence hazard this
    slice's closed `artifact_destination` binding exists to avoid.
    """

    root_a = tmp_path / "root-a"
    cfg = _routed_cfg(root_a)
    baked_run_dir = root_a / "routing" / "unit" / "fixed-run-id"
    OmegaConf.update(cfg, "run.dir", str(baked_run_dir), merge=False, force_add=True)
    OmegaConf.resolve(cfg)  # Bakes every sink's "${run.dir}/..." into a literal string.
    stale_snapshot_path = cfg.callbacks[0].output_path
    assert stale_snapshot_path == str(baked_run_dir / "snapshot.yaml")

    OmegaConf.update(cfg, "run.root", str(tmp_path / "root-b"), merge=False, force_add=True)
    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    # Legacy mode always recomputes the recorded run.dir from root/run_id --
    # it now reflects root-b -- but the sink instantiated from the already
    # literal, already-baked config still carries root-a's stale path:
    # overlaying run.root touched neither the sink nor claimed any
    # destination for it.
    assert context.cfg.run.dir == str(tmp_path / "root-b" / "routing" / "unit" / "fixed-run-id")
    snapshot, _metadata_cb = context.callbacks
    assert str(snapshot.output_path) == stale_snapshot_path


# ---------------------------------------------------------------------------
# T2 round 1 review fixes (F1-F6)
# ---------------------------------------------------------------------------


def test_f1_admitted_run_dir_with_an_extra_segment_is_preserved_not_recomputed(
    tmp_path: Path,
) -> None:
    """A flat-layout admitted run.dir with an intermediate segment -- the real
    HI shape, root/stage/hash -- must be the binding base, not a bare
    root/run_id recompute that drops the segment.
    """

    root = tmp_path / "root"
    admitted_run_dir = root / "Q" / "cellhash"
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "routing", "sector": "unit", "run_name": "routing_unit"},
            "run": {
                "root": str(root),
                "run_id": "cellhash",
                "layout": "flat",
                "dir": str(admitted_run_dir),
            },
            "runtime": {"device": "cpu", "dtype": "float64"},
            "callbacks": [
                {
                    "_target_": "tpen.callback.snapshot.ResolvedConfigSnapshot",
                    "output_path": str(admitted_run_dir / "resolved_config.yaml"),
                },
            ],
            "loggers": [],
        }
    )
    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    destination = ArtifactDestination(physical_root=physical_root)

    context = prepare_run_context(
        cfg, config_path="test.yaml", command="pytest", artifact_destination=destination
    )

    # The bare recompute (flat layout -> root/run_id) would have been
    # root/"cellhash", losing the "Q" segment -- and every eagerly-baked sink
    # path would then fail containment against it instead of routing.
    assert context.cfg.run.dir == str(admitted_run_dir)
    (snapshot_cb,) = context.callbacks
    assert snapshot_cb.output_path == physical_root / "resolved_config.yaml"


def test_d_legacy_mode_never_resolves_an_unused_discarded_run_dir_placeholder(
    tmp_path: Path,
) -> None:
    """(D) review-r2 R2.5: the admitted `run.dir` read itself is gated on
    routed mode. A legacy config's `run.dir` is always OVERWRITTEN by the
    recomputed value and never otherwise consulted, so an unresolvable
    placeholder left there must never be resolved at all -- unlike the
    config-side read this slice's F1 fix (round 2) unconditionally added.
    """

    root = tmp_path / "root"
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "routing", "sector": "unit", "run_name": "routing_unit"},
            "run": {"root": str(root), "run_id": "cellhash", "dir": "${unused_placeholder}"},
            "runtime": {"device": "cpu", "dtype": "float64"},
        }
    )

    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    assert context.cfg.run.dir == str(root / "routing" / "unit" / "cellhash")


def test_f1_legacy_mode_still_recomputes_the_logical_dir_unchanged(tmp_path: Path) -> None:
    """Without a destination, an admitted run.dir in cfg is still OVERWRITTEN
    by the recomputed value -- unaffected legacy behavior (non-goal)."""

    root = tmp_path / "root"
    admitted_run_dir = root / "Q" / "cellhash"
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "routing", "sector": "unit", "run_name": "routing_unit"},
            "run": {
                "root": str(root),
                "run_id": "cellhash",
                "layout": "flat",
                "dir": str(admitted_run_dir),
            },
            "runtime": {"device": "cpu", "dtype": "float64"},
        }
    )

    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    assert context.cfg.run.dir == str(root / "cellhash")
    assert context.cfg.run.dir != str(admitted_run_dir)


def test_f2_a_destination_side_symlink_escape_on_run_start_is_refused(tmp_path: Path) -> None:
    """A symlink planted inside the DESTINATION (not the logical tree) must be
    caught for every manager-owned write, not only the root-level bind."""

    logical_root = tmp_path / "logical"
    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run_start.json").write_text("predecessor evidence\n")
    (physical_root / "run_start.json").symlink_to(outside / "run_start.json")
    destination = ArtifactDestination(physical_root=physical_root)

    cfg = _routed_cfg(logical_root)
    with pytest.raises(ArtifactRoutingError):
        prepare_run_context(
            cfg, config_path="test.yaml", command="pytest", artifact_destination=destination
        )

    assert (outside / "run_start.json").read_text() == "predecessor evidence\n"


def test_f2_a_required_subdirectory_symlink_escape_is_refused(tmp_path: Path) -> None:
    """A symlinked ``checkpoints``/``checks``/``diagnostics`` child must be
    refused during ``make_dirs``, before any write beneath it."""

    logical_root = tmp_path / "logical"
    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (physical_root / "checkpoints").symlink_to(outside)
    destination = ArtifactDestination(physical_root=physical_root)

    cfg = _routed_cfg(logical_root)
    with pytest.raises(ArtifactRoutingError):
        prepare_run_context(
            cfg, config_path="test.yaml", command="pytest", artifact_destination=destination
        )


def test_f3_an_unregistered_sink_bearing_object_is_refused(tmp_path: Path) -> None:
    """A core `WandB`-shaped logger (a ``dir`` attribute, unregistered type)
    must be refused rather than silently passed through."""

    from tpen.run import _route_artifact_sinks

    class _UnregisteredDirLogger:
        def __init__(self, directory: Path) -> None:
            self.dir = directory

        def log(self, record: object) -> None:  # pragma: no cover - not reached
            raise AssertionError("must be refused before any write")

        def finish(self) -> None:  # pragma: no cover - not reached
            raise AssertionError("must be refused before any write")

    logical_run_dir = tmp_path / "logical" / "fixed-run-id"
    physical_root = tmp_path / "allocation-1"
    destination = ArtifactDestination(physical_root=physical_root)
    sink = _UnregisteredDirLogger(directory=logical_run_dir / "wandb")

    with pytest.raises(ArtifactRoutingError):
        _route_artifact_sinks([sink], logical_run_dir=logical_run_dir, destination=destination)


def test_f3_a_sink_free_object_with_no_output_attribute_passes_through(tmp_path: Path) -> None:
    """A registered sink-free type -- one that owns no output path at all --
    stays unaffected.

    review-r2 B: sink-freedom is an explicit, closed allowlist (exact type),
    never inferred from the absence of an attribute -- an arbitrary ad hoc
    class is no longer a valid stand-in for this case.
    """

    from tpen.callback.timing.run_timing import RunTiming
    from tpen.run import _route_artifact_sinks

    logical_run_dir = tmp_path / "logical" / "fixed-run-id"
    physical_root = tmp_path / "allocation-1"
    destination = ArtifactDestination(physical_root=physical_root)
    sink = RunTiming()

    routed = _route_artifact_sinks([sink], logical_run_dir=logical_run_dir, destination=destination)

    assert routed == {}


def test_r3_1_a_subclass_of_a_registered_writer_is_refused_not_routed(
    tmp_path: Path,
) -> None:
    """(R3.1) review-r3: admission is by EXACT type only, no MRO/subclass
    admission. A subclass of a registered writer is refused by
    `_route_artifact_sinks`'s own backstop check, not silently routed
    through its ancestor's registered fields -- a subclass can add its own
    unaudited output path this registry's field list would never see."""

    from tpen.run import _route_artifact_sinks

    class _SubclassOfCSV(CSV):
        pass

    logical_run_dir = tmp_path / "logical" / "fixed-run-id"
    physical_root = tmp_path / "allocation-1"
    destination = ArtifactDestination(physical_root=physical_root)
    sink = _SubclassOfCSV(path=logical_run_dir / "metrics.csv")

    with pytest.raises(ArtifactRoutingError):
        _route_artifact_sinks([sink], logical_run_dir=logical_run_dir, destination=destination)


def test_r3_1_a_configured_writer_subclass_with_a_constructor_write_is_refused_before_construction(
    tmp_path: Path,
) -> None:
    """(R3.1) regression: a configured CSV subclass that writes an
    additional, unaudited path in its constructor must be refused before
    `instantiate()` ever runs it -- not merely matched and routed through its
    registered ancestor's `path` field, which never sees this side write."""

    from tpen.run import _instantiate_sequence

    logical_root = tmp_path / "logical"
    side_write_path = logical_root / "side_write.bin"

    item = OmegaConf.create(
        {
            "_target_": "tests.unit.test_run_artifact_routing._CSVSubclassWithConstructorWrite",
            "path": str(logical_root / "metrics.csv"),
            "side_write": str(side_write_path),
        }
    )

    with pytest.raises(ArtifactRoutingError):
        _instantiate_sequence([item], routed=True)

    assert not side_write_path.exists()


class _CSVSubclassWithConstructorWrite(CSV):
    """Module-level subclass so Hydra's `_target_` can resolve it by path.

    Writes `side_write` from its constructor -- the vector R3.1 describes: an
    admitted-by-MRO subclass could write an additional, unaudited path before
    its inherited `path` field is ever rebound."""

    def __init__(self, *, path: Path, side_write: str | Path, **kwargs: object) -> None:
        Path(side_write).parent.mkdir(parents=True, exist_ok=True)
        Path(side_write).write_text("unaudited constructor write")
        super().__init__(path=path, **kwargs)


def test_f4_runtime_callbacks_are_routed_and_present_before_the_context_returns(
    tmp_path: Path,
) -> None:
    """Runtime callbacks are routed/refused inside `prepare_run_context`,
    strictly before `run_from_config` could ever emit `RunStarted`."""

    from tpen.callback.snapshot import ResolvedConfigSnapshot

    logical_root = tmp_path / "logical"
    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    destination = ArtifactDestination(physical_root=physical_root)
    cfg = _routed_cfg(logical_root)

    probe = prepare_run_context(cfg, config_path="test.yaml", command="pytest")
    logical_run_dir = probe.artifact_manager.logical_run_dir

    runtime_snapshot = ResolvedConfigSnapshot(
        output_path=logical_run_dir / "runtime_resolved_config.yaml"
    )
    context = prepare_run_context(
        cfg,
        config_path="test.yaml",
        command="pytest",
        artifact_destination=destination,
        runtime_callbacks=[runtime_snapshot],
    )

    assert runtime_snapshot in context.callbacks
    assert runtime_snapshot.output_path == physical_root / "runtime_resolved_config.yaml"


def test_f4_an_unregistered_runtime_callback_is_refused_in_routed_mode(tmp_path: Path) -> None:
    class _BadRuntimeObserver:
        def __init__(self, output_dir: Path) -> None:
            self.output_dir = output_dir

        def handle_occurrence(self, occurrence: object, context: object) -> None:
            pass

    logical_root = tmp_path / "logical"
    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    destination = ArtifactDestination(physical_root=physical_root)
    cfg = _routed_cfg(logical_root)

    probe = prepare_run_context(cfg, config_path="test.yaml", command="pytest")
    logical_run_dir = probe.artifact_manager.logical_run_dir
    bad = _BadRuntimeObserver(output_dir=logical_run_dir / "custom")

    with pytest.raises(ArtifactRoutingError):
        prepare_run_context(
            cfg,
            config_path="test.yaml",
            command="pytest",
            artifact_destination=destination,
            runtime_callbacks=[bad],
        )


def test_r3_1_an_explicitly_admitted_runtime_observer_is_admitted_in_routed_mode(
    tmp_path: Path,
) -> None:
    """(R3.1) review-r3: admission by exact type only, never inferred from
    the absence of an output-path-shaped attribute. A runtime observer is
    admitted only when its EXACT type is a member of the closed sink-free
    registry (here `RunTiming`, also used as a configured sink-free type
    elsewhere in this file) -- an ad hoc class with no path attribute is no
    longer, by itself, sufficient."""

    from tpen.callback.timing.run_timing import RunTiming

    logical_root = tmp_path / "logical"
    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    destination = ArtifactDestination(physical_root=physical_root)
    cfg = _routed_cfg(logical_root)
    observer = RunTiming()

    context = prepare_run_context(
        cfg,
        config_path="test.yaml",
        command="pytest",
        artifact_destination=destination,
        runtime_callbacks=[observer],
    )

    assert observer in context.callbacks


def test_r3_1_an_ad_hoc_runtime_observer_with_no_output_attribute_is_still_refused(
    tmp_path: Path,
) -> None:
    """(R3.1) negative-of-the-negative: an ad hoc observer that merely lacks
    an output-path-shaped attribute is NOT, by itself, admitted -- only exact
    membership in the closed registries is. This is the regression the old
    attribute-probe inference would have passed."""

    class _AdHocObserverNoPathAttribute:
        def handle_occurrence(self, occurrence: object, context: object) -> None:
            pass

    logical_root = tmp_path / "logical"
    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    destination = ArtifactDestination(physical_root=physical_root)
    cfg = _routed_cfg(logical_root)

    with pytest.raises(ArtifactRoutingError):
        prepare_run_context(
            cfg,
            config_path="test.yaml",
            command="pytest",
            artifact_destination=destination,
            runtime_callbacks=[_AdHocObserverNoPathAttribute()],
        )


def test_r3_1_a_runtime_filename_writer_with_an_event_write_spy_is_refused_before_any_write(
    tmp_path: Path,
) -> None:
    """(R3.1) regression: an unregistered runtime callback that stores its
    destination as `filename` (not one of the conventional probe names the
    old heuristic checked) must be refused before any write -- exact-type
    admission refuses every unregistered type uniformly, with no attribute
    name able to let one slip through."""

    class _RuntimeFilenameWriter:
        def __init__(self, filename: Path) -> None:
            self.filename = filename
            self.wrote: list[object] = []

        def handle_occurrence(self, occurrence: object, context: object) -> None:
            Path(self.filename).write_text("leak")
            self.wrote.append(occurrence)

    logical_root = tmp_path / "logical"
    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    destination = ArtifactDestination(physical_root=physical_root)
    cfg = _routed_cfg(logical_root)

    probe = prepare_run_context(cfg, config_path="test.yaml", command="pytest")
    logical_run_dir = probe.artifact_manager.logical_run_dir
    bad = _RuntimeFilenameWriter(filename=logical_run_dir / "custom.bin")

    with pytest.raises(ArtifactRoutingError):
        prepare_run_context(
            cfg,
            config_path="test.yaml",
            command="pytest",
            artifact_destination=destination,
            runtime_callbacks=[bad],
        )

    assert bad.wrote == []
    assert not (logical_run_dir / "custom.bin").exists()


def test_f6_rebinding_a_populated_csv_logger_does_not_omit_the_new_files_header(
    tmp_path: Path,
) -> None:
    from tpen.logging.base import LogRecord
    from tpen.run import _route_artifact_sinks

    logical_root = tmp_path / "logical"
    logical_run_dir = logical_root / "fixed-run-id"
    logical_run_dir.mkdir(parents=True)
    logical_csv = logical_run_dir / "metrics.csv"
    logical_csv.write_text("step,namespace,key,value\n0,run,loss,1.0\n")

    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    destination = ArtifactDestination(physical_root=physical_root)

    sink = CSV(path=logical_csv)
    assert sink._wrote_header is True

    _route_artifact_sinks([sink], logical_run_dir=logical_run_dir, destination=destination)
    sink.log(LogRecord(step=1, namespace="run", metrics={"loss": 2.0}))

    new_path = physical_root / "metrics.csv"
    assert sink.path == new_path
    assert new_path.read_text().startswith("step,namespace,key,value\n")
    # The predecessor file is byte-identical to before rebinding.
    assert logical_csv.read_text() == "step,namespace,key,value\n0,run,loss,1.0\n"


# ---------------------------------------------------------------------------
# T2 round 2 review fixes (A-F)
# ---------------------------------------------------------------------------


def test_a_a_nonempty_preexisting_destination_is_refused_before_any_write(tmp_path: Path) -> None:
    """(A) review-r2 R2.1/R2.3: a destination with ANY pre-existing entry is
    refused at preparation -- closing the bootstrap-error and
    default/derived-filename link hazards without touching those owners."""

    logical_root = tmp_path / "logical"
    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    (physical_root / "residue.txt").write_text("from a prior occupant\n")
    destination = ArtifactDestination(physical_root=physical_root)

    with pytest.raises(ArtifactRoutingError):
        prepare_run_context(
            _routed_cfg(logical_root),
            config_path="test.yaml",
            command="pytest",
            artifact_destination=destination,
        )
    # Nothing was created or touched beneath the refused destination.
    assert [p.name for p in physical_root.iterdir()] == ["residue.txt"]


def test_a_a_destination_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    """(A) review-r2 R2.1: the destination itself must be a real directory."""

    logical_root = tmp_path / "logical"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    physical_root = tmp_path / "allocation-1"
    physical_root.symlink_to(elsewhere)
    destination = ArtifactDestination(physical_root=physical_root)

    with pytest.raises(ArtifactRoutingError):
        prepare_run_context(
            _routed_cfg(logical_root),
            config_path="test.yaml",
            command="pytest",
            artifact_destination=destination,
        )


def test_a_an_absent_or_empty_destination_is_still_admitted(tmp_path: Path) -> None:
    """(A) negative: a destination that does not exist, or exists empty, is
    still a valid fresh destination (not a regression on the common case)."""

    logical_root = tmp_path / "logical"
    absent_physical = tmp_path / "allocation-absent"
    empty_physical = tmp_path / "allocation-empty"
    empty_physical.mkdir()

    for physical_root in (absent_physical, empty_physical):
        context = prepare_run_context(
            _routed_cfg(logical_root),
            config_path="test.yaml",
            command="pytest",
            artifact_destination=ArtifactDestination(physical_root=physical_root),
        )
        assert context.run_dir == physical_root


def test_b_an_unregistered_configured_logger_is_refused_before_construction(
    tmp_path: Path,
) -> None:
    """(B) review-r2 R2.2: `_target_` is resolved and admitted BEFORE
    `instantiate()` ever runs the constructor -- WandB (unregistered, no
    sink-free listing) is refused in routed mode, never constructed."""

    logical_root = tmp_path / "logical"
    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    cfg = _routed_cfg(logical_root)
    OmegaConf.set_struct(cfg, False)
    cfg.loggers.append(
        {"_target_": "tpen.logging.wandb.WandB", "project": "p", "dir": "${run.dir}/wandb"}
    )

    with pytest.raises(ArtifactRoutingError):
        prepare_run_context(
            cfg,
            config_path="test.yaml",
            command="pytest",
            artifact_destination=ArtifactDestination(physical_root=physical_root),
        )


def test_b_an_unregistered_logger_constructor_never_runs_before_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(B) review-r2 R2.2, construction-order proof: admission is checked
    BEFORE `instantiate()` ever runs -- not merely before any write. A
    redundant post-construction check alone (defense in depth for direct
    callers of `_route_artifact_sinks`) would still raise, but only AFTER
    the constructor already ran; this test catches that regression even
    though the final exception type would be unchanged.
    """

    from tpen.logging.wandb import WandB

    constructed = []
    original_init = WandB.__init__
    monkeypatch.setattr(
        WandB, "__init__", lambda self, *a, **k: (constructed.append(True), original_init(self, *a, **k))
    )

    logical_root = tmp_path / "logical"
    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    cfg = _routed_cfg(logical_root)
    OmegaConf.set_struct(cfg, False)
    cfg.loggers.append({"_target_": "tpen.logging.wandb.WandB", "project": "p"})

    with pytest.raises(ArtifactRoutingError):
        prepare_run_context(
            cfg,
            config_path="test.yaml",
            command="pytest",
            artifact_destination=ArtifactDestination(physical_root=physical_root),
        )

    assert constructed == []


def test_b_an_unregistered_configured_logger_is_unaffected_without_a_destination(
    tmp_path: Path,
) -> None:
    """(B) negative: the new closed-admission gate is routed-only."""

    logical_root = tmp_path / "logical"
    cfg = _routed_cfg(logical_root)
    OmegaConf.set_struct(cfg, False)
    cfg.loggers.append({"_target_": "tpen.logging.wandb.WandB", "project": "p"})

    context = prepare_run_context(cfg, config_path="test.yaml", command="pytest")

    assert len(context.loggers) == 3


def test_r3_1_a_registered_writer_subclass_is_refused_before_construction(
    tmp_path: Path,
) -> None:
    """(R3.1) review-r3 supersedes (B)'s old MRO admission: a subclass of a
    registered writer is refused, by EXACT type only, before
    `instantiate()` ever runs it -- not matched and admitted through its
    registered ancestor."""

    from tpen.run import _instantiate_sequence

    logical_root = tmp_path / "logical"
    item = OmegaConf.create(
        {
            "_target_": "tests.unit.test_run_artifact_routing._SubclassOfCSVModule",
            "path": str(logical_root / "metrics.csv"),
        }
    )

    with pytest.raises(ArtifactRoutingError):
        _instantiate_sequence([item], routed=True)


def test_r3_1_a_registered_writer_subclass_is_unaffected_without_a_destination(
    tmp_path: Path,
) -> None:
    """(R3.1) negative: the exact-type admission gate is routed-only -- a
    legacy (no-destination) caller's custom writer subclass is unaffected,
    unchanged from before this round's fix."""

    from tpen.run import _instantiate_sequence

    logical_root = tmp_path / "logical"
    item = OmegaConf.create(
        {
            "_target_": "tests.unit.test_run_artifact_routing._SubclassOfCSVModule",
            "path": str(logical_root / "metrics.csv"),
        }
    )

    (instance,) = _instantiate_sequence([item], routed=False)

    assert isinstance(instance, _SubclassOfCSVModule)


class _SubclassOfCSVModule(CSV):
    """Module-level subclass so Hydra's `_target_` can resolve it by path."""


def test_e_without_a_destination_an_invalid_callback_still_writes_failure_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(E) review-r2 legacy failure lifecycle: a legacy (no-destination) run
    with an invalid runtime callback still writes the failure artifact and
    still calls every logger's `finish()`, instead of failing with
    `context=None` (the bug moving validation into `prepare_run_context`,
    unconditionally, would have introduced).
    """

    from tpen.logging.csv import CSV
    from tpen.run import run_from_config

    class _BadRuntimeCallback:
        """Missing `handle_occurrence` -- fails `_validate_callbacks`."""

    finished: list[bool] = []
    original_finish = CSV.finish
    monkeypatch.setattr(CSV, "finish", lambda self: (finished.append(True), original_finish(self)))

    logical_root = tmp_path / "logical"
    cfg = _routed_cfg(logical_root)

    rc = run_from_config(
        cfg, config_path="test.yaml", command="pytest", runtime_callbacks=[_BadRuntimeCallback()]
    )

    assert rc == 1
    assert finished == [True]
    logical_run_dir = tmp_path / "logical" / "routing" / "unit" / "fixed-run-id"
    assert (logical_run_dir / "run_start.json").exists()
    assert (logical_run_dir / "error.json").exists()


def test_r3_4_a_missing_finish_configured_logger_fails_handled_without_masking(
    tmp_path: Path,
) -> None:
    """(R3.4) regression: an invalid configured logger refuses inside
    `prepare_run_context`, before any `RunContext` exists -- so
    `run_from_config`'s unconditional `finally` loop (`logger.finish()` on
    every context logger) never reaches it. Moving this validation past
    context construction (the intervening, now-fixed S3 state) let that
    `AttributeError` mask this handled failure instead of a clean `return 1`.
    """

    from tpen.run import run_from_config

    logical_root = tmp_path / "logical"
    cfg = _routed_cfg(logical_root)
    OmegaConf.set_struct(cfg, False)
    cfg.loggers.append({"_target_": "builtins.object"})

    rc = run_from_config(cfg, config_path="test.yaml", command="pytest")

    assert rc == 1


def test_r3_4_a_missing_finish_configured_logger_raises_the_original_error_when_raising(
    tmp_path: Path,
) -> None:
    """(R3.4) negative-of-the-above in raising mode: the original `TypeError`
    from `_validate_loggers` propagates, not an `AttributeError` from a
    `finally`-loop call to `finish()` on an invalid logger that was never
    bound into any context."""

    from tpen.run import run_from_config

    logical_root = tmp_path / "logical"
    cfg = _routed_cfg(logical_root)
    OmegaConf.set_struct(cfg, False)
    cfg.loggers.append({"_target_": "builtins.object"})

    with pytest.raises(TypeError, match="log"):
        run_from_config(cfg, config_path="test.yaml", command="pytest", raise_exceptions=True)


def test_r3_4_direct_preparation_with_an_invalid_configured_callback_raises_before_any_context(
    tmp_path: Path,
) -> None:
    """(R3.4) regression: `prepare_run_context` itself validates configured
    callbacks unconditionally, before constructing anything -- a direct
    caller (bypassing `run_from_config`) gets the same refusal, with no
    context, directory, or callback left behind."""

    logical_root = tmp_path / "logical"
    cfg = _routed_cfg(logical_root)
    OmegaConf.set_struct(cfg, False)
    cfg.callbacks.append({"_target_": "builtins.object"})

    with pytest.raises(TypeError, match="handle_occurrence"):
        prepare_run_context(cfg, config_path="test.yaml", command="pytest")


def test_r3_4_a_failed_runtime_list_appends_no_runtime_callback_without_a_destination(
    tmp_path: Path,
) -> None:
    """(R3.4) regression: with runtime callbacks `[valid_observer,
    invalid_observer]` and no destination, a failing list appends NEITHER
    runtime observer -- restoring the original behavior where the whole
    list is validated before any of it is merged into `context.callbacks`.
    In the intervening, now-fixed S3 state, both runtime callbacks had
    already been merged inside `prepare_run_context` regardless of
    validation outcome, so the valid observer would have received
    `RunStarted`/`RunFailed` despite the list failing admission."""

    from tpen.run import run_from_config

    class _ValidObserver:
        def __init__(self) -> None:
            self.occurrences: list[object] = []

        def handle_occurrence(self, occurrence: object, context: object) -> None:
            self.occurrences.append(occurrence)

    class _InvalidObserver:
        """Missing `handle_occurrence` -- fails `_validate_callbacks`."""

    logical_root = tmp_path / "logical"
    cfg = _routed_cfg(logical_root)
    valid = _ValidObserver()

    rc = run_from_config(
        cfg,
        config_path="test.yaml",
        command="pytest",
        runtime_callbacks=[valid, _InvalidObserver()],
    )

    assert rc == 1
    assert valid.occurrences == []
