"""The HI firewall refuses a run BEFORE anything is constructed.

L1a's acceptance contract names one falsifier: "a nested forbidden reference
reaches construction". The rejection half is covered in
``tests/unit/test_hi_schema.py``. This module covers the half that a rejection
test cannot see -- that the refusal happened *early enough*.

That distinction is the whole point of the slice. ``prepare_run_context``
creates the run directory and instantiates every logger and callback, so a
firewall that raised after it would produce exactly the same exception while
having already built the thing it was supposed to prevent. Asserting
``pytest.raises`` alone would pass in both worlds.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

import tpen.run
from tpen.config_schema import ClosedSchemaError
from tpen.hi_schema import HI_TRAIN_SCHEMA


@pytest.fixture
def no_construction(monkeypatch):
    """Fail loudly if any construction step is reached.

    Returns a list that stays empty when the firewall did its job. Each
    replaced function records its own name rather than raising, so a test can
    report WHICH construction step was reached instead of only that one was.
    """

    reached: list[str] = []

    def _spy(name):
        def _record(*_args, **_kwargs):
            reached.append(name)
            raise AssertionError(f"{name} was reached; the firewall did not refuse first")

        return _record

    monkeypatch.setattr(tpen.run, "prepare_run_context", _spy("prepare_run_context"))
    monkeypatch.setattr(tpen.run, "_instantiate_runner", _spy("_instantiate_runner"))
    return reached


def _config_with(tmp_path, **sections):
    """Build a schema-declaring config rooted at ``tmp_path``."""

    base = {
        "schema": HI_TRAIN_SCHEMA,
        "experiment": {"name": "hi_firewall", "sector": "atomistic"},
        "run": {"root": str(tmp_path / "outputs"), "run_id": "hi_firewall_0001"},
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.005},
    }
    base.update(sections)
    return OmegaConf.create(base)


def _run(cfg):
    """Invoke the run entrypoint with exceptions surfaced."""

    return tpen.run.run_from_config(cfg, raise_exceptions=True)


class TestNothingIsConstructed:
    def test_a_nested_reference_never_reaches_construction(self, tmp_path, no_construction):
        """The contract's falsifier, with the construction half asserted."""

        cfg = _config_with(tmp_path, system={"nuclei": {"reference_energy": -2.9}})
        with pytest.raises(ClosedSchemaError):
            _run(cfg)
        assert no_construction == []

    def test_no_run_directory_is_created(self, tmp_path):
        """A refused run must leave nothing behind on disk.

        Deliberately does NOT use the ``no_construction`` fixture. That fixture
        replaces ``prepare_run_context``, which is the function that calls
        ``artifact_manager.make_dirs()`` -- so with the spy installed no
        directory could appear whatever the firewall did, and this assertion
        would hold for a reason that has nothing to do with the firewall. The
        real function has to be in place for the absence of a directory to mean
        anything. ``test_a_late_firewall_would_leave_a_directory`` below is the
        positive control that the real function does create one.
        """

        cfg = _config_with(tmp_path, system={"nuclei": {"reference_energy": -2.9}})
        with pytest.raises(ClosedSchemaError):
            _run(cfg)
        assert not (tmp_path / "outputs").exists()
        assert list(tmp_path.iterdir()) == []

    def test_a_late_firewall_would_leave_a_directory(self, tmp_path):
        """Positive control for the assertion above, with no spy installed.

        Calling ``prepare_run_context`` directly shows that reaching it really
        does create the run directory. Without this, "no directory exists"
        could never distinguish an early firewall from a run path that never
        creates directories at all.
        """

        cfg = _config_with(tmp_path)
        tpen.run.prepare_run_context(cfg)
        assert (tmp_path / "outputs").exists()

    # SECOND ROLE, added with the programmatic-caller firewall below. Because
    # this config is CLEAN, it also controls that the firewall inside
    # `prepare_run_context` is not OVER-restrictive: an admissible config must
    # still construct. A rule that refused everything would satisfy every
    # rejection test in this module and would fail only here -- and, later, on
    # the cluster, where a study that cannot start is a far more expensive way
    # to learn it.

    def test_a_forbidden_callback_never_reaches_construction(self, tmp_path, no_construction):
        cfg = _config_with(
            tmp_path,
            callbacks=[{"_target_": "tpen.diagnostics.energy.EnergyDiagnostic"}],
        )
        with pytest.raises(ClosedSchemaError):
            _run(cfg)
        assert no_construction == []

    def test_an_unadmitted_method_never_reaches_construction(self, tmp_path, no_construction):
        cfg = _config_with(tmp_path, optimizer={"_target_": "somewhere.SR"})
        with pytest.raises(ClosedSchemaError):
            _run(cfg)
        assert no_construction == []


class TestTheProgrammaticCallerIsFirewalledToo:
    """The firewall is a property of the construction path, not of one entry point.

    Everything above enters through ``run_from_config``. That leaves the more
    dangerous caller untested: a script, notebook, harness or second runner
    that calls ``prepare_run_context`` directly. ``prepare_run_context`` creates
    the run directory, instantiates and validates every configured logger and
    callback, builds the ``RunContext`` and writes the run-start artifact -- so
    if the check lived only in ``run_from_config``, all of that would happen for
    a configuration the schema refuses, and every test above would still pass.

    These tests are the reason the check is duplicated rather than merely moved.
    """

    def test_a_direct_call_with_a_reference_is_refused(self, tmp_path):
        cfg = _config_with(tmp_path, system={"nuclei": {"reference_energy": -2.9}})
        with pytest.raises(ClosedSchemaError):
            tpen.run.prepare_run_context(cfg)

    def test_a_refused_direct_call_constructs_nothing(self, tmp_path):
        """The refusal must precede construction, not merely accompany it.

        Paired with ``test_a_late_firewall_would_leave_a_directory``, which
        shows the same function DOES create a directory when the config is
        admissible. Without that pairing this assertion would hold in a world
        where ``prepare_run_context`` never created anything.
        """

        cfg = _config_with(tmp_path, system={"nuclei": {"reference_energy": -2.9}})
        with pytest.raises(ClosedSchemaError):
            tpen.run.prepare_run_context(cfg)
        assert not (tmp_path / "outputs").exists()
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        "sections",
        [
            {"callbacks": [{"_target_": "tpen.diagnostics.energy.EnergyDiagnostic"}]},
            {"optimizer": {"_target_": "somewhere.SR"}},
            {"trainer": {"stop_at_energy": -2.9}},
        ],
        ids=["forbidden-callback", "unadmitted-method", "target-triggered-stop"],
    )
    def test_every_refusal_family_holds_on_the_direct_path(self, tmp_path, sections):
        """The direct path enforces the WHOLE schema, not just the reference rule.

        A firewall re-checked at a second site is worth only as much as the
        rule set it re-checks; copying one family across and leaving the rest
        behind would look identical from ``run_from_config``.
        """

        cfg = _config_with(tmp_path, **sections)
        with pytest.raises(ClosedSchemaError):
            tpen.run.prepare_run_context(cfg)
        assert not (tmp_path / "outputs").exists()

    def test_a_non_hi_config_still_constructs_on_the_direct_path(self, tmp_path):
        """Opt-out survives the new check; the firewall did not become mandatory.

        The over-restriction control for the direct path. A configuration that
        declares no HI schema is not firewalled, here exactly as in
        ``run_from_config`` -- otherwise every non-HI caller in the repository
        would have been broken by closing this hole.
        """

        cfg = OmegaConf.create(
            {
                "experiment": {"name": "legacy"},
                "run": {"root": str(tmp_path / "outputs"), "run_id": "hi_firewall_0001"},
                "system": {"reference_energy": -2.9},
            }
        )
        tpen.run.prepare_run_context(cfg)
        assert (tmp_path / "outputs").exists()


class TestTheSpyWouldCatchALateFirewall:
    """Prove the instrument works, so an empty ``reached`` list means something.

    A guard fixture that never fires is indistinguishable from a guard fixture
    that cannot fire. These two tests are the positive control: they show that
    reaching construction really does register.
    """

    def test_a_clean_config_does_reach_construction(self, tmp_path, no_construction):
        cfg = _config_with(tmp_path)
        with pytest.raises(AssertionError, match="prepare_run_context was reached"):
            _run(cfg)
        assert no_construction == ["prepare_run_context"]

    def test_a_config_without_the_schema_key_reaches_construction(self, tmp_path, no_construction):
        """Opt-out is real: a non-HI config is not firewalled."""

        cfg = OmegaConf.create(
            {
                "experiment": {"name": "legacy"},
                "run": {"root": str(tmp_path / "outputs"), "run_id": "hi_firewall_0001"},
                "system": {"reference_energy": -2.9},
            }
        )
        with pytest.raises(AssertionError, match="prepare_run_context was reached"):
            _run(cfg)
        assert no_construction == ["prepare_run_context"]


class TestFailurePath:
    def test_the_cli_path_returns_one_rather_than_raising(self, tmp_path, no_construction):
        """Default CLI behaviour is a handled failure, not a traceback."""

        cfg = _config_with(tmp_path, system={"nuclei": {"reference_energy": -2.9}})
        assert tpen.run.run_from_config(cfg) == 1
        assert no_construction == []
        assert not (tmp_path / "outputs").exists()
