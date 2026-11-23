"""C2.1 configured leaf-program contract and mutation controls."""

from __future__ import annotations

from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.programs import ConfiguredProgram
from tpen_executor.runners import CONFIGURED_RUN_SHIM, ConfiguredProgramCommand, RUN_SHIM, TPENCommand
from tpen_executor.tasks import ConfiguredRun


ROOT = Path(__file__).parents[2]
STAGE_Q = ROOT / "experiments" / "atomistic" / "he-importance" / "run_stage_q.py"
MANIFEST = ROOT / "experiments" / "atomistic" / "he-importance" / "o1_scan_manifest_v1.json"
ROW = "0" * 64


def _program(**changes: str) -> ConfiguredProgram:
    values = {
        "entrypoint": "stage-q",
        "script_path": str(STAGE_Q),
        "manifest": str(MANIFEST),
        "row": ROW,
        "output_root": str(ROOT / "outputs"),
        "output_contract": "attempt-core",
    }
    values.update(changes)
    return ConfiguredProgram(**values)


def test_legacy_tpen_command_argv_matches_golden(tmp_path: Path) -> None:
    """The existing TPEN command remains byte-for-byte compatible."""

    config = str(tmp_path / "config.yaml")
    overrides = ("run.seed=7", "payload.mode=attached")
    assert TPENCommand(config, overrides).args == (
        "-c", RUN_SHIM, "--config", config, "run.seed=7", "payload.mode=attached",
    )
    assert ConfiguredRun("legacy", config, overrides).command().args == TPENCommand(config, overrides).args


def test_configured_program_generates_fixed_stage_q_argv() -> None:
    program = _program()
    args = ConfiguredProgramCommand(program).args
    assert args[:4] == ("-c", CONFIGURED_RUN_SHIM, "--tpen-executor-entry", "stage-q")
    assert args[4:6] == ("--tpen-executor-script", str(STAGE_Q))
    assert args[6:] == ("--manifest", str(MANIFEST), "--row", ROW, "--output-root", str(ROOT / "outputs"))


@pytest.mark.parametrize("entrypoint", ["unknown", "hi.run_stage_q", "python -c"])
def test_unknown_or_dotted_entrypoint_refuses_before_spawn(entrypoint: str, monkeypatch: pytest.MonkeyPatch) -> None:
    spawned = []
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))
    with pytest.raises((TypeError, ValueError)):
        _program(entrypoint=entrypoint)
    assert spawned == []


@pytest.mark.parametrize("field", ["entrypoint", "script_path", "manifest", "row", "output_root", "output_contract"])
def test_program_digest_changes_for_every_bound_field(field: str) -> None:
    base = _program()
    changed = {
        "entrypoint": "stage-q",
        "script_path": str(STAGE_Q),
        "manifest": str(MANIFEST),
        "row": "1" * 64,
        "output_root": str(ROOT / "other-output"),
        "output_contract": "attempt-core",
    }
    if field == "script_path":
        changed[field] = str(STAGE_Q)
    assert digest(base.to_dict(include_schema=False)) == base.program_digest
    # The allowlist constrains values that are operationally meaningful; mutate
    # a canonical field through the wire payload to prove digest sensitivity.
    payload = base.to_dict(include_schema=False)
    payload[field] = changed[field]
    if payload == base.to_dict(include_schema=False):
        payload[field] = payload[field] + "-changed"
    assert digest(payload) != base.program_digest


@pytest.mark.parametrize(
    "changes",
    [
        {"script_path": "relative/run_stage_q.py"},
        {"manifest": "relative/manifest.json"},
        {"manifest": str(ROOT / "wrong-manifest.json")},
        {"row": "not-a-row"},
    ],
)
def test_crossed_or_relative_input_refuses_before_spawn(changes: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    spawned = []
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))
    with pytest.raises((TypeError, ValueError)):
        _program(**changes)
    assert spawned == []


def test_allowlisted_entry_rejects_existing_script_with_wrong_suffix_before_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suffix guard itself is exercised after all earlier path checks pass."""

    spawned = []
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))
    wrong_script = ROOT / "experiments" / "atomistic" / "he-importance" / "test_stage_coordinate.py"
    assert wrong_script.is_file()
    with pytest.raises(ValueError, match="allowlist"):
        _program(script_path=str(wrong_script))
    assert spawned == []


# Required T2 mutation map.  Each ``old`` fragment occurs exactly once in its
# target source, and every killer is a test above.
MUTANTS = (
    {
        "id": "allowlist-bypass",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": "if type(self.entrypoint) is not str or self.entrypoint not in CONFIGURED_PROGRAM_ALLOWLIST:",
        "new": "if type(self.entrypoint) is not str:",
        "killer": "test_unknown_or_dotted_entrypoint_refuses_before_spawn",
    },
    {
        "id": "digest-drops-field",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": 'return digest(self.to_dict(include_schema=False))',
        "new": 'return digest({"entrypoint": self.entrypoint})',
        "killer": "test_program_digest_changes_for_every_bound_field",
    },
    {
        "id": "legacy-argv-drift",
        "target": "executor/src/tpen_executor/runners.py",
        "old": 'return ("-c", RUN_SHIM, "--config", self.config, *self.overrides)',
        "new": 'return ("-m", "tpen.run", "--config", self.config, *self.overrides)',
        "killer": "test_legacy_tpen_command_argv_matches_golden",
    },
    {
        "id": "skip-path-validation",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": 'if tuple(Path(script_path).parts[-len(Path(expected).parts):]) != tuple(Path(expected).parts):',
        "new": 'if False:',
        "killer": "test_allowlisted_entry_rejects_existing_script_with_wrong_suffix_before_spawn",
    },
)


def test_mutant_sources_compile_and_mapping_rows_are_unique() -> None:
    """Run the mutation sanity check with plain Python source compilation."""

    seen = set()
    for mutant in MUTANTS:
        assert set(mutant) == {"id", "target", "old", "new", "killer"}
        assert mutant["id"] not in seen
        seen.add(mutant["id"])
        source_path = ROOT / mutant["target"]
        source = source_path.read_text(encoding="utf-8")
        assert source.count(mutant["old"]) == 1
        mutated = source.replace(mutant["old"], mutant["new"], 1)
        compile(mutated, str(source_path), "exec")
        assert mutant["killer"]
