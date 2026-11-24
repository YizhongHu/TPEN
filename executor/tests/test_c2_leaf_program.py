"""C2.1 configured leaf-program contract and mutation controls."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tpen_executor.dispatch.transport import ConfiguredProgramRouteRefused, LeafRequest
from tpen_executor.programs import ConfiguredProgram
from tpen_executor.runners import CONFIGURED_RUN_SHIM, ConfiguredProgramCommand, RUN_SHIM, TPENCommand
from tpen_executor.tasks import ConfiguredRun


ROOT = Path(__file__).parents[2]
STAGE_Q = ROOT / "experiments" / "atomistic" / "he-importance" / "run_stage_q.py"
MANIFEST = ROOT / "experiments" / "atomistic" / "he-importance" / "o1_scan_manifest_v1.json"
ROW = "0" * 64
EXPECTED_RUN_SHIM_SHA256 = "066b37db2a22c1fd698327f21bbf2f55278023fd59f3e064323da057efa07c03"
EXPECTED_CONFIGURED_RUN_SHIM_SHA256 = "1900a8836e19d91799360cf3a688e77760dcea8a3060a155158d0dc160c43247"


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
    assert hashlib.sha256(RUN_SHIM.encode()).hexdigest() == EXPECTED_RUN_SHIM_SHA256
    assert hashlib.sha256(CONFIGURED_RUN_SHIM.encode()).hexdigest() == EXPECTED_CONFIGURED_RUN_SHIM_SHA256
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


def test_real_program_and_run_variants_have_distinct_digests(tmp_path: Path) -> None:
    """Digest checks compare valid records, not edited dictionaries."""

    base = _program()
    output_variant = _program(output_root=str(ROOT / "other-output"))
    row_variant = _program(row="1" * 64)
    manifest_copy = tmp_path / "o1_scan_manifest_v1.json"
    manifest_copy.write_bytes(MANIFEST.read_bytes() + b"\nvariant")
    manifest_variant = _program(manifest=str(manifest_copy))
    assert base.program_digest != output_variant.program_digest
    assert base.program_digest != row_variant.program_digest
    assert base.program_digest != manifest_variant.program_digest
    run_a = ConfiguredRun("configured", "", (), base)
    run_b = ConfiguredRun("configured", "", (), output_variant)
    assert run_a.program_digest != run_b.program_digest
    legacy_a = ConfiguredRun("legacy", str(tmp_path / "a.yaml"), ())
    legacy_b = ConfiguredRun("legacy", str(tmp_path / "b.yaml"), ("runner.load.mode=train_resume",))
    assert legacy_a.program_digest != legacy_b.program_digest


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


def test_foreign_same_suffix_and_symlink_are_rejected_on_construction_and_decode(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign" / "experiments" / "atomistic" / "he-importance" / "run_stage_q.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("raise SystemExit(0)\n", encoding="utf-8")
    link = tmp_path / "link" / "experiments" / "atomistic" / "he-importance" / "run_stage_q.py"
    link.parent.mkdir(parents=True)
    link.symlink_to(foreign)
    for candidate in (foreign, link):
        with pytest.raises(ValueError, match="trusted"):
            _program(script_path=str(candidate))
        payload = _program().to_dict()
        payload["script_path"] = str(candidate)
        decoded_fields = {
            field: payload[field]
            for field in (
                "entrypoint", "script_path", "manifest", "row", "output_root",
                "output_contract", "manifest_sha256",
            )
        }
        with pytest.raises(ValueError, match="trusted"):
            ConfiguredProgram(**decoded_fields)


def test_manifest_must_exist_be_regular_and_match_bound_sha256(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "o1_scan_manifest_v1.json"
    with pytest.raises(ValueError, match="exist"):
        _program(manifest=str(missing))
    directory = tmp_path / "directory" / "o1_scan_manifest_v1.json"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="regular"):
        _program(manifest=str(directory))
    with pytest.raises(ValueError, match="sha256"):
        _program(manifest_sha256="0" * 64)


def test_configured_run_rejects_unbound_prepared_inputs_before_spawn() -> None:
    with pytest.raises(ValueError, match="until C2.2"):
        ConfiguredRun("configured", "/prepared/config.yaml", (), _program())
    with pytest.raises(ValueError, match="until C2.2"):
        ConfiguredRun("configured", "", ("runner.load.mode=train_resume",), _program())


def test_leaf_request_refuses_configured_program_route_before_submission() -> None:
    fence = SimpleNamespace(allocation_id="allocation", task_id="configured")
    allocation = SimpleNamespace(allocation_id=SimpleNamespace(value="allocation"))
    configured = ConfiguredRun("configured", "", (), _program())
    with pytest.raises(ConfiguredProgramRouteRefused, match="until C2.2"):
        LeafRequest.for_attempt(fence, allocation, configured, None)  # type: ignore[arg-type]


def test_configured_shim_refuses_mismatched_attempt_output(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    marker = tmp_path / "launched"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('launched')\n", encoding="utf-8")
    expected = tmp_path / "expected"
    wrong = tmp_path / "wrong"
    expected.mkdir()
    wrong.mkdir()
    result = subprocess.run(
        [
            sys.executable, "-c", CONFIGURED_RUN_SHIM,
            "--tpen-executor-script", str(script), "--output-root", str(expected),
            f"run.root={wrong}",
        ], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert not marker.exists()
    assert "until C2.2" in result.stderr


# Required T2 mutation map.  Each ``old`` fragment occurs exactly once in its
# target source, and every killer is a test above.
MUTANTS = (
    {
        "id": "allowlist-bypass",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": "if os.path.commonpath((trusted_root, resolved)) != trusted_root or resolved != expected or resolved != script_path:",
        "new": "if False:",
        "killer": "test_foreign_same_suffix_and_symlink_are_rejected_on_construction_and_decode",
    },
    {
        "id": "digest-drops-field",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": "if self.manifest_sha256 and self.manifest_sha256 != computed_manifest_sha256:",
        "new": "if False:",
        "killer": "test_manifest_must_exist_be_regular_and_match_bound_sha256",
    },
    {
        "id": "legacy-argv-drift",
        "target": "executor/src/tpen_executor/runners.py",
        "old": 'return ("-c", RUN_SHIM, "--config", self.config, *self.overrides)',
        "new": 'return ("-m", "tpen.run", "--config", self.config, *self.overrides)',
        "killer": "test_legacy_tpen_command_argv_matches_golden",
    },
    {
        "id": "unbound-prepared-input",
        "target": "executor/src/tpen_executor/tasks.py",
        "old": 'if self.program is not None and (self.config or self.overrides):',
        "new": 'if False:',
        "killer": "test_configured_run_rejects_unbound_prepared_inputs_before_spawn",
    },
    {
        "id": "configured-output-filter",
        "target": "executor/src/tpen_executor/runners.py",
        "old": "    \"    if any(path != declared_output for path in attempt_outputs) or any(arg.startswith('run.layout=') or arg.startswith('run.run_id=') for arg in script_args):\\n\"\n",
        "new": "    \"    if False:\\n\"\n",
        "killer": "test_configured_shim_refuses_mismatched_attempt_output",
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
