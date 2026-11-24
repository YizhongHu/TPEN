"""C2.1 configured leaf-program contract and mutation controls."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from tpen_executor._codec import digest
from tpen_executor.configured_program import AttemptDelta, ConfiguredLoad
from tpen_executor.programs import ConfiguredProgram
from tpen_executor.runners import CONFIGURED_RUN_SHIM, ConfiguredProgramCommand, RUN_SHIM, TPENCommand
from tpen_executor.tasks import ConfiguredRun, ConfiguredRunTable


ROOT = Path(__file__).parents[2]
STAGE_Q = ROOT / "experiments" / "atomistic" / "he-importance" / "run_stage_q.py"
MANIFEST = ROOT / "experiments" / "atomistic" / "he-importance" / "o1_scan_manifest_v1.json"
ROW = "0" * 64
EXPECTED_RUN_SHIM_SHA256 = "066b37db2a22c1fd698327f21bbf2f55278023fd59f3e064323da057efa07c03"
EXPECTED_CONFIGURED_RUN_SHIM_SHA256 = "a3d4e64194cc3364783ba12674a992f76b0822c25af893e8ceb6b4fa1cce1533"


def _program(**changes: str) -> ConfiguredProgram:
    values = {
        "entrypoint": "stage-q",
        "script_path": str(STAGE_Q),
        "manifest": str(MANIFEST),
        "row": ROW,
        "output_root": str(ROOT / "outputs"),
        "output_contract": "attempt-core",
        "load": {"mode": "fresh", "path": None},
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


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux sched affinity and /proc")
def test_configured_shim_applies_placement_prelude_before_marker(tmp_path: Path) -> None:
    """The real configured shim applies affinity, sidecars, and strict host refusal."""

    cpu = min(os.sched_getaffinity(0))
    attempt = tmp_path / "store" / "groups" / ("g" * 64) / "task" / "attempt"
    (attempt / "core").mkdir(parents=True)
    placement = attempt / "child-placement.json"
    delta = AttemptDelta(
        str(attempt / "core"), ConfiguredLoad("fresh", None), "g" * 64, "task", "attempt",
    )
    script = tmp_path / "marker.py"
    marker = tmp_path / "marker"
    script.write_text(
        "import os\n"
        f"from pathlib import Path\nPath({str(marker)!r}).write_text(str(sorted(os.sched_getaffinity(0))))\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update({
        "TPEN_PLACEMENT_OUTPUT": str(placement),
        "TPEN_PLACEMENT_OUTPUT_DIR": str(attempt),
        "TPEN_PLACEMENT_DECLARED_CPUS": str(cpu),
        "TPEN_PLACEMENT_DECLARED_HOST": socket.gethostname(),
        "TPEN_ROUTED_STRICT_HOST": "1",
    })
    result = subprocess.run(
        [
            sys.executable, "-c", CONFIGURED_RUN_SHIM,
            "--tpen-executor-script", str(script), "--output-root", str(tmp_path / "campaign"),
            "--attempt-delta", delta.canonical_bytes.decode("utf-8"),
        ], env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8") == str([cpu])
    assert placement.exists()
    assert (attempt / "executor-child-observation.json").exists()
    assert json.loads(placement.read_text(encoding="utf-8"))["affinity"] == [cpu]

    mismatch_marker = tmp_path / "mismatch-marker"
    mismatch_script = tmp_path / "mismatch.py"
    mismatch_script.write_text(
        f"from pathlib import Path\nPath({str(mismatch_marker)!r}).write_text('launched')\n",
        encoding="utf-8",
    )
    mismatch_environment = dict(environment)
    mismatch_environment["TPEN_PLACEMENT_DECLARED_HOST"] = "not-" + socket.gethostname()
    mismatch = subprocess.run(
        [
            sys.executable, "-c", CONFIGURED_RUN_SHIM,
            "--tpen-executor-script", str(mismatch_script), "--output-root", str(tmp_path / "campaign"),
            "--attempt-delta", delta.canonical_bytes.decode("utf-8"),
        ], env=mismatch_environment, capture_output=True, text=True, check=False,
    )
    assert mismatch.returncode == 2
    assert not mismatch_marker.exists()


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


def test_manifest_content_change_at_one_path_changes_program_and_run_digests(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"row":0}\n')
    first = _program(manifest=str(manifest))
    manifest.write_bytes(b'{ "row": 0 }\n')
    second = _program(manifest=str(manifest))
    assert first.manifest == second.manifest
    assert first.program_digest != second.program_digest
    first_run = ConfiguredRun("configured", "", (), first)
    second_run = ConfiguredRun("configured", "", (), second)
    assert first_run.program_digest != second_run.program_digest


def test_public_configured_decoders_round_trip_and_refuse_bad_payloads(tmp_path: Path) -> None:
    program = _program()
    decoded = ConfiguredProgram.from_mapping(program.to_dict())
    assert decoded.to_dict() == program.to_dict()
    assert decoded.program_digest == program.program_digest
    legacy_payload = program.to_dict()
    legacy_payload.pop("manifest_sha256")
    assert ConfiguredProgram.from_mapping(legacy_payload).program_digest == program.program_digest

    table = ConfiguredRunTable(digest("group"), (ConfiguredRun("configured", "", (), program),))
    table_decoded = ConfiguredRunTable.from_mapping(table.to_dict())
    assert table_decoded.runs[0].program_digest == table.runs[0].program_digest
    assert table_decoded.runs[0].program.program_digest == table.runs[0].program.program_digest

    malformed = program.to_dict()
    malformed.pop("script_path")
    with pytest.raises(ValueError, match="keys"):
        ConfiguredProgram.from_mapping(malformed)
    foreign = tmp_path / "foreign" / "experiments" / "atomistic" / "he-importance" / "run_stage_q.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("raise SystemExit(0)\n", encoding="utf-8")
    link = tmp_path / "link" / "experiments" / "atomistic" / "he-importance" / "run_stage_q.py"
    link.parent.mkdir(parents=True)
    link.symlink_to(foreign)
    for candidate in (foreign, link):
        payload = program.to_dict()
        payload["script_path"] = str(candidate)
        with pytest.raises(ValueError, match="trusted"):
            ConfiguredProgram.from_mapping(payload)


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
        with pytest.raises(ValueError, match="trusted"):
            ConfiguredProgram.from_mapping(payload)


def test_non_v1_manifest_name_is_accepted_and_content_bound(tmp_path: Path) -> None:
    manifest = tmp_path / "o1_scan_manifest_v2.json"
    content = b'{"schema":"stage-q-v2","rows":[]}\n'
    manifest.write_bytes(content)
    program = _program(manifest=str(manifest))
    assert program.manifest == str(manifest)
    assert program.manifest_sha256 == hashlib.sha256(content).hexdigest()


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


def test_configured_command_for_attempt_carries_only_the_typed_delta(tmp_path: Path) -> None:
    configured = ConfiguredRun("configured", "", (), _program())
    command = configured.command_for_attempt(
        str(tmp_path / "groups" / ("g" * 64) / "configured" / "a1"),
        root=str(tmp_path),
        group_digest="g" * 64,
        task_id="configured",
        attempt_id="a1",
    )
    assert command.args[-2] == "--attempt-delta"
    assert "run.root=" not in " ".join(command.args)
    assert '"attempt_core_dir":"' + str(tmp_path / "attempt" / "core") + '"' in command.args[-1]


def test_legacy_and_configured_attempt_bindings_are_separate(tmp_path: Path) -> None:
    config = str(tmp_path / "config.yaml")
    legacy = ConfiguredRun("legacy", config, ("run.seed=7",))
    legacy_args = legacy.command_for_attempt(str(tmp_path / "legacy"), group_digest="g", task_id="legacy", attempt_id="a").args
    assert legacy_args == TPENCommand(
        config, ("run.seed=7", f"run.root={tmp_path / 'legacy'}", "run.layout=flat", "run.run_id=core")
    ).args
    configured = ConfiguredRun("configured", "", (), _program())
    configured_args = configured.command_for_attempt(
        str(tmp_path / "groups" / "g" / "configured" / "a"), root=str(tmp_path), group_digest="g", task_id="configured", attempt_id="a",
    ).args
    assert "--attempt-delta" in configured_args
    assert not any(arg.startswith("run.") for arg in configured_args)


def test_configured_shim_refuses_mismatched_attempt_output(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    marker = tmp_path / "launched"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('launched')\n", encoding="utf-8")
    expected = tmp_path / "expected"
    wrong = tmp_path / "wrong"
    expected.mkdir()
    wrong.mkdir()
    delta = AttemptDelta(
        str(expected / "core"), ConfiguredLoad("fresh", None), "g", "task", "attempt",
    )
    environment = dict(os.environ)
    environment.update({"TPEN_PLACEMENT_OUTPUT": str(expected / "child-placement.json")})
    result = subprocess.run(
        [
            sys.executable, "-c", CONFIGURED_RUN_SHIM,
            "--tpen-executor-script", str(script), "--output-root", str(expected),
            "--attempt-delta", delta.canonical_bytes.decode("utf-8"), f"run.root={wrong}",
        ], env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert not marker.exists()
    assert "until C2.2" in result.stderr


# Required T2 mutation map.  Each ``old`` fragment occurs exactly once in its
# target source, and every killer is a test above.
MUTANTS = (
    {
        "id": "configured-shim-skips-placement-prelude",
        "target": "executor/src/tpen_executor/runners.py",
        "old": "\"        run_roots = [_attempt_leaf]\\n\"",
        "new": "\"        run_roots = []\\n\"",
        "killer": "test_configured_shim_applies_placement_prelude_before_marker",
    },
    {
        "id": "allowlist-bypass",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": "if os.path.commonpath((trusted_root, resolved)) != trusted_root or resolved != expected or resolved != script_path:",
        "new": "if False:",
        "killer": "test_foreign_same_suffix_and_symlink_are_rejected_on_construction_and_decode",
    },
    {
        "id": "manifest-hash-mismatch-guard",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": "if self.manifest_sha256 and self.manifest_sha256 != computed_manifest_sha256:",
        "new": "if False:",
        "killer": "test_manifest_must_exist_be_regular_and_match_bound_sha256",
    },
    {
        "id": "decoder-set-of-sets",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": "if (keys != expected and keys != expected | {\"manifest_sha256\"}) or data.get(\"schema\") != cls.schema:",
        "new": "if set(data) not in {expected, expected | {\"manifest_sha256\"}} or data.get(\"schema\") != cls.schema:",
        "killer": "test_public_configured_decoders_round_trip_and_refuse_bad_payloads",
    },
    {
        "id": "digest-drops-manifest-hash",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": "            \"manifest_sha256\": self.manifest_sha256,\n",
        "new": "",
        "killer": "test_manifest_content_change_at_one_path_changes_program_and_run_digests",
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
    {
        "id": "parsl-admits-configured-before-dfk",
        "target": "executor/src/tpen_executor/dispatch/parsl.py",
        "old": "        super(ParslDispatcher, self).validate(group, runs, lifecycle, env, timeout)\n",
        "new": "        pass\n",
        "killer": "test_configured_parsl_public_admission_accepts_typed_binding_before_dfk",
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
