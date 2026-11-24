"""C2.2 common configured-program output/start binding controls."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tpen_executor._codec import digest
from tpen_executor.configured_program import AttemptDelta, ConfiguredLoad, ConfiguredProgram
from tpen_executor.dispatch.transport import LeafRequest
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.tasks import ConfiguredRun


ROOT = Path(__file__).parents[2]
STAGE_Q = ROOT / "experiments" / "atomistic" / "he-importance" / "run_stage_q.py"
MANIFEST = ROOT / "experiments" / "atomistic" / "he-importance" / "o1_scan_manifest_v1.json"
GROUP = "g" * 64
ROW = "0" * 64
DOC_EXAMPLE = (
    '{"attempt_core_dir":"/store/groups/g/t/a/core","attempt_id":"a",'
    '"group_digest":"g","load":{"mode":"fresh","path":null},'
    '"schema":"tpen-executor/attempt-delta-v1","task_id":"t"}'
)
DOC_EXAMPLE_SHA256 = "95f504768be45378d0d6a9396a88fc85b6524bbfda6ced9b9723b2701f775159"


def _program(tmp_path: Path, *, load: dict[str, object] | None = None) -> ConfiguredProgram:
    return ConfiguredProgram(
        "stage-q", str(STAGE_Q), str(MANIFEST), ROW, str(tmp_path / "campaign"),
        load={"mode": "fresh", "path": None} if load is None else load,
    )


def _configured(tmp_path: Path, *, load: dict[str, object] | None = None) -> ConfiguredRun:
    return ConfiguredRun("task", "", (), _program(tmp_path, load=load))


def _request(tmp_path: Path, *, load: dict[str, object] | None = None) -> LeafRequest:
    root = tmp_path / "store"
    output = root / "groups" / GROUP / "task" / "attempt"
    fence = SimpleNamespace(
        allocation_id="allocation", group_digest=GROUP, task_id="task", attempt_id=SimpleNamespace(value="attempt"),
        root=str(root), output_dir=str(output),
    )
    allocation = SimpleNamespace(
        allocation_id=SimpleNamespace(value="allocation"), backend_job_id=SimpleNamespace(value="job"),
    )
    return LeafRequest.for_attempt(
        fence, allocation, _configured(tmp_path, load=load),
        PreparedPythonEnvironment(sys.executable, env={}), 60,
    )


def test_attempt_delta_encoder_matches_normative_doc_example() -> None:
    delta = AttemptDelta(
        "/store/groups/g/t/a/core", ConfiguredLoad("fresh", None), "g", "t", "a",
    )
    assert delta.canonical_bytes.decode("utf-8") == DOC_EXAMPLE
    assert hashlib.sha256(delta.canonical_bytes).hexdigest() == DOC_EXAMPLE_SHA256
    assert delta.digest == DOC_EXAMPLE_SHA256


def test_load_mode_is_explicit_and_train_resume_is_not_own_leaf(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fresh|train_resume"):
        ConfiguredLoad("implicit", "/checkpoint")
    with pytest.raises(ValueError, match="keys"):
        ConfiguredLoad.from_mapping({"path": None})
    own_leaf = tmp_path / "store" / "groups" / GROUP / "task" / "attempt" / "checkpoint"
    with pytest.raises(ValueError, match="own attempt leaf"):
        AttemptDelta(
            str(own_leaf.parent / "core"), ConfiguredLoad("train_resume", str(own_leaf)),
            GROUP, "task", "attempt",
        )


def test_local_and_parsl_sources_use_one_typed_binding_without_duplicate_run_overrides() -> None:
    contract = (ROOT / "executor/src/tpen_executor/dispatch/contract.py").read_text(encoding="utf-8")
    parsl = (ROOT / "executor/src/tpen_executor/dispatch/parsl.py").read_text(encoding="utf-8")
    assert contract.count("configured.command_for_attempt(") == 1
    assert parsl.count("configured.command_for_attempt(") == 1
    assert "f\"run.root={output_dir}\"" not in contract
    assert "f\"run.root={output_dir}\"" not in parsl


def test_serialized_leaf_carries_delta_and_rejects_crossed_output(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert request.program is not None and request.attempt_delta is not None
    args = request.science_args()
    assert "--attempt-delta" in args
    assert not any(argument.startswith("run.") for argument in args)
    output_root_index = args.index("--output-root")
    assert args[output_root_index + 1] == request.program.output_root
    decoded = LeafRequest.decode(request.encode())
    assert decoded == request

    foreign = dataclasses.replace(
        request.attempt_delta,
        attempt_core_dir=str(tmp_path / "foreign" / "groups" / GROUP / "task" / "attempt" / "core"),
    )
    payload = request.to_dict()
    payload["attempt_delta"] = foreign.to_dict()
    payload["payload_digest"] = digest({"program_digest": request.program.program_digest, "attempt_delta_digest": foreign.digest})
    with pytest.raises(ValueError, match="attempt delta does not bind"):
        LeafRequest.from_mapping(payload)


def test_common_configured_binding_has_no_legacy_run_overrides(tmp_path: Path) -> None:
    args = _configured(tmp_path).command_for_attempt(
        f"/store/groups/{GROUP}/task/attempt", root="/store", group_digest=GROUP, task_id="task", attempt_id="attempt",
    ).args
    assert "--attempt-delta" in args
    assert not any(argument.startswith("run.") for argument in args)


@pytest.mark.parametrize("crossed", ["group", "task", "attempt"])
def test_crossed_identity_is_refused_before_child_on_all_carriers(tmp_path: Path, crossed: str) -> None:
    """The shared equation rejects crossed local/Parsl bindings and wire requests."""

    root = tmp_path / "store"
    valid_output = root / "groups" / GROUP / "task" / "attempt"
    values = {"group": GROUP, "task": "task", "attempt": "attempt"}
    values[crossed] = {"group": "foreign-group", "task": "foreign-task", "attempt": "foreign-attempt"}[crossed]
    crossed_output = root / "groups" / values["group"] / values["task"] / values["attempt"]
    configured = _configured(tmp_path)
    child_calls: list[str] = []
    for carrier in ("local", "parsl"):
        try:
            configured.command_for_attempt(
                str(crossed_output), root=str(root), group_digest=GROUP, task_id="task", attempt_id="attempt",
            )
        except ValueError as error:
            assert "placement fence" in str(error)
        else:
            child_calls.append(carrier)
    assert child_calls == []
    positive = configured.command_for_attempt(
        str(valid_output), root=str(root), group_digest=GROUP, task_id="task", attempt_id="attempt",
    )
    assert "--attempt-delta" in positive.args

    request = _request(tmp_path)
    payload = request.to_dict()
    payload["output_dir"] = str(crossed_output)
    with pytest.raises(ValueError):
        LeafRequest.from_mapping(payload)
    assert request == LeafRequest.decode(request.encode())


def test_all_carriers_use_the_shared_full_argv_binding(tmp_path: Path) -> None:
    """Legacy and configured argv stay identical for local, Parsl, and wire seams."""

    configured = _configured(tmp_path)
    request = _request(tmp_path)
    output = request.output_dir
    expected_configured = configured.command_for_attempt(
        output, root=request.root, group_digest=GROUP, task_id="task", attempt_id="attempt",
    ).args
    assert request.science_args() == expected_configured

    legacy_config = "/prepared/config.yaml"
    legacy_overrides = ("payload.mode=attached",)
    legacy = dataclasses.replace(
        request,
        program=None,
        attempt_delta=None,
        config=legacy_config,
        overrides=legacy_overrides,
        payload_digest=digest({"config": legacy_config, "overrides": list(legacy_overrides)}),
    )
    expected_legacy = ConfiguredRun("task", legacy_config, legacy_overrides).command_for_attempt(output).args
    assert legacy.science_args() == expected_legacy

    contract = (ROOT / "executor/src/tpen_executor/dispatch/contract.py").read_text(encoding="utf-8")
    parsl = (ROOT / "executor/src/tpen_executor/dispatch/parsl.py").read_text(encoding="utf-8")
    transport = (ROOT / "executor/src/tpen_executor/dispatch/transport.py").read_text(encoding="utf-8")
    assert contract.count("configured.command_for_attempt(") == 1
    assert parsl.count("configured.command_for_attempt(") == 1
    assert transport.count("command_for_attempt(") == 1
    assert "ConfiguredProgramCommand" not in transport


def test_attempt_binding_digest_changes_with_placement_delta(tmp_path: Path) -> None:
    configured = _configured(tmp_path)
    resumed = _configured(tmp_path, load={"mode": "train_resume", "path": "/predecessor/core"})
    assert configured.program is not None and resumed.program is not None
    assert configured.program.program_digest != resumed.program.program_digest
    first = configured.attempt_binding_digest(f"/store/groups/{GROUP}/task/a", root="/store", group_digest=GROUP, task_id="task", attempt_id="a")
    second = configured.attempt_binding_digest(f"/store/groups/{GROUP}/task/b", root="/store", group_digest=GROUP, task_id="task", attempt_id="b")
    assert first != second


MUTANTS = (
    {
        "id": "configured-local-duplicate-run-injection",
        "target": "executor/src/tpen_executor/tasks.py",
        "old": "        return ConfiguredProgramCommand(self.program, delta)\n",
        "new": "        command = ConfiguredProgramCommand(self.program, delta)\n        return type(\"MutatedConfiguredCommand\", (), {\"args\": command.args + (f\"run.root={output_dir}\", \"run.layout=flat\", \"run.run_id=core\")})()\n",
        "killer": "test_common_configured_binding_has_no_legacy_run_overrides",
    },
    {
        "id": "parsl-duplicate-run-injection",
        "target": "executor/src/tpen_executor/dispatch/parsl.py",
        "old": "                args = configured.command_for_attempt(\n                    output_dir,\n                    root=fence.root,\n                    group_digest=group.identity,\n                    task_id=member.task_id,\n                    attempt_id=fence.attempt_id.value,\n                ).args\n",
        "new": "                args = configured.command_for_attempt(\n                    output_dir,\n                    root=fence.root,\n                    group_digest=group.identity,\n                    task_id=member.task_id,\n                    attempt_id=fence.attempt_id.value,\n                ).args + (f\"run.root={output_dir}\", \"run.layout=flat\", \"run.run_id=core\")\n",
        "killer": "test_local_and_parsl_sources_use_one_typed_binding_without_duplicate_run_overrides",
    },
    {
        "id": "transport-duplicate-run-injection",
        "target": "executor/src/tpen_executor/dispatch/transport.py",
        "old": "            return ConfiguredRun(self.task_id, \"\", (), self.program).command_for_attempt(\n                output_dir,\n                root=self.root,\n                group_digest=self.group_digest,\n                task_id=self.task_id,\n                attempt_id=self.attempt_id,\n            ).args\n",
        "new": "            return ConfiguredRun(self.task_id, \"\", (), self.program).command_for_attempt(\n                output_dir,\n                root=self.root,\n                group_digest=self.group_digest,\n                task_id=self.task_id,\n                attempt_id=self.attempt_id,\n            ).args + (f\"run.root={output_dir}\", \"run.layout=flat\", \"run.run_id=core\")\n",
        "killer": "test_serialized_leaf_carries_delta_and_rejects_crossed_output",
    },
    {
        "id": "cross-output-dir",
        "target": "executor/src/tpen_executor/dispatch/transport.py",
        "old": "            if self.attempt_delta != expected_delta:\n                raise ValueError(\"attempt delta does not bind the request fence\")\n",
        "new": "            if False:\n                raise ValueError(\"attempt delta does not bind the request fence\")\n",
        "killer": "test_serialized_leaf_carries_delta_and_rejects_crossed_output",
    },
    {
        "id": "misdirect-hi-output-root",
        "target": "executor/src/tpen_executor/runners.py",
        "old": "            *self.program.argv(),\n",
        "new": "            \"--manifest\", self.program.manifest, \"--row\", self.program.row, \"--output-root\", self.attempt_delta.attempt_core_dir if self.attempt_delta else self.program.output_root,\n",
        "killer": "test_serialized_leaf_carries_delta_and_rejects_crossed_output",
    },
    {
        "id": "own-leaf-resume-path-check",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": "            if os.path.commonpath((attempt_leaf, self.load.path)) == attempt_leaf:\n                raise ValueError(\"train_resume load path must not lie in its own attempt leaf\")\n",
        "new": "            if False:\n                raise ValueError(\"train_resume load path must not lie in its own attempt leaf\")\n",
        "killer": "test_load_mode_is_explicit_and_train_resume_is_not_own_leaf",
    },
    {
        "id": "missing-load-mode-defaults-to-fresh",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": "        if not isinstance(data, Mapping) or set(data) != {\"mode\", \"path\"}:\n            raise ValueError(\"ConfiguredLoad keys differ\")\n",
        "new": "        if not isinstance(data, Mapping):\n            raise ValueError(\"ConfiguredLoad keys differ\")\n        if \"mode\" not in data:\n            data = {**data, \"mode\": \"fresh\", \"path\": None}\n        if set(data) != {\"mode\", \"path\"}:\n            raise ValueError(\"ConfiguredLoad keys differ\")\n",
        "killer": "test_load_mode_is_explicit_and_train_resume_is_not_own_leaf",
    },
    {
        "id": "delta-digest-omission",
        "target": "executor/src/tpen_executor/tasks.py",
        "old": "        return digest({\"program_digest\": self.program.program_digest, \"attempt_delta_digest\": delta.digest})\n",
        "new": "        return self.program.program_digest\n",
        "killer": "test_attempt_binding_digest_changes_with_placement_delta",
    },
)


def test_mutant_sources_compile_and_mapping_rows_are_unique() -> None:
    seen: set[str] = set()
    for mutant in MUTANTS:
        assert set(mutant) == {"id", "target", "old", "new", "killer"}
        assert mutant["id"] not in seen
        seen.add(mutant["id"])
        source_path = ROOT / mutant["target"]
        source = source_path.read_text(encoding="utf-8")
        assert source.count(mutant["old"]) == 1
        compile(source.replace(mutant["old"], mutant["new"], 1), str(source_path), "exec")
        assert mutant["killer"]
