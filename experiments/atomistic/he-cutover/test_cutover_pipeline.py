from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Siblings are loaded study-scoped, not by bare import: experiments/ has several
# same-named modules and the first study loaded would otherwise own the bare name
# for every study after it. See experiments/toolkit/study_imports.py.
#
# The loader is reached BY PATH rather than by putting the repository root on
# sys.path. A study directory that mutates sys.path is the mechanism behind the
# very defect this import exists to fix, and he-cutover's gateway test forbids it
# outright -- so the fix must not reintroduce it in order to install itself.
import importlib.util as _tpen_importlib  # noqa: E402
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

if "_tpen_study_imports" not in _tpen_sys.modules:
    _tpen_spec = _tpen_importlib.spec_from_file_location(
        "_tpen_study_imports",
        _TpenPath(__file__).resolve().parents[3] / "experiments" / "toolkit" / "study_imports.py",
    )
    _tpen_module = _tpen_importlib.module_from_spec(_tpen_spec)
    _tpen_sys.modules["_tpen_study_imports"] = _tpen_module
    _tpen_spec.loader.exec_module(_tpen_module)
sibling = _tpen_sys.modules["_tpen_study_imports"].sibling

cutover_plan = sibling(__file__, 'cutover_plan')
pipeline = sibling(__file__, 'pipeline')
from experiments.toolkit.dispatch import DispatchRecord
from experiments.toolkit.parsl_attach import validate_pbs_nodefile


class RecordingExecutor:
    def __init__(self, fail_stage=None, materialize_completion=True):
        self.stages = []
        self.interpreters = []
        self.fail_stage = fail_stage
        self.materialize_completion = materialize_completion

    def dispatch(self, dispatches, *, context):
        context.validate()
        stage = dispatches[0].stage
        self.stages.append(stage)
        self.interpreters.extend(dispatch.argv[0] for dispatch in dispatches)
        if stage == self.fail_stage:
            raise RuntimeError("injected failure")
        records = []
        for dispatch in dispatches:
            if stage == "02_train" and self.materialize_completion:
                status = Path(dispatch.completion.status_path)
                status.parent.mkdir(parents=True, exist_ok=True)
                status.write_text('{"status":"completed"}\n')
                checkpoint = Path(dispatch.completion.checkpoint_path)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text("complete\n")
                checkpoint_dir = checkpoint.parent
                (checkpoint_dir / "manifest.json").write_text("{}\n")
            elif stage == "03_eval" and self.materialize_completion:
                status = Path(dispatch.completion.status_path)
                status.parent.mkdir(parents=True, exist_ok=True)
                status.write_text('{"status":"completed"}\n')
            records.append(DispatchRecord.accepted(dispatch, backend="fake", launcher_job_id=context.allocation_id, submitted_command=dispatch.argv))
        return tuple(records)


def _plans(tmp_path):
    grid = cutover_plan.load_grid(Path(__file__).with_name("smoke_grid.yaml"))
    return cutover_plan.build_plans(grid, facility="cannon", results_root=tmp_path / "results", plan_id="p")[:2]


def test_facility_binding_branches_are_exercised_end_to_end(tmp_path: Path) -> None:
    cannon = pipeline.allocation_context(facility="cannon", run_root=tmp_path / "cannon", environ={"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "MIG-owned-by-slurm"})
    polaris = pipeline.allocation_context(facility="polaris", run_root=tmp_path / "polaris", environ={"PBS_JOBID": "2.server"})
    assert cannon.visibility_values == ()
    assert polaris.visibility_values == ("0", "1", "2", "3")


def test_nodes_per_block_intent_keeps_absent_and_one_on_legacy_path(tmp_path: Path) -> None:
    absent = pipeline.allocation_context(facility="polaris", run_root=tmp_path / "absent", environ={"PBS_JOBID": "1.server"})
    one = pipeline.allocation_context(facility="polaris", run_root=tmp_path / "one", environ={"PBS_JOBID": "1.server", "TPEN_NODES_PER_BLOCK": "1"})
    assert absent.nodes_per_block is None
    assert one.nodes_per_block is None
    assert absent.to_dict() | {"run_root": str(tmp_path / "one")} == one.to_dict()


def test_nodes_per_block_intent_two_selects_multinode_path(tmp_path: Path) -> None:
    context = pipeline.allocation_context(
        facility="polaris",
        run_root=tmp_path / "two",
        environ={"PBS_JOBID": "1.server", "TPEN_NODES_PER_BLOCK": "2"},
    )
    assert context.nodes_per_block == 2
    assert context.visibility_values == ("0", "1", "2", "3")


def test_nodefile_guard_compares_hosts_with_independent_pipeline_intent(tmp_path: Path) -> None:
    nodefile = tmp_path / "PBS_NODEFILE"
    nodefile.write_text("node-01\n")
    context = pipeline.allocation_context(
        facility="polaris",
        run_root=tmp_path / "two",
        environ={"PBS_JOBID": "1.server", "TPEN_NODES_PER_BLOCK": "2", "PBS_NODEFILE": str(nodefile)},
    )
    assert context.nodes_per_block == 2
    with pytest.raises(RuntimeError, match=r"actual host count 1.*expected 2"):
        validate_pbs_nodefile(nodefile, requested_node_count=context.nodes_per_block)


def test_nodefile_guard_accepts_two_hosts_against_intent_two(tmp_path: Path) -> None:
    nodefile = tmp_path / "PBS_NODEFILE"
    nodefile.write_text("node-01\nnode-02\n")
    context = pipeline.allocation_context(
        facility="polaris",
        run_root=tmp_path / "two",
        environ={"PBS_JOBID": "1.server", "TPEN_NODES_PER_BLOCK": "2"},
    )
    assert validate_pbs_nodefile(nodefile, requested_node_count=context.nodes_per_block) == (
        "node-01", "node-02"
    )


def test_preflight_and_science_record_the_same_unresolved_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = tmp_path / "venv"
    executable = prefix / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "executable", str(executable))
    train, evaluation = _plans(tmp_path)
    executor = RecordingExecutor()

    code = pipeline.run_pipeline(train_plan=train, eval_plan=evaluation, facility="cannon", launch_dir=tmp_path / "launch", admission_id="a", executor=executor, environ={"SLURM_JOB_ID": "1"})

    assert code == 0
    assert executor.interpreters == [str(executable)] * 4
    assert all(interpreter != str(executable.resolve()) for interpreter in executor.interpreters)


@pytest.mark.parametrize(
    ("facility", "scheduler_env"),
    [("cannon", {"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "MIG-owned-by-slurm"}), ("polaris", {"PBS_JOBID": "2.server"})],
)
def test_pipeline_orders_probe_train_barrier_eval_and_verification_matches_exit(tmp_path: Path, facility: str, scheduler_env: dict[str, str]) -> None:
    grid = cutover_plan.load_grid(Path(__file__).with_name("smoke_grid.yaml"))
    train, evaluation = cutover_plan.build_plans(grid, facility=facility, results_root=tmp_path / "results", plan_id="p")[:2]
    executor = RecordingExecutor()
    code = pipeline.run_pipeline(train_plan=train, eval_plan=evaluation, facility=facility, launch_dir=tmp_path / "launch", admission_id="a", executor=executor, environ=scheduler_env)
    verification = json.loads((tmp_path / "launch/verification.json").read_text())
    assert executor.stages == ["01_preflight", "02_train", "03_eval"]
    assert (code, verification["exit_code"], verification["complete"]) == (0, 0, True)
    assert (tmp_path / "launch/02_train/dispatch_specs.jsonl").is_file()
    assert (tmp_path / "launch/03_eval/dispatch_specs.jsonl").is_file()


def test_preflight_failure_prevents_all_science_and_is_truthful(tmp_path: Path) -> None:
    train, evaluation = _plans(tmp_path)
    executor = RecordingExecutor(fail_stage="01_preflight")
    code = pipeline.run_pipeline(train_plan=train, eval_plan=evaluation, facility="cannon", launch_dir=tmp_path / "launch", admission_id="a", executor=executor, environ={"SLURM_JOB_ID": "1"})
    verification = json.loads((tmp_path / "launch/verification.json").read_text())
    assert executor.stages == ["01_preflight"]
    assert (code, verification["exit_code"], verification["complete"]) == (1, 1, False)


def test_missing_train_completion_prevents_eval_and_is_truthful(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train, evaluation = _plans(tmp_path)
    executor = RecordingExecutor(materialize_completion=False)
    checkpoint_checks = []
    monkeypatch.setattr(pipeline.hev1.eval_stage, "require_complete_checkpoint", checkpoint_checks.append)
    code = pipeline.run_pipeline(train_plan=train, eval_plan=evaluation, facility="cannon", launch_dir=tmp_path / "launch", admission_id="a", executor=executor, environ={"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "MIG-owned-by-slurm"})
    verification = json.loads((tmp_path / "launch/verification.json").read_text())
    assert executor.stages == ["01_preflight", "02_train"]
    assert checkpoint_checks == []
    assert not (tmp_path / "launch/03_eval/dispatch_specs.jsonl").exists()
    assert "training completion barrier failed" in verification["error"]
    assert (code, verification["exit_code"], verification["complete"]) == (1, 1, False)
