"""Run one planned He-v1 fixed-model evaluation chain inside its allocation.

Each row is one independent chain over one predeclared retained checkpoint of
one training seed. The checkpoint it restores is passed explicitly and must
exist as a COMPLETE checkpoint directory before the run starts: an evaluation
that silently restored a different (or partial) checkpoint would produce a
number attributed to the wrong model.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

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

canary = sibling(__file__, 'canary')
driver = sibling(__file__, 'driver')
layout = sibling(__file__, 'layout')

plan_stage = sibling(__file__, 'plan')

#: Marker written by the checkpoint writer when a directory is fully written.
#: Spelled here rather than imported because ``experiments/`` may not import
#: ``tpen``; the drift is guarded by a test.
COMPLETE_MARKER = "COMPLETE"

#: Model weights file inside a checkpoint directory. The A6-C trajectory
#: summary content-hashes THIS FILE, while ``load.path`` restores from the
#: DIRECTORY that contains it. The asymmetry is real and has already cost this
#: lane one job: passing the directory where the file is wanted raises
#: ``IsADirectoryError`` inside the summary, after the chain has been sampled.
CHECKPOINT_MODEL_FILE = "model.pt"

#: Evaluator identity recorded in the trajectory join. Constant for this study.
EVALUATOR_ID = "tpen_he_v1_eval"

#: Source receipt written outside the configured run directory before restore.
CHECKPOINT_BINDING_RECEIPT = "checkpoint_binding.json"


def require_checkpoint_model_file(checkpoint_dir: str | Path) -> Path:
    """Return the ``model.pt`` inside a complete checkpoint directory.

    Raises
    ------
    driver.DriverError
        If the weights file is absent. Checked here, before the allocation
        spends anything, because the summary that needs it runs LAST: a missing
        or mis-typed checkpoint path cannot fail early on its own and would
        otherwise surface only after the whole chain has been sampled.
    """

    model_file = Path(checkpoint_dir) / CHECKPOINT_MODEL_FILE
    if not model_file.is_file():
        raise driver.DriverError(
            f"checkpoint {checkpoint_dir} has no {CHECKPOINT_MODEL_FILE}; the trajectory "
            "statistics identity content-hashes that file, and restoring from the "
            "directory does not supply it"
        )
    return model_file


def config_identity_hash(
    config_path: str | Path,
    overrides: Sequence[str],
    *,
    identity_values: Mapping[str, Any] | None = None,
) -> str:
    """Return the deterministic config hash recorded in the join identity.

    Computed over the parsed config document plus the row's overrides, both
    before injection. Hashing the resolved document after the identity is
    injected would be self-referential -- the hash would be an input to the
    thing it describes -- so the inputs are deliberately the two things that
    fully determine the run and do not depend on the hash.
    """

    try:
        config_document = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        config_payload = json.dumps(
            config_document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OSError, UnicodeDecodeError, yaml.YAMLError, TypeError, ValueError) as exc:
        raise ValueError(
            f"config {config_path} cannot be represented by the canonical JSON identity"
        ) from exc
    return _config_identity_digest(config_payload, overrides, identity_values)


def legacy_config_identity_hash(
    config_path: str | Path,
    overrides: Sequence[str],
    *,
    identity_values: Mapping[str, Any] | None = None,
    source_git_sha: str | None = None,
) -> str:
    """Return the pre-canonical byte-based config identity.

    This is retained solely so collection can re-join receipts written before
    config canonicalisation. New evaluation rows must use
    :func:`config_identity_hash`. When ``source_git_sha`` is supplied, the
    bytes are read from that historical evaluation revision; this is required
    when the current checkout has since reordered the YAML document.
    """

    path = Path(config_path)
    config_payload = (
        path.read_bytes()
        if source_git_sha is None
        else _config_bytes_at_revision(path, source_git_sha)
    )
    return _config_identity_digest(
        config_payload, overrides, identity_values
    )


@lru_cache(maxsize=32)
def _config_bytes_at_revision(config_path: Path, source_git_sha: str) -> bytes:
    """Read and cache one historical config payload for a collection attempt."""

    try:
        repo_root = Path(
            subprocess.run(
                ["git", "-C", str(config_path.parent), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        relative_path = config_path.resolve().relative_to(repo_root.resolve())
        return subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "show",
                f"{source_git_sha}:{relative_path.as_posix()}",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise ValueError(
            f"config {config_path} is unavailable at evaluation revision {source_git_sha}"
        ) from exc


def _config_identity_digest(
    config_payload: bytes,
    overrides: Sequence[str],
    identity_values: Mapping[str, Any] | None,
) -> str:
    """Hash one config payload with the shared identity suffix convention."""

    digest = hashlib.sha256()
    digest.update(config_payload)
    digest.update(b"\0overrides\0")
    digest.update(json.dumps(sorted(str(item) for item in overrides)).encode("utf-8"))
    if identity_values is not None:
        digest.update(b"\0identity-values\0")
        digest.update(
            json.dumps(
                identity_values,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the SHA256 digest of one checkpoint payload file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_replay_semantics_overrides(
    checkpoint_dir: Path, *, binding: Mapping[str, Any] | None = None
) -> list[str]:
    """Return fail-closed replay-provenance overrides for one checkpoint.

    The evaluation config intentionally leaves source and checkpoint fields
    missing.  This driver reads them from the complete checkpoint immediately
    before launching the row, so a result cannot claim a different source or
    model payload than the directory it restores.
    """

    manifest_path = checkpoint_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    provenance = manifest.get("provenance")
    if not isinstance(files, Mapping) or files.get("model") != CHECKPOINT_MODEL_FILE:
        raise driver.DriverError(
            "checkpoint manifest must identify model.pt as its model payload for replay"
        )
    if not isinstance(provenance, Mapping):
        raise driver.DriverError("checkpoint manifest lacks replay provenance")
    source_git_sha = provenance.get("git_sha")
    source_tpen_version = provenance.get("tpen_version")
    if not isinstance(source_git_sha, str) or not isinstance(source_tpen_version, str):
        raise driver.DriverError(
            "checkpoint manifest lacks source git SHA or TPEN version for replay"
        )
    model_file = require_checkpoint_model_file(checkpoint_dir)
    if binding is not None:
        expected = {
            "source_git_sha": binding.get("training_source_sha"),
            "source_tpen_version": binding.get("source_tpen_version"),
            "checkpoint_schema_version": binding.get("checkpoint_schema_version"),
            "checkpoint_kind": binding.get("checkpoint_kind"),
            "checkpoint_model_sha256": binding.get("model_sha256"),
        }
        actual = {
            "source_git_sha": source_git_sha,
            "source_tpen_version": source_tpen_version,
            "checkpoint_schema_version": manifest.get("schema_version"),
            "checkpoint_kind": manifest.get("kind"),
            "checkpoint_model_sha256": _file_sha256(model_file),
        }
        if actual != expected:
            raise driver.DriverError(
                f"checkpoint replay identity changed after allocation: expected={expected}, "
                f"actual={actual}"
            )
    return [
        f"load.replay_semantics.source_git_sha={source_git_sha}",
        f"load.replay_semantics.source_tpen_version={source_tpen_version}",
        f"load.replay_semantics.checkpoint_schema_version={int(manifest['schema_version'])}",
        f"load.replay_semantics.checkpoint_kind={manifest['kind']}",
        f"load.replay_semantics.checkpoint_model_sha256={_file_sha256(model_file)}",
    ]


def configure_canary_evaluation(cfg: Any, row: Mapping[str, Any]) -> Any:
    """Select exactly the task graph declared by one canary row."""

    from omegaconf import OmegaConf  # noqa: PLC0415 - driver-only dependency

    task_names = list(row.get("task_names", []))
    if not task_names or any(not str(name).strip() for name in task_names):
        raise driver.DriverError("canary row must declare one or more task names")
    if len(set(task_names)) != len(task_names):
        raise driver.DriverError("canary row task_names must be unique")
    selected = []
    for name in task_names:
        try:
            task = cfg.evaluation_tasks[name]
        except (KeyError, TypeError):
            raise driver.DriverError(f"canary row selected unknown evaluation task {name!r}") from None
        selected.append(task)
    cfg.evaluation_sampler.seed = int(row["seed"])
    cfg.evaluation_sampler.n_walkers = int(row["n_walkers"])
    cfg.evaluation_sampler.burn_in = int(row["burn_in"])
    cfg.evaluation_sampler.n_steps = int(row["stride"])
    for task in selected:
        if str(task.name) != "mcmc_energy":
            if str(task.name) == "factor_response_re_equilibrated" and row.get("factor_arm"):
                # Re-equilibrated factor rows are trajectory evaluations.  The
                # nested generator is deliberately configured here, rather
                # than relying on the sampler's one-batch snapshot default.
                arm = OmegaConf.create(dict(row["factor_arm"]))
                task.generator.arm = arm
                trajectory_generator = task.generator.generator
                trajectory_generator.n_draws = int(row["n_draws"])
                trajectory_generator.discard_draws = int(row["discard_draws"])
                trajectory_generator.chunk_size = int(row["chunk_size"])
                # The generator cap limits its final snapshot, while the
                # writer cap protects the complete draw-by-walker artifact;
                # both must be derived from the row rather than a stale YAML
                # literal that truncates the declared trajectory.
                trajectory_generator.max_samples = int(row["record_capacity"])
                writers = [
                    summary for summary in task.summaries
                    if str(summary.get("_target_"))
                    == "tpen.evaluation.summaries.SampledRecordWriter"
                ]
                if len(writers) != 1:
                    raise driver.DriverError(
                        "re-equilibrated factor graph lost its single SampledRecordWriter"
                    )
                writers[0].max_samples = int(row["record_capacity"])
                for calculator in task.calculators:
                    if str(calculator.get("_target_")) == "tpen.evaluation.calculators.FactorArmCalculator":
                        calculator.arm = arm
            continue
        task.generator.n_draws = int(row["n_draws"])
        task.generator.discard_draws = int(row["discard_draws"])
        task.generator.chunk_size = int(row["chunk_size"])
        local_energy = [
            calculator for calculator in task.calculators
            if str(calculator.get("_target_"))
            == "tpen.evaluation.calculators.LocalEnergyCalculator"
        ]
        writers = [
            summary for summary in task.summaries
            if str(summary.get("_target_"))
            == "tpen.evaluation.summaries.SampledRecordWriter"
        ]
        if len(local_energy) != 1 or len(writers) != 1:
            raise driver.DriverError(
                "generic eval graph lost its single LocalEnergyCalculator or SampledRecordWriter"
            )
        local_energy[0].chunk_size = int(row["chunk_size"])
        writers[0].max_samples = int(row["record_capacity"])
    # Clone only after every reduced-scale value is applied. OmegaConf.create
    # owns a distinct evaluator task; mutating the declaration afterwards does
    # not update that clone.
    cfg.evaluator.tasks = OmegaConf.create(selected)
    cfg.callbacks.append(OmegaConf.create({"_target_": "tpen.callback.ArtifactIndex"}))
    cfg.callbacks.append(OmegaConf.create({"_target_": "tpen.callback.FailureLog"}))
    return cfg


def trajectory_identity_overrides(
    row: Mapping[str, object],
    *,
    plan_attempt_id: str,
    checkpoint_dir: Path,
    config_sha256: str,
) -> list[str]:
    """Return the six A6-C identity overrides for one evaluation row.

    Every field the producer requires is supplied explicitly. The config
    declares them ``???`` so a forgotten one fails at resolution rather than
    producing a sidecar that cannot be joined.
    """

    return [
        f"trajectory_identity.stage={row['stage']}",
        f"trajectory_identity.run_id={row['row_id']}",
        f"trajectory_identity.attempt_id={plan_attempt_id}",
        f"trajectory_identity.evaluator_id={EVALUATOR_ID}",
        f"trajectory_identity.checkpoint_file={require_checkpoint_model_file(checkpoint_dir)}",
        f"trajectory_identity.config_sha256={config_sha256}",
    ]


def require_complete_checkpoint(path: str | Path) -> Path:
    """Return ``path`` once it is a complete checkpoint directory.

    Raises
    ------
    driver.DriverError
        If the directory is missing, or is missing its manifest or completion
        marker. A partially written checkpoint restores a model nobody planned.
    """

    checkpoint_dir = Path(path)
    if not checkpoint_dir.is_dir():
        raise driver.DriverError(f"checkpoint directory does not exist: {checkpoint_dir}")
    missing = [
        name
        for name in ("manifest.json", COMPLETE_MARKER)
        if not (checkpoint_dir / name).is_file()
    ]
    if missing:
        raise driver.DriverError(
            f"checkpoint {checkpoint_dir} is incomplete; missing: {missing}"
        )
    return checkpoint_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse evaluation-driver arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    driver.add_common_arguments(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint-dir", help="Complete checkpoint directory this chain restores."
    )
    source.add_argument(
        "--checkpoint-source-map",
        help="External immutable source map required by a canary plan.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one evaluation chain."""

    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    manifest = plan_stage.read_manifest(results_root, args.plan_attempt_id)
    row = plan_stage.row_by_id(manifest, args.row_id)
    if str(row["kind"]) != "eval":
        raise driver.DriverError(f"row {args.row_id!r} is not an evaluation row")

    binding: Mapping[str, Any] | None = None
    source_receipt: Mapping[str, Any] | None = None
    if row.get("canary_protocol") == canary.CANARY_SCHEMA:
        if args.checkpoint_source_map is None:
            raise driver.DriverError("canary row requires --checkpoint-source-map")
        sources = canary.reconcile_manifest_sources(
            manifest, args.checkpoint_source_map
        )
        source = canary.source_for_row(row, sources)
        binding = dict(row["checkpoint_source"])
        source_receipt = source.receipt()
        checkpoint_dir = require_complete_checkpoint(source.checkpoint_dir)
    else:
        if args.checkpoint_dir is None:
            raise driver.DriverError("generic evaluation row requires --checkpoint-dir")
        checkpoint_dir = require_complete_checkpoint(args.checkpoint_dir)

    config_sha256 = config_identity_hash(
        driver.STUDY_DIR.parents[2] / str(row["config"]),
        [str(item) for item in row["overrides"]],
        identity_values={
            "canary_protocol": row.get("canary_protocol"),
            "checkpoint_source": row.get("checkpoint_source"),
            "task_names": row.get("task_names"),
            "n_walkers": row.get("n_walkers"),
            "n_draws": row.get("n_draws"),
            "burn_in": row.get("burn_in"),
            "discard_draws": row.get("discard_draws"),
            "stride": row.get("stride"),
            "chunk_size": row.get("chunk_size"),
            "record_capacity": row.get("record_capacity"),
        }
        if binding is not None
        else None,
    )
    if source_receipt is not None:
        result_dir = layout.row_dir(
            results_root, str(row["stage"]), str(row["row_id"]), args.plan_attempt_id
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        layout.write_json(result_dir / CHECKPOINT_BINDING_RECEIPT, dict(source_receipt))
    return driver.run_row(
        row,
        results_root=results_root,
        plan_attempt_id=args.plan_attempt_id,
        launch_attempt_id=args.launch_attempt_id,
        # `load.path` is the DIRECTORY; the identity's `checkpoint_file` is the
        # model.pt inside it. Both are supplied because they are different
        # paths for different consumers, not two spellings of one.
        extra_overrides=[
            f"load.path={checkpoint_dir}",
            *trajectory_identity_overrides(
                row,
                plan_attempt_id=args.plan_attempt_id,
                checkpoint_dir=checkpoint_dir,
                config_sha256=config_sha256,
            ),
            *checkpoint_replay_semantics_overrides(checkpoint_dir, binding=binding),
        ],
        config_transform=(
            (lambda cfg: configure_canary_evaluation(cfg, row))
            if binding is not None
            else None
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
