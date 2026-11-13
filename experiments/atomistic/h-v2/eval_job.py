"""Study-side H-v2 evaluation fan-out (TO 8a5e926f, operator EVAL option b).

Replaces ``attached.EVAL_ROUTE_UNSUPPORTED_CODE``'s unconditional refusal
with the real pieces item E owns: checkpoint selection and hash
verification against a TRAIN accepted receipt's own retained-checkpoint
inventory (never a directory glob), the frozen science/pilot/part-(g) seed
grids, the pilot SEL step (the instrument's ``W`` estimator, TO 5cfcca3e),
and the study-side eval receipt writer/reader.

Orchestrator decision 4: this reader stays OUTSIDE ``collect.py`` --
collector adoption of eval receipts is a follow-up after this item and
372ef393 both merge. The acceptance contract: EVAL is "a study-side job,
NOT an executor dispatch." Unlike ``attached.py``'s TRAIN/single-generation
EVAL route (which dispatches through ``tpen_executor`` and is blocked for
EVAL on an unrelated core/executor checkpoint-evidence contract -- TO item
9d3aa0a3, ``attached.EVAL_ROUTE_UNSUPPORTED_CODE``), this module runs each
fan-out generation directly: a ``SCIENCE_PYTHON`` subprocess calls
``tpen.run.run_from_config`` explicitly on a loaded config (never
``tpen.run.main``, which is not an experiment-facing surface per
``experiments/README.md``, and never ``python -m tpen.run``, which has no
``__main__`` guard and silently does nothing), then this module verifies
the written evidence itself (``status.json``, the source commit, every
declared artifact's sha256) -- reaching core only through
``run_from_config``/the ``tpen`` command, per ``experiments/README.md``;
this module itself never imports ``tpen`` directly.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

HERE = Path(__file__).parent


def _h_v2_module(name: str) -> Any:
    """Load an h-v2 sibling under the shared ``h_v2_<name>`` key (see ``attached.py``)."""

    key = f"h_v2_{name}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _h_v2_module("protocol")

#: Schema of the eval receipt this module writes and reads. Distinct from
#: ``attached.H_V2_ACCEPTED_RECEIPT_SCHEMA`` (one accepted TRAIN/EVAL
#: generation) -- this is the FAN-OUT'S OWN consolidated receipt, aggregating
#: every generation's accepted receipt plus the identity fields the
#: acceptance contract requires (eval commit, TRAIN commit, TRAIN receipt
#: sha256, resolved config + sha256, every checkpoint and output sha256).
EVAL_RECEIPT_SCHEMA = "tpen-experiments/h-v2-eval-receipt-v1"

#: Part (g)'s single fresh-evaluation generation (pilot-plan-20261110 S7):
#: the g TRAIN payload's lone retained non-zero checkpoint (1,000 updates),
#: evaluated with one stream.
GPU_ROUTE_G_EVAL_UPDATE = 1000
GPU_ROUTE_G_EVAL_STREAM = 0


class EvalJobRefusal(ValueError):
    """A named refusal while building or consuming the H-v2 eval receipt/fan-out."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def derive_packed_validation_tolerance(
    single_stream_repeats: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Derive a comparison tolerance from same-seed single-stream repeats.

    The calibration population is deliberately restricted to single-stream
    observations.  Consecutive same-seed repeat differences are the measured
    noise population; packed-minus-single rows are not accepted as a
    calibration input because packed instability must not enlarge its own
    acceptance threshold.  The returned ``variance_reference`` is carried
    into :func:`compare_packed_single_stream`, so calibration and comparison
    use one denominator.
    """

    if len(single_stream_repeats) < 3:
        raise ValueError("at least three single-stream repeats are required")
    energies: list[float] = []
    variances: list[float] = []
    identity_fields = ("seed", "checkpoint", "checkpoint_sha256", "lineage")
    first_identity = {field: single_stream_repeats[0].get(field) for field in identity_fields}
    for row in single_stream_repeats:
        if any(row.get(field) is None for field in identity_fields):
            raise ValueError("single-stream calibration identity is incomplete")
        if any(row.get(field) != first_identity[field] for field in identity_fields):
            raise ValueError("single-stream calibration repeats must share identity")
        try:
            energy = float(row["energy"])
            variance = float(row["variance"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("single-stream calibration requires energy and variance") from error
        if not math.isfinite(energy) or not math.isfinite(variance):
            raise ValueError("single-stream calibration values must be finite")
        if variance < 0:
            raise ValueError("single-stream calibration variance must be non-negative")
        energies.append(energy)
        variances.append(variance)

    energy_differences = [right - left for left, right in zip(energies, energies[1:])]
    variance_differences = [right - left for left, right in zip(variances, variances[1:])]
    variance_reference = statistics.mean(variances)
    if variance_reference <= 0:
        raise ValueError("single-stream repeat variance reference must be positive")
    return {
        "energy_abs": 3.0 * statistics.stdev(energy_differences),
        "variance_relative": 3.0 * statistics.stdev(variance_differences) / variance_reference,
        "variance_reference": variance_reference,
    }


def compare_packed_single_stream(
    packed: Mapping[str, float], single_stream: Mapping[str, float], *, tolerance: Mapping[str, float]
) -> None:
    """Validate one packed result against the same-seed single-stream result.

    Raises
    ------
    EvalJobRefusal
        Code ``packed-validation-mismatch`` when either metric exceeds the
        tolerance derived by :func:`derive_packed_validation_tolerance`.
    """

    identity_fields = ("seed", "checkpoint", "checkpoint_sha256", "lineage")
    if any(field not in packed or packed[field] is None for field in identity_fields):
        raise EvalJobRefusal("packed-validation-identity", "packed identity is absent or null")
    if any(field not in single_stream or single_stream[field] is None for field in identity_fields):
        raise EvalJobRefusal("packed-validation-identity", "single-stream identity is absent or null")
    if any(packed[field] != single_stream[field] for field in identity_fields):
        raise EvalJobRefusal("packed-validation-identity", "packed and single-stream records are not same-seed/checkpoint")
    metric_values = (packed.get("energy"), single_stream.get("energy"), packed.get("variance"), single_stream.get("variance"))
    tolerance_values = (
        tolerance.get("energy_abs"),
        tolerance.get("variance_relative"),
        tolerance.get("variance_reference"),
    )
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (*metric_values, *tolerance_values)):
        raise EvalJobRefusal("packed-validation-nonfinite", "metrics and tolerances must be finite")
    if any(float(value) < 0 for value in tolerance_values):
        raise EvalJobRefusal("packed-validation-tolerance", "tolerances must be non-negative")
    if float(packed["variance"]) < 0 or float(single_stream["variance"]) < 0:
        raise EvalJobRefusal("packed-validation-variance", "variance must be non-negative")
    if float(tolerance["variance_reference"]) <= 0:
        raise EvalJobRefusal("packed-validation-tolerance", "variance reference must be positive")
    energy_delta = abs(float(packed["energy"]) - float(single_stream["energy"]))
    variance_delta = abs(float(packed["variance"]) - float(single_stream["variance"])) / float(
        tolerance["variance_reference"]
    )
    if energy_delta > float(tolerance["energy_abs"]) or variance_delta > float(tolerance["variance_relative"]):
        raise EvalJobRefusal(
            "packed-validation-mismatch",
            f"energy delta={energy_delta:g}, variance relative delta={variance_delta:g}, tolerance={dict(tolerance)!r}",
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_receipt(path: str | Path) -> tuple[dict[str, Any], str]:
    """Return ``(receipt, sha256)`` for the TRAIN accepted receipt at *path*.

    Raises
    ------
    EvalJobRefusal
        Code ``train-receipt-missing``, if *path* does not exist.
    """

    receipt_path = Path(path)
    if not receipt_path.is_file():
        raise EvalJobRefusal("train-receipt-missing", f"no TRAIN accepted receipt at {receipt_path}")
    return json.loads(receipt_path.read_text(encoding="utf-8")), _sha256_file(receipt_path)


def verify_train_receipt_identity(
    receipt: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_lineage: str,
    expected_payload: str,
) -> None:
    """Refuse a TRAIN receipt that does not match this EVAL job's own expectations.

    Acceptance contract: "an explicit expected TRAIN commit; refuses on any
    mismatch of commit, lineage, payload, seed namespace or config
    identity." Every segment's recorded source commit must equal
    *expected_commit*; the receipt's own ``payload`` and its
    ``science_config_identity``/pilot-identity ``lineage`` field must equal
    *expected_payload*/*expected_lineage*.

    F3 (review r1) / D4 (review r2 delta): a correct commit/lineage/payload
    is not sufficient -- the receipt's seed namespace (checked against
    :data:`protocol.PILOT_PAYLOADS`' own entry, a fixed code-level mapping,
    never a frozen-input file) and the presence of every required identity
    field are checked too.

    D4 (review r2 delta): the overlay/contract-inputs/materialized-config
    identity hashes are NOT re-derived by hashing THIS checkout's OWN
    current frozen-input files -- a real EVAL-only submission runs against a
    TRAIN receipt an EARLIER commit wrote (the acceptance contract's own
    words), so "this checkout" and "the commit that wrote the receipt" are
    routinely different, and re-hashing the current files would re-bind a
    historical receipt to a checkout it never saw. These fields are instead
    required present (the TRAIN side, not this job, is responsible for
    writing them correctly) and authenticated by their OWN DISTINCT
    ``resolved_config`` ARTIFACT'S ACTUAL BYTES on disk being re-hashed and
    compared to its own recorded sha256 -- a tampered or stale artifact file
    is refused even though its string-recorded hash was never touched.
    ``materialized_config_sha256`` (the INPUT yaml TRAIN submitted) and each
    segment's ``resolved_config_sha256`` (core's RESOLVED runtime yaml) are
    NOT compared to each other: they are different artifacts by design
    (interpolation/placeholder resolution changes the bytes), and a prior
    equality check here falsely refused every real TRAIN receipt.

    Raises
    ------
    EvalJobRefusal
        Code ``train-receipt-empty``, ``wrong-train-payload``,
        ``wrong-train-commit``, ``wrong-train-lineage``,
        ``wrong-train-seed-namespace``, ``wrong-train-identity-missing``, or
        ``wrong-train-resolved-config-bytes``.
    """

    if receipt.get("payload") != expected_payload:
        raise EvalJobRefusal(
            "wrong-train-payload",
            f"TRAIN receipt payload {receipt.get('payload')!r} does not match expected {expected_payload!r}",
        )
    segments = receipt.get("segments") or []
    if not segments:
        raise EvalJobRefusal("train-receipt-empty", "TRAIN receipt carries no segments")
    for segment in segments:
        if segment.get("source_commit") != expected_commit:
            raise EvalJobRefusal(
                "wrong-train-commit",
                f"TRAIN receipt segment ran commit {segment.get('source_commit')!r}, "
                f"expected {expected_commit!r}",
            )
    identity = receipt.get("science_config_identity") or {}
    if identity.get("lineage") != expected_lineage:
        raise EvalJobRefusal(
            "wrong-train-lineage",
            f"TRAIN receipt lineage {identity.get('lineage')!r} does not match expected {expected_lineage!r}",
        )

    pilot_spec = protocol.PILOT_PAYLOADS.get(expected_payload)
    if pilot_spec is not None:
        if identity.get("seed_namespace") != pilot_spec["seed_namespace"]:
            raise EvalJobRefusal(
                "wrong-train-seed-namespace",
                f"TRAIN receipt seed_namespace {identity.get('seed_namespace')!r} does not match "
                f"expected {pilot_spec['seed_namespace']!r}",
            )
        required_identity_fields = ("payload_overlay_sha256", "contract_inputs_sha256", "materialized_config_sha256")
    else:
        required_identity_fields = ("overlay_sha256", "contract_inputs_sha256", "materialized_config_sha256")
    for field in required_identity_fields:
        if not identity.get(field):
            raise EvalJobRefusal(
                "wrong-train-identity-missing", f"TRAIN receipt science_config_identity is missing {field!r}"
            )

    for segment in segments:
        artifact = (segment.get("artifacts") or {}).get("resolved_config")
        if not artifact or not artifact.get("path"):
            raise EvalJobRefusal(
                "wrong-train-resolved-config-bytes", "TRAIN receipt segment has no resolved_config artifact"
            )
        recorded_sha256 = artifact.get("sha256")
        actual_sha256 = _sha256_file(Path(artifact["path"]))
        if actual_sha256 != recorded_sha256:
            raise EvalJobRefusal(
                "wrong-train-resolved-config-bytes",
                f"{artifact['path']} sha256 {actual_sha256} does not match the TRAIN receipt's "
                f"recorded {recorded_sha256!r}",
            )


def select_checkpoint(receipt: Mapping[str, Any], *, update: int) -> dict[str, Any]:
    """Return the TRAIN receipt's inventory entry whose ``update`` equals *update*.

    TO 63690220 orchestrator decision 1: E selects checkpoints ONLY through
    this inventory (keyed by the manifest's ``completed_updates``, never the
    directory's own resume cursor) -- never a directory glob.

    Raises
    ------
    EvalJobRefusal
        Code ``checkpoint-not-in-inventory``, if no entry matches.
    """

    inventory = receipt.get("checkpoint_inventory") or []
    for entry in inventory:
        if entry.get("update") == update:
            return entry
    available = sorted(entry.get("update") for entry in inventory)
    raise EvalJobRefusal(
        "checkpoint-not-in-inventory", f"no checkpoint inventory entry for update={update} (available: {available!r})"
    )


#: F2 (review r1): every file core's model-only restore opens for ONE
#: checkpoint -- the model weights, the manifest recording
#: ``completed_updates``, and the ``COMPLETE`` marker written only once that
#: checkpoint finished saving. An inventory entry missing any of these is
#: refused outright, before this module ever opens the checkpoint directory:
#: never load from a directory whose full read closure is not inventoried.
REQUIRED_CHECKPOINT_CLOSURE_FILES: tuple[str, ...] = ("model.pt", "manifest.json", "COMPLETE")


def verify_checkpoint_files(entry: Mapping[str, Any]) -> Path:
    """Verify every inventoried file's sha256, then return the checkpoint directory.

    Reads ONLY the files the inventory entry names -- never a directory
    listing or glob. A file on disk but absent from the inventory is simply
    never read.

    Raises
    ------
    EvalJobRefusal
        Code ``checkpoint-closure-incomplete``, ``checkpoint-file-missing``,
        or ``checkpoint-hash-mismatch``.
    """

    files = entry.get("files") or {}
    missing_from_inventory = sorted(set(REQUIRED_CHECKPOINT_CLOSURE_FILES) - set(files))
    if missing_from_inventory:
        raise EvalJobRefusal(
            "checkpoint-closure-incomplete",
            f"inventory entry for update={entry.get('update')} is missing required closure file(s) "
            f"{missing_from_inventory!r}",
        )
    checkpoint_dir = Path(entry["path"])
    for relative, expected_sha256 in files.items():
        file_path = checkpoint_dir / relative
        if not file_path.is_file():
            raise EvalJobRefusal("checkpoint-file-missing", f"inventoried file {file_path} is missing")
        actual = _sha256_file(file_path)
        if actual != expected_sha256:
            raise EvalJobRefusal(
                "checkpoint-hash-mismatch",
                f"{file_path} sha256 {actual} does not match inventoried {expected_sha256}",
            )
    return checkpoint_dir


def materialize_verified_checkpoint(entry: Mapping[str, Any], destination: str | Path) -> Path:
    """Copy *entry*'s inventoried files into *destination*, verifying each hash while copying.

    F2 (review r1): :func:`verify_checkpoint_files` leaves a check/use
    window -- core opens the ORIGINAL checkpoint directory after this
    function returns, so a swap between verification and load would pass
    hashes it never re-checks. This function instead reads, verifies, and
    writes each inventoried file's bytes into a private destination this job
    alone controls; the caller points core at *destination*, never the
    original directory, eliminating that window. Reads ONLY the files the
    inventory names -- never a directory listing or glob, same as
    :func:`verify_checkpoint_files`.

    D3 (review r2 delta): *destination* is created FRESH, PRIVATE and
    EXCLUSIVE (never reused or silently merged into an existing directory,
    so a pre-planted file or symlink there is refused rather than followed).
    Every inventoried relative path is checked for containment -- an escape
    (``..``/absolute) is refused before any read. ``manifest.json``'s own
    parsed ``completed_updates`` must equal *entry*'s ``update`` label --
    an inventory entry whose declared update disagrees with what the
    manifest it points at actually records is refused.

    Raises
    ------
    EvalJobRefusal
        Code ``checkpoint-closure-incomplete``, ``checkpoint-file-missing``,
        ``checkpoint-hash-mismatch``, ``checkpoint-path-escape``,
        ``checkpoint-snapshot-escape``, or ``checkpoint-wrong-update``.
    """

    files = entry.get("files") or {}
    missing_from_inventory = sorted(set(REQUIRED_CHECKPOINT_CLOSURE_FILES) - set(files))
    if missing_from_inventory:
        raise EvalJobRefusal(
            "checkpoint-closure-incomplete",
            f"inventory entry for update={entry.get('update')} is missing required closure file(s) "
            f"{missing_from_inventory!r}",
        )
    source_dir = Path(entry["path"])
    destination = Path(destination)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise EvalJobRefusal(
            "checkpoint-snapshot-escape", f"snapshot destination {destination} already exists"
        ) from None
    for relative, expected_sha256 in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise EvalJobRefusal(
                "checkpoint-path-escape", f"inventoried path {relative!r} is not contained in the checkpoint directory"
            )
        source_path = source_dir / relative_path
        if not source_path.is_file():
            raise EvalJobRefusal("checkpoint-file-missing", f"inventoried file {source_path} is missing")
        data = source_path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256:
            raise EvalJobRefusal(
                "checkpoint-hash-mismatch",
                f"{source_path} sha256 {actual} does not match inventoried {expected_sha256}",
            )
        if relative == "manifest.json":
            manifest = json.loads(data.decode("utf-8"))
            if manifest.get("completed_updates") != entry.get("update"):
                raise EvalJobRefusal(
                    "checkpoint-wrong-update",
                    f"{source_path} completed_updates={manifest.get('completed_updates')!r} does not match "
                    f"the inventory entry's update={entry.get('update')!r}",
                )
        target_path = destination / relative_path
        if target_path.exists() or target_path.is_symlink():
            raise EvalJobRefusal(
                "checkpoint-snapshot-escape", f"{target_path} unexpectedly already exists in a fresh snapshot"
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)
    return destination


def resolve_checkpoint(
    receipt: Mapping[str, Any], *, update: int, snapshot_dir: str | Path | None = None
) -> Path:
    """Select *update*'s inventory entry and verify its files in one call.

    Parameters
    ----------
    snapshot_dir : path, optional
        When given, the verified files are copied into this job-local
        directory (:func:`materialize_verified_checkpoint`) and that
        directory is returned, closing F2's check/use window. When omitted
        (the default), the original checkpoint directory is verified and
        returned unchanged, for a caller that cannot yet supply a private
        destination.
    """

    entry = select_checkpoint(receipt, update=update)
    if snapshot_dir is not None:
        return materialize_verified_checkpoint(entry, snapshot_dir)
    return verify_checkpoint_files(entry)


def science_fanout(*, lineage_index: int) -> tuple[dict[str, Any], ...]:
    """Return science's 27x4=108 fan-out generations (design target C2).

    Each entry: ``{"ordinal", "update", "stream", "seed"}``, ordinal 0..26
    indexing ``protocol.science_evaluated_checkpoint_updates()``, stream
    0..3, seed from ``protocol.science_eval_seed``.
    """

    updates = protocol.science_evaluated_checkpoint_updates()
    return tuple(
        {
            "ordinal": ordinal,
            "update": update,
            "stream": stream,
            "seed": protocol.science_eval_seed(lineage_index, ordinal, stream),
        }
        for ordinal, update in enumerate(updates)
        for stream in range(4)
    )


def pilot_fanout(
    stage: str, *, c1: int | None = None, payload: str = "capacity-pilot"
) -> tuple[dict[str, Any], ...]:
    """Return one pilot stage's fan-out generations (amendment r3/r4 E1).

    Parameters
    ----------
    stage : {"1", "2", "2-core"}
    c1 : int, optional
        Required for stage 2/2-core: the SEL basin ordinal
        (:func:`select_pilot_basin_ordinal`).
    payload : str, default="capacity-pilot"
        Stage-1 ordinal sweep source (amendment 20261112-r1:
        :func:`protocol.pilot_stage1_grid`). Stage 2/2-core has no
        generalized form -- it remains ``capacity-pilot``'s own fixed
        5-checkpoint window and is never reached for another payload (the
        caller refuses ``--pilot-stage 2``/``2-core`` for any other payload
        before this function is called).

    Raises
    ------
    EvalJobRefusal
        Code ``pilot-stage2-needs-c1`` or ``unknown-pilot-stage``.
    """

    if stage == "1":
        pairs = protocol.pilot_stage1_grid(payload)
    elif stage in ("2", "2-core"):
        if c1 is None:
            raise EvalJobRefusal("pilot-stage2-needs-c1", f"--pilot-stage {stage} requires c1 from the SEL step")
        pairs = protocol.pilot_stage2_grid(c1, core=(stage == "2-core"))
    else:
        raise EvalJobRefusal("unknown-pilot-stage", f"--pilot-stage must be one of 1/2/2-core, got {stage!r}")
    return tuple(
        {
            "ordinal": ordinal,
            "update": ordinal * 1000,
            "stream": stream,
            "seed": protocol.pilot_eval_seed(ordinal, stream),
        }
        for ordinal, stream in pairs
    )


def gpu_route_g_fanout() -> tuple[dict[str, Any], ...]:
    """Return part (g)'s single fresh-evaluation generation."""

    return (
        {
            "ordinal": GPU_ROUTE_G_EVAL_UPDATE // 1000,
            "update": GPU_ROUTE_G_EVAL_UPDATE,
            "stream": GPU_ROUTE_G_EVAL_STREAM,
            "seed": protocol.gpu_route_g_eval_seed(),
        },
    )


#: Run via ``science_python -c <this> <module>``: a bare import probe, never
#: a direct ``tpen`` import from this study module itself (the attached
#: subprocess boundary already used by ``attached.assert_candidate_instantiates``).
#: F8: catches ONLY ``ModuleNotFoundError`` for the target module (or one of
#: its parent packages) as genuine absence -- any OTHER exception (a broken
#: dependency import inside an otherwise-present module, a syntax error,
#: etc.) propagates uncaught, giving a nonzero exit the caller refuses on,
#: never a false "ABSENT".
_READOUT_GENERATOR_PROBE_SCRIPT = """
import importlib
import sys

name = sys.argv[1]
try:
    importlib.import_module(name)
except ModuleNotFoundError as error:
    if error.name == name or (error.name is not None and name.startswith(error.name + ".")):
        print("ABSENT")
        raise SystemExit(0)
    raise
print("PRESENT")
"""


def probe_readout_trajectory_generator(science_python: str | Path) -> bool:
    """Return whether the readout-trajectory generator module exists on *science_python*.

    E5 (orchestrator decision 2(a)): enabled for every eval payload; if
    absent on this job's own base, this probe is never skipped silently --
    the caller records the deviation (``readout_generator_enabled: false``)
    on the eval receipt instead.

    F8 (review r1): a probe is only ever PRESENT or explicit ABSENT. A
    nonzero exit, a malformed/empty stdout, or a broken dependency import
    (which also raises ``ImportError`` and would otherwise be
    indistinguishable from a genuinely missing target module) is refused
    outright -- never silently converted into "treat readout as disabled".

    Raises
    ------
    EvalJobRefusal
        Code ``readout-probe-failed``, if the probe exits nonzero or its
        stdout is neither exactly ``PRESENT`` nor exactly ``ABSENT``.
    """

    module_name = protocol.READOUT_TRAJECTORY_GENERATOR_TARGET.rsplit(".", 1)[0]
    result = subprocess.run(
        [str(science_python), "-c", _READOUT_GENERATOR_PROBE_SCRIPT, module_name],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout.strip()
    if result.returncode != 0 or output not in ("PRESENT", "ABSENT"):
        raise EvalJobRefusal(
            "readout-probe-failed",
            f"readout-trajectory-generator probe did not report a clean PRESENT/ABSENT "
            f"(rc={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr[-500:]!r})",
        )
    return output == "PRESENT"


#: Run via ``science_python -c <this> <config path>``: loads the config and
#: calls ``tpen.run.run_from_config`` directly -- never ``tpen.run.main``
#: (not an experiment-facing surface per ``experiments/README.md``; this is
#: the exact boundary the orchestrator's binding correction names) and never
#: ``python -m tpen.run`` (operator policy 2026-11-06: no ``__main__`` guard
#: will be added to ``tpen/run.py``, so that invocation silently exits 0 and
#: does nothing).
_RUN_FROM_CONFIG_SCRIPT = """
import sys
from omegaconf import OmegaConf
from tpen.run import run_from_config

config_path = sys.argv[1]
cfg = OmegaConf.load(config_path)
raise SystemExit(run_from_config(cfg, config_path=config_path, command=f"eval_job {config_path}"))
"""

#: Literal unresolved ``${run.dir}/`` prefix every evaluator task's
#: ``output_dir`` carries in these frozen configs (mirrors ``attached.py``'s
#: private ``_RUN_DIR_PREFIX``, duplicated here rather than imported since
#: that name is private to ``attached.py``).
_RUN_DIR_PREFIX = "${run.dir}/"


def run_generation_direct(
    *,
    science_python: str | Path,
    config: Mapping[str, Any],
    config_path: str | Path,
    expected_source_commit: str,
    cwd: str | Path | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Write *config*, run it directly through ``tpen.run.run_from_config``, and verify its output.

    EVAL is "a study-side job, NOT an executor dispatch" (acceptance
    contract): this calls the real ``tpen`` route directly, then verifies
    evidence itself -- ``status.json`` completed, the source commit, and
    every declared artifact's sha256 -- instead of going through
    ``tpen_executor``'s manifest/accepted-segment contract or core's
    checkpoint-gated ``inspect_run`` (TO item 9d3aa0a3's blocker, which
    applies only to the executor route this bypasses).

    Returns
    -------
    dict
        ``{"resolved_config_sha256", "trajectory_csv_sha256", "artifacts"}``
        -- the shape :func:`generation_record` expects.

    Raises
    ------
    EvalJobRefusal
        Code ``eval-run-bad-layout``, ``eval-run-failed``,
        ``eval-run-missing-artifact``, ``eval-run-wrong-commit``, or
        ``eval-run-uncommitted-tail``.
    """

    if (config.get("run") or {}).get("layout") != "flat":
        raise EvalJobRefusal("eval-run-bad-layout", "generation config must use run.layout=flat")
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")
    run_dir = Path(config["run"]["root"]) / config["run"]["run_id"]

    result = subprocess.run(
        [str(science_python), "-c", _RUN_FROM_CONFIG_SCRIPT, str(config_path)],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise EvalJobRefusal(
            "eval-run-failed",
            f"tpen run failed (rc={result.returncode}): stdout={result.stdout[-2000:]!r} "
            f"stderr={result.stderr[-2000:]!r}",
        )

    status_path = run_dir / "status.json"
    if not status_path.is_file():
        raise EvalJobRefusal("eval-run-missing-artifact", f"{status_path} is missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise EvalJobRefusal(
            "eval-run-failed", f"{status_path} reports status={status.get('status')!r}, expected 'completed'"
        )

    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise EvalJobRefusal("eval-run-missing-artifact", f"{metadata_path} is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_commit = metadata.get("git_commit")
    if actual_commit != expected_source_commit:
        raise EvalJobRefusal(
            "eval-run-wrong-commit",
            f"run metadata git_commit {actual_commit!r} does not match expected {expected_source_commit!r}",
        )

    required = {"resolved_config": "resolved_config.yaml", "metadata": "metadata.json"}
    for task in config["evaluator"]["tasks"]:
        relative = str(task["output_dir"])
        if relative.startswith(_RUN_DIR_PREFIX):
            relative = relative[len(_RUN_DIR_PREFIX):]
        name = task["name"]
        required[f"{name}_csv"] = f"{relative}/sampled_eval_table.csv"
        required[f"{name}_metadata"] = f"{relative}/sampled_eval_table.metadata.json"
        if task.get("generator", {}).get("_target_") == protocol.READOUT_TRAJECTORY_GENERATOR_TARGET:
            required[f"{name}_raw_readout"] = f"{relative}/raw_readout.csv"
            required[f"{name}_raw_readout_metadata"] = f"{relative}/raw_readout.json"

    artifacts: dict[str, dict[str, str]] = {}
    for name, relative in required.items():
        artifact_path = run_dir / relative
        if not artifact_path.is_file():
            raise EvalJobRefusal(
                "eval-run-missing-artifact", f"{run_dir} is missing declared artifact {name!r} ({artifact_path})"
            )
        artifacts[name] = {"path": str(artifact_path), "sha256": _sha256_file(artifact_path)}

    # F6 (review r1): the receipt previously carried only the explicitly
    # named required artifacts above, dropping status.json/config.yaml/
    # metrics.jsonl (and anything else core writes that this module does
    # not name one of its own tasks after) from the written-file hash map.
    # Every OTHER file actually present under run_dir is captured too, keyed
    # by its path relative to run_dir, so nothing written is un-receipted.
    already_declared = {Path(entry["path"]) for entry in artifacts.values()}
    for file_path in sorted(run_dir.rglob("*")):
        if not file_path.is_file() or file_path in already_declared:
            continue
        key = str(file_path.relative_to(run_dir))
        artifacts[key] = {"path": str(file_path), "sha256": _sha256_file(file_path)}

    trajectory_metadata = json.loads(Path(artifacts["trajectory_metadata"]["path"]).read_text(encoding="utf-8"))
    trajectory_csv_sha256 = trajectory_metadata.get("csv_sha256")
    if trajectory_csv_sha256 != artifacts["trajectory_csv"]["sha256"]:
        raise EvalJobRefusal("eval-run-uncommitted-tail", f"{run_dir} trajectory csv is stale (uncommitted tail)")

    return {
        "resolved_config_sha256": artifacts["resolved_config"]["sha256"],
        "trajectory_csv_sha256": trajectory_csv_sha256,
        "artifacts": artifacts,
    }


def generation_record(
    *,
    ordinal: int,
    update: int,
    stream: int,
    seed: int,
    task_id: str,
    run_result: Mapping[str, Any],
    checkpoint_files: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return one generation's eval-receipt entry from :func:`run_generation_direct`'s result.

    E3: called once per generation on completion, so a wall-killed job
    leaves every already-completed generation's output sha256s verifiable
    even though the job itself still exits non-zero (see ``run_job.py``).

    Parameters
    ----------
    checkpoint_files : mapping, optional
        F6 (review r1): the TRAIN inventory entry's own ``{relative path:
        sha256}`` map for the checkpoint this generation restored --
        recorded so a consumer can bind this generation to the EXACT input
        checkpoint bytes it ran against, not only the fact that
        verification passed at dispatch time.
    """

    return {
        "ordinal": ordinal,
        "update": update,
        "stream": stream,
        "seed": seed,
        "task_id": task_id,
        "resolved_config_sha256": run_result.get("resolved_config_sha256"),
        "trajectory_csv_sha256": run_result.get("trajectory_csv_sha256"),
        # Full {name: {path, sha256}} per declared output artifact -- every
        # file written's sha256 (the acceptance contract's own words), with
        # its path kept too so a later consumer (the SEL step) can read the
        # trajectory CSV without re-deriving its location.
        "artifacts": {name: dict(entry) for name, entry in (run_result.get("artifacts") or {}).items()},
        "checkpoint_files": dict(checkpoint_files) if checkpoint_files is not None else {},
    }


def trajectory_csv_path(generation: Mapping[str, Any]) -> Path:
    """Return one generation record's trajectory CSV path.

    Raises
    ------
    EvalJobRefusal
        Code ``generation-missing-trajectory``, if the record carries no
        ``trajectory_csv`` artifact.
    """

    artifacts = generation.get("artifacts") or {}
    entry = artifacts.get("trajectory_csv")
    if not entry or not entry.get("path"):
        raise EvalJobRefusal(
            "generation-missing-trajectory", f"generation {generation.get('task_id')!r} has no trajectory_csv artifact"
        )
    return Path(entry["path"])


def build_eval_receipt(
    *,
    eval_commit: str,
    train_commit: str,
    train_receipt_path: str | Path,
    train_receipt_sha256: str,
    payload: str,
    generations: Sequence[Mapping[str, Any]],
    readout_generator_enabled: bool,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the fan-out's own consolidated eval-receipt payload.

    Parameters
    ----------
    payload : {"science", "capacity-pilot", "gpu-route-g"}
    generations : sequence of mapping
        Each from :func:`generation_record`, in completion order.
    selection : mapping, optional
        The pilot SEL step's selection record (:func:`build_selection_record`),
        carried only for the pilot's stage-1 receipt.
    """

    return {
        "schema": EVAL_RECEIPT_SCHEMA,
        "eval_commit": eval_commit,
        "train_commit": train_commit,
        "train_receipt_path": str(train_receipt_path),
        "train_receipt_sha256": train_receipt_sha256,
        "payload": payload,
        "readout_generator_enabled": bool(readout_generator_enabled),
        "selection": dict(selection) if selection is not None else None,
        "generations": [dict(generation) for generation in generations],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_eval_receipt(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Build and atomically publish the H-v2 eval receipt."""

    payload = build_eval_receipt(**kwargs)
    _atomic_write_json(Path(path), payload)
    return payload


def read_eval_receipt(path: str | Path) -> dict[str, Any]:
    """Read and schema-check one H-v2 eval receipt -- the consumer-side reader.

    TO 8a5e926f orchestrator decision 4: this reader stays OUTSIDE
    ``collect.py``; collector adoption of eval receipts is a follow-up after
    this item and 372ef393 both merge.

    Raises
    ------
    EvalJobRefusal
        Code ``eval-receipt-missing`` or ``eval-receipt-bad-schema``.
    """

    payload, _ = _read_eval_receipt_bytes(path)
    return payload


def read_eval_receipt_with_hash(path: str | Path) -> tuple[dict[str, Any], str]:
    """Return ``(receipt, sha256)``, the sha256 computed from the EXACT bytes parsed.

    D5 (review r2 delta): a caller that reads a receipt and separately
    re-reads the file later to hash it (``path.read_bytes()`` after the
    fact) can bind the wrong bytes if the file is atomically replaced in
    between -- the published selection/final-selection record would then
    carry a hash for bytes nobody actually computed from. This function
    hashes the SAME buffer it parses, closing that window.

    Raises
    ------
    EvalJobRefusal
        Code ``eval-receipt-missing`` or ``eval-receipt-bad-schema``.
    """

    return _read_eval_receipt_bytes(path)


def _read_eval_receipt_bytes(path: str | Path) -> tuple[dict[str, Any], str]:
    receipt_path = Path(path)
    if not receipt_path.is_file():
        raise EvalJobRefusal("eval-receipt-missing", f"no eval receipt at {receipt_path}")
    raw = receipt_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema") != EVAL_RECEIPT_SCHEMA:
        raise EvalJobRefusal(
            "eval-receipt-bad-schema",
            f"{receipt_path} schema {payload.get('schema')!r} != {EVAL_RECEIPT_SCHEMA!r}",
        )
    return payload, hashlib.sha256(raw).hexdigest()


def read_stream_local_energies(trajectory_csv: str | Path) -> list[float]:
    """Return the ``local_energy`` column of one evaluation's trajectory CSV.

    Used only by the SEL step's ``W`` computation (descriptive); it does not
    replace ``collect.py``'s identity/integrity-reconciled record reading for
    any admission decision.
    """

    with Path(trajectory_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [float(row["local_energy"]) for row in reader]


def read_verified_stream_local_energies(generation: Mapping[str, Any]) -> list[float]:
    """Verify one generation's trajectory CSV bytes against its own recorded hash, then read it.

    D2/D5 (review r2 delta): shared by :func:`compute_pilot_selection` and
    the final-SEL reader (``run_job.run_pilot_final_sel``) so every
    consumer of a generation's trajectory CSV authenticates it the same
    way -- never trusting a receipt's recorded hash without re-checking the
    bytes actually on disk.

    Raises
    ------
    EvalJobRefusal
        Code ``sel-trajectory-hash-mismatch``, if the CSV's current bytes do
        not match the generation's own recorded ``trajectory_csv`` sha256.
    """

    path = trajectory_csv_path(generation)
    recorded_sha256 = generation.get("artifacts", {}).get("trajectory_csv", {}).get("sha256")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != recorded_sha256:
        raise EvalJobRefusal(
            "sel-trajectory-hash-mismatch",
            f"{path} sha256 {actual_sha256} does not match the receipt's recorded {recorded_sha256!r} "
            f"for ordinal {generation.get('ordinal')} stream {generation.get('stream')}",
        )
    return read_stream_local_energies(path)


def compute_pilot_selection(
    stage1_receipt: Mapping[str, Any], *, payload: str = "capacity-pilot"
) -> dict[str, Any]:
    """Compute the SEL step from a completed pilot stage-1 eval receipt.

    Reads every stage-1 generation's trajectory CSV, computes
    :func:`protocol.w_statistic` per ordinal, and returns
    :func:`build_selection_record`'s payload (``b1_eval_receipt_sha256`` is
    the caller's job to fill in from the receipt file it read this payload
    from, since this function only sees the in-memory receipt).

    F4 (review r1): every stream-0 trajectory CSV is re-hashed and checked
    against the receipt's OWN recorded ``trajectory_csv`` sha256 before its
    bytes are read for ``W`` (:func:`read_verified_stream_local_energies`)
    -- a CSV replaced on disk without updating the receipt it is bound to is
    refused, never silently included. Stage 1 must carry every one of
    *payload*'s stream-0 generations (:func:`protocol.pilot_stage1_ordinals`,
    ``0..20`` for ``capacity-pilot``); a receipt missing any of them refuses
    even if the missing ordinals fall outside every candidate triple
    ``select_pilot_basin_ordinal`` happens to need.

    D5 (review r2 delta): an ordinal appearing MORE THAN ONCE among stream-0
    generations is refused (never silently overwritten); each generation's
    recorded seed must equal :func:`protocol.pilot_eval_seed`'s own formula
    for its ordinal; every local-energy sample must be finite (not NaN/inf)
    -- none of this is implied by "all ordinals present with matching
    hashes".

    Parameters
    ----------
    payload : str, default="capacity-pilot"
        Amendment 20261112-r1: selects the expected stage-1 ordinal set
        (:func:`protocol.pilot_stage1_ordinals`) and SEL window
        (:func:`protocol.pilot_sel_window_hi`).

    Raises
    ------
    EvalJobRefusal
        Code ``sel-incomplete-stage1``, if stage 1 is missing any of
        *payload*'s stream-0 ordinals; code ``sel-duplicate-ordinal``,
        ``sel-wrong-seed``, or ``sel-nonfinite-energy``; code
        ``sel-trajectory-hash-mismatch``, if a trajectory CSV's current
        bytes do not match the receipt's own recorded hash for it.
    """

    w_by_ordinal: dict[int, float] = {}
    seen_ordinals: set[int] = set()
    for generation in stage1_receipt.get("generations") or []:
        if generation.get("stream") != 0:
            continue
        ordinal = generation["ordinal"]
        if ordinal in seen_ordinals:
            raise EvalJobRefusal(
                "sel-duplicate-ordinal", f"B1 receipt carries more than one stream-0 generation for ordinal {ordinal}"
            )
        seen_ordinals.add(ordinal)
        expected_seed = protocol.pilot_eval_seed(ordinal, 0)
        if generation.get("seed") != expected_seed:
            raise EvalJobRefusal(
                "sel-wrong-seed",
                f"B1 generation for ordinal {ordinal} has seed {generation.get('seed')!r}, "
                f"expected {expected_seed!r}",
            )
        energies = read_verified_stream_local_energies(generation)
        if not all(math.isfinite(energy) for energy in energies):
            raise EvalJobRefusal(
                "sel-nonfinite-energy", f"B1 generation for ordinal {ordinal} has a non-finite local_energy sample"
            )
        w_by_ordinal[ordinal] = protocol.w_statistic(energies)
    missing = sorted(set(protocol.pilot_stage1_ordinals(payload)) - seen_ordinals)
    if missing:
        raise EvalJobRefusal(
            "sel-incomplete-stage1", f"B1 receipt is missing stream-0 generations for ordinals {missing!r}"
        )
    c1, triple_means = protocol.select_pilot_basin_ordinal(w_by_ordinal, hi=protocol.pilot_sel_window_hi(payload))
    return {"c1": c1, "triple_means": triple_means}


def build_selection_record(
    *,
    c1: int,
    triple_means: Mapping[int, float],
    b1_eval_receipt_sha256: str,
    payload: str = "capacity-pilot",
) -> dict[str, Any]:
    """Return the SEL step's selection record (amendment r3/r4 E1).

    Stage 2 refuses unless its own B1 receipt sha256 matches this record's
    ``b1_eval_receipt_sha256`` and ``3 <= c1 <= hi``
    (:func:`verify_selection_record`).

    Operator ruling (operator-rulings-pilot-20261110-1900, item 3):
    ``interior_minimum`` is ``False`` when *c1* is at the window edge (3 or
    *hi*) -- there is no full neighboring triple on both sides, so stage 2
    still runs (the window is defined) but the record itself says so. *hi*
    is :func:`protocol.pilot_sel_window_hi` (amendment 20261112-r1), keyed
    by *payload*.
    """

    hi = protocol.pilot_sel_window_hi(payload)
    if not 3 <= c1 <= hi:
        raise EvalJobRefusal("sel-c1-out-of-range", f"c1 must be in 3..{hi}, got {c1}")
    return {
        "c1": c1,
        "triple_means": {str(center): mean for center, mean in triple_means.items()},
        "b1_eval_receipt_sha256": b1_eval_receipt_sha256,
        "interior_minimum": protocol.is_interior_minimum(c1, hi=hi),
    }


def verify_selection_record(
    selection: Mapping[str, Any], *, b1_eval_receipt_sha256: str, payload: str = "capacity-pilot"
) -> int:
    """Return ``c1`` from *selection*, refusing a mismatched or out-of-range record.

    Raises
    ------
    EvalJobRefusal
        Code ``sel-record-mismatch`` or ``sel-c1-out-of-range``.
    """

    if selection.get("b1_eval_receipt_sha256") != b1_eval_receipt_sha256:
        raise EvalJobRefusal(
            "sel-record-mismatch",
            f"selection record b1_eval_receipt_sha256 {selection.get('b1_eval_receipt_sha256')!r} "
            f"does not match the B1 receipt {b1_eval_receipt_sha256!r}",
        )
    hi = protocol.pilot_sel_window_hi(payload)
    c1 = selection.get("c1")
    if not isinstance(c1, int) or not 3 <= c1 <= hi:
        raise EvalJobRefusal("sel-c1-out-of-range", f"selection record c1 {c1!r} is not an int in 3..{hi}")
    return c1


__all__ = [
    "EVAL_RECEIPT_SCHEMA",
    "GPU_ROUTE_G_EVAL_STREAM",
    "GPU_ROUTE_G_EVAL_UPDATE",
    "EvalJobRefusal",
    "build_eval_receipt",
    "build_selection_record",
    "compute_pilot_selection",
    "generation_record",
    "materialize_verified_checkpoint",
    "REQUIRED_CHECKPOINT_CLOSURE_FILES",
    "gpu_route_g_fanout",
    "load_train_receipt",
    "pilot_fanout",
    "probe_readout_trajectory_generator",
    "read_eval_receipt",
    "read_eval_receipt_with_hash",
    "read_stream_local_energies",
    "read_verified_stream_local_energies",
    "resolve_checkpoint",
    "run_generation_direct",
    "science_fanout",
    "select_checkpoint",
    "trajectory_csv_path",
    "verify_checkpoint_files",
    "verify_selection_record",
    "verify_train_receipt_identity",
    "write_eval_receipt",
]
