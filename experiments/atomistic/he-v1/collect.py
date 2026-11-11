"""Assemble planned He-v1 rows into one table keyed by immutable identity.

Two failures this stage exists to prevent, both of which have happened in this
repository or its receipts:

a missing value is not a zero and not a blank
    Every cell states its own presence. A blank cell parses to NaN and is then
    silently dropped, so a median over two of nine rows renders exactly like a
    median over nine; a zero is worse, because it participates. Aggregates
    therefore always carry ``n_present``/``n_absent``.

rows are joined on identity, not on position
    A row is identified by its run id, its launch attempt, the hash of the
    checkpoint it ran, the hash of its resolved config, its seed, and its GPU
    stratum -- requested AND delivered. A delivered/requested mismatch fails
    the row here as it does in the allocation, and an evaluation whose restored
    checkpoint does not hash to its training row's retained checkpoint fails
    too, because that number belongs to a different model than the one claimed.

a metric name is not a metric
    Two evaluation tasks can share a summary class and therefore share every
    metric NAME while measuring different things. In the He config
    ``full_model_antisymmetry`` and ``spatial_exchange_symmetry`` both use
    ``TransformConsistencySummary``, so ``triplet_fraction_mean_under_psi_orig_sq``
    exists twice and means two different things: under full label exchange it is
    identically ``1.0`` by construction (``Psi -> -Psi`` gives ``u = 0`` and sign
    ratio ``-1``, so ``f = (1 - s*sech(u))/2 = 1``), while under spatial exchange
    it is the singlet-purity diagnostic. A request may therefore be qualified --
    ``eval/spatial_exchange_symmetry.triplet_fraction_mean_under_psi_orig_sq``
    names one task -- and an UNQUALIFIED request for a colliding name still
    fails loudly and names the namespaces rather than picking one. The collector
    never resolves that ambiguity by guessing; it only lets the caller express
    which task was meant.

Gating is delegated in full to ``gates.evaluate_atom_gates`` (layer L2); this
module adds no gate logic of its own. Before the tolerances are predeclared in
H-F1 every value gate legitimately reports ``absent`` with its observed value
retained -- that is the honest state of an ungated run, not a defect, and it is
never patched over with a placeholder threshold.

This module imports no ``tpen`` (``experiments/README.md``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

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

absence = sibling(__file__, 'absence')
canary = sibling(__file__, 'canary')
eval_stage = sibling(__file__, 'eval')

launch_stage = sibling(__file__, 'launch')
layout = sibling(__file__, 'layout')
plan_stage = sibling(__file__, 'plan')


def _load_gates() -> ModuleType:
    """Load layer L2's ``gates`` module by path.

    The study directory is not an importable package (its name contains a
    hyphen), so the checked-in experiment code loads its siblings by file
    location. ``gates.py`` is owned by another layer and is imported, never
    modified.
    """

    path = STUDY_DIR / "gates.py"
    spec = importlib.util.spec_from_file_location("he_v1_gates", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load gates module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gates = _load_gates()

#: Energy columns retained for every evaluation row.
#:
#: TWO ESTIMANDS LIVE HERE AND THEY ARE NOT INTERCHANGEABLE. The ``local_energy_*``
#: names without ``trajectory`` describe the FINAL DRAW ONLY -- a deliberate slice
#: that discards ~99.6% of the sampled records -- while the ``trajectory`` family
#: describes the whole retained chain. Both were already produced per row; only the
#: snapshot was ever selected here, so it reached the report under an unqualified
#: name and read as a trajectory estimate. ``report.py`` now names the estimator of
#: every column, and the trajectory pair is the headline.
#:
#: ``local_energy_stderr`` is an IID standard error over the snapshot and is
#: labelled as such wherever it is rendered; it is never presented as an MCSE.
#: ``local_energy_mcse`` is the correlation-corrected bar and is the one to quote.
#:
#: ABSENT IS NOT ZERO, AND ABSENT MUST NOT FALL BACK. ``TrajectoryStatisticsSummary``
#: withholds the payload statistics when Geyer's initial positive sequence never
#: terminated inside the data, so a row can legitimately report
#: ``..._trajectory_statistics_available`` false while carrying no mean, MCSE, ESS
#: or tau -- measured on both ``*-diagnostics`` rows of the 42-row study, whose
#: single-draw shape leaves no lags to sum. Such a row renders ``absent``. Falling
#: back to the snapshot there would reinstate exactly the conflation this list
#: exists to end, and would do it on precisely the rows that cannot support the
#: claim. ``..._plateau_reached`` is retained beside the MCSE because a truncated
#: sequence UNDERSTATES it, and a reader cannot otherwise tell a well-estimated bar
#: from a clipped one.
ENERGY_METRIC_KEYS: tuple[str, ...] = (
    # Final-draw snapshot. Retained for continuity and comparison, no longer the
    # headline.
    "local_energy_mean",
    "local_energy_stderr",
    "local_energy_variance",
    "local_energy_n_finite",
    "local_energy_nonfinite_count",
    # Whole-trajectory estimates. Computed per row all along; selected by nothing
    # until now, which is the entire defect.
    "local_energy_trajectory_mean",
    "local_energy_mcse",
    "local_energy_trajectory_variance",
    "local_energy_ess",
    "local_energy_tau_int",
    "local_energy_stderr_iid",
    "local_energy_mcse_inflation",
    # Quality of the two blocks above: whether the estimate resolved at all,
    # whether its error bar is truncated, and how many draws it spans.
    "local_energy_trajectory_statistics_available",
    "local_energy_plateau_reached",
    "local_energy_trajectory_total_draws",
    "reference_energy",
    "energy_error",
    "energy_abs_error",
)

#: Every metric the gates read, retained whether or not it gated.
GATE_METRIC_KEYS: tuple[str, ...] = tuple(gates.ATOM_GATE_METRIC_KEYS)

# These summaries intentionally publish metrics only.  Their implementations
# live in tpen/evaluation/summaries/trace.py and return SummaryResult(metrics=)
# without artifacts; an empty index entry is therefore their declared output,
# not a missing producer.  Every other task remains required to emit an
# artifact, so adding a metrics-only task requires an explicit review here.
METRICS_ONLY_TASKS = frozenset(
    {
        "full_model_antisymmetry",
        "spatial_exchange_symmetry",
        "trace_equivariance",
    }
)

#: Separator between a namespace and a metric name in a qualified request, as in
#: ``eval/spatial_exchange_symmetry.triplet_fraction_mean_under_psi_orig_sq``.
#: The namespace itself is slash-separated, exactly as the run logs it, so the
#: last ``.`` is the split point and a bare name never contains one.
NAMESPACE_SEPARATOR = "."

#: Reserved gate-spec key: a mapping from a bare metric name to the single
#: namespace it must be read from, for the retained column and for the gates
#: alike. It is stripped before the spec reaches ``gates.evaluate_atom_gates``,
#: which owns its own strict threshold-key check and must not be handed a key it
#: does not recognize.
METRIC_NAMESPACE_SPEC_KEY = "metric_namespaces"

CANARY_ARTIFACT_FILENAMES: Mapping[str, str] = {
    "sampled_eval_table": "sampled_eval_table.csv",
    "local_energy_trajectory_statistics": "trajectory_statistics.jsonl",
    "sampler_trajectory_diagnostics": "sampler_trajectory_diagnostics.json",
}

CANARY_ARTIFACT_KINDS: Mapping[str, str] = {
    "sampled_eval_table": "csv",
    "local_energy_trajectory_statistics": "trajectory_statistics_sidecar",
    "sampler_trajectory_diagnostics": "sampler_trajectory_diagnostics",
}

_SAMPLER_DIAGNOSTICS_KEYS = frozenset(
    {
        "schema",
        "n_walkers",
        "draw_stride",
        "sampler_burn_in",
        "proposal_scale",
        "intermediate_sampler_steps_observed",
        "intermediate_sampler_steps_unobserved_reason",
        "sampler_internal_burn_in_states_observed",
        "discarded_draw_acceptance_rate_series",
        "retained_draw_acceptance_rate_series",
        "discarded_draws",
        "retained_draws",
        "metrics",
    }
)

_SAMPLER_DRAW_DIAGNOSTICS_KEYS = frozenset(
    {
        "collection_index",
        "region_index",
        "acceptance_rate",
        "n_walkers",
        "burn_in",
        "draw_stride",
        "transition_count",
        "proposal_scale",
        "seed",
        "minimum_electron_nucleus_radius",
    }
)

COLLECTED_FILENAME = "collected.json"
ROWS_CSV = "rows.csv"
GATES_CSV = "gates.csv"


class CollectError(RuntimeError):
    """The collected rows do not form a table that can be reported honestly."""


def read_metrics_jsonl(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[str]]]:
    """Read one ``metrics.jsonl`` into namespaced and flat mappings.

    Returns
    -------
    namespaced : dict
        ``"<namespace>/<key>" -> value`` for every logged record.
    flat : dict
        ``"<key>" -> value`` for keys that carry one value across every
        namespace that logged them. A name logged twice with the SAME value is
        not ambiguous: either namespace answers the question identically.
    ambiguous : dict
        ``"<key>" -> [namespace, ...]`` for keys logged under more than one
        namespace with differing values. They are excluded from ``flat`` rather
        than resolved by guesswork: picking one silently would attribute a
        number to the wrong task. The namespaces are carried here so a failure
        can name them instead of only naming the key.
    """

    path = Path(path)
    namespaced: dict[str, Any] = {}
    by_key: dict[str, list[tuple[str, Any]]] = {}
    if not path.is_file():
        return {}, {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        record = json.loads(text)
        namespace = str(record.get("namespace") or "").strip("/")
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            full = f"{namespace}/{key}" if namespace else str(key)
            namespaced[full] = value
            by_key.setdefault(str(key), []).append((namespace, value))
    flat: dict[str, Any] = {}
    ambiguous: dict[str, list[str]] = {}
    for key, entries in by_key.items():
        values = {json.dumps(value, sort_keys=True) for _namespace, value in entries}
        if len(values) == 1:
            flat[key] = entries[0][1]
        else:
            ambiguous[key] = sorted({namespace for namespace, _value in entries})
    return namespaced, flat, ambiguous


def logged_namespaces(namespaced: Mapping[str, Any]) -> set[str]:
    """Return every namespace one row actually logged under."""

    return {str(full).rpartition("/")[0] for full in namespaced}


def split_metric_request(request: str) -> tuple[str | None, str]:
    """Split ``"<namespace>.<key>"`` into its parts.

    A bare request yields ``(None, request)``. The split is on the LAST
    :data:`NAMESPACE_SEPARATOR`, because the namespace is slash-separated and
    the metric names emitted by the summaries carry no dot.
    """

    text = str(request)
    namespace, separator, key = text.rpartition(NAMESPACE_SEPARATOR)
    if not separator or not namespace or not key:
        return None, text
    return namespace, key


def resolve_metric_request(
    request: str,
    *,
    namespaced: Mapping[str, Any],
    flat: Mapping[str, Any],
    ambiguous: Mapping[str, Sequence[str]],
    namespaces: set[str],
) -> tuple[Any, str | None]:
    """Resolve one requested metric against one row's logged metrics.

    Returns
    -------
    value : Any
        The measured value, or :data:`absence.ABSENT` when this row has none.
    reason : str or None
        A row failure reason when the request cannot be honoured, else ``None``.

    Notes
    -----
    Three cases are deliberately distinct, because collapsing any two of them
    is how a number gets attributed to the wrong task:

    - a bare name that collides FAILS and names the colliding namespaces; the
      caller must say which task it meant. This is the pre-existing refusal to
      guess, now scoped to the metrics actually requested rather than fired by
      any collision anywhere in the log;
    - a qualified name whose namespace THIS row logged, but which that
      namespace never emitted, FAILS -- the row ran the task and the metric is
      still missing, so the request is wrong or the task is broken;
    - a qualified name whose namespace this row never logged is ABSENT, not a
      failure. A train row does not run the evaluation tasks, and an eval-only
      column must not fail every train row. A request naming a namespace that
      NO row logged is caught once, study-wide, by
      :func:`require_requested_namespaces`.
    """

    namespace, key = split_metric_request(request)
    if namespace is None:
        if key in ambiguous:
            return absence.ABSENT, (
                f"metric key {key!r} is logged under several namespaces "
                f"{list(ambiguous[key])} with differing values; request it as "
                f"'<namespace>{NAMESPACE_SEPARATOR}{key}' to say which task is meant"
            )
        return flat.get(key), None
    full = f"{namespace}/{key}"
    if full in namespaced:
        return namespaced[full], None
    if namespace in namespaces:
        return absence.ABSENT, (
            f"qualified metric key {request!r} names namespace {namespace!r}, which this "
            f"row logged, but that namespace emitted no {key!r}"
        )
    return absence.ABSENT, None


def require_requested_namespaces(
    rows: Sequence[Mapping[str, Any]], requests: Iterable[str]
) -> None:
    """Reject a qualified request whose namespace no row in the attempt logged.

    Per row, an unlogged namespace is honest absence (a train row runs no
    evaluation task). Across the whole attempt it is a mis-typed or stale
    request, and letting it render ``absent`` in every row would look exactly
    like a metric nobody emitted -- the silent failure this stage exists to
    prevent. It is therefore raised, not recorded: re-collecting is cheap and
    the request has to be corrected.
    """

    logged: set[str] = set()
    for row in rows:
        logged.update(str(name) for name in row["logged_namespaces"])
    unmatched = sorted(
        request
        for request in requests
        if (namespace := split_metric_request(request)[0]) is not None
        and namespace not in logged
    )
    if unmatched:
        raise CollectError(
            f"qualified metric requests name namespaces no row logged: {unmatched}; "
            f"namespaces logged in this attempt: {sorted(logged)}"
        )


def split_gate_spec(spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Split a gate spec into thresholds and namespace bindings.

    ``gates.evaluate_atom_gates`` rejects any spec key it does not recognize --
    a mistyped threshold would otherwise disable exactly the gate it was meant
    to set. The namespace bindings are therefore removed here rather than
    passed through, and the thresholds reach the gates untouched.
    """

    thresholds = {
        key: value for key, value in spec.items() if key != METRIC_NAMESPACE_SPEC_KEY
    }
    raw = spec.get(METRIC_NAMESPACE_SPEC_KEY)
    if raw is None:
        return thresholds, {}
    if not isinstance(raw, Mapping):
        raise CollectError(
            f"gate spec {METRIC_NAMESPACE_SPEC_KEY!r} must be a mapping of "
            f"'<metric>: <namespace>', got {raw!r}"
        )
    bindings: dict[str, str] = {}
    for metric, namespace in raw.items():
        if not isinstance(namespace, str) or not namespace.strip():
            raise CollectError(
                f"gate spec {METRIC_NAMESPACE_SPEC_KEY}[{metric!r}] must be a namespace "
                f"string, got {namespace!r}"
            )
        bindings[str(metric)] = namespace.strip().strip("/")
    return thresholds, bindings


def resolve_metric_bindings(
    bindings: Mapping[str, str],
    *,
    namespaced: Mapping[str, Any],
    flat: Mapping[str, Any],
    ambiguous: Mapping[str, Sequence[str]],
    namespaces: set[str],
    reasons: list[str],
) -> dict[str, Any]:
    """Resolve every bound bare metric to the value of its declared namespace.

    A binding is a caller stating which task a name refers to. It is resolved
    once per row and then used both for the retained column and for the mapping
    the gates read, so a bound name can never gate one task's value while
    reporting another's.
    """

    bound: dict[str, Any] = {}
    for metric, namespace in sorted(bindings.items()):
        value, reason = resolve_metric_request(
            f"{namespace}{NAMESPACE_SEPARATOR}{metric}",
            namespaced=namespaced,
            flat=flat,
            ambiguous=ambiguous,
            namespaces=namespaces,
        )
        if reason is not None:
            reasons.append(reason)
        bound[metric] = value
    return bound


def gate_metric_view(flat: Mapping[str, Any], bound: Mapping[str, Any]) -> dict[str, Any]:
    """Return the metrics mapping the gates read, with bindings applied.

    A bound metric is read from its declared namespace and from nowhere else,
    which is what lets a tolerance gate spatial-exchange singlet purity without
    ever seeing the full-model triplet fraction that is ``1.0`` by construction.
    An unbound metric keeps today's behaviour: unambiguous names resolve, and a
    colliding name is simply absent from the view, so its gate reports
    ``absent`` rather than gating a guess.
    """

    view = dict(flat)
    for metric, value in bound.items():
        # The raw value is handed on unchanged: the gates own what a non-finite
        # or wrongly typed metric means, and they fail closed on it. Only a
        # metric this row never logged is removed from the view.
        if value is absence.ABSENT or value is None:
            view.pop(metric, None)
        else:
            view[metric] = value
    return view


def file_sha256(path: str | Path) -> str | Any:
    """Return the SHA-256 of one file, or :data:`absence.ABSENT` when missing."""

    path = Path(path)
    if not path.is_file():
        return absence.ABSENT
    return plan_stage.file_sha256(path)


def directory_sha256(path: str | Path) -> str | Any:
    """Return a content hash of a directory tree, or absent when missing.

    The hash covers relative paths and file contents in sorted order, so it
    identifies the checkpoint rather than the moment it was written.
    """

    root = Path(path)
    if not root.is_dir():
        return absence.ABSENT
    digest = hashlib.sha256()
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(plan_stage.file_sha256(file_path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_gate_spec(path: str | Path | None, *, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the tolerance spec, from a file when given, else from the plan.

    An empty spec is a legitimate state, not a missing input: the tolerance
    numbers are predeclared in H-F1, and until then every value gate reports
    ``absent`` with its observed value retained.
    """

    if path is None:
        return dict(manifest.get("gate_spec") or {})
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise CollectError(f"gate spec {path} is not a mapping")
    return dict(payload)


def reconcile_canary_row(
    row: Mapping[str, Any],
    *,
    source: canary.CheckpointSource,
    result_dir: Path,
    run_dir: Path,
    plan_attempt_id: str,
    manifest: Mapping[str, Any],
    row_record: Any,
    metadata: Any,
    identity_diagnostics: MutableMapping[str, Any] | None = None,
    source_git_sha: str | None = None,
) -> list[str]:
    """Reconcile one canary's source, execution, records, and trajectory receipt."""

    reasons: list[str] = []
    binding_receipt = _required_json(
        result_dir / eval_stage.CHECKPOINT_BINDING_RECEIPT,
        "checkpoint binding receipt",
        reasons,
    )
    try:
        expected_receipt = source.receipt()
    except canary.CanaryError as exc:
        reasons.append(f"immutable checkpoint source changed: {exc}")
        expected_receipt = None
    if expected_receipt is not None and binding_receipt != expected_receipt:
        reasons.append("checkpoint binding receipt disagrees with the live immutable source")

    if not isinstance(row_record, Mapping) or row_record.get("row") != dict(row):
        reasons.append("executed row record disagrees with the planned canary row")
    if _json_field(row_record, "plan_attempt_id") != plan_attempt_id:
        reasons.append("executed row record has the wrong plan attempt id")
    if _json_field(metadata, "run_id") != row["row_id"]:
        reasons.append("run metadata has the wrong row id")
    if _json_field(metadata, "git_commit") != manifest.get("evaluation_git_sha"):
        reasons.append("run metadata has the wrong evaluation source SHA")
    if _json_field(metadata, "dirty_worktree") is not False:
        reasons.append("run metadata does not attest a clean evaluation worktree")

    config_path = run_dir / "resolved_config.yaml"
    config = _required_yaml(config_path, "resolved config", reasons)
    expected_config_sha = eval_stage.config_identity_hash(
        STUDY_DIR.parents[2] / str(row["config"]),
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
        },
    )
    legacy_config_error: str | None = None
    try:
        legacy_config_sha: str | None = eval_stage.legacy_config_identity_hash(
            STUDY_DIR.parents[2] / str(row["config"]),
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
            },
            source_git_sha=source_git_sha,
        )
    except ValueError as exc:
        legacy_config_sha = None
        legacy_config_error = str(exc)
    _reconcile_canary_config(config, row=row, source=source, reasons=reasons)

    index = _required_json(
        run_dir / "diagnostics" / "index.json", "artifact index", reasons
    )
    task_artifacts = _reconcile_canary_index(
        index,
        run_dir=run_dir,
        task_names=row.get("task_names", []),
        row=row,
        reasons=reasons,
    )
    if "mcmc_energy" in row.get("task_names", []):
        task_dir = run_dir / "mcmc_energy"
        artifacts = task_artifacts.get("mcmc_energy", {})
        csv_path = task_dir / "sampled_eval_table.csv"
        record_metadata_path = task_dir / "sampled_eval_table.metadata.json"
        trajectory_path = task_dir / "trajectory_statistics.jsonl"
        sampler_diagnostics_path = task_dir / "sampler_trajectory_diagnostics.json"
        record_metadata = _required_json(
            record_metadata_path, "trajectory record metadata", reasons
        )
        _reconcile_canary_records(
            csv_path,
            record_metadata,
            row=row,
            artifact=artifacts.get("sampled_eval_table"),
            reasons=reasons,
        )
        config_identity_match = _reconcile_canary_trajectory(
            trajectory_path,
            row=row,
            source=source,
            canonical_config_sha256=expected_config_sha,
            legacy_config_sha256=legacy_config_sha,
            legacy_config_error=legacy_config_error,
            plan_attempt_id=plan_attempt_id,
            artifact=artifacts.get("local_energy_trajectory_statistics"),
            reasons=reasons,
        )
        if identity_diagnostics is not None:
            identity_diagnostics["config_identity_match"] = config_identity_match
        _reconcile_canary_sampler_diagnostics(
            sampler_diagnostics_path,
            row=row,
            artifact=artifacts.get("sampler_trajectory_diagnostics"),
            reasons=reasons,
        )
    return reasons


def _reconcile_canary_config(
    config: Any,
    *,
    row: Mapping[str, Any],
    source: canary.CheckpointSource,
    reasons: list[str],
) -> None:
    if not isinstance(config, Mapping):
        return
    evaluator = config.get("evaluator")
    sampler = config.get("evaluation_sampler")
    tasks = evaluator.get("tasks") if isinstance(evaluator, Mapping) else None
    expected_names = list(row.get("task_names", []))
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)) or len(tasks) != len(expected_names):
        reasons.append(
            "resolved evaluator task list disagrees with the canary row: "
            f"expected={expected_names}, actual={len(tasks) if isinstance(tasks, Sequence) else None}"
        )
        return
    actual_names = [task.get("name") if isinstance(task, Mapping) else None for task in tasks]
    if actual_names != expected_names:
        reasons.append(
            f"resolved evaluator task names disagree with the canary row: expected={expected_names}, actual={actual_names}"
        )
    for task in tasks:
        if not isinstance(task, Mapping) or task.get("name") != "mcmc_energy":
            continue
        if not isinstance(sampler, Mapping) or (
            sampler.get("n_walkers"), sampler.get("burn_in"), sampler.get("n_steps")
        ) != (row["n_walkers"], row["burn_in"], row["stride"]):
            reasons.append("resolved sampler scale disagrees with the canary row")
        generator = task.get("generator")
        if not isinstance(generator, Mapping) or (
            generator.get("n_draws"), generator.get("discard_draws"), generator.get("chunk_size")
        ) != (row["n_draws"], row["discard_draws"], row["chunk_size"]):
            reasons.append("resolved trajectory generator scale disagrees with the canary row")
        writers = [
            value for value in task.get("summaries", [])
            if isinstance(value, Mapping)
            and value.get("_target_") == "tpen.evaluation.summaries.SampledRecordWriter"
        ]
        if len(writers) != 1 or writers[0].get("max_samples") != row["record_capacity"]:
            reasons.append("resolved trajectory writer capacity disagrees with the canary row")
    replay = config.get("load", {}).get("replay_semantics", {})
    expected_replay = {
        "source_git_sha": source.training_source_sha,
        "source_tpen_version": row["checkpoint_source"]["source_tpen_version"],
        "checkpoint_schema_version": 2,
        "checkpoint_kind": "tpen.checkpoint",
        "checkpoint_model_sha256": source.model_sha256,
    }
    if not isinstance(replay, Mapping) or any(
        replay.get(key) != value for key, value in expected_replay.items()
    ):
        reasons.append("resolved replay semantics disagree with the immutable source")


def _reconcile_canary_index(
    index: Any,
    *,
    run_dir: Path,
    task_names: Sequence[str],
    row: Mapping[str, Any],
    reasons: list[str],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    tasks = index.get("tasks") if isinstance(index, Mapping) else None
    expected_names = list(task_names)
    if not isinstance(tasks, list) or len(tasks) != len(expected_names):
        reasons.append(
            f"artifact index task list disagrees with the canary row: expected={expected_names}, "
            f"actual={len(tasks) if isinstance(tasks, list) else None}"
        )
        return {}
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    actual_names: list[str] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            reasons.append("artifact index contains a non-mapping task")
            continue
        name = str(task.get("name") or "")
        actual_names.append(name)
        expected_namespace = f"eval/{name}"
        if name not in expected_names:
            reasons.append(f"artifact index contains an undeclared task {name!r}")
        if task.get("namespace") != expected_namespace:
            reasons.append(
                f"artifact index task {name!r} namespace disagrees: "
                f"expected={expected_namespace!r}, actual={task.get('namespace')!r}"
            )
        if task.get("status") != "success":
            reasons.append(f"artifact index task {name!r} is not successful")
        artifacts = task.get("artifacts")
        if not isinstance(artifacts, list):
            reasons.append(f"{name} artifact index has no artifact list")
            continue
        named_artifacts = [
            artifact
            for artifact in artifacts
            if isinstance(artifact, Mapping) and str(artifact.get("name") or "").strip()
        ]
        by_name = {str(artifact["name"]): artifact for artifact in named_artifacts}
        if len(by_name) != len(artifacts) or (
            not by_name and name not in METRICS_ONLY_TASKS
        ):
            reasons.append(
                f"{name} artifact index has duplicate, invalid, or unexpected empty artifacts"
            )
        if name == "mcmc_energy":
            expected = set(CANARY_ARTIFACT_FILENAMES)
            if len(artifacts) != len(expected) or set(by_name) != expected:
                reasons.append(
                    f"mcmc_energy artifacts mismatch: expected={sorted(expected)}, "
                    f"actual={[artifact.get('name') if isinstance(artifact, Mapping) else None for artifact in artifacts]}"
                )
        task_dir = run_dir / name
        indexed_output_dir = Path(str(task.get("output_dir") or ""))
        if not indexed_output_dir.is_absolute() or indexed_output_dir.resolve() != task_dir.resolve():
            reasons.append(f"{name} artifact index output directory disagrees with its task directory")
        for artifact_name, artifact in by_name.items():
            path = Path(str(artifact.get("path") or ""))
            if not path.is_absolute():
                reasons.append(f"artifact {artifact_name!r} path is not absolute")
            else:
                try:
                    path.resolve().relative_to(task_dir.resolve())
                except ValueError:
                    reasons.append(f"artifact {artifact_name!r} is outside its indexed task directory")
                if not path.is_file():
                    reasons.append(f"artifact {artifact_name!r} path does not exist")
            if name == "mcmc_energy":
                expected_kind = CANARY_ARTIFACT_KINDS.get(artifact_name)
                if expected_kind is not None and artifact.get("kind") != expected_kind:
                    reasons.append(f"artifact {artifact_name!r} kind disagrees with its known contract")
                expected_filename = CANARY_ARTIFACT_FILENAMES.get(artifact_name)
                if expected_filename is not None and path.resolve() != (task_dir / expected_filename).resolve():
                    reasons.append(f"artifact {artifact_name!r} path disagrees with its known filename")
        _reconcile_canary_task_content(
            name, by_name, row=row, reasons=reasons
        )
        result[name] = by_name
    if actual_names != expected_names:
        reasons.append(f"artifact index task names disagree with the canary row: expected={expected_names}, actual={actual_names}")
    return result


def _reconcile_canary_task_content(
    task_name: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    row: Mapping[str, Any],
    reasons: list[str],
) -> None:
    """Check the content shape of declared non-trajectory task artifacts.

    The index/path checks establish provenance, not semantic completeness. The
    factor-response producer has a fixed seven-arm contract, while the
    re-equilibrated factor task publishes one complete sampled table per row.
    Keeping these checks named here makes a future task addition an explicit
    review point instead of silently inheriting a weaker generic check.
    """

    if task_name == "factor_response":
        artifact = artifacts.get("factor_response_common_configuration")
        if artifact is None:
            reasons.append("factor_response is missing its common-configuration artifact")
            return
        # Common-configuration compares seven factor arms on one sampler
        # snapshot.  Its producer intentionally emits one batch per walker;
        # the row's n_draws belongs to co-selected trajectory tasks, not this
        # snapshot task.
        capacity = int(row["n_walkers"])
        metadata = artifact.get("metadata")
        expected = {
            "comparison_kind": "common_configuration",
            "rows": capacity * 7,
            "arm_count": 7,
            "configuration_count": capacity,
            "model_state_restored": True,
        }
        if not isinstance(metadata, Mapping) or any(
            metadata.get(key) != value for key, value in expected.items()
        ):
            reasons.append(
                "factor_response artifact metadata disagrees with the seven-arm "
                f"common-configuration contract: expected={expected}"
            )
    elif task_name == "factor_response_re_equilibrated":
        artifact = artifacts.get("sampled_eval_table")
        if artifact is None:
            reasons.append("re-equilibrated factor task is missing its sampled table artifact")
            return
        capacity = int(row["record_capacity"])
        metadata = artifact.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("rows") != capacity:
            reasons.append(
                "re-equilibrated factor sampled table metadata disagrees with the row capacity"
            )


def _reconcile_canary_records(
    csv_path: Path,
    record_metadata: Any,
    *,
    row: Mapping[str, Any],
    artifact: Mapping[str, Any] | None,
    reasons: list[str],
) -> None:
    if not csv_path.is_file() or not isinstance(record_metadata, Mapping):
        if not csv_path.is_file():
            reasons.append(f"missing trajectory record CSV: {csv_path}")
        return
    expected_rows = int(row["record_capacity"])
    expected_metadata = {
        "schema": "trajectory_records/v1",
        "row_semantics": "complete_draw_walker_grid",
        "observable": "local_energy",
        "draw_count": row["n_draws"],
        "walker_count": row["n_walkers"],
        "draw_stride": row["stride"],
        "burn_in_draws": row["discard_draws"],
        "row_count": expected_rows,
    }
    if any(record_metadata.get(key) != value for key, value in expected_metadata.items()):
        reasons.append("trajectory record metadata disagrees with the planned draw/walker grid")
    actual_sha = plan_stage.file_sha256(csv_path)
    if record_metadata.get("csv_sha256") != actual_sha:
        reasons.append("trajectory record CSV hash disagrees with its metadata")
    if record_metadata.get("byte_count") != csv_path.stat().st_size:
        reasons.append("trajectory record CSV byte count disagrees with its metadata")
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        reasons.append(f"trajectory record CSV is unreadable: {exc}")
        return
    coordinates = [
        (int(record["sample_index"]), int(record["draw_index"]), int(record["walker_index"]))
        for record in records
        if {"sample_index", "draw_index", "walker_index"} <= set(record)
    ]
    expected_coordinates = [
        (draw * int(row["n_walkers"]) + walker, draw, walker)
        for draw in range(int(row["n_draws"]))
        for walker in range(int(row["n_walkers"]))
    ]
    if len(records) != expected_rows or coordinates != expected_coordinates:
        reasons.append("trajectory record CSV is not the complete ordered draw/walker grid")
    artifact_metadata = artifact.get("metadata") if isinstance(artifact, Mapping) else None
    if not isinstance(artifact_metadata, Mapping) or any(
        artifact_metadata.get(key) != value
        for key, value in {
            "rows": expected_rows,
            "n_total": expected_rows,
            "draw_count": row["n_draws"],
            "walker_count": row["n_walkers"],
            "truncated": False,
            "selection": "complete_draw_walker_grid",
            "content_id": actual_sha,
            "bytes": csv_path.stat().st_size,
        }.items()
    ):
        reasons.append("sampled_eval_table artifact metadata disagrees with the retained grid")


def _reconcile_canary_trajectory(
    path: Path,
    *,
    row: Mapping[str, Any],
    source: canary.CheckpointSource,
    canonical_config_sha256: str,
    legacy_config_sha256: str | None,
    legacy_config_error: str | None,
    plan_attempt_id: str,
    artifact: Mapping[str, Any] | None,
    reasons: list[str],
) -> str | None:
    if not path.is_file():
        reasons.append(f"missing trajectory-statistics sidecar: {path}")
        return None
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        reasons.append(f"trajectory-statistics sidecar is unreadable: {exc}")
        return None
    if len(records) != 1 or not isinstance(records[0], Mapping):
        reasons.append("trajectory-statistics sidecar must contain exactly one receipt")
        return None
    receipt = records[0]
    config_identity_match = _match_config_identity(
        receipt.get("config_sha256"),
        canonical=canonical_config_sha256,
        legacy=legacy_config_sha256,
    )
    expected_identity = {
        "stage": row["stage"],
        "run_id": row["row_id"],
        "attempt_id": plan_attempt_id,
        "checkpoint_sha256": source.model_sha256,
        "config_sha256": (
            canonical_config_sha256
            if config_identity_match != "legacy"
            else legacy_config_sha256
        ),
        "observable": "local_energy",
        "evaluator_id": eval_stage.EVALUATOR_ID,
    }
    if config_identity_match is None:
        if legacy_config_error is not None:
            reasons.append(
                f"legacy config identity unavailable: {legacy_config_error}"
            )
        reasons.append(
            "trajectory-statistics receipt config identity matches neither canonical "
            "nor legacy hash"
        )
    if any(receipt.get(key) != value for key, value in expected_identity.items()):
        reasons.append("trajectory-statistics receipt join identity mismatch")
    shape = receipt.get("shape")
    if not isinstance(shape, Mapping) or any(
        shape.get(key) != value
        for key, value in {
            "walker_count": row["n_walkers"],
            "draw_count": row["n_draws"],
            "total_draws": row["record_capacity"],
            "draw_stride": row["stride"],
            "burn_in_draws": row["discard_draws"],
        }.items()
    ):
        reasons.append("trajectory-statistics shape disagrees with the retained grid")
    if receipt.get("status") not in {"available", "unresolved"}:
        reasons.append("trajectory-statistics receipt does not attest a retained trajectory")
    artifact_metadata = artifact.get("metadata") if isinstance(artifact, Mapping) else None
    if not isinstance(artifact_metadata, Mapping) or any(
        artifact_metadata.get(key) != value
        for key, value in {**expected_identity, "status": receipt.get("status")}.items()
    ):
        reasons.append("trajectory-statistics artifact metadata disagrees with its sidecar")
    return config_identity_match


def _match_config_identity(
    observed: Any, *, canonical: str, legacy: str | None
) -> str | None:
    """Classify one receipt hash without accepting an unknown identity."""

    if observed == canonical:
        return "canonical"
    if legacy is not None and observed == legacy:
        return "legacy"
    return None


def _reconcile_canary_sampler_diagnostics(
    path: Path,
    *,
    row: Mapping[str, Any],
    artifact: Mapping[str, Any] | None,
    reasons: list[str],
) -> None:
    payload = _required_json(path, "sampler trajectory diagnostics", reasons)
    if not isinstance(payload, Mapping):
        if payload is not None:
            reasons.append("sampler trajectory diagnostics must be a mapping")
        return

    discarded = payload.get("discarded_draws")
    retained = payload.get("retained_draws")
    discarded_series = payload.get("discarded_draw_acceptance_rate_series")
    retained_series = payload.get("retained_draw_acceptance_rate_series")
    proposal_scale = payload.get("proposal_scale")
    expected_top_level = {
        "schema": "sampler_trajectory_diagnostics/v1",
        "n_walkers": row["n_walkers"],
        "draw_stride": row["stride"],
        "sampler_burn_in": row["burn_in"],
        "intermediate_sampler_steps_observed": False,
        "sampler_internal_burn_in_states_observed": False,
    }
    expected_metrics = {
        "trajectory_retained_draw_count": row["n_draws"],
        "trajectory_discarded_draw_count": row["discard_draws"],
        "trajectory_n_walkers": row["n_walkers"],
        "trajectory_draw_stride": row["stride"],
        "trajectory_sampler_burn_in": row["burn_in"],
        "trajectory_proposal_scale": proposal_scale,
        "trajectory_retained_value_count": row["record_capacity"],
        "trajectory_discarded_value_count": row["discard_draws"]
        * row["n_walkers"],
        "trajectory_retained_transition_count": row["record_capacity"]
        * row["stride"],
        "trajectory_discarded_transition_count": row["discard_draws"]
        * row["n_walkers"]
        * row["stride"],
        "trajectory_intermediate_sampler_steps_observed": False,
    }
    metrics = payload.get("metrics")
    sidecar_matches = (
        set(payload) == _SAMPLER_DIAGNOSTICS_KEYS
        and all(payload.get(key) == value for key, value in expected_top_level.items())
        and isinstance(payload.get("intermediate_sampler_steps_unobserved_reason"), str)
        and bool(payload.get("intermediate_sampler_steps_unobserved_reason"))
        and _sampler_diagnostic_draws_match(
            discarded,
            row=row,
            count=row["discard_draws"],
            collection_offset=0,
            proposal_scale=proposal_scale,
        )
        and _sampler_diagnostic_draws_match(
            retained,
            row=row,
            count=row["n_draws"],
            collection_offset=row["discard_draws"],
            proposal_scale=proposal_scale,
        )
        and isinstance(discarded_series, list)
        and isinstance(retained_series, list)
        and discarded_series
        == [draw["acceptance_rate"] for draw in discarded]
        and retained_series == [draw["acceptance_rate"] for draw in retained]
        and isinstance(metrics, Mapping)
        and all(metrics.get(key) == value for key, value in expected_metrics.items())
    )
    if not sidecar_matches:
        reasons.append("sampler trajectory diagnostics disagree with the planned sampler/grid")

    expected_artifact_metadata = {
        "schema": "sampler_trajectory_diagnostics/v1",
        "retained_draw_count": row["n_draws"],
        "discarded_draw_count": row["discard_draws"],
        "draw_stride": row["stride"],
        "intermediate_sampler_steps_observed": False,
    }
    artifact_metadata = artifact.get("metadata") if isinstance(artifact, Mapping) else None
    if not isinstance(artifact_metadata, Mapping) or (
        set(artifact_metadata) != set(expected_artifact_metadata)
        or any(
            artifact_metadata.get(key) != value
            for key, value in expected_artifact_metadata.items()
        )
    ):
        reasons.append("sampler trajectory diagnostics artifact metadata mismatch")


def _sampler_diagnostic_draws_match(
    draws: Any,
    *,
    row: Mapping[str, Any],
    count: int,
    collection_offset: int,
    proposal_scale: Any,
) -> bool:
    if not isinstance(draws, list) or len(draws) != count:
        return False
    for region_index, draw in enumerate(draws):
        if not isinstance(draw, Mapping) or set(draw) != _SAMPLER_DRAW_DIAGNOSTICS_KEYS:
            return False
        acceptance_rate = draw.get("acceptance_rate")
        radius = draw.get("minimum_electron_nucleus_radius")
        seed = draw.get("seed")
        if (
            isinstance(acceptance_rate, bool)
            or not isinstance(acceptance_rate, int | float)
            or not 0.0 <= acceptance_rate <= 1.0
            or (radius is not None and (isinstance(radius, bool) or not isinstance(radius, int | float) or radius < 0.0))
            or (seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)))
        ):
            return False
        expected = {
            "collection_index": collection_offset + region_index,
            "region_index": region_index,
            "n_walkers": row["n_walkers"],
            "burn_in": row["burn_in"],
            "draw_stride": row["stride"],
            "transition_count": row["n_walkers"] * row["stride"],
            "proposal_scale": proposal_scale,
            "seed": row["seed"],
        }
        if any(draw.get(key) != value for key, value in expected.items()):
            return False
    return True


def _required_json(path: Path, label: str, reasons: list[str]) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        reasons.append(f"missing or invalid {label}: {path}: {exc}")
        return None
    return payload


def _required_yaml(path: Path, label: str, reasons: list[str]) -> Any:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        reasons.append(f"missing or invalid {label}: {path}: {exc}")
        return None
    return payload


def collect_row(
    row: Mapping[str, Any],
    *,
    results_root: Path,
    plan_attempt_id: str,
    manifest: Mapping[str, Any],
    gate_spec: Mapping[str, Any],
    metric_namespaces: Mapping[str, str] | None = None,
    extra_metric_keys: Sequence[str] = (),
    hash_checkpoints: bool = True,
    checkpoint_sources: Mapping[str, canary.CheckpointSource] | None = None,
) -> dict[str, Any]:
    """Collect one planned row into a table row with explicit absence.

    ``gate_spec`` carries thresholds only; the namespace bindings arrive
    separately as ``metric_namespaces`` because the gates reject any spec key
    outside their own recognized set. A binding decides both the retained
    column and the value the gates read, so one name cannot report one task's
    number while gating another's.
    """

    row_id = str(row["row_id"])
    kind = str(row["kind"])
    result_dir = layout.row_dir(results_root, str(row["stage"]), row_id, plan_attempt_id)
    run_dir = result_dir / row_id
    reasons: list[str] = []

    receipt = _read_json_or_absent(result_dir / "allocation_receipt.json", reasons)
    row_record = _read_json_or_absent(result_dir / "row.json", reasons)
    metadata = _read_json_or_absent(run_dir / "metadata.json", reasons)
    status = _read_json_or_absent(run_dir / "status.json", reasons)
    namespaced, flat, ambiguous = read_metrics_jsonl(run_dir / "metrics.jsonl")
    namespaces = logged_namespaces(namespaced)
    if not namespaced:
        reasons.append("no metrics were logged")

    requested_stratum = str(row["resources"]["stratum"])
    requested_partition = str(row["resources"]["partition"])
    delivered_partition = _receipt_field(receipt, "partition")
    delivered_device = _receipt_field(receipt, "delivered_device")
    delivered_matches = _receipt_field(receipt, "delivered_matches_requested")
    if absence.is_absent(delivered_matches):
        reasons.append("no allocation receipt: the delivered GPU was never verified")
    elif delivered_matches is not True:
        reasons.append(
            f"delivered device {delivered_device!r} does not match requested stratum "
            f"{requested_stratum!r}"
        )

    checkpoint_step = row.get("checkpoint_step")
    checkpoint_hash: Any = absence.ABSENT
    checkpoint_dir: Any = absence.ABSENT
    identity_diagnostics: dict[str, Any] = {}
    if kind == "eval":
        if row.get("canary_protocol") == canary.CANARY_SCHEMA:
            if checkpoint_sources is None:
                reasons.append("canary collection has no validated checkpoint source map")
            else:
                try:
                    source = canary.source_for_row(row, checkpoint_sources)
                    checkpoint_dir = source.checkpoint_dir
                    checkpoint_hash = (
                        source.model_sha256 if hash_checkpoints else absence.ABSENT
                    )
                    reasons.extend(
                        reconcile_canary_row(
                            row,
                            source=source,
                            result_dir=result_dir,
                            run_dir=run_dir,
                            plan_attempt_id=plan_attempt_id,
                            manifest=manifest,
                            row_record=row_record,
                            metadata=metadata,
                            identity_diagnostics=identity_diagnostics,
                            source_git_sha=manifest.get("evaluation_git_sha"),
                        )
                    )
                except canary.CanaryError as exc:
                    reasons.append(f"immutable checkpoint source reconciliation failed: {exc}")
        else:
            checkpoint_dir = launch_stage.checkpoint_dir_for_eval_row(
                results_root, row, plan_attempt_id, manifest=manifest
            )
            checkpoint_hash = (
                directory_sha256(checkpoint_dir) if hash_checkpoints else absence.ABSENT
            )
            if hash_checkpoints and absence.is_absent(checkpoint_hash):
                reasons.append(f"restored checkpoint is missing: {checkpoint_dir}")

    identity = {
        "row_id": row_id,
        "kind": kind,
        "seed": int(row["seed"]),
        "checkpoint_step": absence.cell(checkpoint_step),
        "chain": absence.cell(row.get("chain")),
        "chain_seed": absence.cell(row.get("chain_seed")),
        "run_id": absence.cell(_json_field(metadata, "run_id")),
        "plan_attempt_id": str(plan_attempt_id),
        "plan_hash": str(manifest["plan_hash"]),
        "launch_attempt_id": absence.cell(_json_field(row_record, "launch_attempt_id")),
        "job_id": absence.cell(_receipt_field(receipt, "job_id")),
        "hostname": absence.cell(_receipt_field(receipt, "hostname")),
        "requested_stratum": requested_stratum,
        "requested_constraint": str(row["resources"].get("constraint") or ""),
        "delivered_device": absence.cell(delivered_device),
        "partition": absence.cell(delivered_partition),
        "requested_partition": requested_partition,
        "config_sha256": absence.cell(file_sha256(run_dir / "resolved_config.yaml")),
        "checkpoint_dir": absence.cell(
            None if absence.is_absent(checkpoint_dir) else str(checkpoint_dir)
        ),
        "checkpoint_sha256": absence.cell(checkpoint_hash),
    }

    bound = resolve_metric_bindings(
        dict(metric_namespaces or {}),
        namespaced=namespaced,
        flat=flat,
        ambiguous=ambiguous,
        namespaces=namespaces,
        reasons=reasons,
    )

    metric_keys = list(dict.fromkeys([*ENERGY_METRIC_KEYS, *GATE_METRIC_KEYS, *extra_metric_keys]))
    # Gate metric keys are requested BY THE COLLECTOR on the gates' behalf, not
    # by a caller who named them. That distinction decides what a collision on
    # one of them means, and getting it wrong resurrects a measured regression:
    # at merged dev an unrequested collision failed all three smoke rows, and
    # the refusal to guess was deliberately rescoped to "the metrics actually
    # requested". A gate metric that collides and is NOT bound is therefore
    # ABSENT here -- its gate then reports `absent` with the collision retained
    # in the row's diagnostics -- rather than failing a row nobody asked to
    # gate. A caller's own `--metric-key` request keeps the strict refusal.
    #
    # This became load-bearing when the singlet-purity gates were added: their
    # metric names are shared with `full_model_antisymmetry`, so every gate
    # metric key they introduced collides on every eval row by construction.
    #
    # BUT TOLERATING THE COLLISION IS ITSELF THE DANGEROUS DIRECTION when the
    # caller declared a tolerance for that metric. A declared threshold is a
    # statement of intent to enforce, so an unresolved collision on it means THE
    # DECLARED CHECK DID NOT RUN -- and `absent` does not read as `failed`. That
    # is how the singlet-purity gates could silently become no-ops if a future
    # edit dropped their `metric_namespaces` binding: every gate would report
    # absent, every row would pass, and the physics check H-R5's namespace
    # mechanism exists to enable would quietly not happen.
    # So leniency is scoped to metrics NOBODY ARMED. An armed metric that cannot
    # be resolved fails the row and names the namespaces, as it always did.
    auto_requested = set(GATE_METRIC_KEYS) - set(extra_metric_keys)
    armed = set(gates.enforcing_metrics(gate_spec))
    metrics: dict[str, Any] = {}
    for key in metric_keys:
        namespace, name = split_metric_request(key)
        if namespace is None and name in bound:
            # An explicit binding answers the collision for the retained column
            # too. Reporting a bound name as unresolved while gating it would
            # put two different numbers under one heading.
            metrics[key] = absence.cell(bound[name])
            continue
        value, reason = resolve_metric_request(
            key,
            namespaced=namespaced,
            flat=flat,
            ambiguous=ambiguous,
            namespaces=namespaces,
        )
        tolerated = namespace is None and key in auto_requested and key not in armed
        if reason is not None and not tolerated:
            reasons.append(reason)
        metrics[key] = absence.cell(value)

    gate_rows: list[dict[str, Any]] = []
    if kind == "eval":
        for outcome in gates.evaluate_atom_gates(gate_metric_view(flat, bound), spec=gate_spec):
            gate_rows.append(
                {
                    "name": outcome.name,
                    "status": outcome.status,
                    "value": absence.cell(outcome.value),
                    "threshold": absence.cell(outcome.threshold),
                    "reason": outcome.reason,
                }
            )
        failed_gates = [gate["name"] for gate in gate_rows if gate["status"] == "fail"]
        if failed_gates:
            reasons.append(f"failed gates: {failed_gates}")

    run_status = _json_field(status, "status")
    if absence.is_absent(run_status):
        reasons.append("no run status.json: the row did not finish")
    elif str(run_status) != "completed":
        reasons.append(f"run status is {run_status!r}")

    return {
        "identity": identity,
        "status": "fail" if reasons else "pass",
        "reasons": reasons,
        "run_dir": str(run_dir),
        "result_dir": str(result_dir),
        "run_status": absence.cell(run_status),
        "metrics": metrics,
        "gates": gate_rows,
        "gate_counts": _gate_counts(gate_rows),
        # Diagnostic, not a verdict: every colliding name this row logged,
        # whether or not anything asked for it. Only a REQUESTED collision
        # fails the row, and it fails with its namespaces named.
        "ambiguous_metric_keys": sorted(ambiguous),
        "ambiguous_metric_namespaces": {key: list(ambiguous[key]) for key in sorted(ambiguous)},
        "logged_namespaces": sorted(namespaces),
        "namespaced_metric_count": len(namespaced),
        "config_identity_match": identity_diagnostics.get("config_identity_match"),
    }


def cross_check_checkpoint_identity(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail rows whose restored checkpoint disagrees across chains.

    Four chains over one checkpoint must have restored the same bytes. If two
    chains of the same (seed, step) hash differently, at least one number is
    attributed to a model that did not produce it.
    """

    by_checkpoint: dict[tuple[int, Any], dict[str, str]] = {}
    for row in rows:
        identity = row["identity"]
        if identity["kind"] != "eval":
            continue
        digest = absence.cell_value(identity["checkpoint_sha256"])
        if absence.is_absent(digest):
            continue
        key = (identity["seed"], absence.cell_value(identity["checkpoint_step"]))
        by_checkpoint.setdefault(key, {})[identity["row_id"]] = str(digest)
    for key, digests in by_checkpoint.items():
        distinct = set(digests.values())
        if len(distinct) > 1:
            for row in rows:
                identity = row["identity"]
                if identity["kind"] != "eval":
                    continue
                if (identity["seed"], absence.cell_value(identity["checkpoint_step"])) != key:
                    continue
                row["status"] = "fail"
                row["reasons"].append(
                    f"chains over seed={key[0]} step={key[1]} restored differing checkpoint "
                    f"hashes: {sorted(distinct)}"
                )


def require_unique_identities(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject two collected rows that claim the same identity."""

    seen: dict[tuple[Any, ...], str] = {}
    for row in rows:
        identity = row["identity"]
        key = (
            identity["row_id"],
            identity["plan_attempt_id"],
            absence.cell_value(identity["run_id"]),
        )
        if key in seen:
            raise CollectError(
                f"rows {seen[key]!r} and {identity['row_id']!r} share identity {key!r}"
            )
        seen[key] = str(identity["row_id"])


def summarize(rows: Sequence[Mapping[str, Any]], *, keys: Sequence[str]) -> dict[str, Any]:
    """Aggregate evaluation-row metrics, keeping coverage visible."""

    eval_rows = [row for row in rows if row["identity"]["kind"] == "eval"]
    summaries: dict[str, Any] = {}
    for key in keys:
        values = [absence.cell_value(row["metrics"].get(key, absence.cell(None))) for row in eval_rows]
        summaries[key] = absence.summarize_values(values).to_dict()
    return summaries


def _canary_completion(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[bool | None, list[str]]:
    """Check coverage against the plan's declared shape and row identities."""

    if manifest.get("canary_schema") != canary.CANARY_SCHEMA:
        return None, []

    declared_rows = list(manifest.get("rows", []))
    declared_ids = [str(row.get("row_id")) for row in declared_rows]
    actual_ids = [str(row.get("identity", {}).get("row_id")) for row in rows]
    reasons: list[str] = []
    declared_count = int(manifest.get("n_rows", len(declared_rows)))
    if declared_count != len(declared_rows):
        reasons.append(
            f"manifest declares {declared_count} rows but contains {len(declared_rows)}"
        )
    if len(rows) != declared_count:
        reasons.append(f"collected {len(rows)} rows but plan declares {declared_count}")

    expected_counts = Counter(declared_ids)
    actual_counts = Counter(actual_ids)
    missing = sorted(
        row_id
        for row_id, count in (expected_counts - actual_counts).items()
        for _ in range(count)
    )
    unexpected = sorted(
        row_id
        for row_id, count in (actual_counts - expected_counts).items()
        for _ in range(count)
    )
    if missing:
        reasons.append(f"missing rows={missing}")
    if unexpected:
        reasons.append(f"unexpected rows={unexpected}")

    failing = sorted(
        str(row.get("identity", {}).get("row_id"))
        for row in rows
        if row.get("status") != "pass"
    )
    if failing:
        reasons.append(f"failing rows={failing}")
    return not reasons, reasons


def collect(
    *,
    results_root: Path,
    plan_attempt_id: str,
    collect_attempt_id: str,
    gate_spec: Mapping[str, Any],
    gate_spec_source: str,
    metric_namespaces: Mapping[str, str] | None = None,
    extra_metric_keys: Sequence[str] = (),
    hash_checkpoints: bool = True,
    checkpoint_source_map: str | Path | None = None,
) -> dict[str, Any]:
    """Collect one plan attempt into a durable table.

    ``gate_spec`` may carry the reserved ``metric_namespaces`` binding block; it
    is split out here so the gates receive thresholds only. Bindings passed in
    ``metric_namespaces`` win over the ones declared in the spec, which is
    what lets a re-collect qualify a metric without editing a config.
    """

    manifest = plan_stage.read_manifest(results_root, plan_attempt_id)
    checkpoint_sources: Mapping[str, canary.CheckpointSource] | None = None
    if manifest.get("canary_schema") == canary.CANARY_SCHEMA:
        if checkpoint_source_map is None:
            raise CollectError("canary collection requires --checkpoint-source-map")
        try:
            checkpoint_sources = canary.reconcile_manifest_sources(
                manifest, checkpoint_source_map
            )
        except canary.CanaryError as exc:
            raise CollectError(f"immutable checkpoint sources are incomplete: {exc}") from exc
    thresholds, spec_bindings = split_gate_spec(gate_spec)
    bindings = {**spec_bindings, **dict(metric_namespaces or {})}
    rows = [
        collect_row(
            row,
            results_root=results_root,
            plan_attempt_id=plan_attempt_id,
            manifest=manifest,
            gate_spec=thresholds,
            metric_namespaces=bindings,
            extra_metric_keys=extra_metric_keys,
            hash_checkpoints=hash_checkpoints,
            checkpoint_sources=checkpoint_sources,
        )
        for row in manifest["rows"]
    ]
    require_unique_identities(rows)
    cross_check_checkpoint_identity(rows)
    metric_keys = list(dict.fromkeys([*ENERGY_METRIC_KEYS, *GATE_METRIC_KEYS, *extra_metric_keys]))
    require_requested_namespaces(
        rows,
        [
            *metric_keys,
            *(f"{namespace}{NAMESPACE_SEPARATOR}{metric}" for metric, namespace in bindings.items()),
        ],
    )
    canary_complete, canary_completion_reasons = _canary_completion(manifest, rows)
    collected = {
        "schema_version": plan_stage.SCHEMA_VERSION,
        "study": str(manifest["study"]),
        "plan_attempt_id": str(plan_attempt_id),
        "plan_hash": str(manifest["plan_hash"]),
        "collect_attempt_id": str(collect_attempt_id),
        "gate_spec": dict(thresholds),
        # Bindings are not tolerances: a spec that only says WHICH namespace a
        # metric comes from still declares no threshold, and every value gate
        # must keep reporting 'absent' with its observed value retained.
        "gate_spec_declared": bool(thresholds),
        "metric_namespaces": dict(bindings),
        "gate_spec_source": gate_spec_source,
        "checkpoint_hashing": bool(hash_checkpoints),
        "metric_keys": metric_keys,
        "n_rows": len(rows),
        "n_pass": sum(1 for row in rows if row["status"] == "pass"),
        "n_fail": sum(1 for row in rows if row["status"] == "fail"),
        "canary_complete": canary_complete,
        "canary_completion_reasons": canary_completion_reasons,
        "checkpoint_source_map_sha256": manifest.get("source_map_sha256"),
        "summaries": summarize(rows, keys=metric_keys),
        "rows": rows,
    }
    return collected


def write_collected(collected: Mapping[str, Any], *, results_root: Path) -> Path:
    """Write the collected table, its CSV, and the per-gate CSV."""

    attempt_id = str(collected["collect_attempt_id"])
    directory = layout.collect_attempt_dir(results_root, attempt_id)
    layout.write_json(directory / COLLECTED_FILENAME, dict(collected))
    _write_rows_csv(directory / ROWS_CSV, collected)
    _write_gates_csv(directory / GATES_CSV, collected)
    layout.write_latest(layout.stage_dir(results_root, layout.STAGE_COLLECT), attempt_id)
    return directory


def require_complete_canary_collection(collected: Mapping[str, Any]) -> None:
    """Fail closed unless every row declared by the plan reconciled completely."""

    if collected.get("canary_complete") is None:
        return
    if collected.get("canary_complete") is not True:
        reasons = list(collected.get("canary_completion_reasons", []))
        if not reasons:
            reasons = [
                "failing rows="
                + str(
                    [
                        row["identity"]["row_id"]
                        for row in collected.get("rows", [])
                        if row.get("status") != "pass"
                    ]
                )
            ]
        raise CollectError(
            "canary collection is incomplete; " + "; ".join(reasons)
        )


def _write_rows_csv(path: Path, collected: Mapping[str, Any]) -> None:
    metric_keys = list(collected["metric_keys"])
    fieldnames = [
        "row_id",
        "kind",
        "status",
        "seed",
        "checkpoint_step",
        "chain",
        "run_id",
        "job_id",
        "partition",
        "requested_partition",
        "requested_stratum",
        "requested_constraint",
        "delivered_device",
        "config_sha256",
        "config_identity_match",
        "checkpoint_sha256",
        "run_status",
        *metric_keys,
        "reasons",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in collected["rows"]:
            identity = row["identity"]
            record = {
                "row_id": identity["row_id"],
                "kind": identity["kind"],
                "status": row["status"],
                "seed": identity["seed"],
                "checkpoint_step": _render_cell(identity["checkpoint_step"]),
                "chain": _render_cell(identity["chain"]),
                "run_id": _render_cell(identity["run_id"]),
                "job_id": _render_cell(identity["job_id"]),
                "partition": _render_cell(identity["partition"]),
                "requested_partition": identity["requested_partition"],
                "requested_stratum": identity["requested_stratum"],
                "requested_constraint": identity["requested_constraint"],
                "delivered_device": _render_cell(identity["delivered_device"]),
                "config_sha256": _render_cell(identity["config_sha256"]),
                "config_identity_match": _render_cell(row.get("config_identity_match")),
                "checkpoint_sha256": _render_cell(identity["checkpoint_sha256"]),
                "run_status": _render_cell(row["run_status"]),
                "reasons": "; ".join(row["reasons"]) or "none",
            }
            for key in metric_keys:
                record[key] = _render_cell(row["metrics"].get(key, absence.cell(None)))
            writer.writerow(record)


def _write_gates_csv(path: Path, collected: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "gate", "status", "value", "threshold", "reason"],
        )
        writer.writeheader()
        for row in collected["rows"]:
            for gate_row in row["gates"]:
                writer.writerow(
                    {
                        "row_id": row["identity"]["row_id"],
                        "gate": gate_row["name"],
                        "status": gate_row["status"],
                        "value": _render_cell(gate_row["value"]),
                        "threshold": _render_cell(gate_row["threshold"]),
                        "reason": gate_row["reason"],
                    }
                )


def _render_cell(cell: Any) -> str:
    return absence.render(absence.cell_value(cell))


def _gate_counts(gate_rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"pass": 0, "fail": 0, "absent": 0}
    for gate_row in gate_rows:
        counts[str(gate_row["status"])] = counts.get(str(gate_row["status"]), 0) + 1
    return counts


def _read_json_or_absent(path: Path, reasons: list[str]) -> Any:
    if not path.is_file():
        reasons.append(f"missing artifact: {path.name}")
        return absence.ABSENT
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        reasons.append(f"unreadable artifact {path.name}: {exc}")
        return absence.ABSENT


def _json_field(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        return absence.present_or_absent(payload.get(key))
    return absence.ABSENT


def _receipt_field(receipt: Any, key: str) -> Any:
    return _json_field(receipt, key)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse collect command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, help="Durable study results root.")
    parser.add_argument("--plan-attempt-id", default=None, help="Plan attempt (defaults to latest).")
    parser.add_argument("--collect-attempt-id", default=None, help="Explicit collect attempt id.")
    parser.add_argument(
        "--gate-spec",
        default=None,
        help="Tolerance spec YAML; defaults to the spec recorded in the plan manifest.",
    )
    parser.add_argument(
        "--metric-key",
        action="append",
        default=[],
        help=(
            "Additional metric key to retain per row; repeatable. Qualify a name that "
            "several tasks emit as '<namespace>.<key>', e.g. "
            "'eval/spatial_exchange_symmetry.triplet_fraction_mean_under_psi_orig_sq'. "
            "An unqualified colliding name fails the row and names its namespaces."
        ),
    )
    parser.add_argument(
        "--metric-namespace",
        action="append",
        default=[],
        metavar="METRIC=NAMESPACE",
        help=(
            "Bind one bare metric name to the single namespace it is read from, for its "
            "retained column and for the gates alike; repeatable. Overrides the gate "
            "spec's 'metric_namespaces' block."
        ),
    )
    parser.add_argument(
        "--skip-checkpoint-hash",
        action="store_true",
        help="Skip checkpoint hashing; the hash then renders as absent, never as matching.",
    )
    parser.add_argument(
        "--checkpoint-source-map",
        default=None,
        help="External immutable source map required by a canary plan.",
    )
    return parser.parse_args(argv)


def parse_metric_namespace_arguments(entries: Sequence[str]) -> dict[str, str]:
    """Parse ``METRIC=NAMESPACE`` bindings from the command line."""

    bindings: dict[str, str] = {}
    for entry in entries:
        metric, separator, namespace = str(entry).partition("=")
        if not separator or not metric.strip() or not namespace.strip():
            raise CollectError(
                f"--metric-namespace expects 'METRIC=NAMESPACE', got {entry!r}"
            )
        bindings[metric.strip()] = namespace.strip().strip("/")
    return bindings


def main(argv: Sequence[str] | None = None) -> int:
    """Collect one plan attempt."""

    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    plan_attempt_id = layout.resolve_attempt_id(results_root, layout.STAGE_PLAN, args.plan_attempt_id)
    manifest = plan_stage.read_manifest(results_root, plan_attempt_id)
    gate_spec = load_gate_spec(args.gate_spec, manifest=manifest)
    collect_attempt_id = args.collect_attempt_id or plan_stage.now_attempt_id()
    try:
        collected = collect(
            results_root=results_root,
            plan_attempt_id=plan_attempt_id,
            collect_attempt_id=collect_attempt_id,
            gate_spec=gate_spec,
            gate_spec_source=str(args.gate_spec) if args.gate_spec else "plan_manifest",
            metric_namespaces=parse_metric_namespace_arguments(args.metric_namespace),
            extra_metric_keys=args.metric_key,
            hash_checkpoints=not args.skip_checkpoint_hash,
            checkpoint_source_map=args.checkpoint_source_map,
        )
        directory = write_collected(collected, results_root=results_root)
        require_complete_canary_collection(collected)
    except Exception as exc:
        failure_dir = layout.collect_attempt_dir(results_root, collect_attempt_id)
        layout.write_json(
            failure_dir / "failure.json",
            {
                "schema": "he-v1-collect-failure/v1",
                "plan_attempt_id": plan_attempt_id,
                "collect_attempt_id": collect_attempt_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    print(
        f"[he-v1] collected {collected['n_rows']} rows "
        f"({collected['n_pass']} pass, {collected['n_fail']} fail) into {directory}"
    )
    for metric, namespace in sorted(collected["metric_namespaces"].items()):
        print(f"[he-v1] metric {metric!r} is read from namespace {namespace!r} only")
    if not collected["gate_spec_declared"]:
        print(
            "[he-v1] no tolerances declared: value gates report 'absent' with observed "
            "values retained (thresholds are predeclared in H-F1)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
