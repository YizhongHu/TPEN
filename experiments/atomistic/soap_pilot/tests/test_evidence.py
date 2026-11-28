"""Consumer-side JSONL and optimizer-state evidence controls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.atomistic.soap_pilot.checks import Verdict
from experiments.atomistic.soap_pilot.evidence import (
    EvidenceFormatError,
    MetricRecord,
    check_metric_records,
    load_metric_records,
)
from experiments.atomistic.soap_pilot.checks import check_disconnected_optimizer_state
from tests.helpers.run_context import make_run_context


def test_loader_reads_only_declared_metrics_file_and_preserves_namespaces(tmp_path: Path) -> None:
    (tmp_path / "unrelated.jsonl").write_text("not evidence\n", encoding="utf-8")
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"step": 100, "namespace": "optimizer", "metrics": {"loss": 1.0}}),
                json.dumps({"step": 100, "namespace": "timing", "metrics": {"elapsed": 2.0}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_metric_records(tmp_path) == (
        MetricRecord(100, "optimizer", {"loss": 1.0}),
        MetricRecord(100, "timing", {"elapsed": 2.0}),
    )
    assert check_metric_records(load_metric_records(tmp_path)).passed


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        json.dumps({"step": True, "namespace": "run", "metrics": {}}),
        json.dumps({"step": 1, "namespace": "run", "metrics": []}),
    ],
)
def test_loader_rejects_malformed_envelopes(tmp_path: Path, payload: str) -> None:
    (tmp_path / "metrics.jsonl").write_text(payload + "\n", encoding="utf-8")
    with pytest.raises(EvidenceFormatError):
        load_metric_records(tmp_path)


def test_loader_rejects_truncated_last_jsonl_record(tmp_path: Path) -> None:
    """A partial final line is format loss, not an empty evidence set."""

    (tmp_path / "metrics.jsonl").write_text(
        '{"step": 1, "namespace": "run", "metrics": {"loss": 1.0}}\n'
        '{"step": 2, "namespace": "run", "metrics": {"loss": ',
        encoding="utf-8",
    )
    with pytest.raises(EvidenceFormatError, match="invalid JSON"):
        load_metric_records(tmp_path)


def test_missing_or_empty_metrics_are_unmeasured(tmp_path: Path) -> None:
    assert check_metric_records(load_metric_records(tmp_path)).verdict is Verdict.UNMEASURED
    (tmp_path / "metrics.jsonl").write_text("", encoding="utf-8")
    assert check_metric_records(load_metric_records(tmp_path)).verdict is Verdict.UNMEASURED


def test_nonfinite_jsonl_metric_fails_the_finite_only_consumer_check(tmp_path: Path) -> None:
    (tmp_path / "metrics.jsonl").write_text(
        '{"step": 1, "namespace": "run", "metrics": {"loss": NaN}}\n',
        encoding="utf-8",
    )
    result = check_metric_records(load_metric_records(tmp_path))
    assert result.verdict is Verdict.PRODUCT_FAILURE
    assert "non-finite" in result.detail


@pytest.mark.parametrize(
    "value", [{"nested": {"loss": float("inf")}}, {"nested": {"inner": {"loss": float("-inf")}}}]
)
def test_nested_nonfinite_jsonl_metric_fails_the_finite_only_consumer_check(
    tmp_path: Path, value: object
) -> None:
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "namespace": "run", "metrics": value}, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    result = check_metric_records(load_metric_records(tmp_path))
    assert result.verdict is Verdict.PRODUCT_FAILURE
    assert "non-finite" in result.detail


def test_optimizer_state_aggregates_rendered_groups_at_terminal_step() -> None:
    records = [
        MetricRecord(20, "train/optimizer_state/embedding", {"n_no_optimizer_state": 99}),
        MetricRecord(100, "train/optimizer_state/embedding", {"n_no_optimizer_state": 8}),
        MetricRecord(100, "train/optimizer_state/readout", {"n_no_optimizer_state": 8, "n_no_optimizer_state_elements": 492000}),
    ]
    groups = ("embedding", "readout")
    assert check_disconnected_optimizer_state(records, groups=groups, terminal_step=100).passed

    wrong = records[:-1] + [MetricRecord(100, "train/optimizer_state/readout", {"n_no_optimizer_state": 7})]
    assert check_disconnected_optimizer_state(wrong, groups=groups, terminal_step=100).verdict is Verdict.PRODUCT_FAILURE

    missing = records[:2]
    assert check_disconnected_optimizer_state(missing, groups=groups, terminal_step=100).verdict is Verdict.UNMEASURED


def test_configured_overlay_logger_round_trips_into_consumer(tmp_path: Path) -> None:
    """The overlay logger, not composer output, is the consumer boundary."""

    repo = Path(__file__).parents[4]
    overlay_path = repo / "experiments/atomistic/stage1/configs/overlays/hooke-diagnostics-v2.yaml"
    overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    logger_config = next(item for item in overlay["loggers"] if item["_target_"] == "tpen.logging.JSONL")
    run_dir = tmp_path / "run"
    logger_config = {**logger_config, "path": str(run_dir / "metrics.jsonl")}
    logger = instantiate(OmegaConf.create(logger_config))

    context = make_run_context(tmp_path / "context", loggers=[logger], run_id="overlay-logger")
    context.log({"energy": 1.25}, step=7, namespace="optimizer")
    context.log({"loss": 0.5}, step=7, namespace="timing")

    records = load_metric_records(run_dir)
    assert records == (
        MetricRecord(7, "optimizer", {"energy": 1.25}),
        MetricRecord(7, "timing", {"loss": 0.5}),
    )
    assert check_metric_records(records).passed
