"""Pin the P10 smoke summary to the executor's child-observation writer."""

from __future__ import annotations

from pathlib import Path

from tpen_executor.runners import CHILD_OBSERVATION_FILENAME, RUN_SHIM


TEMPLATE = Path(__file__).with_name("templates") / "p10-smoke.pbs"


def test_template_reads_product_child_observation_from_attempt_root() -> None:
    """The smoke summary must follow the executor writer's output location."""

    text = TEMPLATE.read_text(encoding="utf-8")
    assert CHILD_OBSERVATION_FILENAME in text
    assert "attempt_root = Path(attempt[\"output_dir\"])" in text
    assert "attempt_root / child_observation_filename" in text
    assert "attempt_root / \"core\"" not in text


def test_run_shim_uses_the_same_product_filename_constant() -> None:
    """The child writer and template must share one filename authority."""

    assert f"Path(run_roots[-1], {CHILD_OBSERVATION_FILENAME!r})" in RUN_SHIM
