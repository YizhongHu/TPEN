"""Contract tests for the append-only trajectory-statistics sidecar."""

import json

import pytest

from tpen.statistics.receipt import (
    TrajectoryStatisticsIdentity,
    TrajectoryStatisticsReceipt,
)
from pathlib import Path

from tpen.statistics.sidecar import DuplicateReceiptError, TrajectoryStatisticsSidecar

from .test_receipt import _receipt


def _identity(**overrides: str) -> TrajectoryStatisticsIdentity:
    """Build a valid identity for sidecar join tests."""

    return TrajectoryStatisticsIdentity(
        stage=overrides.get("stage", "eval"),
        run_id=overrides.get("run_id", "run-1"),
        attempt_id=overrides.get("attempt_id", "attempt-1"),
        checkpoint_sha256=overrides.get("checkpoint_sha256", "a" * 64),
        config_sha256=overrides.get("config_sha256", "b" * 64),
        observable=overrides.get("observable", "local_energy"),
        evaluator_id=overrides.get("evaluator_id", "local_energy/v1"),
    )


def _stored_receipt(**identity_overrides: str) -> TrajectoryStatisticsReceipt:
    """Build a receipt whose identity can be varied one field at a time."""

    return _receipt(identity=_identity(**identity_overrides))


def test_missing_sidecar_reads_empty_without_creating_file(tmp_path) -> None:
    """Read-only inspection must not create a durable sidecar."""

    path = tmp_path / "nested" / "statistics.jsonl"
    sidecar = TrajectoryStatisticsSidecar(path)

    assert sidecar.read() == ()
    assert len(sidecar) == 0
    assert not path.exists()


def test_append_round_trips_and_creates_parent(tmp_path) -> None:
    """The first append creates the requested parent directory on demand."""

    path = tmp_path / "nested" / "statistics.jsonl"
    receipt = _stored_receipt()
    sidecar = TrajectoryStatisticsSidecar(path)

    sidecar.append(receipt)

    assert path.parent.is_dir()
    assert sidecar.read() == (receipt,)


def test_multiple_appends_preserve_file_order(tmp_path) -> None:
    """JSONL order is the publication order and must remain observable."""

    sidecar = TrajectoryStatisticsSidecar(tmp_path / "statistics.jsonl")
    receipts = tuple(_stored_receipt(run_id=f"run-{index}") for index in range(3))

    for receipt in receipts:
        sidecar.append(receipt)

    assert sidecar.read() == receipts


def test_get_joins_on_all_seven_identity_fields(tmp_path) -> None:
    """Observable and evaluator changes cannot collide with an existing join key."""

    sidecar = TrajectoryStatisticsSidecar(tmp_path / "statistics.jsonl")
    receipts = (
        _stored_receipt(),
        _stored_receipt(stage="train"),
        _stored_receipt(run_id="run-2"),
        _stored_receipt(attempt_id="attempt-2"),
        _stored_receipt(checkpoint_sha256="c" * 64),
        _stored_receipt(config_sha256="d" * 64),
        _stored_receipt(observable="gradient_norm"),
        _stored_receipt(evaluator_id="local_energy/v2"),
    )
    sidecar.extend(receipts)

    for receipt in receipts:
        assert sidecar.get(receipt.identity) == receipt
    assert sidecar.get(_identity(observable="other_observable")) is None


def test_duplicate_append_leaves_file_byte_identical(tmp_path) -> None:
    """Duplicate rejection must happen before opening the append handle."""

    path = tmp_path / "statistics.jsonl"
    sidecar = TrajectoryStatisticsSidecar(path)
    receipt = _stored_receipt()
    sidecar.append(receipt)
    before = path.read_bytes()

    with pytest.raises(DuplicateReceiptError):
        sidecar.append(receipt)

    assert path.read_bytes() == before
    assert len(path.read_bytes()) == len(before)


def test_extend_duplicate_inside_batch_writes_nothing(tmp_path) -> None:
    """Whole-batch validation prevents a new prefix from leaking on failure."""

    path = tmp_path / "statistics.jsonl"
    sidecar = TrajectoryStatisticsSidecar(path)
    existing = _stored_receipt()
    sidecar.append(existing)
    before = path.read_bytes()
    new_receipt = _stored_receipt(run_id="run-new")

    with pytest.raises(DuplicateReceiptError):
        sidecar.extend((new_receipt, new_receipt))

    assert path.read_bytes() == before


def test_extend_empty_is_noop_without_creating_file(tmp_path) -> None:
    """An empty publication batch has no filesystem side effect."""

    path = tmp_path / "missing" / "statistics.jsonl"
    TrajectoryStatisticsSidecar(path).extend(())

    assert not path.exists()


def test_evaluator_revision_is_a_distinct_publishable_identity(tmp_path) -> None:
    """Bumping evaluator_id is the documented way to publish a revision."""

    sidecar = TrajectoryStatisticsSidecar(tmp_path / "statistics.jsonl")
    original = _stored_receipt()
    revised = _stored_receipt(evaluator_id="local_energy/v2")

    sidecar.append(original)
    sidecar.append(revised)

    assert sidecar.read() == (original, revised)


@pytest.mark.parametrize(
    "contents,line_number",
    (
        ("\nnot-json\n", 2),
        (json.dumps(_stored_receipt().to_dict()) + "\n" + json.dumps({**_stored_receipt().to_dict(), "status": "available", "statistics": None}) + "\n", 2),
    ),
    ids=("invalid-json", "invalid-receipt"),
)
def test_read_reports_path_and_line_for_corrupt_records(tmp_path, contents: str, line_number: int) -> None:
    """Corruption is loud and attributable to its exact JSONL location."""

    path = tmp_path / "statistics.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{path}:{line_number}"):
        TrajectoryStatisticsSidecar(path).read()


def test_read_skips_blank_lines(tmp_path) -> None:
    """Human-added blank separators are harmless in an otherwise valid file."""

    path = tmp_path / "statistics.jsonl"
    receipt = _stored_receipt()
    path.write_text("\n" + json.dumps(receipt.to_dict(), sort_keys=True) + "\n\n", encoding="utf-8")

    assert TrajectoryStatisticsSidecar(path).read() == (receipt,)


def test_jsonl_keys_are_sorted_and_views_agree_with_read(tmp_path) -> None:
    """The wire format is canonical and all collection views share one read."""

    sidecar = TrajectoryStatisticsSidecar(tmp_path / "statistics.jsonl")
    receipts = (_stored_receipt(), _stored_receipt(run_id="run-2"))
    sidecar.extend(receipts)

    lines = sidecar.path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert all(isinstance(record, dict) for record in records)
    assert all(list(record) == sorted(record) for record in records)
    assert sidecar.identities() == tuple(receipt.identity.as_key() for receipt in sidecar.read())
    assert tuple(sidecar) == sidecar.read()
    assert len(sidecar) == len(sidecar.read())


def test_append_does_not_join_onto_an_unterminated_last_line(tmp_path) -> None:
    """The sidecar is structurally the checkpoint publication catalog.

    Append-only JSONL; :meth:`read` raises on a malformed row; :meth:`extend`
    reads the file before it writes. So ONE torn row blocks every later append
    rather than merely losing itself -- the same amplification that makes the
    publication catalog the most consequential instance of this exposure, but
    with no typed diagnosis and no repair recipe.

    This writer was very nearly left tearing: the census that scoped the
    torn-append fix classified it as a batch writer from the shape of the loop
    in ``extend``, when its production call path is ``append`` ->
    ``extend((receipt,))`` -- one record per open.
    """

    path = tmp_path / "trajectory_statistics.jsonl"
    path.write_text('{"torn": ', encoding="utf-8")
    sidecar = TrajectoryStatisticsSidecar(path)

    with pytest.raises(ValueError):
        sidecar.read()

    # This exercises the PRIMITIVE directly, not ``sidecar.append``, and that
    # is forced rather than chosen: ``extend`` calls ``identities()`` ->
    # ``read()`` before writing, so on a torn file ``append`` raises and can
    # never reach its write. That fail-loud behaviour is the point. The
    # consequence is that this test alone cannot detect a sidecar that stopped
    # routing through the primitive -- see
    # ``test_append_routes_one_receipt_through_the_shared_primitive``, which
    # pins the mechanism.
    sidecar.path.write_text('{"torn": ', encoding="utf-8")
    receipt = _stored_receipt()
    from tpen.durable_append import append_record
    append_record(path, json.dumps(receipt.to_dict(), sort_keys=True))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"torn": ', "the torn bytes must keep their own line"
    # Exact round trip: the line was produced by json.dumps of this very
    # mapping, so comparing the whole dict depends on no key name.
    assert json.loads(lines[1]) == receipt.to_dict()


def test_append_routes_one_receipt_through_the_shared_primitive(
    monkeypatch, tmp_path
) -> None:
    """Mechanism pin: ``append`` must reach ``append_record`` exactly once.

    Without this, a sidecar that went back to opening its own append handle
    would keep every behavioural test in this file green, because on a clean
    file a hand-rolled body-then-newline loop produces identical bytes. The
    torn-tail test above cannot catch it either, since ``append`` raises on a
    torn file before it writes. Measured: a mutant using ``open(mode="a")``
    bypassed the primitive with the whole suite still green.
    """

    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        "tpen.statistics.sidecar.append_record",
        lambda path, record: calls.append((Path(path), record)),
    )
    receipt = _stored_receipt()
    path = tmp_path / "trajectory_statistics.jsonl"

    TrajectoryStatisticsSidecar(path).append(receipt)

    assert calls == [(path, json.dumps(receipt.to_dict(), sort_keys=True))]


def test_a_receipt_appended_after_a_clean_row_adds_no_blank_line(tmp_path) -> None:
    """Routing through the primitive must not perturb the normal path."""

    sidecar = TrajectoryStatisticsSidecar(tmp_path / "trajectory_statistics.jsonl")
    sidecar.append(_stored_receipt())
    sidecar.append(_stored_receipt(evaluator_id="local_energy/v2"))

    text = sidecar.path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\n\n" not in text
    assert len(sidecar.read()) == 2


def test_the_sidecar_emits_ascii_only_bytes(tmp_path) -> None:
    """The sidecar calls ``json.dumps`` itself, so it needs its own pin.

    ``tpen.durable_append``'s docstring states the ASCII invariant is pinned by
    a test. That test lives in the primitive's own (torch-free) file and can
    only reach the writers importable there -- MEASURED: setting
    ``ensure_ascii=False`` here left the entire 2154-test suite green, while the
    same mutation in ``artifacts.py`` was killed. The instrument worked; the
    coverage did not reach.

    This matters most here of all six writers: the sidecar writes
    ``trajectory_statistics.jsonl``, which ``experiments/atomistic/he-v1/
    collect.py:926`` reads as UTF-8 text. A torn write splitting a multi-byte
    character raises ``UnicodeDecodeError`` out of the file iteration itself,
    bypassing the torn-row diagnosis entirely.
    """

    path = tmp_path / "trajectory_statistics.jsonl"

    TrajectoryStatisticsSidecar(path).append(_stored_receipt(run_id="run-\u00e9\u4e2d"))

    raw = path.read_bytes()
    assert raw.isascii(), f"sidecar emitted non-ASCII bytes: {raw!r}"
