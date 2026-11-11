"""Torch-free tests for the single-write, torn-tail-safe append primitive.

The defect these pin: ``append_jsonl`` could leave an unterminated line that the
*next* append concatenated onto -- destroying a later, entirely valid record
rather than merely losing the interrupted one.

The CAUSE is size-dependent, and the original diagnosis was corrected by
measurement. For a record above the 8 KiB text buffer the two writes really do
flush separately (measured: 40,014 of 40,015 bytes on disk before the newline),
which is the ``occurrences.jsonl`` case. For a smaller record -- a catalog row,
a receipt row, most log lines -- both writes coalesce into a single flush, so
the two-call structure was NOT the trigger there; the exposure is an error or
process termination DURING that coalesced flush, after only a prefix was
stored. It is NOT a short write on its own: the old text-mode path wrapped a
``BufferedWriter``, which retries a short raw count (measured: raw calls
``[9, 1]``, final file complete), as this module's own docstring says. Do not
repeat the two-separate-writes story about small records: it is measurably
wrong for them.

Two things make a test here easy to get wrong, so they are stated up front:

* **Appending twice proves nothing.** The failure is a crash *between* writes.
  Every test below either constructs the torn state on disk explicitly or
  induces a real short write, then appends through the production path.
* **The single-write property is a syscall count, not a file size.** An
  on-disk-size measurement observes flush visibility and cannot distinguish one
  write from two that coalesced in a buffer. It is pinned here with a handle
  that counts ``write`` calls.
"""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import pytest

from tpen.artifacts import append_jsonl
from tpen.distributed import (
    ExecutionTopology,
    ProfileRecord,
    ProfileScope,
    RankLocalJSONLWriter,
)
from tpen.durable_append import PartialAppendError, append_record, ends_without_newline
from tpen.logging.base import LogRecord
from tpen.logging.jsonl import JSONL
from tpen.process_resources import ProcessResourceResult

REPO_ROOT = Path(__file__).resolve().parents[2]


class _RecordingHandle:
    """Wrap a binary handle and record the size of every ``write`` call."""

    def __init__(self, inner, calls: list[int], *, truncate_by: int = 0) -> None:
        self._inner = inner
        self._calls = calls
        self._truncate_by = truncate_by

    def write(self, data: bytes) -> int:
        self._calls.append(len(data))
        if self._truncate_by:
            # A real short write: the kernel stores a prefix and reports it.
            return self._inner.write(data[: -self._truncate_by])
        return self._inner.write(data)

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._inner.__exit__(*exc_info)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture
def write_calls(monkeypatch: pytest.MonkeyPatch):
    """Count writes on binary-append handles only, leaving reads untouched."""

    def _install(*, truncate_by: int = 0) -> list[int]:
        calls: list[int] = []
        real_open = Path.open

        def spy(self, mode="r", *args, **kwargs):
            handle = real_open(self, mode, *args, **kwargs)
            if "a" in mode and "b" in mode:
                return _RecordingHandle(handle, calls, truncate_by=truncate_by)
            return handle

        monkeypatch.setattr(Path, "open", spy)
        return calls

    return _install


# --- I1: exactly one write call per record -----------------------------------


def test_a_record_reaches_the_kernel_in_exactly_one_write(write_calls, tmp_path: Path) -> None:
    calls = write_calls()
    path = tmp_path / "log.jsonl"

    append_record(path, '{"a": 1}')

    assert calls == [len(b'{"a": 1}\n')], "record and newline must be one write, not two"


def test_a_record_larger_than_the_text_buffer_is_still_one_write(
    write_calls, tmp_path: Path
) -> None:
    # Over 8 KiB is where the old text-mode path measurably flushed mid-record.
    calls = write_calls()
    path = tmp_path / "big.jsonl"
    record = json.dumps({"blob": "x" * 40_000}, sort_keys=True)

    append_record(path, record)

    assert calls == [len(record.encode()) + 1]
    assert path.read_text(encoding="utf-8") == record + "\n"


def test_closing_out_a_torn_line_does_not_cost_a_second_write(
    write_calls, tmp_path: Path
) -> None:
    """The repair newline rides in the same write, not a preceding one.

    Terminating the torn line separately would reintroduce the very two-write
    window the primitive exists to remove.
    """

    path = tmp_path / "log.jsonl"
    path.write_text('{"a": 1}\n{"torn": ', encoding="utf-8")
    calls = write_calls()

    append_record(path, '{"b": 2}')

    assert calls == [len(b'\n{"b": 2}\n')], "repair newline must ride in the same write"


# --- I2: a torn tail never merges with the next record -----------------------


def test_a_new_record_does_not_join_onto_an_unterminated_line(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text('{"a": 1}\n{"torn": ', encoding="utf-8")

    append_record(path, '{"b": 2}')

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"a": 1}', '{"torn": ', '{"b": 2}']
    assert not any("torn" in line and '"b"' in line for line in lines), (
        "the torn bytes and the new record must never share a line"
    )
    assert json.loads(lines[-1]) == {"b": 2}, "the new record must survive intact"


def test_the_torn_state_produced_by_a_real_short_write_also_does_not_merge(
    write_calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Construct the torn state through the actual failure, not by hand.

    Dropping the final byte of the write drops exactly the terminator, which is
    the shape a crash between flush and completion leaves behind.
    """

    path = tmp_path / "log.jsonl"
    append_record(path, '{"a": 1}')
    write_calls(truncate_by=1)

    with pytest.raises(PartialAppendError):
        append_record(path, '{"b": 2}')

    assert ends_without_newline(path), "the failed append must leave an unterminated line"

    # Lift the induced failure: the next append is the production path recovering.
    monkeypatch.undo()
    append_record(path, '{"c": 3}')

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"a": 1}', '{"b": 2}', '{"c": 3}']
    assert json.loads(lines[-1]) == {"c": 3}


# --- I3: a short write is detected, never silent -----------------------------


def test_a_short_write_raises_instead_of_reporting_success(
    write_calls, tmp_path: Path
) -> None:
    path = tmp_path / "log.jsonl"
    write_calls(truncate_by=3)

    with pytest.raises(PartialAppendError) as caught:
        append_record(path, '{"a": 1}')

    assert "partial append" in str(caught.value)
    assert isinstance(caught.value, OSError), "callers already handle OSError here"


# --- I5: the guard is inert on well-formed files -----------------------------


def test_appending_to_a_terminated_file_inserts_no_blank_line(tmp_path: Path) -> None:
    """The over-restrictive direction: a guard that always repairs corrupts."""

    path = tmp_path / "log.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")

    append_record(path, '{"b": 2}')

    assert path.read_text(encoding="utf-8") == '{"a": 1}\n{"b": 2}\n'


def test_a_fresh_file_gets_no_leading_newline(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "log.jsonl"

    append_record(path, '{"a": 1}')

    assert path.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_ends_without_newline_reports_false_for_absent_and_empty_files(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.jsonl"
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")

    assert ends_without_newline(missing) is False
    assert ends_without_newline(empty) is False


# --- one record is one line --------------------------------------------------


def test_a_record_containing_a_newline_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one record is one line"):
        append_record(tmp_path / "log.jsonl", '{"a": 1}\n{"b": 2}')


def test_a_json_escaped_newline_is_accepted(tmp_path: Path) -> None:
    """The over-restrictive direction: escaped newlines are ordinary content."""

    path = tmp_path / "log.jsonl"
    record = json.dumps({"text": "first\nsecond"}, sort_keys=True)
    assert "\n" not in record

    append_record(path, record)

    assert json.loads(path.read_text(encoding="utf-8")) == {"text": "first\nsecond"}


# --- the writers that route through the primitive ----------------------------


def _torn(path: Path) -> None:
    path.write_text('{"a": 1}\n{"torn": ', encoding="utf-8")


def test_append_jsonl_does_not_join_onto_an_unterminated_line(tmp_path: Path) -> None:
    path = tmp_path / "publications.jsonl"
    _torn(path)

    append_jsonl(path, {"b": 2})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[-1] == '{"b": 2}'
    assert lines[-2] == '{"torn": '


def test_jsonl_logger_does_not_join_onto_an_unterminated_line(tmp_path: Path) -> None:
    """`tpen/logging/jsonl.py` reimplemented the idiom under its own name.

    "``append_jsonl`` repaired while ``logging/jsonl.py`` still tears" is the
    named failure mode for this slice; this is the test that would catch it.
    """

    path = tmp_path / "metrics.jsonl"
    _torn(path)

    JSONL(path).log(LogRecord(step=1, namespace="train", metrics={"loss": 0.5}))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[-2] == '{"torn": '
    assert json.loads(lines[-1])["metrics"] == {"loss": 0.5}


# --- census, re-run at this head as an anti-regression guard ------------------

MODE_CHARS = set("rwxab+t")

# The one text-mode append writer deliberately NOT routed through the primitive,
# tracked as item bc8925a8 rather than left silent.
#
# csv.py is a GENUINE many-rows-per-open writer: one open emits a header plus
# one line per scalar metric, so covering it needs a batch shape whose failure
# mode nobody has measured. That is the whole reason it is out.
#
# statistics/sidecar.py was originally excluded alongside it for the same stated
# reason, and that reason was WRONG. It is batch-CAPABLE but its production call
# path is ``append`` -> ``extend((receipt,))`` -- ONE record per open. It is now
# routed. See the instrument defect noted below.
KNOWN_BATCH_WRITERS = {
    Path("tpen/logging/csv.py"),
}

# Modules that legitimately open an append handle because they ARE the
# primitive, rather than because they bypassed it.
PRIMITIVE_MODULES = {
    Path("tpen/durable_append.py"),
}


def _opens_append_handle(source: Path) -> bool:
    """Report whether `source` opens a file in an append mode, via AST.

    Deliberately NOT a regex. The original instrument matched only a positional
    string literal immediately inside ``open(``, so ``open(mode="a")`` evaded it
    -- measured: an unsafe sidecar mutant using that spelling bypassed the
    primitive while this test stayed green.

    Two further evasions were then found against the AST version and are closed
    here: a starred literal argument (``Path.open(*("a",), ...)``) arrives as
    ``ast.Starred`` rather than ``ast.Constant``, and ``from os import O_APPEND
    as APPEND`` defeats a name-only check. Both escalate to "append", which is
    the conservative direction: a false positive is a test failure someone
    reads, a false negative is an unsafe writer certified safe.

    NOT every unresolvable mode escalates, and an earlier version of this
    docstring wrongly claimed it did. Only ``ast.Starred`` args and ``**kwargs``
    do. ``open(p, MODE)`` with a NAMED mode resolves to no constant and returns
    False -- measured. That is a gap, not a policy, and it is the same gap the
    limit paragraph below describes.

    KNOWN LIMIT, stated rather than iterated on. **A static census is a guard
    against ACCIDENT, not against a determined bypass.** A mode assembled at
    runtime, an append handle obtained from a helper this function cannot see
    through, or a writer outside ``tpen/`` all defeat it, and closing each new
    spelling as it is found is an infinite regress. It exists so that someone
    reintroducing the idiom by habit is stopped, and it should not be mistaken
    for proof that no bypass exists.
    """

    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        # os.O_APPEND, a bare O_APPEND, or any aliased import of it.
        if isinstance(node, ast.Attribute) and node.attr == "O_APPEND":
            return True
        if isinstance(node, ast.Name) and node.id == "O_APPEND":
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            if any(alias.name == "O_APPEND" for alias in node.names):
                return True
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_open = (isinstance(func, ast.Name) and func.id == "open") or (
            isinstance(func, ast.Attribute) and func.attr == "open"
        )
        if not is_open:
            continue
        # Unresolvable argument shapes are treated as append, not as clean.
        if any(isinstance(a, ast.Starred) for a in node.args):
            return True
        if any(kw.arg is None for kw in node.keywords):
            return True
        candidates = [a for a in node.args if isinstance(a, ast.Constant)]
        candidates += [
            kw.value
            for kw in node.keywords
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant)
        ]
        for const in candidates:
            value = const.value
            if (
                isinstance(value, str)
                and value
                and set(value) <= MODE_CHARS
                and "a" in value
            ):
                return True
    return False


def test_no_one_record_writer_opens_its_own_append_handle() -> None:
    """Behaviour-scoped census: the property is a write pattern, not a name.

    A name-scoped search for callers of ``append_jsonl`` reports three sites and
    misses the modules that reimplement the idiom -- which is how the original
    census undercounted seven writers as three. This parses for the *pattern*
    instead, so a newly added append writer anywhere in ``tpen/`` fails this
    test and must be classified.

    KNOWN INSTRUMENT DEFECTS, recorded because the next census will be run by
    someone who was not here, and this instrument has now been wrong twice:

    1. **Classify by the CALL, not the definition.** A reader who classifies a
       site as one-record or batch from the shape of the write LOOP at the
       DEFINITION site gets it wrong whenever a batch-capable writer is only
       ever CALLED one record at a time. That is exactly what happened to
       ``statistics/sidecar.py``, excluded as a batch writer when its sole
       production caller passes a single receipt. Check callers first.
    2. **A regex over source text encodes a spelling, not a property.** The
       previous version of this test matched only positional ``open("a")``.
       ``open(mode="a")`` evaded it, and an unsafe mutant using that spelling
       kept this test green. Hence the AST.
    """

    found = {
        source.relative_to(REPO_ROOT)
        for source in (REPO_ROOT / "tpen").rglob("*.py")
        if _opens_append_handle(source)
    }

    assert found == KNOWN_BATCH_WRITERS | PRIMITIVE_MODULES, (
        "append-handle owners changed; classify each as one-record (route "
        f"through append_record) or batch. Unexpected: "
        f"{found - KNOWN_BATCH_WRITERS - PRIMITIVE_MODULES}, missing: "
        f"{(KNOWN_BATCH_WRITERS | PRIMITIVE_MODULES) - found}"
    )


def test_the_removed_trailing_newline_helper_has_exactly_one_replacement() -> None:
    """One invariant, one mechanism -- two would be free to drift apart."""

    receipt = (REPO_ROOT / "tpen" / "checkpoint" / "receipt.py").read_text(encoding="utf-8")

    assert "def _ensure_trailing_newline" not in receipt


# --- buffering=0 is a load-bearing token, and nothing above pins it ----------


class _DropsOneByteOnce(io.RawIOBase):
    """A raw handle whose FIRST write stores one byte fewer than asked.

    This is the shape POSIX permits for a regular file: a partial store with a
    short count returned, and no error.  Every later write is honoured in full,
    so a layer that RETRIES the remainder completes the record -- which is
    precisely the behaviour these tests must be able to see.
    """

    def __init__(self, inner: io.FileIO) -> None:
        self._inner = inner
        self._shortened = False

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        payload = bytes(data)
        if not self._shortened and len(payload) > 1:
            self._shortened = True
            return self._inner.write(payload[:-1])
        return self._inner.write(payload)

    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            super().close()


@pytest.fixture
def short_write_below_the_opened_layer(monkeypatch: pytest.MonkeyPatch):
    """Inject the short write BELOW whatever layer ``append_record`` opens.

    The ``write_calls`` fixture wraps the object ``Path.open`` returns, so it
    sits ABOVE any buffering and cannot observe it.  This fixture instead
    honours the ``buffering`` the production code asks for: with ``buffering=0``
    the caller receives the raw shim and sees the short count; with any other
    buffering it receives a ``BufferedWriter`` over the shim, which returns
    ``len(data)`` unconditionally and loops the raw handle until every byte is
    stored.  The outcome is therefore decided by the production open, not by
    the test's own wrapper.
    """

    real_open = Path.open

    def spy(self, mode="r", *args, **kwargs):
        if "a" in mode and "b" in mode:
            buffering = args[0] if args else kwargs.get("buffering", -1)
            shim = _DropsOneByteOnce(io.FileIO(str(self), "a"))
            return shim if buffering == 0 else io.BufferedWriter(shim)
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)


def test_the_append_handle_is_unbuffered_so_a_short_write_is_reported_not_retried(
    short_write_below_the_opened_layer, tmp_path: Path
) -> None:
    """``buffering=0`` is the token that makes BOTH I1 and I3 true.

    Deleting it leaves every other test in this file green: the recording
    handle counts one ``write`` either way, and ``BufferedWriter.write`` returns
    the full length, so ``written != len(encoded)`` can never fire and
    ``PartialAppendError`` becomes unreachable.  Measured before this test
    existed: removing ``buffering=0`` killed 0 of 25 tests.
    """

    path = tmp_path / "log.jsonl"

    with pytest.raises(PartialAppendError):
        append_record(path, '{"a": 1}')


def test_the_production_append_opens_a_raw_unbuffered_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pin the property directly: the handle must be raw, not buffered.

    Stated as ``RawIOBase`` rather than ``buffering == 0`` so a refactor to a
    different unbuffered spelling still passes, while any buffered handle --
    which silently retries short writes and may split one record across several
    syscalls -- fails.
    """

    real_open = Path.open
    handles: list[object] = []

    def spy(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        if "a" in mode and "b" in mode:
            handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", spy)

    append_record(tmp_path / "log.jsonl", '{"a": 1}')

    assert len(handles) == 1
    assert isinstance(handles[0], io.RawIOBase), (
        "append_record must open an unbuffered raw handle: a BufferedWriter "
        "returns len(data) from write() unconditionally, which defeats the "
        "short-write check, and loops the raw handle, which defeats one-write."
    )


# --- the unstated ASCII invariant the torn-row diagnosis depends on ----------


def test_every_routed_writer_emits_ascii_only_bytes(tmp_path: Path) -> None:
    """The torn-row DIAGNOSIS silently depends on records being pure ASCII.

    ``iter_publications`` opens the catalog as UTF-8 text.  A torn write that
    split a multi-byte character raises ``UnicodeDecodeError`` out of the file
    iteration itself -- before the ``except json.JSONDecodeError`` that produces
    ``IncompletePublicationRecordError`` -- so the operator would get no
    diagnosis, no repair recipe and no line number.  That cannot happen today
    only because every routed writer uses ``json.dumps``'s default
    ``ensure_ascii=True``, an invariant stated nowhere and pinned by nothing.
    This states it.

    COVERAGE LIMIT, and it is the whole reason the other pins exist. This file
    is deliberately torch-free, so it can only reach the routed writers that are
    importable without torch. Two more call ``json.dumps`` themselves and are
    pinned in their own suites -- ``test_the_sidecar_emits_ascii_only_bytes``
    and ``test_the_failure_log_emits_ascii_only_bytes``. MEASURED before those
    existed: ``ensure_ascii=False`` in either module left the full 2154-test
    suite green, while the same mutation in ``artifacts.py`` was killed. The
    instrument worked; the coverage did not reach.

    ``checkpoint/receipt.py`` needs no separate pin: it delegates to
    ``append_jsonl`` and therefore inherits this one. ``distributed.py``'s
    serialization-error FALLBACK also needs none, but for a different reason
    worth recording so nobody adds an undiscriminating test for it -- its
    payload is ASCII BY CONSTRUCTION, carrying only an enum value, two numbers,
    and field KEYS which are ASCII literals. A mutation there would survive for
    want of reachability, not for want of coverage.
    """

    direct = tmp_path / "direct.jsonl"
    append_jsonl(direct, {"text": "é中\U0001f600"})

    metrics = tmp_path / "metrics.jsonl"
    JSONL(metrics).log(LogRecord(step=1, namespace="é", metrics={"loss": 0.5}))

    topology = ExecutionTopology(
        global_rank=0, global_size=1, local_rank=0, local_size=1,
        node_rank=0, node_size=1, host="éhost", pid=1, device="cpu", job_id=None,
    )
    profiles = tmp_path / "profiles"
    RankLocalJSONLWriter(profiles, topology).write(
        ProfileRecord(
            ProfileScope.PROCESS,
            0.0,
            topology,
            process=ProcessResourceResult(
                user_cpu_seconds=1, system_cpu_seconds=1,
                read_block_operations=1, write_block_operations=1,
                voluntary_context_switches=1, involuntary_context_switches=1,
                peak_rss_mb=1,
            ),
        )
    )

    written = [direct, metrics, *profiles.rglob("*.jsonl")]
    assert len(written) >= 3
    for path in written:
        raw = path.read_bytes()
        assert raw.isascii(), f"{path} emitted non-ASCII bytes: {raw!r}"


def test_a_successful_append_writes_exactly_the_expected_bytes(tmp_path: Path) -> None:
    """Pin the byte delta, which "does not merge" and "stays readable" do not.

    Measured before this existed: changing the written terminator from LF to a
    bare CR left all 29 durability and catalog tests green, while every newly
    written file ended in ``\r`` and ``ends_without_newline`` immediately
    classified it as torn. A mutant may also append junk AFTER a correct record
    and still satisfy a non-merge assertion. Both are excluded here.
    """

    path = tmp_path / "log.jsonl"
    before = b'{"a": 1}\n{"torn": '
    record = '{"b": 2}'
    path.write_bytes(before)

    append_record(path, record)

    assert path.read_bytes() == before + b"\n" + record.encode() + b"\n"
    assert ends_without_newline(path) is False


def test_a_fresh_append_writes_exactly_the_record_and_one_lf(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"

    append_record(path, '{"a": 1}')

    assert path.read_bytes() == b'{"a": 1}\n'


@pytest.mark.parametrize(
    "separator",
    ("\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"),
    ids=("lf", "cr", "vt", "ff", "fs", "gs", "rs", "nel", "ls", "ps"),
)
def test_a_record_containing_a_physical_line_separator_is_rejected(
    tmp_path: Path, separator: str
) -> None:
    """The accepted alphabet is the union over what production readers do.

    Not one mechanism: the tpen readers iterate a text handle (LF, CR, CRLF)
    while the science collectors under experiments/ use splitlines, which also
    splits on VT, FF, FS, GS, RS, NEL, U+2028 and U+2029. Measured: eight of
    these ten split under splitlines and NOT under file iteration, so a record
    carrying one was written as one record, read as one by the sidecar, and
    read as TWO by the collector, with the call returning success.
    """

    with pytest.raises(ValueError, match="one record is one line"):
        append_record(tmp_path / "log.jsonl", f'{{"a": 1}}{separator}{{"b": 2}}')


def test_a_record_ending_in_a_separator_is_rejected(tmp_path: Path) -> None:
    """A trailing separator is also a split, and the LF/CR check missed it."""

    with pytest.raises(ValueError, match="one record is one line"):
        append_record(tmp_path / "log.jsonl", '{"a": 1}\n')


def test_an_empty_record_is_still_accepted(tmp_path: Path) -> None:
    """`"".splitlines()` is `[]`, not `[""]`, so the guard needs its `if record`.

    Without that guard the check would reject an empty record, changing
    behaviour for a case the previous LF/CR check accepted.
    """

    path = tmp_path / "log.jsonl"

    append_record(path, "")

    assert path.read_bytes() == b"\n"


@pytest.mark.parametrize(
    "snippet",
    (
        'open("a")',
        'p.open("a")',
        'p.open(mode="a")',
        'open(path, "ab")',
        'p.open(*("a",), encoding="utf-8")',
        'p.open(**kwargs)',
        'import os\nos.open(p, os.O_APPEND)',
        'from os import O_APPEND\nos_open(p, O_APPEND)',
        'from os import O_APPEND as APPEND\nos_open(p, APPEND)',
    ),
    ids=(
        "positional", "path-positional", "mode-keyword", "two-char",
        "starred-literal", "double-star-kwargs",
        "os-attribute", "imported-name", "aliased-import",
    ),
)
def test_the_census_detects_every_known_append_spelling(
    snippet: str, tmp_path: Path
) -> None:
    """The census instrument itself, pinned against the spellings it has missed.

    Two of these are regressions it actually had: a starred literal argument
    arrives as ``ast.Starred`` rather than ``ast.Constant``, and an aliased
    ``from os import O_APPEND as APPEND`` defeats a name-only check. Both were
    found by review AFTER the AST version replaced a regex that had itself
    missed ``mode=``. Testing the instrument directly is cheaper than
    discovering the next gap through a mutant that slipped past the census.
    """

    source = tmp_path / "candidate.py"
    source.write_text(snippet + "\n", encoding="utf-8")

    assert _opens_append_handle(source) is True


@pytest.mark.parametrize(
    "snippet",
    ('open("r")', 'p.open("rb")', 'p.open(mode="w")', 'json.dumps(x)', 'p.read_text()'),
    ids=("read", "read-binary", "write-keyword", "unrelated-call", "read-text"),
)
def test_the_census_does_not_flag_non_append_opens(
    snippet: str, tmp_path: Path
) -> None:
    """The over-restrictive direction: flagging every open would be useless.

    A census that fires on reads would be permanently red and would be silenced
    rather than fixed, which is worse than one gap.
    """

    source = tmp_path / "candidate.py"
    source.write_text(snippet + "\n", encoding="utf-8")

    assert _opens_append_handle(source) is False

