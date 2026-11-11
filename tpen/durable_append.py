"""Single-write, torn-tail-safe appends for line-oriented artifact files.

TPEN's run artifacts are append-only, line-oriented files -- ``publications.jsonl``,
``events.jsonl``, ``occurrences.jsonl``, the rank-local resource profiles, and the
metric logs.  They live on Netscratch, which is NFS.  This module owns the single
primitive every one-record-per-open writer uses to extend such a file, so that the
torn-write invariant has exactly one implementation rather than one per call site.

What this module does NOT rely on
---------------------------------
It relies on **neither** of the two guarantees a reader might reasonably assume.

*Atomicity of a single* ``write(2)`` *to a regular file.*  POSIX grants atomicity
only for writes to a pipe or FIFO below ``PIPE_BUF``.  A write to a regular file
may store fewer bytes than requested -- on ``ENOSPC``, on inode exhaustion, or on
a signal -- and it is allowed to do so silently by returning a short count.

*``O_APPEND`` offset resolution on NFS.*  NFSv3 has no server-side append
operation.  The client resolves the current end of file and then writes at that
offset, so two clients can obtain the same offset and overwrite one another.
``O_APPEND`` therefore does not hold across NFS clients at all.  Under a single
writer this is immaterial; under concurrent writers it is unfixed here and is
the separate concern of rank-sharding in the distributed program.

What it does provide
--------------------
1. **One write call per record.**  The record and its terminating newline are
   encoded together and handed to an *unbuffered* binary handle, so exactly one
   ``write`` reaches the kernel.  This narrows the interruption window to a
   single kernel-visible operation; it does not make that operation atomic.
2. **Detection instead of silence.**  A short write raises
   :class:`PartialAppendError` rather than returning normally, so *a reported
   short write cannot be mistaken for success*.  That is the whole of the
   claim.  It is NOT "a caller can never mistake a torn record for a stored
   one": a full-count return means the bytes were handed to the kernel, not
   that they are durable.  No ``fsync`` is issued, and on NFS a later client,
   server, or system failure can still lose or tear a record this function
   reported as written.
3. **Bounded damage.**  If a record is nevertheless left unterminated, the next
   append terminates it *in the same single write* that stores the new record.
   The new record therefore lands on its own line instead of being concatenated
   onto the torn bytes.  Without this, one interrupted append destroys the *next*
   record too -- a strictly worse outcome than losing the interrupted one.

The resulting guarantee is precise and deliberately modest: **damage from a torn
write is bounded to the torn record itself and is detectable when the file is
read.**  It is *not* "torn writes cannot occur", which is unachievable on the
filesystem these files actually live on.

Three things this module relies on that are easy to miss
--------------------------------------------------------
**A short write is NOT retried, and that is a behaviour change.**  The previous
text-mode path wrapped a ``BufferedWriter``, which loops the raw handle until
every byte is stored, so a partial store was completed transparently.  This
module raises instead.  Under ``O_APPEND`` with a single writer a bounded retry
would be safe -- the remainder lands at the new end of file -- so completion was
available and was deliberately not taken: a caller that believes a record was
stored when it was not is the failure this module exists to prevent, and under
``ENOSPC``, the dominant case on a quota-bound filesystem, a retry fails anyway.
The cost is real: a partial store followed by a signal now loses a record the
old path would have completed.  Revisiting this is tracked separately.

**Callers that swallow ``OSError`` now swallow this too.**  Subclassing
``OSError`` keeps existing handlers working, which is the point -- but "callers
already treat storage failures that way" is not uniformly true.  Of the sites
reached from an append, ``tpen/checkpoint/receipt.py`` logs a warning, and
``tpen/callback/resource_usage.py`` has one commented best-effort ``pass`` and
one bare ``pass`` with no log at all.  At the latter a partial append is now
silent AND leaves a torn line behind.  That is bounded to rank-local telemetry,
whose reader skips bad rows, but it is not the same as being handled.

**Records must encode to ASCII.**  The readers open these files as UTF-8 text.
A torn write that split a multi-byte character raises ``UnicodeDecodeError``
from the file iteration itself, before any per-row error handling -- so the
torn-row diagnosis is bypassed and the operator gets no repair guidance.  This
cannot happen today only because every routed writer uses ``json.dumps`` with
its default ``ensure_ascii=True``.  That invariant is depended on here and is
pinned by a test; do not weaken it without changing the readers.
"""

from __future__ import annotations

import io
from pathlib import Path

__all__ = ["PartialAppendError", "append_record", "ends_without_newline"]


class PartialAppendError(OSError):
    """The sole raw write reported fewer bytes than the complete append payload.

    Note "payload", not "record": the payload is the record PLUS its terminator
    and any repair newline.  A one-byte-short write can store every byte of the
    record and omit only the terminator -- measured: ``stored 8 of 9`` with all
    eight record bytes on disk and parseable.  Saying "fewer bytes than the
    record" would be false in exactly that case.

    Means exactly one thing: **the requested record was not committed**, where
    committed is defined by this module and its readers as terminated by a
    newline.  It does NOT mean the record's bytes are absent.  It
    does NOT describe the file's resulting final state, which depends on how
    many bytes landed and whether a torn predecessor was being repaired -- the
    file may end mid-record, or may end in a newline with none of the new
    record present.  Callers that need the state must inspect it.

    Subclasses :class:`OSError` so that existing ``except OSError`` handlers
    around artifact appends keep catching it; a partial append is a storage
    failure and callers already treat storage failures that way.
    """


def ends_without_newline(path: Path) -> bool:
    """Report whether `path` holds an unterminated final line.

    A missing or empty file has no unterminated line.  Only the final line of a
    file can lack a terminator, because a newline is precisely what separates it
    from a following line -- so this single byte is a complete test.

    Parameters
    ----------
    path : pathlib.Path
        File to inspect.

    Returns
    -------
    bool
        True when the file is non-empty and its last byte is not ``b"\\n"``.
    """

    try:
        with path.open("rb") as handle:
            if handle.seek(0, io.SEEK_END) == 0:
                return False
            handle.seek(-1, io.SEEK_END)
            return handle.read(1) != b"\n"
    except FileNotFoundError:
        # Losing the race against creation is indistinguishable from "no file",
        # and both mean there is no unterminated line to close out.
        return False


def append_record(path: str | Path, record: str) -> None:
    """Append `record` as one newline-terminated line using a single write.

    Parameters
    ----------
    path : str or pathlib.Path
        File to extend.  Parent directories are created if absent.
    record : str
        One complete record, without its trailing terminator.  Must contain no
        separator that ANY production reader of these files treats as a line
        ending.

        The check is derived from the WIDEST reader rather than from a list of
        characters, because the readers do not share a mechanism.  Some iterate
        a text handle (universal newlines: ``\n``, ``\r``, ``\r\n``); others --
        the science collectors under ``experiments/`` -- use
        ``Path.read_text(...).splitlines()``, which ALSO splits on VT, FF, FS,
        GS, RS, NEL, U+2028 and U+2029.  The union is therefore the
        ``str.splitlines`` set, and ``record.splitlines() != [record]`` IS that
        union, so it cannot drift out of sync with either mechanism the way a
        hard-coded character list would.

        An earlier version of this guard enumerated ``\n`` and ``\r`` from ONE
        reader mechanism and was under-restrictive: a record carrying, say,
        U+2028 was written as one record, read as one by the sidecar, and read
        as TWO by the science collector, with this function returning success.

        The readers disagreeing with EACH OTHER is a separate live defect that
        this guard does not fix -- it only stops this module contributing new
        instances.  Tracked as its own item.

    Raises
    ------
    ValueError
        If `record` contains any separator recognised by ``str.splitlines``,
        including a trailing one.
    PartialAppendError
        If the write stored fewer bytes than the encoded record.  The requested
        record was NOT committed; see that exception's note on what the file's
        final state may be.
    """

    # Derived from the widest production reader, not from a character list:
    # str.splitlines is a superset of universal-newline file iteration, and the
    # science collectors use it. An empty record has no separator to find --
    # "".splitlines() is [] rather than [""] -- so it is excluded explicitly.
    if record and record.splitlines() != [record]:
        raise ValueError(
            "record must not contain a line separator; one record is one line. "
            "The accepted set is derived from str.splitlines, the widest "
            "mechanism any production reader of these files uses, so this "
            "rejects VT, FF, FS, GS, RS, NEL, U+2028 and U+2029 as well as "
            "line feed and carriage return -- and a trailing separator too."
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Closing out a torn predecessor is folded into THIS record's single write.
    # Terminating it in a separate write would reintroduce exactly the two-write
    # window the primitive exists to remove.
    prefix = "\n" if ends_without_newline(target) else ""
    encoded = f"{prefix}{record}\n".encode()

    # buffering=0 yields a raw FileIO: one write call in, one write syscall out,
    # with no BufferedWriter chunking the payload or retrying a short write.
    with target.open("ab", buffering=0) as handle:
        written = handle.write(encoded)

    if written != len(encoded):
        raise PartialAppendError(
            f"partial append to {target}: stored {written} of {len(encoded)} bytes. "
            "The requested record was NOT committed. Inspect the file before "
            "assuming its final state: this does NOT necessarily leave an "
            "unterminated last line. If a torn predecessor was being repaired, "
            "the only stored byte may be the repair newline, in which case the "
            "file ends in a newline and contains nothing of the new record."
        )
