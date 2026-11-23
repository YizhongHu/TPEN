"""Build and validate bounded names used for Stage-1 jobs.

This module is deliberately standard-library-only because exact-source
science checkouts may contain ``experiments/`` and project metadata without
the operational ``scripts/`` tree.
"""

import argparse
import re
from typing import Any, Mapping, Optional


MAX_JOB_NAME_LENGTH = 40
_COMPONENT_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")

# ``root`` and ``exec`` are the two explicit arms of the devsuite creator.
PURPOSES = frozenset({
    "train", "eval", "smoke", "rehearsal", "verify", "mutants", "gate",
    "collect", "envbuild", "archive", "diag", "root", "exec",
})


def _component(value: object, *, field: str) -> str:
    """Validate one shell-safe, lower-case name component."""

    if not isinstance(value, str) or _COMPONENT_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must contain only lower-case letters, digits, '.', '_' or '-'")
    return value


def validate_job_name(name: object) -> str:
    """Validate a complete Slurm job name and return it unchanged."""

    if not isinstance(name, str) or _COMPONENT_RE.fullmatch(name) is None:
        raise ValueError("job name contains unsupported characters")
    if len(name) > MAX_JOB_NAME_LENGTH:
        raise ValueError(
            f"job name is too long ({len(name)} > {MAX_JOB_NAME_LENGTH} characters)"
        )
    return name


def build_job_name(
    subject: str,
    purpose: str,
    *,
    variant: Optional[str] = None,
    lane: str = "s1",
    sha: Optional[str] = None,
) -> str:
    """Build ``<subject>-<purpose>[-<variant>]-<lane>[-<sha8>]``."""

    subject = _component(subject, field="subject")
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {sorted(PURPOSES)}")
    lane = _component(lane, field="lane")
    parts = [subject, purpose]
    if variant is not None:
        parts.append(_component(variant, field="variant"))
    parts.append(lane)
    if sha is not None:
        if _SHA_RE.fullmatch(sha) is None:
            raise ValueError("sha must be a full 40-character hexadecimal commit")
        parts.append(sha[:8].lower())
    return validate_job_name("-".join(parts))


def stage1_row_job_name(row: Mapping[str, Any]) -> str:
    """Build the job name for a rendered Stage-1 manifest row.

    The row's ``lineage`` is the public identity used in the name.  A
    ``lineage_seed`` without that index is rejected so naming cannot silently
    expose the seed label instead of the lineage index.
    """

    kind = row.get("kind")
    if kind not in {"smoke", "train", "eval"}:
        raise ValueError("manifest row has an invalid kind")
    subject = row.get("system") or row.get("sector")
    if not isinstance(subject, str):
        raise ValueError("manifest row needs a system or sector for its job name")
    if "lineage_seed" in row and "lineage" not in row:
        raise ValueError("job name requires lineage index, not lineage_seed")
    lineage = row.get("lineage", 0)
    if type(lineage) is not int or lineage < 0:
        raise ValueError("lineage must be a non-negative integer")
    variant = None
    if kind in {"train", "eval"}:
        variant = f"l{lineage}"
    if kind == "eval":
        checkpoint = row.get("checkpoint", 0)
        stream = row.get("stream", 0)
        if type(checkpoint) is not int or checkpoint < 0:
            raise ValueError("checkpoint must be a non-negative integer")
        if type(stream) is not int or stream < 0:
            raise ValueError("stream must be a non-negative integer")
        variant = f"{variant}-c{checkpoint}-s{stream}"
    return build_job_name(subject, kind, variant=variant, lane="s1", sha=row.get("source_sha"))


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--lane", default="s1")
    parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    print(build_job_name(args.subject, args.purpose, variant=args.variant,
                         lane=args.lane, sha=args.sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
