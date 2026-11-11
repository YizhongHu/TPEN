"""Immutable descriptions of prepared input bundles."""

from __future__ import annotations

import os
from typing import Any, ClassVar

from ._codec import canonical_bytes, decode_json, fields, sha256_text
from ._records import frozen_record


def _absolute_path(value: Any) -> str:
    """Validate an absolute bundle root without reading it."""

    if type(value) is not str or not value:
        raise TypeError("root must be a non-empty string")
    if not os.path.isabs(value):
        raise ValueError("root must be an absolute path")
    return value


@frozen_record
class PreparedInputBundle:
    """Description of immutable inputs prepared for an owner process.

    Parameters
    ----------
    root : str
        Absolute path to the prepared bundle.  Construction never creates,
        deletes, or checks this path.
    manifest_digest : str or None
        Optional SHA-256 digest of the bundle manifest.
    """

    schema: ClassVar[str] = "tpen-executor/prepared-input-bundle-v1"
    root: str
    manifest_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _absolute_path(self.root))
        if self.manifest_digest is not None:
            object.__setattr__(self, "manifest_digest", sha256_text(self.manifest_digest, "manifest_digest"))

    def to_dict(self) -> dict[str, Any]:
        """Return the strict wire representation."""

        return {"schema": self.schema, "root": self.root, "manifest_digest": self.manifest_digest}

    def encode(self) -> bytes:
        """Encode this descriptor as canonical JSON bytes."""

        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "PreparedInputBundle":
        """Decode one strict mapping."""

        fields(data, {"schema", "root", "manifest_digest"}, cls.schema, "PreparedInputBundle")
        return cls(data["root"], data["manifest_digest"])

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "PreparedInputBundle":
        """Decode canonical JSON or an already-decoded mapping."""

        return cls.from_mapping(decode_json(data))


__all__ = ["PreparedInputBundle"]
