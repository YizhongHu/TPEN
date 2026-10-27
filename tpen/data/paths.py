"""Virtual-support path metadata for equivariant real-space mixing.

Path generation is intentionally separated from neural modules. The metadata
objects here are deterministic constructors/readers; training code should load
saved cache files instead of silently regenerating path orderings.
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from enum import Enum
from pathlib import Path
from typing import Literal

from tpen.data.indices import ordered_tuples


OutputEmbedding = Literal["canonical", "full"]


class LinearPathPolicy(str, Enum):
    """Closed set of linear path-family selection policies."""

    COORDINATE_NEIGHBOR = "coordinate_neighbor"
    ORBIT_COMPLETE = "orbit_complete"
    EXPLICIT = "explicit"


def normalize_linear_path_policy(value: LinearPathPolicy | str) -> LinearPathPolicy:
    """Normalize a closed configuration value to a typed policy."""

    if isinstance(value, LinearPathPolicy):
        return value
    try:
        return LinearPathPolicy(value)
    except ValueError as error:
        raise ValueError(f"Unsupported linear path policy {value!r}") from error
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
DEFAULT_PATH_FILES = {
    "canonical": CACHE_DIR / "paths_canonical.json",
    "full": CACHE_DIR / "paths_full.json",
}


@dataclass(frozen=True)
class NormalizedOrders:
    """Immutable, canonical collection of positive tuple orders."""

    values: tuple[int, ...]

    def __post_init__(self) -> None:
        values = tuple(int(value) for value in self.values)
        if any(value <= 0 for value in values):
            raise ValueError("orders must be positive")
        object.__setattr__(self, "values", tuple(sorted(set(values))))

    @classmethod
    def from_tuple(cls, values: tuple[int, ...]) -> "NormalizedOrders":
        """Normalize an order tuple without accepting semantic mappings."""

        return cls(tuple(values))


@dataclass(frozen=True)
class NormalizedChannels:
    """Immutable ``(order, channels)`` pairs in canonical order."""

    values: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        values = tuple((int(order), int(channels)) for order, channels in self.values)
        if any(order <= 0 or channels < 0 for order, channels in values):
            raise ValueError("orders must be positive and channels must be non-negative")
        if len({order for order, _ in values}) != len(values):
            raise ValueError("channels must be unique by order")
        object.__setattr__(self, "values", tuple(sorted(values)))

    @classmethod
    def from_tuple(cls, values: tuple[tuple[int, int], ...]) -> "NormalizedChannels":
        """Normalize channel pairs supplied as tuples."""

        return cls(tuple(values))

    def for_order(self, order: int) -> int:
        """Return the channel count for ``order`` or raise ``KeyError``."""

        for candidate, channels in self.values:
            if candidate == order:
                return channels
        raise KeyError(order)


@dataclass(frozen=True)
class SupportPath:
    """Describe one unary path by its two injections into a common support."""

    output_order: int
    input_order: int
    tau_out: tuple[int, ...]
    tau_in: tuple[int, ...]
    normalization: Literal["sum", "completion_mean"] = "completion_mean"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tau_out", tuple(self.tau_out))
        object.__setattr__(self, "tau_in", tuple(self.tau_in))
        validate_linear_output_plus_input_cover_support(self)

    @property
    def support_order(self) -> int:
        """Return the inferred size of the canonical common support."""

        return self.output_order + self.input_order - self.overlap

    @property
    def overlap(self) -> int:
        """Return the number of shared support labels."""

        return len(set(self.tau_out) & set(self.tau_in))

    def as_tuple(self) -> tuple[object, ...]:
        """Return the canonical semantic representation."""

        return (self.output_order, self.input_order,
                self.tau_out, self.tau_in, self.normalization)


@dataclass(frozen=True)
class PathEntry:
    """Associate a typed support path with its output and input orders."""

    output_order: int
    input_order: int
    path: SupportPath

    def __post_init__(self) -> None:
        if (self.output_order, self.input_order) != (self.path.output_order, self.path.input_order):
            raise ValueError("PathEntry orders must match SupportPath")

    def as_tuple(self) -> tuple[object, ...]:
        """Return the canonical semantic representation."""

        return (self.output_order, self.input_order, self.path.as_tuple())

    @property
    def support_path(self) -> SupportPath:
        """Return the underlying unary support path."""

        return self.path


@dataclass(frozen=True)
class OutputPathLayout:
    """Immutable ordered path entries contributing to one output order."""

    output_order: int
    entries: tuple[PathEntry, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(entry.output_order != self.output_order for entry in entries):
            raise ValueError("all entries must contribute to output_order")
        # Path position is part of the contract: aggregation weights index this
        # sequence, and composite producers contribute ordered path families.
        keys = [entry.as_tuple() for entry in entries]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate semantic paths are not allowed")
        object.__setattr__(self, "entries", entries)

    @property
    def count(self) -> int:
        """Return the number of paths for this output order."""

        return len(self.entries)


@dataclass(frozen=True)
class PathLayout:
    """Immutable common path layout shared by producers and aggregation."""

    outputs: tuple[OutputPathLayout, ...]
    input_orders: NormalizedOrders
    output_orders: NormalizedOrders
    input_channels: NormalizedChannels
    output_channels: NormalizedChannels
    version: str = "path-layout-v1"
    family_slices: tuple["PathFamilyLayout", ...] = ()

    def __post_init__(self) -> None:
        outputs = tuple(sorted(self.outputs, key=lambda layout: layout.output_order))
        if tuple(layout.output_order for layout in outputs) != self.output_orders.values:
            raise ValueError("outputs must cover output_orders in canonical order")
        for order in self.input_orders.values:
            self.input_channels.for_order(order)
        for order in self.output_orders.values:
            self.output_channels.for_order(order)
        entries = [entry.as_tuple() for layout in outputs for entry in layout.entries]
        if len(entries) != len(set(entries)):
            raise ValueError("duplicate semantic paths are not allowed")
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "family_slices", tuple(self.family_slices))
        if self.family_slices:
            if tuple(slice_.family for slice_ in self.family_slices) != tuple(
                sorted(slice_.family for slice_ in self.family_slices)
            ):
                raise ValueError("family slices must use deterministic family order")
            for slice_ in self.family_slices:
                for output in slice_.outputs:
                    if output.output_order not in self.output_orders.values:
                        raise ValueError("family slice output order is not configured")

    @property
    def counts(self) -> tuple[tuple[int, int], ...]:
        """Return ``(output_order, path_count)`` pairs."""

        return tuple((order, self.count_for_order(order)) for order in self.output_orders.values)

    def count_for_order(self, order: int) -> int:
        """Return the union path count for one output order."""

        if self.family_slices:
            return sum(slice_.count_for_order(order) for slice_ in self.family_slices)
        for layout in self.outputs:
            if layout.output_order == order:
                return layout.count
        raise KeyError(order)

    @property
    def output_layouts(self) -> tuple[OutputPathLayout, ...]:
        """Return output layouts under the descriptive alias."""

        return self.outputs

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of semantic layout values."""

        # Covers version, channel/order contracts, and declared path position.
        # It deliberately excludes Python class names, field names, and repr
        # formatting so implementation refactors do not change science identity.
        family_values = tuple(slice_.as_tuple() for slice_ in self.family_slices)
        semantic_values = (self.version, self.input_orders.values, self.output_orders.values,
                           self.input_channels.values, self.output_channels.values,
                           tuple(tuple(entry.as_tuple() for entry in layout.entries)
                                 for layout in self.outputs), family_values)
        payload = json.dumps(semantic_values, separators=(",", ":"), ensure_ascii=True).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LinearPathMetadata:
    """Immutable deterministic metadata for unary linear mixing paths.

    Parameters
    ----------
    outputs : tuple of OutputPathLayout
        Contractual path sequences grouped by output order.
    input_orders, output_orders : NormalizedOrders
        Orders represented by the metadata. Input orders are owned by this
        value rather than discovered from runtime feature tensors.
    policy : LinearPathPolicy
        Selection policy used to construct the metadata.
    """

    outputs: tuple[OutputPathLayout, ...]
    input_orders: NormalizedOrders
    output_orders: NormalizedOrders
    policy: LinearPathPolicy

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", normalize_linear_path_policy(self.policy))
        outputs = tuple(sorted(tuple(self.outputs), key=lambda layout: layout.output_order))
        if tuple(layout.output_order for layout in outputs) != self.output_orders.values:
            raise ValueError("linear outputs must cover output_orders in canonical order")
        for layout in outputs:
            for entry in layout.entries:
                if entry.input_order not in self.input_orders.values:
                    raise ValueError("linear path input order is not configured")
        entries = [entry.as_tuple() for layout in outputs for entry in layout.entries]
        if len(entries) != len(set(entries)):
            raise ValueError("duplicate semantic linear paths are not allowed")
        object.__setattr__(self, "outputs", outputs)

    @classmethod
    def generate(
        cls,
        *,
        max_order: int,
        policy: LinearPathPolicy | str = LinearPathPolicy.COORDINATE_NEIGHBOR,
        input_orders: tuple[int, ...] | None = None,
        output_orders: tuple[int, ...] | None = None,
        normalization: Literal["sum", "completion_mean"] = "completion_mean",
        explicit: Sequence[SupportPath] | None = None,
    ) -> "LinearPathMetadata":
        """Construct deterministic metadata without runtime or file access.

        ``coordinate_neighbor`` emits same-order identity and one-coordinate
        replacement paths. ``orbit_complete`` emits every canonical partial
        matching for each configured order pair. ``explicit`` preserves the
        supplied path order and rejects paths outside the configured orders.

        Parameters
        ----------
        explicit : collections.abc.Sequence[SupportPath] or None
            Explicit paths to retain in the supplied order. Every element is
            checked because collection type checks may inspect only the first
            element.
        """

        policy = normalize_linear_path_policy(policy)
        max_order = int(max_order)
        if max_order <= 0:
            raise ValueError("max_order must be positive")
        inputs = NormalizedOrders(tuple(range(1, max_order + 1)) if input_orders is None else input_orders)
        outputs = NormalizedOrders(tuple(range(1, max_order + 1)) if output_orders is None else output_orders)
        if policy is LinearPathPolicy.COORDINATE_NEIGHBOR and inputs.values != outputs.values:
            raise ValueError("coordinate_neighbor requires matching input and output orders")
        by_output: list[OutputPathLayout] = []
        if explicit is None:
            explicit_paths: tuple[SupportPath, ...] = ()
        elif isinstance(explicit, (str, bytes)) or not isinstance(explicit, Sequence):
            raise TypeError("explicit must be a sequence of SupportPath values")
        else:
            invalid_index = next(
                (index for index, path in enumerate(explicit) if not isinstance(path, SupportPath)),
                None,
            )
            if invalid_index is not None:
                raise TypeError(f"explicit[{invalid_index}] must be a SupportPath")
            explicit_paths = tuple(explicit)
        for output_order in outputs.values:
            selected: list[SupportPath] = []
            if policy is LinearPathPolicy.COORDINATE_NEIGHBOR:
                selected.append(
                    SupportPath(output_order, output_order, tuple(range(output_order)), tuple(range(output_order)), normalization)
                )
                for slot in range(output_order):
                    tau_in = tuple(
                        output_order if input_slot == slot else input_slot
                        for input_slot in range(output_order)
                    )
                    selected.append(SupportPath(output_order, output_order, tuple(range(output_order)), tau_in, normalization))
            elif policy is LinearPathPolicy.ORBIT_COMPLETE:
                for input_order in inputs.values:
                    selected.extend(enumerate_linear_support_paths(output_order, input_order, normalization=normalization))
            else:
                selected.extend(path for path in explicit_paths if path.output_order == output_order)
            entries = tuple(PathEntry(output_order, path.input_order, path) for path in selected)
            by_output.append(OutputPathLayout(output_order, entries))
        if policy is LinearPathPolicy.EXPLICIT:
            configured = {(path.output_order, path.input_order, path.as_tuple()) for path in explicit_paths}
            actual = {
                (entry.output_order, entry.input_order, entry.path.as_tuple())
                for layout in by_output
                for entry in layout.entries
            }
            if actual != configured:
                raise ValueError("explicit paths must use configured output orders")
        return cls(tuple(by_output), inputs, outputs, policy)

    def all_paths(self) -> tuple[SupportPath, ...]:
        """Return paths in their contractual output/path-axis order."""

        return tuple(entry.path for layout in self.outputs for entry in layout.entries)

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of linear path semantics."""

        semantic_values = (
            "linear-path-metadata-v1",
            self.policy.value,
            self.input_orders.values,
            self.output_orders.values,
            tuple(tuple(entry.as_tuple() for entry in layout.entries) for layout in self.outputs),
        )
        payload = json.dumps(semantic_values, separators=(",", ":"), ensure_ascii=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def paths_for_output_order(self, output_order: int) -> tuple[SupportPath, ...]:
        """Return the static path sequence for one output order."""

        for layout in self.outputs:
            if layout.output_order == output_order:
                return tuple(entry.path for entry in layout.entries)
        raise KeyError(output_order)


@dataclass(frozen=True)
class VirtualPath:
    """Describe one bilinear virtual-support mixing path.

    Parameters
    ----------
    s : int
        Virtual support order.
    m : int
        Output tuple order.
    m1, m2 : int
        Left and right input tuple orders.
    local_id : int
        Stable path index inside a fixed ``(s, m, m1, m2)`` block.
    global_id : int
        Stable path index across the whole path file.
    tau, tau1, tau2 : tuple of int
        Injective maps into the virtual support, represented as zero-based
        images.
    """

    s: int
    m: int
    m1: int
    m2: int
    local_id: int
    global_id: int
    tau: tuple[int, ...]
    tau1: tuple[int, ...]
    tau2: tuple[int, ...]

    @property
    def input_support(self) -> set[int]:
        """Return virtual labels covered by the two input injections."""

        return set(self.tau1) | set(self.tau2)

    def as_tuple(self) -> tuple[int, int, int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Return the mathematical tuple ``(s, m, m1, m2, tau, tau1, tau2)``."""

        return (self.s, self.m, self.m1, self.m2, self.tau, self.tau1, self.tau2)


@dataclass(frozen=True)
class PathFamilyOutput:
    """Immutable ordered paths for one family and output order."""

    family: Literal["linear", "tensor_product"]
    output_order: int
    paths: tuple[SupportPath | VirtualPath, ...]

    def __post_init__(self) -> None:
        paths = tuple(self.paths)
        expected = self.output_order
        for path in paths:
            actual = path.output_order if isinstance(path, SupportPath) else path.m
            if actual != expected:
                raise ValueError("family paths must match their output order")
        if len({path.as_tuple() for path in paths}) != len(paths):
            raise ValueError("duplicate family paths are not allowed")
        object.__setattr__(self, "paths", paths)

    @property
    def count(self) -> int:
        """Return the static number of paths in this family slice."""

        return len(self.paths)

    def as_tuple(self) -> tuple[object, ...]:
        """Return the recursive fingerprint representation."""

        return (self.family, self.output_order, tuple(path.as_tuple() for path in self.paths))


@dataclass(frozen=True)
class PathFamilyLayout:
    """Immutable ordered output slices for one producer family."""

    family: Literal["linear", "tensor_product"]
    outputs: tuple[PathFamilyOutput, ...]

    def __post_init__(self) -> None:
        outputs = tuple(sorted(self.outputs, key=lambda output: output.output_order))
        if any(output.family != self.family for output in outputs):
            raise ValueError("family output labels must agree")
        if len({output.output_order for output in outputs}) != len(outputs):
            raise ValueError("family output orders must be unique")
        object.__setattr__(self, "outputs", outputs)

    def count_for_order(self, order: int) -> int:
        """Return this family's count for an order, including zero slices."""

        for output in self.outputs:
            if output.output_order == order:
                return output.count
        return 0

    def as_tuple(self) -> tuple[object, ...]:
        """Return the recursive fingerprint representation."""

        return (self.family, tuple(output.as_tuple() for output in self.outputs))


def compose_path_layout(
    *,
    linear: LinearPathMetadata | None,
    tensor_product: PathMetadata | None,
    input_orders: NormalizedOrders,
    output_orders: NormalizedOrders,
    input_channels: NormalizedChannels,
    output_channels: NormalizedChannels,
    max_virtual_order: int | None = None,
) -> PathLayout:
    """Compose deterministic linear-then-TP family slices into one layout.

    The function only combines already-materialized metadata. It never reads
    runtime tensors or regenerates checked-in tensor-product JSON.
    """

    families: list[PathFamilyLayout] = []
    outputs: list[PathFamilyOutput] = []
    if linear is not None:
        for order in output_orders.values:
            outputs.append(PathFamilyOutput("linear", order, tuple(
                path for path in linear.paths_for_output_order(order)
                if path.input_order in input_orders.values
            )))
        families.append(PathFamilyLayout("linear", tuple(outputs)))
    if tensor_product is not None:
        limit = tensor_product.max_virtual_order if max_virtual_order is None else int(max_virtual_order)
        tp_outputs = tuple(
            PathFamilyOutput("tensor_product", order, tuple(
                path for path in tensor_product.paths_for_output_order(order)
                if path.s <= limit and path.m1 in input_orders.values and path.m2 in input_orders.values
            ))
            for order in output_orders.values
        )
        families.append(PathFamilyLayout("tensor_product", tp_outputs))
    if not families:
        raise ValueError("at least one producer family is required")
    # ``outputs`` retains the legacy unary view; family_slices is authoritative
    # for union counts and semantic identity.
    unary_outputs = tuple(
        OutputPathLayout(order, tuple(
            PathEntry(order, path.input_order, path)
            for family in families if family.family == "linear"
            for family_output in family.outputs if family_output.output_order == order
            for path in family_output.paths if isinstance(path, SupportPath)
        ))
        for order in output_orders.values
    )
    return PathLayout(
        outputs=unary_outputs,
        input_orders=input_orders,
        output_orders=output_orders,
        input_channels=input_channels,
        output_channels=output_channels,
        family_slices=tuple(families),
    )


PathFamily = dict[int, dict[int, dict[int, dict[int, list[VirtualPath]]]]]


class PathMetadata:
    """Thin reader/constructor for saved virtual-support path metadata.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON metadata path.
    """

    schema_version = 1
    path_order_version = "lexicographic-v1"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = self._load_json(self.path)
        self.max_order = int(self.data["max_order"])
        self.max_virtual_order = int(self.data["max_virtual_order"])
        self.output_embedding = self.data["output_embedding"]
        self.paths = self._parse_paths(self.data["paths"])

    @classmethod
    def load(cls, path: str | Path) -> "PathMetadata":
        """Load metadata from `path`."""

        return cls(path)

    @classmethod
    def generate(
        cls,
        *,
        max_order: int,
        max_virtual_order: int,
        output_embedding: OutputEmbedding,
    ) -> "PathMetadata":
        """Generate deterministic metadata without writing it to disk."""

        paths = generate_virtual_paths(
            max_order=max_order,
            max_virtual_order=max_virtual_order,
            output_embedding=output_embedding,
        )
        metadata = cls.__new__(cls)
        metadata.path = None
        metadata.max_order = int(max_order)
        metadata.max_virtual_order = int(max_virtual_order)
        metadata.output_embedding = output_embedding
        metadata.paths = paths
        metadata.data = metadata.to_json_data()
        return metadata

    def save(self, path: str | Path) -> None:
        """Write this metadata to `path` as stable compact JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_json_data(), separators=(",", ":"), sort_keys=True) + "\n")
        self.path = output

    def get(self, s: int, m: int, m1: int, m2: int) -> list[VirtualPath]:
        """Return paths for one fixed ``(s, m, m1, m2)`` block."""

        return self.paths.get(s, {}).get(m, {}).get(m1, {}).get(m2, [])

    def paths_for_output_order(self, m: int) -> list[VirtualPath]:
        """Return all paths contributing to output order `m`."""

        return [path for path in self.all_paths() if path.m == m]

    def paths_for_virtual_order(self, s: int) -> list[VirtualPath]:
        """Return all paths with virtual support order `s`."""

        return [path for path in self.all_paths() if path.s == s]

    def all_paths(self) -> list[VirtualPath]:
        """Return every path in stable global-id order."""

        paths = [
            path
            for by_m in self.paths.values()
            for by_m1 in by_m.values()
            for by_m2 in by_m1.values()
            for block in by_m2.values()
            for path in block
        ]
        return sorted(paths, key=lambda path: path.global_id)

    def to_json_data(self) -> dict[str, object]:
        """Return the JSON-serializable metadata representation."""

        return {
            "schema_version": self.schema_version,
            "index_base": 0,
            "max_order": self.max_order,
            "max_virtual_order": self.max_virtual_order,
            "output_embedding": self.output_embedding,
            "path_order_version": self.path_order_version,
            "path_storage_format": "nested-injections-v1",
            "paths": _serialize_paths(self.paths),
        }

    @staticmethod
    def _load_json(path: Path) -> dict[str, object]:
        data = json.loads(path.read_text())
        if int(data.get("schema_version", -1)) != PathMetadata.schema_version:
            raise ValueError(f"Unsupported path metadata schema in {path}")
        if int(data.get("index_base", -1)) != 0:
            raise ValueError(f"Path metadata must use zero-based indices: {path}")
        if data.get("path_order_version") != PathMetadata.path_order_version:
            raise ValueError(f"Unsupported path ordering in {path}")
        return data

    @staticmethod
    def _parse_paths(data: object) -> PathFamily:
        if not isinstance(data, list):
            raise TypeError("Path metadata paths must use compact nested-list storage")
        return _parse_compact_paths(data)


def _parse_compact_paths(data: list[object]) -> PathFamily:
    paths: PathFamily = {}
    global_id = 0
    for s, by_m in enumerate(data):
        if s == 0 or not by_m:
            continue
        if not isinstance(by_m, list):
            raise TypeError(f"paths[{s}] must be a list")
        paths[s] = {}
        for m, by_m1 in enumerate(by_m):
            if m == 0 or not by_m1:
                continue
            if not isinstance(by_m1, list):
                raise TypeError(f"paths[{s}][{m}] must be a list")
            paths[s][m] = {}
            for m1, by_m2 in enumerate(by_m1):
                if m1 == 0 or not by_m2:
                    continue
                if not isinstance(by_m2, list):
                    raise TypeError(f"paths[{s}][{m}][{m1}] must be a list")
                paths[s][m][m1] = {}
                for m2, block in enumerate(by_m2):
                    if m2 == 0 or block is None:
                        continue
                    if not isinstance(block, list):
                        raise TypeError(f"paths[{s}][{m}][{m1}][{m2}] must be a list")
                    parsed = []
                    for local_id, item in enumerate(block):
                        if not (
                            isinstance(item, list)
                            and len(item) == 3
                            and all(isinstance(component, list) for component in item)
                        ):
                            raise TypeError("compact path entries must be [tau, tau1, tau2]")
                        path = VirtualPath(
                            s=s,
                            m=m,
                            m1=m1,
                            m2=m2,
                            local_id=local_id,
                            global_id=global_id,
                            tau=tuple(int(value) for value in item[0]),
                            tau1=tuple(int(value) for value in item[1]),
                            tau2=tuple(int(value) for value in item[2]),
                        )
                        validate_virtual_path(path, max_virtual_order=s)
                        parsed.append(path)
                        global_id += 1
                    paths[s][m][m1][m2] = parsed
    return paths


def load_default_path_metadata(output_embedding: OutputEmbedding) -> PathMetadata:
    """Load saved project path metadata for an output embedding.

    Parameters
    ----------
    output_embedding : {"canonical", "full"}
        Saved path family to load.

    Returns
    -------
    PathMetadata
        Metadata loaded from ``tpen/cache``.
    """

    if output_embedding not in DEFAULT_PATH_FILES:
        raise ValueError(f"Unsupported output_embedding {output_embedding!r}")
    return PathMetadata.load(DEFAULT_PATH_FILES[output_embedding])


def generate_virtual_paths(
    *,
    max_order: int,
    max_virtual_order: int,
    output_embedding: OutputEmbedding,
) -> PathFamily:
    """Generate deterministic virtual-support paths.

    Parameters
    ----------
    max_order : int
        Maximum input/output body order.
    max_virtual_order : int
        Maximum hidden virtual support order.
    output_embedding : {"canonical", "full"}
        Whether the output map ``tau`` is fixed to ``(0, ..., m - 1)`` or all
        injective output maps are used.
    """

    if max_order <= 0:
        raise ValueError(f"max_order must be positive, got {max_order}")
    if max_virtual_order <= 0:
        raise ValueError(f"max_virtual_order must be positive, got {max_virtual_order}")
    if output_embedding not in {"canonical", "full"}:
        raise ValueError(f"Unsupported output_embedding {output_embedding!r}")

    paths: PathFamily = {}
    global_id = 0
    for s in range(1, max_virtual_order + 1):
        paths[s] = {}
        for m in range(1, min(max_order, s) + 1):
            paths[s][m] = {}
            # Canonical output embeddings fix tau = (0, ..., m - 1).
            # This is a gauge choice for the output injection, not independent
            # canonicalization of all injections. Relative input injections
            # tau1/tau2 remain part of the path data and carry the interaction
            # degrees of freedom.
            output_maps = [tuple(range(m))] if output_embedding == "canonical" else ordered_tuples(s, m)
            for m1 in range(1, min(max_order, s) + 1):
                paths[s][m][m1] = {}
                # Injections from [m1] into the virtual support
                left_maps = ordered_tuples(s, m1)
                for m2 in range(1, min(max_order, s) + 1):
                    # Injections from [m2] into the virtual support
                    right_maps = ordered_tuples(s, m2)
                    block: list[VirtualPath] = []
                    local_id = 0
                    # Iterate over all injections
                    for tau in output_maps:
                        for tau1 in left_maps:
                            for tau2 in right_maps:
                                path = VirtualPath(
                                    s=s,
                                    m=m,
                                    m1=m1,
                                    m2=m2,
                                    local_id=local_id,
                                    global_id=global_id,
                                    tau=tuple(tau),
                                    tau1=tuple(tau1),
                                    tau2=tuple(tau2),
                                )
                                if path.input_support != set(range(s)):
                                    continue
                                validate_virtual_path(path, max_virtual_order=max_virtual_order)
                                block.append(path)
                                local_id += 1
                                global_id += 1
                    paths[s][m][m1][m2] = sorted(
                        block,
                        key=lambda path: (path.s, path.m, path.m1, path.m2, path.tau, path.tau1, path.tau2),
                    )
    return paths


def iter_path_blocks(paths: PathFamily) -> Iterator[tuple[tuple[int, int, int, int], list[VirtualPath]]]:
    """Yield ``((s, m, m1, m2), paths)`` blocks in deterministic order."""

    for s in sorted(paths):
        for m in sorted(paths[s]):
            for m1 in sorted(paths[s][m]):
                for m2 in sorted(paths[s][m][m1]):
                    yield (s, m, m1, m2), paths[s][m][m1][m2]


def validate_virtual_path(path: VirtualPath, *, max_order: int | None = None, max_virtual_order: int | None = None) -> None:
    """Validate one virtual-support path."""

    limit = max_virtual_order if max_virtual_order is not None else max_order
    if limit is not None and path.s > limit:
        raise ValueError(f"Virtual support order {path.s} exceeds max_virtual_order {limit}")
    if path.m <= 0 or path.m1 <= 0 or path.m2 <= 0:
        raise ValueError("Path orders must be positive")
    if path.m > path.s:
        raise ValueError("m must be <= s")
    for name, order, injection in (
        ("tau", path.m, path.tau),
        ("tau1", path.m1, path.tau1),
        ("tau2", path.m2, path.tau2),
    ):
        if len(injection) != order:
            raise ValueError(f"{name} length must match its order")
        if len(set(injection)) != len(injection):
            raise ValueError(f"{name} must be injective")
        if any(label < 0 or label >= path.s for label in injection):
            raise ValueError(f"{name} labels must land in the virtual support")
    validate_tp_input_cover_support(path)


def validate_tp_input_cover_support(path: VirtualPath) -> None:
    """Validate the explicit TP input-cover-support contract."""

    if path.input_support != set(range(path.s)):
        raise ValueError("left and right injections must cover the virtual support")


def validate_linear_output_plus_input_cover_support(path: SupportPath) -> None:
    """Validate a unary path's canonical common-support contract.

    The canonical labels remove the arbitrary gauge of naming completion
    variables. Noncanonical records are rejected rather than silently merged.
    """

    if path.output_order <= 0 or path.input_order <= 0:
        raise ValueError("path orders must be positive")
    if path.normalization not in {"sum", "completion_mean"}:
        raise ValueError(f"Unsupported path normalization {path.normalization!r}")
    if path.tau_out != tuple(range(path.output_order)):
        raise ValueError("tau_out must use canonical output-slot labels")
    if len(path.tau_in) != path.input_order or len(set(path.tau_in)) != path.input_order:
        raise ValueError("tau_in must be an injective input-slot map")
    overlap = sum(label < path.output_order for label in path.tau_in)
    expected_free = iter(range(path.output_order, path.output_order + path.input_order - overlap))
    expected = tuple(label if label < path.output_order else next(expected_free) for label in path.tau_in)
    if path.tau_in != expected:
        raise ValueError("tau_in must use canonical matched and unmatched labels")


def enumerate_linear_support_paths(
    output_order: int,
    input_order: int,
    *,
    normalization: Literal["sum", "completion_mean"] = "completion_mean",
) -> tuple[SupportPath, ...]:
    """Enumerate canonical complete partial-matching paths.

    Paths are ordered by input order, overlap, matched slot pairs, then
    unmatched input-slot order, as fixed by the linear-mixing contract.
    """

    if output_order <= 0 or input_order <= 0:
        raise ValueError("path orders must be positive")
    records: list[tuple[tuple[object, ...], SupportPath]] = []
    output_slots = tuple(range(output_order))
    input_slots = tuple(range(input_order))
    for overlap in range(min(output_order, input_order) + 1):
        for matched_outputs in combinations(output_slots, overlap):
            for matched_inputs in combinations(input_slots, overlap):
                for output_assignment in permutations(matched_outputs):
                    matching = tuple(sorted(zip(output_assignment, matched_inputs)))
                    by_input = {input_slot: output_slot for output_slot, input_slot in matching}
                    tau_in = tuple(
                        by_input.get(
                            input_slot,
                            output_order
                            + sum(prior not in by_input for prior in input_slots[: input_slot + 1])
                            - 1,
                        )
                        for input_slot in input_slots
                    )
                    path = SupportPath(output_order, input_order, output_slots, tau_in, normalization)
                    key = (input_order, overlap, matching,
                           tuple(slot for slot in input_slots if slot not in by_input))
                    records.append((key, path))
    return tuple(path for _, path in sorted(records, key=lambda item: item[0]))


def _serialize_paths(paths: PathFamily) -> list[object]:
    serialized: list[object] = []
    for (s, m, m1, m2), block in iter_path_blocks(paths):
        by_m = _ensure_list_index(serialized, s)
        by_m1 = _ensure_list_index(by_m, m)
        by_m2 = _ensure_list_index(by_m1, m1)
        _ensure_list_index(by_m2, m2)
        by_m2[m2] = [[list(path.tau), list(path.tau1), list(path.tau2)] for path in block]
    return serialized


def _ensure_list_index(items: list[object], index: int) -> list[object]:
    while len(items) <= index:
        items.append([])
    value = items[index]
    if not isinstance(value, list):
        raise TypeError(f"Expected list at compact path index {index}")
    return value


__all__ = [
    "LinearPathMetadata",
    "LinearPathPolicy",
    "NormalizedChannels",
    "NormalizedOrders",
    "PathFamily",
    "PathFamilyLayout",
    "PathFamilyOutput",
    "PathEntry",
    "PathLayout",
    "PathMetadata",
    "OutputPathLayout",
    "SupportPath",
    "VirtualPath",
    "enumerate_linear_support_paths",
    "compose_path_layout",
    "generate_virtual_paths",
    "iter_path_blocks",
    "load_default_path_metadata",
    "normalize_linear_path_policy",
    "validate_virtual_path",
    "validate_linear_output_plus_input_cover_support",
    "validate_tp_input_cover_support",
]
