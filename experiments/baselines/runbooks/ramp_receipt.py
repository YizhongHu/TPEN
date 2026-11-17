"""Write and gate mechanical BL-R1 ramp-row receipts.

The helper deliberately makes no scientific judgement.  It records a producer
process outcome and rejects a ramp marker unless every named row wrote one
successful receipt.  It is independent of scheduler state, so its logic is
covered by a small offline test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Sequence

import yaml


def record(path: Path, *, row: str, returncode: int, producer_label: str) -> None:
    """Persist one outcome inside a pre-created row directory, once only."""

    if not row or "/" in row or "\\" in row:
        raise ValueError("row must be a non-empty leaf name")
    if not producer_label:
        raise ValueError("producer_label must be non-empty")
    if path.exists():
        raise FileExistsError(f"receipt already exists: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"receipt parent is not an existing directory: {path.parent}")
    path.write_text(
        json.dumps(
            {
                "schema": "bl-r1-ramp-row/v1",
                "row": row,
                "returncode": returncode,
                "producer_label": producer_label,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def gate(root: Path, expected_rows: Sequence[str]) -> dict[str, object]:
    """Return the complete receipt account or raise on any failed/missing row."""

    rows = list(expected_rows)
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("expected rows must be non-empty and unique")
    observed: list[dict[str, object]] = []
    failures: list[str] = []
    for row in rows:
        path = root / row / "receipt.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{row}: no readable receipt ({exc})")
            continue
        observed.append(payload)
        if payload.get("schema") != "bl-r1-ramp-row/v1" or payload.get("row") != row:
            failures.append(f"{row}: receipt identity mismatch")
        if not payload.get("producer_label"):
            failures.append(f"{row}: missing producer label")
        if payload.get("returncode") != 0:
            failures.append(f"{row}: producer returncode={payload.get('returncode')!r}")
    report: dict[str, object] = {"schema": "bl-r1-ramp-gate/v1", "rows": observed, "failures": failures}
    if failures:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


_FERMINET_TRACE_COLUMNS = ("step", "energy", "ewmean", "ewvar", "pmove", "walker_variance")
_FERMINET_CONFIG_HEADER = re.compile(
    r"(?m)^[IWEF]\d{4} \d{2}:\d{2}:\d{2}\.\d{6} \d+ main\.py:34\] System config:\s*$"
)
_LOG_RECORD = re.compile(r"(?m)^[IWEF]\d{4} ")
_FERMINET_ATOM_TAG = "tag:yaml.org,2002:python/object:ferminet.utils.system.Atom"


class _FerminetSafeLoader(yaml.SafeLoader):
    """Safe YAML loader for the exact, producer-logged FermiNet config."""


def _construct_tuple(loader: _FerminetSafeLoader, node: yaml.Node) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node, deep=True))


def _construct_mapping(
    loader: _FerminetSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


def _reject_python_tag(loader: _FerminetSafeLoader, tag_suffix: str, node: yaml.Node) -> Any:
    del loader
    raise yaml.constructor.ConstructorError(
        None,
        None,
        f"unsupported Python YAML tag: {tag_suffix}",
        node.start_mark,
    )


_FerminetSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)
_FerminetSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)
_FerminetSafeLoader.add_constructor(_FERMINET_ATOM_TAG, _construct_mapping)
_FerminetSafeLoader.add_multi_constructor("tag:yaml.org,2002:python/", _reject_python_tag)


def _producer_yaml_block(stderr: str) -> str:
    markers = list(_FERMINET_CONFIG_HEADER.finditer(stderr))
    if len(markers) != 1:
        raise ValueError(f"expected one pinned FermiNet System config record, found {len(markers)}")
    start = markers[0].end()
    terminator = _LOG_RECORD.search(stderr, start)
    if terminator is None:
        raise ValueError("producer system config record is truncated at EOF")
    block = stderr[start : terminator.start()]
    if not block.strip():
        raise ValueError("producer System config record is truncated")
    return block


def _mapping_value(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"incomplete producer config record: missing={path!r}")
        value = value[component]
    return value


def _sequence_value(value: Any, *, path: str, length: int, item_type: type | tuple[type, ...]) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"producer config {path} must be a sequence")
    if len(value) != length:
        raise ValueError(f"producer config {path} must contain {length} values")
    if item_type is int:
        valid = all(type(item) is int for item in value)
    else:
        valid = all(isinstance(item, item_type) and not isinstance(item, bool) for item in value)
    if not valid:
        raise ValueError(f"producer config {path} has invalid value types")
    return list(value)


def parse_ferminet_logged_config(stderr: str, *, steps: int, expected_network_type: str) -> dict[str, object]:
    """Parse and validate FermiNet's complete, producer-logged config dump."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if not expected_network_type:
        raise ValueError("expected_network_type must be non-empty")
    try:
        config = yaml.load(_producer_yaml_block(stderr), Loader=_FerminetSafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid producer config YAML: {exc}") from exc
    if not isinstance(config, Mapping):
        raise ValueError("producer System config must be a YAML mapping")

    required = {
        "batch_size": 4096,
        "debug.check_nan": True,
        "debug.deterministic": True,
        "debug.seed": 0,
        "optim.iterations": steps,
        "optim.reset_if_nan": False,
        "pretrain.iterations": 0,
    }
    observed: dict[str, object] = {}
    output_names = {
        "optim.iterations": "iterations",
        "pretrain.iterations": "pretrain_iterations",
    }
    for path, wanted in required.items():
        value = _mapping_value(config, path)
        observed[output_names.get(path, path.rsplit(".", 1)[-1])] = value
        if value != wanted:
            raise ValueError(f"producer config {path}={value!r}, expected {wanted!r}")

    network_type = _mapping_value(config, "network.network_type")
    if network_type != expected_network_type:
        raise ValueError(f"producer network_type={network_type!r}, expected {expected_network_type!r}")
    system = _mapping_value(config, "system")
    electrons = _sequence_value(_mapping_value(system, "electrons"), path="system.electrons", length=2, item_type=int)
    if electrons != [1, 0]:
        raise ValueError(f"producer system electrons={electrons!r}, expected (1, 0)")
    molecule = _mapping_value(system, "molecule")
    if not isinstance(molecule, list) or len(molecule) != 1 or not isinstance(molecule[0], Mapping):
        raise ValueError("producer nucleus identity is incomplete: expected one atom")
    atom = molecule[0]
    atomic_number = _mapping_value(atom, "atomic_number")
    nucleus_charge = _mapping_value(atom, "charge")
    nucleus_coords = _sequence_value(_mapping_value(atom, "coords"), path="system.molecule[0].coords", length=3, item_type=Real)
    _mapping_value(atom, "symbol")
    if _mapping_value(system, "charge") != 0 or atomic_number != 1 or nucleus_charge != 1.0:
        raise ValueError("producer nucleus identity is not one H nucleus with charge 1")
    if nucleus_coords != [0.0, 0.0, 0.0]:
        raise ValueError(f"producer nucleus origin={nucleus_coords!r}, expected (0.0, 0.0, 0.0)")

    observed.update(
        {
            "network_type": network_type,
            "nucleus_count": len(molecule),
            "nucleus_atomic_number": atomic_number,
            "nucleus_charge": nucleus_charge,
            "nucleus_coords": list(nucleus_coords),
            "electrons": list(electrons),
        }
    )
    return observed


def validate_ferminet_trace(path: Path, *, steps: int) -> dict[str, object]:
    """Require the pinned FermiNet training schema and a complete finite trace."""

    if steps < 1:
        raise ValueError("steps must be positive")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(_FERMINET_TRACE_COLUMNS):
            raise ValueError(f"unexpected FermiNet schema: {reader.fieldnames!r}")
        rows = list(reader)
    if len(rows) != steps:
        raise ValueError(f"expected {steps} trace rows, found {len(rows)}")
    for expected_step, row in enumerate(rows):
        if set(row) != set(_FERMINET_TRACE_COLUMNS):
            raise ValueError(f"row {expected_step}: malformed columns")
        for column in _FERMINET_TRACE_COLUMNS:
            value = row[column]
            if value in (None, ""):
                raise ValueError(f"row {expected_step}: missing {column}")
            try:
                number = float(value)
            except ValueError as exc:
                raise ValueError(f"row {expected_step}: non-numeric {column}") from exc
            if not math.isfinite(number):
                raise ValueError(f"row {expected_step}: non-finite {column}")
        if int(float(row["step"])) != expected_step or float(row["step"]) != expected_step:
            raise ValueError(f"row {expected_step}: unexpected step {row['step']!r}")
    return {"schema": "bl-r1-ferminet-trace/v1", "path": str(path), "steps": steps}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ramp_required_paths() -> list[str]:
    """Return the fixed relative evidence set for the two-row BL-R1 ramp."""

    required = [
        "ramp/mode.json",
        "ramp/gpu-inventory.txt",
        "ramp/ferminet-gate.json",
        "ramp/ramp.complete",
    ]
    for row in ("ferminet-codebase-ferminet-seed0", "ferminet-codebase-psiformer-seed0"):
        required.extend(
            [
                f"ramp/{row}/{name}"
                for name in (
                    "stdout.log", "stderr.log", "start.txt", "end.txt", "timing.txt", "receipt.json", "backend.json",
                    "producer-config.json", "train_stats.csv",
                )
            ]
        )
    return required


def inventory(root: Path, output: Path, required: Sequence[str]) -> dict[str, object]:
    """Write a deterministic, partial-safe artifact inventory, then flag gaps."""

    root = root.resolve()
    entries: list[dict[str, object]] = []
    missing: list[str] = []
    for relative in sorted(set(required)):
        candidate = (root / relative).resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError(f"inventory path escapes root: {relative}")
        entry: dict[str, object] = {"path": relative}
        if candidate.is_file():
            entry.update({"status": "present", "bytes": candidate.stat().st_size, "sha256": _sha256(candidate)})
        else:
            entry["status"] = "missing"
            missing.append(relative)
        entries.append(entry)
    output.write_text("\n".join(json.dumps(entry, sort_keys=True) for entry in entries) + "\n", encoding="utf-8")
    report = {"schema": "bl-r1-artifact-inventory/v1", "entries": entries, "missing": missing}
    if missing:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


def check_qstat_poll(path: Path) -> dict[str, str]:
    """Validate the run-count contract for one preserved PBS observation."""

    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    state = fields.get("job_state")
    run_count = fields.get("run_count")
    if state in {"Q", "H", "W", "T"}:
        if run_count not in (None, "0"):
            raise ValueError(f"state {state} requires absent or zero run_count, found {run_count!r}")
        return {"state": state, "phase": "queue"}
    if state in {"R", "B", "E", "F"}:
        if run_count != "1":
            raise ValueError(f"state {state} requires run_count=1, found {run_count!r}")
        if state == "F":
            if fields.get("Exit_status") != "0":
                raise ValueError(f"finished job requires Exit_status=0, found {fields.get('Exit_status')!r}")
            return {"state": state, "phase": "terminal"}
        return {"state": state, "phase": "run"}
    raise ValueError(f"unknown or missing PBS job_state: {state!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the receipt writer or gate."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--path", type=Path, required=True)
    record_parser.add_argument("--row", required=True)
    record_parser.add_argument("--returncode", type=int, required=True)
    record_parser.add_argument("--producer-label", required=True)
    gate_parser = subparsers.add_parser("gate")
    gate_parser.add_argument("--root", type=Path, required=True)
    gate_parser.add_argument("--row", action="append", required=True)
    trace_parser = subparsers.add_parser("validate-ferminet-trace")
    trace_parser.add_argument("--path", type=Path, required=True)
    trace_parser.add_argument("--steps", type=int, required=True)
    config_parser = subparsers.add_parser("validate-ferminet-config")
    config_parser.add_argument("--path", type=Path, required=True)
    config_parser.add_argument("--steps", type=int, required=True)
    config_parser.add_argument("--expected-network-type", required=True)
    config_parser.add_argument("--output", type=Path, required=True)
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--root", type=Path, required=True)
    inventory_parser.add_argument("--output", type=Path, required=True)
    inventory_parser.add_argument("--required", action="append", default=[])
    inventory_parser.add_argument("--ramp-required", action="store_true")
    poll_parser = subparsers.add_parser("poll-check")
    poll_parser.add_argument("--path", type=Path, required=True)
    poll_parser.add_argument("--field", choices=("state", "phase"))
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            record(args.path, row=args.row, returncode=args.returncode, producer_label=args.producer_label)
        elif args.command == "gate":
            print(json.dumps(gate(args.root, args.row), sort_keys=True))
        elif args.command == "validate-ferminet-trace":
            print(json.dumps(validate_ferminet_trace(args.path, steps=args.steps), sort_keys=True))
        elif args.command == "validate-ferminet-config":
            result = parse_ferminet_logged_config(args.path.read_text(encoding="utf-8"), steps=args.steps, expected_network_type=args.expected_network_type)
            args.output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(result, sort_keys=True))
        elif args.command == "inventory":
            required = args.required + (ramp_required_paths() if args.ramp_required else [])
            print(json.dumps(inventory(args.root, args.output, required), sort_keys=True))
        else:
            result = check_qstat_poll(args.path)
            print(result[args.field] if args.field else json.dumps(result, sort_keys=True))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
