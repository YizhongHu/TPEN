"""Artifact-based verdict for a retained Stage-Q GPU run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import math
import os
import tempfile


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _predicate(name: str, expected: object, observed: object, status: str | None = None) -> dict:
    result = status or ("PASS" if expected == observed else "FAIL")
    return {"name": name, "status": result, "expected": expected, "observed": observed}


def _numbers_finite(value: object) -> bool:
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_numbers_finite(item) for item in value)
    return True


def _torch_tensors_finite(root: Path) -> bool:
    """Load retained model tensors on CPU and reject non-finite values."""
    model_files = list(root.rglob("model.pt")) + list(root.rglob("model.pth"))
    if not model_files:
        return False
    try:
        import torch
        for path in model_files:
            state = torch.load(path, map_location="cpu", weights_only=True)
            values = state.values() if isinstance(state, dict) else (state,)
            if not all(not hasattr(value, "is_floating_point") or bool(torch.isfinite(value).all()) for value in values):
                return False
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _find(root: Path, names: tuple[str, ...]) -> list[Path]:
    return [path for name in names for path in root.rglob(name)]


def qualify(manifest_path: Path, evidence: Path, *, artifact_root: Path | None = None) -> int:
    """Write the atomic qualification verdict and return its process code."""
    manifest = _json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("runtime_kind") != "polaris-gpu-qualification":
        raise ValueError("manifest has wrong runtime_kind")
    root = artifact_root or Path(str(manifest["output_root"]))
    cell = manifest["cell"]
    predicates: list[dict] = []
    selection = _find(root, ("selection_receipt.json",))
    resolved = _find(root, ("resolved_config.yaml",))
    predicates.append(_predicate("cell_complete", True, any(path.parent.joinpath("COMPLETE").is_file() for path in selection)))
    predicates.append(_predicate("selection_receipt", True, bool(selection)))
    predicates.append(_predicate("resolved_config", True, bool(resolved)))
    checkpoint_manifests = _find(root, ("manifest.json", "checkpoint-manifest.json"))
    trainer_states = _find(root, ("trainer.json",))
    completed_values: list[int] = []
    for path in checkpoint_manifests + trainer_states:
        try:
            value = _json(path)
            if isinstance(value, dict):
                for key in ("completed_updates", "next_iteration", "updates"):
                    if isinstance(value.get(key), int):
                        completed_values.append(value[key])
        except (OSError, json.JSONDecodeError):
            continue
    final_dirs = list(root.rglob(f"step_{cell['updates']:06d}")) + list(root.rglob(f"step_{cell['updates']:04d}"))
    final_counters: list[int] = []
    for directory in final_dirs:
        for filename in ("manifest.json", "trainer.json"):
            try:
                value = _json(directory / filename)
                if isinstance(value, dict):
                    final_counters.extend(value.get(key) for key in ("completed_updates", "next_iteration") if isinstance(value.get(key), int))
            except (OSError, json.JSONDecodeError):
                pass
    counters_ok = len(final_counters) >= 2 and all(value == cell["updates"] for value in final_counters)
    predicates.append(_predicate("completed_updates", cell["updates"], final_counters or completed_values, "PASS" if counters_ok else "FAIL"))
    expected_steps = range(cell["checkpoint_every"], cell["updates"] + 1, cell["checkpoint_every"])
    step_results = []
    for step in expected_steps:
        candidates = list(root.rglob(f"step_{step:06d}")) + list(root.rglob(f"step_{step:04d}"))
        step_results.append({"step": step, "complete": any(path.joinpath("COMPLETE").is_file() for path in candidates)})
    predicates.append(_predicate("checkpoint_complete", True, all(row["complete"] for row in step_results), "PASS" if step_results and all(row["complete"] for row in step_results) else "FAIL"))
    tensor_files = [path for path in root.rglob("*.json") if path.name in {"final-tensors.json", "tensors.json"}]
    finite = all(_numbers_finite(_json(path)) for path in tensor_files) if tensor_files else _torch_tensors_finite(root)
    predicates.append(_predicate("final_tensors_finite", True, finite, "PASS" if finite else "FAIL"))
    owners = sorted({str(path.parent) for path in selection + resolved})
    predicates.append(_predicate("artifact_owners", "selection and resolved owners recorded", owners, "PASS" if owners else "FAIL"))
    resolved_identity = {"context": cell["context"], "walkers": cell["walkers"]}
    if selection:
        try:
            selected = _json(selection[0]).get("selection", {}).get("selected", {})
            resolved_identity = {"context": selected.get("context", cell["context"]), "walkers": cell["walkers"]}
        except (AttributeError, OSError, json.JSONDecodeError):
            pass
    predicates.append(_predicate("resolved_cell_identity", {"context": cell["context"], "walkers": cell["walkers"]}, resolved_identity, "PASS" if resolved_identity == {"context": cell["context"], "walkers": cell["walkers"]} else "FAIL"))
    if not checkpoint_manifests or not trainer_states:
        predicates.append(_predicate("artifact_counters", True, False, "UNMEASURED"))
    status = "PASS" if all(row["status"] == "PASS" for row in predicates) else ("UNMEASURED" if any(row["status"] == "UNMEASURED" for row in predicates) else "FAIL")
    verdict = {"schema": 1, "runtime_kind": manifest["runtime_kind"], "status": status, "exit_code": 0 if status == "PASS" else 2, "manifest": str(manifest_path), "predicates": predicates, "artifact_root": str(root)}
    evidence.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=evidence, prefix="qualification.", suffix=".tmp", delete=False, encoding="utf-8") as stream:
        json.dump(verdict, stream, indent=2, sort_keys=True); stream.write("\n"); temporary = Path(stream.name)
    temporary.replace(evidence / "qualification.json")
    return int(verdict["exit_code"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args(argv)
    return qualify(args.manifest, args.evidence, artifact_root=args.artifact_root)


if __name__ == "__main__":
    raise SystemExit(main())
