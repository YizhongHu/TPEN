"""Hazard generators frozen before reading the slice-A implementation.

Generate independent axes: construction mechanism, filesystem spelling, location,
and container depth. This module deliberately imports neither the validator nor
its target vocabulary. Receipts bind its original bytes by SHA-256.
"""

from copy import deepcopy
from pathlib import Path


def construction_specs(manifest: str, module_file: str, witness: str):
    """Yield named keyword and positional constructions that can read or execute."""
    # Discovering a symbol and executing a module are separate hazard families.
    for converter in ("all", "object"):
        yield f"relative-import-{converter}", {
            "_target_": "builtins.__import__", "name": "hi_manifest",
            "globals": {"__package__": "tpen"}, "level": 1,
            "_convert_": converter,
        }
    yield "relative-importlib", {
        "_target_": "importlib.import_module", "name": ".hi_manifest", "package": "tpen",
    }
    yield "execute-file", {"_target_": "runpy.run_path", "path_name": module_file}
    for target, argument in (
        ("runpy.run_module", "mod_name"),
        ("hydra.utils.get_class", "path"),
        ("hydra.utils.get_object", "path"),
        ("hydra.utils.get_method", "path"),
        ("pydoc.locate", "path"),
        ("pkgutil.resolve_name", "name"),
    ):
        yield target, {"_target_": target, argument: "tpen.hi_manifest"}
    # Indirection constructs the name from components instead of spelling it whole.
    yield "joined-import", {
        "_target_": "importlib.import_module", "name": {
            "_target_": "builtins.str.join", "_args_": [".", ["tpen", "hi_manifest"]],
        },
    }
    yield "getattr-package", {
        "_target_": "builtins.getattr", "_args_": [
            {"_target_": "importlib.import_module", "name": "tpen"}, "hi_manifest",
        ],
    }
    for target, argument in (
        ("omegaconf.OmegaConf.load", "file_"),
        ("builtins.open", "file"), ("io.open", "file"),
        ("codecs.open", "filename"), ("io.FileIO", "file"),
    ):
        yield target, {"_target_": target, argument: manifest}
    yield "copyfile", {"_target_": "shutil.copyfile", "src": manifest, "dst": witness}
    yield "copy", {"_target_": "shutil.copy", "src": manifest, "dst": witness}
    for target, argument in (("json.load", "fp"), ("yaml.safe_load", "stream"),
                             ("yaml.full_load", "stream")):
        yield target, {"_target_": target, argument: {
            "_target_": "builtins.open", "file": manifest,
        }}
    for method in ("read_text", "read_bytes", "open"):
        yield f"pathlib-{method}", {
            "_target_": f"pathlib.Path.{method}", "self": {
                "_target_": "pathlib.Path", "_args_": [manifest],
            },
        }
    yield "positional-open", {"_target_": "builtins.open", "_args_": [manifest]}


def file_spellings(path: Path, symlink: Path):
    """Include lexical and filesystem aliases; the caller owns the symlink."""
    yield "absolute", str(path.resolve())
    yield "dot", f"{path.parent}/./{path.name}"
    yield "parent", f"{path.parent}/../{path.parent.name}/{path.name}"
    yield "symlink", str(symlink)
    yield "case", str(path.with_name(path.name.upper()))
    yield "relative", f"./{path.name}"
    yield "double-separator", f"{path.parent}//{path.name}"


def mapping_paths(value, path=()):
    """Derive placements from every mapping, including mappings inside sequences."""
    if isinstance(value, dict):
        yield path
        for key, child in value.items():
            yield from mapping_paths(child, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from mapping_paths(child, (*path, index))


def nested_spec(spec, depth, container):
    """Wrap a construction without changing its eventual target."""
    result = deepcopy(spec)
    for index in range(depth):
        if container == "mapping":
            result = {f"argument_{index}": result}
        elif container == "list":
            result = [result]
        elif container == "tuple":
            result = (result,)
        else:
            raise ValueError(container)
    return result


def place_spec(config, path, spec, *, key="future_parameter", depth=0,
               container="mapping"):
    """Insert into a discovered mapping, without consulting allowed positions."""
    result = deepcopy(config)
    node = result
    for part in path:
        node = node[part]
    node[key] = nested_spec(spec, depth, container)
    return result


def renamed_keys(config):
    """Mutate each existing root section's argument names as a separate axis."""
    for section, value in config.items():
        if isinstance(value, dict):
            for key in value:
                if not key.startswith("_"):
                    yield (section,), f"{key}s"
    # This measured rename must remain expressible even if the base omits load.
    yield ("runner",), "loads"
