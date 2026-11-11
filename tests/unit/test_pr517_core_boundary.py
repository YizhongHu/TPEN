"""Reviewer tests for the PR 517 core/experiment and schema boundaries.

These are intentionally core tests: they do not import or fixture any code from
``experiments``.  The source census is static and therefore cannot prove that
computed import names or runtime control-flow paths are unreachable; it covers
the literal forms used by this repository.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.config_schema import ClosedSchemaError
from tpen.hi_manifest import HI_EVALUATION_SCHEMA, load_evaluation_manifest, reference_energy
from tpen.hi_schema import HI_TRAIN_SCHEMA, validate_hi_train_config


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_Tpen_ROOT = _SOURCE_ROOT / "tpen"


def _is_experiments_name(name: str) -> bool:
    """Return whether ``name`` is the experiments package or a descendant."""

    return name == "experiments" or name.startswith("experiments.")


def _literal_string(node: ast.AST | None) -> str | None:
    """Return a literal string, deliberately rejecting computed expressions."""

    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _forbidden_imports(source: str) -> set[str]:
    """Find literal imports of ``experiments`` without matching prose strings.

    Direct imports, ``from`` imports, and the importlib/``__import__`` aliases
    are all resolved from syntax.  This modest static check intentionally does
    not claim to reason about computed names or control-flow reachability.
    """

    tree = ast.parse(source)
    importlib_names = {"importlib"}
    import_module_names = set()
    builtin_import_names = {"__import__"}
    builtins_names = {"builtins"}
    hits: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if alias.name == "importlib":
                    importlib_names.add(bound)
                elif alias.name == "builtins":
                    builtins_names.add(bound)
                elif alias.name == "builtins.__import__":
                    builtin_import_names.add(bound)
                if _is_experiments_name(alias.name):
                    hits.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                continue
            for alias in node.names:
                bound = alias.asname or alias.name
                if module == "importlib" and alias.name == "import_module":
                    import_module_names.add(bound)
                if module == "builtins" and alias.name == "__import__":
                    builtin_import_names.add(bound)
            if _is_experiments_name(module):
                hits.add(module)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        is_import_module = (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
            and isinstance(function.value, ast.Name)
            and function.value.id in importlib_names
        ) or (isinstance(function, ast.Name) and function.id in import_module_names)
        is_builtin_import = (
            isinstance(function, ast.Name) and function.id in builtin_import_names
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr == "__import__"
            and isinstance(function.value, ast.Name)
            and function.value.id in builtins_names
        )
        if not (is_import_module or is_builtin_import):
            continue
        name = _literal_string(node.args[0]) if node.args else None
        if name is None:
            for keyword in node.keywords:
                if keyword.arg == "name":
                    name = _literal_string(keyword.value)
                    break
        if name is not None and _is_experiments_name(name):
            hits.add(name)
    return hits


def test_t1_core_source_has_no_experiments_imports() -> None:
    """Recursively census the nonempty, known core source tree from this file."""

    paths = sorted(_Tpen_ROOT.rglob("*.py"))
    assert paths, "source corpus is empty; boundary test would pass vacuously"
    print(f"T1 recursive tpen Python files: {len(paths)}")
    assert (_Tpen_ROOT / "run.py") in paths
    assert (_Tpen_ROOT / "hi_schema.py") in paths
    assert (_Tpen_ROOT / "hi_manifest.py") in paths
    findings = {str(path.relative_to(_SOURCE_ROOT)): _forbidden_imports(path.read_text()) for path in paths}
    findings = {path: hits for path, hits in findings.items() if hits}
    print(f"T1 corpus triggered imports: {findings!r}")
    assert findings == {}, findings


@pytest.mark.parametrize(
    ("name", "source", "expected"),
    [
        ("direct-dotted", "import experiments.atomistic", {"experiments.atomistic"}),
        ("from-package", "from experiments import atomistic", {"experiments"}),
        ("from-dotted", "from experiments.atomistic import x", {"experiments.atomistic"}),
        ("importlib-attribute", "import importlib as il\nil.import_module('experiments.atomistic')", {"experiments.atomistic"}),
        ("importlib-bare", "from importlib import import_module\nimport_module('experiments.atomistic')", {"experiments.atomistic"}),
        ("historical-hyphenated-edge", 'from importlib import import_module\nimport_module("experiments.atomistic.he-importance.stage_coordinate")', {"experiments.atomistic.he-importance.stage_coordinate"}),
        ("importlib-keyword-alias", "from importlib import import_module as load\nload(name='experiments.atomistic')", {"experiments.atomistic"}),
        ("builtin-attribute", "import builtins as b\nb.__import__('experiments.atomistic')", {"experiments.atomistic"}),
        ("builtin-bare-alias", "from builtins import __import__ as load\nload('experiments.atomistic')", {"experiments.atomistic"}),
        ("identifier-spelling-1", "experiments = 'experiments.atomistic'", set()),
        ("identifier-spelling-2", "experiments_atomistic = 'experiments.atomistic'", set()),
        ("schema-identifier-train", "HI_TRAIN_SCHEMA = 'tpen.hi.train.v1'", set()),
        ("schema-identifier-evaluation", "HI_EVALUATION_SCHEMA = 'tpen.hi.evaluation.v1'", set()),
        ("docstring-spelling-1", '"experiments.atomistic import x"', set()),
        ("docstring-spelling-2", '"from experiments.atomistic import x"', set()),
        ("comment-trap-1", "# experiments.atomistic import x", set()),
        ("comment-trap-2", "# import experiments.atomistic", set()),
        ("docstring-naivegrep-trap", '"import experiments.atomistic"', set()),
        ("harmless-string", "'experiments.atomistic'\n\"experiments\"", set()),
        ("computed-concatenation-excluded", "from importlib import import_module\nimport_module('experi' + 'ments.atomistic')", set()),
        ("nonliteral-name", "import importlib\nimportlib.import_module(module_name)", set()),
        ("sibling-experiment-utils", "import tpen.experiments_utils", set()),
        ("relative-experiments-import", "from .experiments import x", set()),
        ("relative-importlib-alias", "from .importlib import import_module as load\nload('experiments.atomistic')", set()),
    ],
)
def test_t1_static_import_controls(name: str, source: str, expected: set[str]) -> None:
    """Self-controls pin executable forms and benign identifier/prose sites."""

    actual = _forbidden_imports(source)
    print(f"T1 control {name}: triggered={sorted(actual)!r}")
    assert actual == expected


def test_t2_schema_modules_are_importable_without_forbidden_namespace(tmp_path: Path) -> None:
    """Exercise core imports and train/evaluation schemas in a clean process."""

    script = textwrap.dedent(
        r'''
        import builtins
        import importlib
        import importlib.abc
        import json
        import sys
        from pathlib import Path

        blocked = ("tpen.hi", "experiments")
        attempts = []
        source_root = Path(sys.argv[2]).resolve()

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
                    attempts.append(fullname)
                    raise ModuleNotFoundError("blocked by T2: " + fullname)
                return None

        sys.meta_path.insert(0, Blocker())
        assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in blocked)
        for name, importer in (
            ("tpen.hi", lambda: __import__("tpen.hi")),
            ("experiments", lambda: importlib.import_module("experiments.atomistic.he-importance.stage_coordinate")),
        ):
            try:
                importer()
            except ModuleNotFoundError:
                pass
        assert attempts == ["tpen.hi", "experiments"]
        positive_control_attempts = list(attempts)
        attempts.clear()

        import tpen
        import tpen.hi_manifest as hi_manifest
        import tpen.hi_schema as hi_schema
        import tpen.run as run
        from omegaconf import OmegaConf

        assert tpen and run and hi_schema and hi_manifest
        assert "" not in sys.path and str(Path.cwd()) not in sys.path
        assert attempts == []
        assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in blocked)
        provenance = {
            name: getattr(module, "__file__", None)
            for name, module in {
                "tpen": tpen,
                "tpen.run": run,
                "tpen.hi_schema": hi_schema,
                "tpen.hi_manifest": hi_manifest,
            }.items()
        }
        assert all(isinstance(path, str) and path for path in provenance.values())
        assert all(source_root == Path(path).resolve() or source_root in Path(path).resolve().parents for path in provenance.values())
        assert hi_schema.HI_TRAIN_SCHEMA == "tpen.hi.train.v1"
        assert hi_manifest.HI_EVALUATION_SCHEMA == "tpen.hi.evaluation.v1"

        train = OmegaConf.create({
            "schema": hi_schema.HI_TRAIN_SCHEMA,
            "optimizer": {"_target_": "torch.optim.Adam", "_partial_": True, "lr": 0.005},
        })
        hi_schema.validate_hi_train_config(train, env={})
        assert attempts == []
        assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in blocked)

        reference_cfg = OmegaConf.create({
            "schema": hi_schema.HI_TRAIN_SCHEMA,
            "optimizer": {"_target_": "torch.optim.Adam", "_partial_": True, "lr": 0.005},
            "system": {"reference_energy": -2.9},
        })
        try:
            hi_schema.validate_hi_train_config(reference_cfg, env={})
        except hi_schema.ClosedSchemaError as error:
            assert any(r.rule == "forbidden-surface:reference" and r.path == "system.reference_energy" for r in error.rejections)
        else:
            raise AssertionError("reference-bearing training config unexpectedly passed")

        for bad in (
            OmegaConf.create({"experiment": {"name": hi_schema.HI_EXPERIMENT_NAME}}),
            OmegaConf.create({"schema": hi_manifest.HI_EVALUATION_SCHEMA, "experiment": {"name": hi_schema.HI_EXPERIMENT_NAME}}),
        ):
            try:
                hi_schema.validate_hi_train_config(bad, env={})
            except hi_schema.ClosedSchemaError as error:
                assert any(r.rule == "undeclared-schema" and r.path == "schema" for r in error.rejections)
            else:
                raise AssertionError("wrong/missing HI schema unexpectedly passed")

        path = Path(sys.argv[1])
        path.write_text("schema: tpen.hi.evaluation.v1\nreference:\n  energy: -2.9\n  qualification: infinite_mass_nonrelativistic\n  units: hartree\n  system_id: he_atom\n")
        manifest = hi_manifest.load_evaluation_manifest(path)
        reference = hi_manifest.reference_energy(manifest)
        assert reference.energy == -2.9 and reference.system_id == "he_atom"
        assert attempts == []
        assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in blocked)
        path.write_text("schema: tpen.hi.train.v1\n")
        try:
            hi_manifest.load_evaluation_manifest(path)
        except ValueError as error:
            assert hi_manifest.HI_EVALUATION_SCHEMA in str(error)
        else:
            raise AssertionError("wrong evaluation schema unexpectedly passed")
        assert attempts == []
        assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in blocked)
        print(json.dumps({"status": "ok", "attempts": attempts, "positive_control_attempts": positive_control_attempts, "provenance": provenance}, sort_keys=True))
        ''',
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "manifest.yaml"), str(_SOURCE_ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    print(json.dumps(report, sort_keys=True))
    assert report["status"] == "ok"
    assert report["positive_control_attempts"] == ["tpen.hi", "experiments"]
    assert report["attempts"] == []
    assert set(report["provenance"]) == {"tpen", "tpen.run", "tpen.hi_schema", "tpen.hi_manifest"}
