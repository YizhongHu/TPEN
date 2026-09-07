from __future__ import annotations

import ast
from pathlib import Path

import yaml
from omegaconf import OmegaConf

# Siblings are loaded study-scoped, not by bare import: experiments/ has several
# same-named modules and the first study loaded would otherwise own the bare name
# for every study after it. See experiments/toolkit/study_imports.py.
#
# The loader is reached BY PATH rather than by putting the repository root on
# sys.path. A study directory that mutates sys.path is the mechanism behind the
# very defect this import exists to fix, and he-cutover's gateway test forbids it
# outright -- so the fix must not reintroduce it in order to install itself.
import importlib.util as _tpen_importlib  # noqa: E402
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

if "_tpen_study_imports" not in _tpen_sys.modules:
    _tpen_spec = _tpen_importlib.spec_from_file_location(
        "_tpen_study_imports",
        _TpenPath(__file__).resolve().parents[3] / "experiments" / "toolkit" / "study_imports.py",
    )
    _tpen_module = _tpen_importlib.module_from_spec(_tpen_spec)
    _tpen_sys.modules["_tpen_study_imports"] = _tpen_module
    _tpen_spec.loader.exec_module(_tpen_module)
sibling = _tpen_sys.modules["_tpen_study_imports"].sibling

from experiments.toolkit.ast_bindings import sys_module_names  # noqa: E402

run_train_row = sibling(__file__, 'run_train_row')


ROOT = Path(__file__).resolve().parents[3]
HEV1_BARE_MODULES = {"layout", "strata", "plan", "driver", "eval", "collect", "canary", "launch"}


def _differences(before, after, prefix=""):
    differences = set()
    if isinstance(before, dict) and isinstance(after, dict):
        for key in before.keys() | after.keys():
            differences |= _differences(before.get(key), after.get(key), f"{prefix}.{key}".strip("."))
    elif isinstance(before, list) and isinstance(after, list):
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            label = left.get("_target_", str(index)) if isinstance(left, dict) else str(index)
            differences |= _differences(left, right, f"{prefix}.{label}")
    elif before != after:
        differences.add(prefix)
    return differences


def test_smoke_training_diff_is_exactly_three_scale_fields() -> None:
    path = ROOT / "experiments/atomistic/he-v1/configs/train.yaml"
    cfg = OmegaConf.load(path)
    before = OmegaConf.to_container(cfg, resolve=False)
    run_train_row.configure_smoke_training(cfg, {"max_steps": 25, "n_walkers": 16})
    after = OmegaConf.to_container(cfg, resolve=False)
    assert _differences(before, after) == {"trainer.max_steps", "sampler.n_walkers", "callbacks.tpen.callback.Checkpoint.schedule.every_n"}


def test_smoke_training_finds_checkpoint_by_target_when_not_last() -> None:
    path = ROOT / "experiments/atomistic/he-v1/configs/train.yaml"
    cfg = OmegaConf.load(path)
    checkpoint = next(callback for callback in cfg.callbacks if callback.get("_target_") == "tpen.callback.Checkpoint")
    cfg.callbacks.append({"_target_": "tests.DummyCallback", "every_n_steps": 999})

    run_train_row.configure_smoke_training(cfg, {"max_steps": 25, "n_walkers": 16})

    assert checkpoint.schedule.every_n == 25
    assert cfg.callbacks[-1].every_n_steps == 999


def test_production_training_preserves_frozen_checkpoint_cadence() -> None:
    path = ROOT / "experiments/atomistic/he-v1/configs/train.yaml"
    cfg = OmegaConf.load(path)
    checkpoint = next(callback for callback in cfg.callbacks if callback.get("_target_") == "tpen.callback.Checkpoint")
    run_train_row.configure_training(
        cfg, {"scale": "production", "max_steps": 300000, "n_walkers": 4096}
    )
    assert checkpoint.schedule.every_n == 25000


def test_profiles_contain_policy_but_no_filesystem_roots() -> None:
    profiles = Path(__file__).with_name("profiles")
    cannon = yaml.safe_load((profiles / "cannon.yaml").read_text())
    polaris = yaml.safe_load((profiles / "polaris.yaml").read_text())
    assert cannon["runtimes"]["tpen-cu126"]["binding"] == "inherit"
    assert polaris["runtimes"]["tpen-polaris"]["available_accelerators"] == ["0", "1", "2", "3"]
    for path in profiles.iterdir():
        payload = yaml.safe_load(path.read_text())
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, str):
                assert not value.startswith("/")


# Binding analysis has ONE owner. This rule and the sys.modules rule in
# tests/unit/experiments/test_study_module_identity.py had a copy each, and the
# copies were wrong in the same way at the same time -- the duplication shape
# that produced two of this slice's own defects.
def _cross_study_violations(study: Path) -> set[str]:
    """Return gateway violations for the he-cutover study directory.

    THE SUPPORTED OPERATION SET, stated exactly rather than implied: a CALL to
    ``sys.path.insert``, ``.append`` or ``.extend``, where the ``sys`` binding
    is resolved by ``experiments.toolkit.ast_bindings.sys_module_names``.

    Other ways to change the same list are NOT detected, each MEASURED against a
    running mutant rather than assumed: ``sys.path += [...]`` (augmented
    assignment), ``sys.path = [...]`` (whole-list replacement),
    ``sys.path.remove(...)``, and ``sys.path[:0] = [...]`` (slice assignment).
    See ``sys_module_names`` for the binding forms that also evade it.

    Deliberately narrow. The realistic case is a study author reaching for a
    sibling the ordinary way; an author routing around the guard is not the
    threat model, and no guard of this shape would stop one.
    """

    violations = set()
    for path in study.glob("*.py"):
        if path.name == "hev1.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        sys_aliases = sys_module_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name.split(".", 1)[0] in HEV1_BARE_MODULES for alias in node.names):
                violations.add(f"{path.name}:bare-import")
                continue
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0] in HEV1_BARE_MODULES:
                violations.add(f"{path.name}:bare-from-import")
                continue
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                node.func.attr in {"insert", "append", "extend"}
                and isinstance(owner, ast.Attribute)
                and owner.attr == "path"
                and isinstance(owner.value, ast.Name)
                and owner.value.id in sys_aliases
            ):
                violations.add(f"{path.name}:sys-path-{node.func.attr}")
    return violations


def test_hev1_is_the_only_cross_study_accessor_and_import_gateway() -> None:
    study = Path(__file__).resolve().parent
    assert _cross_study_violations(study) == set()
    assert not (study / "configs").exists()
    assert not ({path.stem for path in study.glob("*.py")} & HEV1_BARE_MODULES)


def test_sys_path_rule_resolves_aliases_not_spellings(tmp_path: Path) -> None:
    """An aliased ``sys`` must not evade the sys.path rule.

    Matching ``owner.value.id == "sys"`` matched a spelling. Ordinary code in
    this repository imports ``sys`` under an alias, so the literal match was
    already evadable by practice rather than only in principle -- the study
    bootstrap is the existence proof.
    """

    (tmp_path / "aliased.py").write_text(
        "import sys as _s\nfrom pathlib import Path\n_s.path.insert(0, str(Path('/tmp')))\n",
        encoding="utf-8",
    )

    assert _cross_study_violations(tmp_path) == {"aliased.py:sys-path-insert"}


def test_sys_path_rule_ignores_an_unrelated_path_attribute(tmp_path: Path) -> None:
    """``.path.insert`` on something that is not the sys module is not a violation."""

    (tmp_path / "unrelated.py").write_text(
        "class Cfg:\n    def __init__(self):\n        self.path = []\n"
        "cfg = Cfg()\ncfg.path.insert(0, 'x')\n",
        encoding="utf-8",
    )

    assert _cross_study_violations(tmp_path) == set()


def test_sys_path_rule_does_not_accuse_a_rebound_alias(tmp_path: Path) -> None:
    """FALSE POSITIVE, measured: a rebound name must not be treated as ``sys``.

    An over-restrictive rule passes every test written for it and surfaces only
    when it refuses somebody's legitimate code, so this direction needs its own
    exhibit rather than trust.
    """

    (tmp_path / "entry.py").write_text(
        "import sys as s\n"
        "class Cfg:\n    def __init__(self):\n        self.path = []\n"
        "s = Cfg()\n"
        "s.path.append('x')\n",
        encoding="utf-8",
    )

    assert _cross_study_violations(tmp_path) == set()


def test_sys_path_rule_does_not_accuse_a_locally_shadowed_name(tmp_path: Path) -> None:
    """FALSE POSITIVE, measured: a function-local shadow is not the sys module."""

    (tmp_path / "entry.py").write_text(
        "import sys\n"
        "def go(sys):\n"
        "    sys.path.append('x')\n",
        encoding="utf-8",
    )

    assert _cross_study_violations(tmp_path) == set()
