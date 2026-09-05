"""Train and evaluation hold the reference apart, and the boundary is checked.

The consolidated authority requires separate train and evaluation
manifests/processes, with only the evaluation side resolving the literature
value, and names "import tests" as one of the mechanisms that must enforce it.
This module is that import test, plus the two ends it separates.

A separation that is only documented is a comment. What makes it real is that
``tpen.hi_schema`` and ``tpen.run`` cannot reach ``tpen.hi_manifest`` -- so a
reference is not merely unused by a training process, it is unreachable from
one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.hi_manifest import (
    HI_EVALUATION_SCHEMA,
    load_evaluation_manifest,
    reference_energy,
)
from tpen.hi_schema import HI_TRAIN_SCHEMA, validate_hi_train_config

CONTROL_CONFIG = Path("experiments/atomistic/he-importance/configs/train.yaml")
EVALUATION_MANIFEST = Path("experiments/atomistic/he-importance/manifests/evaluation.yaml")

# Modules that are on the training path and must not reach the reference.
TRAIN_PATH_MODULES = ("tpen/hi_schema.py", "tpen/run.py", "tpen/config_schema.py")

REFERENCE_MODULE = "tpen.hi_manifest"


def _imported_modules(path: Path) -> set[str]:
    """Return every module name a source file imports, statically.

    Parsed with ``ast`` rather than by importing, so the answer describes the
    file rather than whatever happens to be in ``sys.modules`` from an earlier
    test. Covers both ``import x`` and ``from x import y``.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class TestImportSeparation:
    @pytest.mark.parametrize("module_path", TRAIN_PATH_MODULES)
    def test_no_training_module_imports_the_reference_holder(self, module_path: str) -> None:
        imported = _imported_modules(Path(module_path))
        assert REFERENCE_MODULE not in imported, (
            f"{module_path} imports {REFERENCE_MODULE}; the reference must be unreachable "
            "from the training path, not merely unused by it"
        )

    def test_the_parser_would_notice_an_import(self) -> None:
        """Positive control: prove the instrument can see what it looks for.

        Without this, "no training module imports it" would pass equally well
        for a parser that returned an empty set -- which is exactly what a
        typo in the module name would produce.
        """

        assert REFERENCE_MODULE in _imported_modules(Path(__file__))


class TestEveryHIConfigDeclaresTheSchema:
    """Second, independent net against a config that forgets the marker.

    The run-time family rule in ``tpen.hi_schema`` catches a config that says
    it is helium-importance but omits ``schema:``. It is the net that covers
    configs L2 will GENERATE, which never live in the repository and which no
    directory scan can see.

    This is the other net: a static scan of the HI config directory, which
    catches a repo config that omitted the marker AND the experiment name and
    would therefore slip past the run-time rule. Neither net covers the other's
    population, which is why both exist.
    """

    HI_CONFIG_DIR = Path("experiments/atomistic/he-importance/configs")

    def test_the_directory_is_not_empty(self) -> None:
        """A scan over zero files passes vacuously and protects nothing."""

        assert list(self.HI_CONFIG_DIR.glob("*.yaml"))

    def test_every_config_declares_the_hi_train_schema(self) -> None:
        for path in sorted(self.HI_CONFIG_DIR.glob("*.yaml")):
            cfg = OmegaConf.load(path)
            declared = OmegaConf.select(cfg, "schema", default=None)
            assert declared == HI_TRAIN_SCHEMA, (
                f"{path} declares schema {declared!r}; every config in the "
                "helium-importance train family must declare "
                f"{HI_TRAIN_SCHEMA!r}, or it silently receives no enforcement"
            )

    def test_every_config_actually_passes_the_firewall(self) -> None:
        """Declaring the schema is not the same as satisfying it."""

        for path in sorted(self.HI_CONFIG_DIR.glob("*.yaml")):
            validate_hi_train_config(OmegaConf.load(path), env={})


class TestTheTrainConfigHoldsNoReference:
    def test_the_control_config_passes_the_firewall(self) -> None:
        cfg = OmegaConf.load(CONTROL_CONFIG)
        validate_hi_train_config(cfg, env={})

    def test_the_control_config_contains_no_reference_value(self) -> None:
        """Read as text, so an interpolation cannot hide the literal."""

        text = CONTROL_CONFIG.read_text(encoding="utf-8")
        assert "-2.903724377034119598" not in text

    def test_adding_a_reference_to_the_control_config_is_refused(self) -> None:
        """Red arm. Without it, the passing test above proves only that the
        firewall is quiet, not that it is watching this file."""

        from tpen.config_schema import ClosedSchemaError

        cfg = OmegaConf.load(CONTROL_CONFIG)
        cfg.system.reference_energy = -2.903724377034119598
        with pytest.raises(ClosedSchemaError):
            validate_hi_train_config(cfg, env={})


class TestTheManifestHoldsTheReference:
    def test_the_manifest_carries_the_qualified_reference(self) -> None:
        reference = reference_energy(load_evaluation_manifest(EVALUATION_MANIFEST))
        assert reference.energy == pytest.approx(-2.903724377034119598)
        assert reference.qualification == "infinite_mass_nonrelativistic"
        assert reference.units == "hartree"
        assert reference.system_id == "he_atom"

    def test_the_manifest_declares_the_evaluation_schema(self) -> None:
        manifest = load_evaluation_manifest(EVALUATION_MANIFEST)
        assert manifest.schema == HI_EVALUATION_SCHEMA

    def test_loading_the_train_config_as_a_manifest_is_refused(self, ) -> None:
        """The mistake this guards is silent, which is why it is guarded.

        A training config read as a manifest would simply have no reference in
        it, and "no reference found" is indistinguishable from success unless
        the loader refuses the file outright.
        """

        with pytest.raises(ValueError, match="expected 'tpen.hi.evaluation.v1'"):
            load_evaluation_manifest(CONTROL_CONFIG)

    def test_an_unqualified_reference_is_refused(self, tmp_path) -> None:
        """A reference with no qualification cannot be compared honestly.

        Nothing would record which physics produced it, so a discrepancy
        against a differently qualified value would read as model error.
        """

        path = tmp_path / "manifest.yaml"
        path.write_text(
            f"schema: {HI_EVALUATION_SCHEMA}\nreference:\n  energy: -2.9\n  units: hartree\n"
            "  system_id: he_atom\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="qualification"):
            reference_energy(load_evaluation_manifest(path))
