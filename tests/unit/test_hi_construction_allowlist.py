"""Global construction admission, with a separately frozen hazard generator."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from tests.helpers.hi_firewall_corpus import (
    construction_specs,
    file_spellings,
    mapping_paths,
    place_spec,
    renamed_keys,
)
from tpen.config_schema import ClosedSchemaError
from tpen.hi_schema import HI_TRAIN_SCHEMA, validate_hi_train_config


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "experiments/atomistic/he-importance/configs/train.yaml"
MANIFEST = ROOT / "experiments/atomistic/he-importance/manifests/evaluation.yaml"
MODULE = ROOT / "tpen/hi_manifest.py"
RAW_CONTROL = OmegaConf.to_container(OmegaConf.load(CONTROL), resolve=False)
RULE = "unadmitted-free-form-target"
SPEC_IDS = [name for name, _ in construction_specs("manifest", "module", "witness")]
PATH_IDS = [name for name, _ in file_spellings(Path("/tmp/data/evaluation.yaml"), Path("link"))]


def minimal_config(**sections):
    """Supply only the independent schema prerequisites."""
    return OmegaConf.create({
        "schema": HI_TRAIN_SCHEMA,
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.005},
        **sections,
    })


def target_findings(cfg):
    with pytest.raises(ClosedSchemaError) as caught:
        validate_hi_train_config(cfg, env={})
    return [finding for finding in caught.value.rejections if finding.rule == RULE]


@pytest.mark.parametrize("spelling", PATH_IDS)
@pytest.mark.parametrize("mechanism", SPEC_IDS)
def test_generated_mechanisms_and_filesystem_spellings(mechanism, spelling, tmp_path):
    link = tmp_path / "opaque-input"
    link.symlink_to(MANIFEST)
    path = dict(file_spellings(MANIFEST, link))[spelling]
    spec = dict(construction_specs(path, str(MODULE), str(tmp_path / "witness")))[mechanism]
    cfg = minimal_config(runner={"_target_": "tpen.runner.Train", "loads": spec})
    findings = target_findings(cfg)
    assert "runner.loads._target_" in {f.path for f in findings}
    assert not (tmp_path / "witness").exists()


PLACEMENTS = list(mapping_paths(RAW_CONTROL))


@pytest.mark.parametrize("path", PLACEMENTS, ids=lambda p: "/".join(map(str, p)) or "root")
@pytest.mark.parametrize("depth,container", [(0, "mapping"), (1, "mapping"), (2, "mapping"),
                                            (2, "list"), (2, "tuple"), (5, "list")])
@pytest.mark.parametrize("mechanism", ["omegaconf.OmegaConf.load", "relative-importlib"])
def test_generated_locations_and_depths(path, depth, container, mechanism):
    specs = dict(construction_specs(str(MANIFEST), str(MODULE), "unused-witness"))
    cfg = OmegaConf.create(place_spec(RAW_CONTROL, path, specs[mechanism],
                                     depth=depth, container=container))
    findings = target_findings(cfg)
    assert any("future_parameter" in f.path for f in findings)


@pytest.mark.parametrize("path,key", list(renamed_keys(RAW_CONTROL)))
def test_renaming_an_argument_never_admits_a_target(path, key):
    spec = {"_target_": "importlib.import_module", "name": ".hi_manifest", "package": "tpen"}
    cfg = OmegaConf.create(place_spec(RAW_CONTROL, path, spec, key=key))
    assert any(f.path.endswith(f".{key}._target_") for f in target_findings(cfg))


@pytest.mark.parametrize("convert", ["none", "partial", "all", "object"])
@pytest.mark.parametrize("recursive", [True, False])
@pytest.mark.parametrize("partial", [True, False])
def test_hydra_execution_flags_do_not_exempt_nested_arguments(convert, recursive, partial):
    cfg = minimal_config(loggers=[{
        "_target_": "tpen.logging.CSV", "_convert_": convert,
        "_recursive_": recursive, "_partial_": partial,
        "path": {"first": [{"second": {
            "_target_": "importlib.import_module", "name": ".hi_manifest", "package": "tpen",
        }}]},
    }])
    assert {f.path for f in target_findings(cfg)} == {"loggers[0].path.first[0].second._target_"}


@pytest.mark.parametrize("target", [None, 7, False, {"name": "tpen.logging.CSV"}])
def test_non_string_construction_identities_are_refused(target):
    cfg = minimal_config(run={"run_id": "x", "future": {"_target_": target}})
    assert {f.path for f in target_findings(cfg)} == {"run.future._target_"}


@pytest.mark.parametrize("target", [
    "foreign.CSV", "tpen.logging.CSV.extra", "tpen.logging.csv.CSV",
    " tpen.logging.CSV", "tpen.logging.CSV ", "tpen.logging.CSVSibling",
    "tpen.logging.CSV.__init__", "tpen.logging.CSV.__init__.__globals__.get",
])
def test_admission_compares_the_complete_identity(target):
    cfg = minimal_config(runtime={"future": {"_target_": target}})
    assert {f.path for f in target_findings(cfg)} == {"runtime.future._target_"}


@pytest.mark.parametrize("suffix", ["${missing.node}", "${oc.env:HI_CONSTRUCTION_WITNESS,ok}"])
def test_raw_target_findings_survive_resolution_failure_or_refusal(suffix):
    cfg = minimal_config(run={"run_id": suffix}, loggers=[{
        "_target_": "tpen.logging.CSV", "path": {
            "_target_": "importlib.import_module", "name": ".hi_manifest", "package": "tpen",
        },
    }])
    with pytest.raises(ClosedSchemaError) as caught:
        validate_hi_train_config(cfg, env={})
    findings = caught.value.rejections
    assert any(f.rule == RULE and f.tree == "raw" and f.path == "loggers[0].path._target_"
               for f in findings)
    assert any(f.rule == "unresolvable" or "resolver" in f.rule for f in findings)


@pytest.mark.parametrize("target", ["tpen.logging.CSV", "importlib.import_module"])
def test_interpolated_target_is_qualified_after_resolution(target):
    cfg = minimal_config(runtime={"identity": target}, run={"run_id": "x"},
                         loggers=[{"_target_": "${runtime.identity}", "path": "unused.csv"}])
    if target == "tpen.logging.CSV":
        validate_hi_train_config(cfg, env={})
    else:
        assert {(f.path, f.tree) for f in target_findings(cfg)} == {
            ("loggers[0]._target_", "resolved"),
        }


def test_interpolated_subtree_is_also_walked():
    cfg = minimal_config(runtime={"spec": {"_target_": "importlib.import_module"}},
                         loggers=[{"_target_": "tpen.logging.CSV", "path": "${runtime.spec}"}])
    assert {f.path for f in target_findings(cfg)} == {
        "runtime.spec._target_", "loggers[0].path._target_",
    }


def test_deferred_factory_arguments_are_still_qualified():
    cfg = minimal_config(model={
        "_target_": "hydra.utils.instantiate", "_recursive_": False, "config": {
            "_target_": "torch.nn.ModuleList", "modules": [{
                "_target_": "omegaconf.OmegaConf.load", "file_": str(MANIFEST),
            }],
        },
    })
    assert {f.path for f in target_findings(cfg)} == {"model.config.modules[0]._target_"}


# Independent admit-side vocabulary: control targets plus the existing schema
# contracts and valid wrapper/variant tests. Never parameterize this from the
# implementation allowlist: removing an admission must leave a failing node.
VALID_EXTRA_TARGETS = [
    "hydra.utils.instantiate", "torch.nn.ModuleList", "torch.nn.Tanh",
    "tpen.nn.ReplaceUpdater", "tpen.nn.BoundedTwoCoefficientJastrow", "tpen.runner.Evaluate",
    "tpen.training.LegacyAutogradUpdate", "tpen.training.update.LegacyAutogradUpdate",
    "tpen.callback.ArtifactIndex", "tpen.callback.FailureLog", "tpen.callback.RunTiming",
    "tpen.callback.TrainPhaseTiming", "tpen.callback.TrainStepTiming", "tpen.callback.DiagnosticTiming",
]


def _node_at(tree, path):
    for part in path:
        tree = tree[part]
    return tree


CONTROL_TARGETS = sorted({_node_at(RAW_CONTROL, p)["_target_"] for p in PLACEMENTS
                          if "_target_" in _node_at(RAW_CONTROL, p)})


@pytest.mark.parametrize("target", sorted(set(CONTROL_TARGETS + VALID_EXTRA_TARGETS)))
def test_every_existing_valid_identity_remains_admitted_at_arbitrary_depth(target):
    cfg = minimal_config(runtime={"future": [{"argument": {"_target_": target}}]})
    validate_hi_train_config(cfg, env={})


@pytest.mark.parametrize("mechanism", ["copyfile", "relative-importlib", "execute-file",
                                       "relative-import-all", "relative-import-object"])
def test_falsifiers_are_live_without_validation_and_blocked_with_it(mechanism, tmp_path):
    """Use fresh processes so the importing control cannot contaminate the guard arm."""
    spec = dict(construction_specs(str(MANIFEST), str(MODULE), str(tmp_path / "witness")))[mechanism]
    cfg = minimal_config(loggers=[{"_target_": "tpen.logging.CSV", "path": spec}])
    payload = OmegaConf.to_yaml(cfg)
    script = r'''
import json, sys
from pathlib import Path
import tpen
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tpen.config_schema import ClosedSchemaError
from tpen.hi_schema import validate_hi_train_config
assert Path(tpen.__file__).resolve().is_relative_to(Path.cwd()), tpen.__file__
assert "tpen.hi_manifest" not in sys.modules
cfg = OmegaConf.create(sys.stdin.read())
if sys.argv[1] == "guard":
    try:
        validate_hi_train_config(cfg, env={})
    except ClosedSchemaError as error:
        assert any(f.rule == "unadmitted-free-form-target" for f in error.rejections)
    else:
        raise AssertionError("configuration passed validation")
    assert "tpen.hi_manifest" not in sys.modules
    assert not Path(sys.argv[2]).exists()
else:
    # Exercise the inner construction, whose effects precede any outer error.
    result = instantiate(cfg.loggers[0].path)
    if sys.argv[3] == "copyfile":
        assert "2.9037243770341" in Path(sys.argv[2]).read_text()
    elif sys.argv[3] == "execute-file":
        assert callable(result["reference_energy"])
    else:
        assert "tpen.hi_manifest" in sys.modules
print(json.dumps({"mode": sys.argv[1], "mechanism": sys.argv[3], "passed": True}))
'''
    # Guard first: the deliberately live control subsequently creates evidence.
    for mode in ("guard", "control"):
        result = subprocess.run([sys.executable, "-c", script, mode, str(tmp_path / "witness"),
                                 mechanism], input=payload, text=True, capture_output=True, cwd=ROOT)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mechanism", ["copyfile", "relative-importlib"])
def test_actual_runner_sequence_seam_cannot_reach_the_falsifier(mechanism, tmp_path):
    """The production sequence consumer is exercised on torch-enabled hosts."""
    pytest.importorskip("torch")
    from tpen.run import _instantiate_sequence

    witness = tmp_path / "witness"
    spec = dict(construction_specs(str(MANIFEST), str(MODULE), str(witness)))[mechanism]
    cfg = minimal_config(loggers=[{"_target_": "tpen.logging.CSV", "path": spec}])
    before = sys.modules.get("tpen.hi_manifest")
    with pytest.raises(ClosedSchemaError):
        validate_hi_train_config(cfg, env={})
        _instantiate_sequence(cfg.loggers)
    assert sys.modules.get("tpen.hi_manifest") is before
    assert not witness.exists()


@pytest.mark.parametrize("target", sorted(set(CONTROL_TARGETS + VALID_EXTRA_TARGETS)))
def test_admitted_identity_resolves_to_an_existing_callable(target):
    """Verify the independently enumerated vocabulary on a torch-enabled host."""
    pytest.importorskip("torch")
    from hydra.utils import get_object

    assert callable(get_object(target))
