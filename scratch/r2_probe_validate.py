from omegaconf import OmegaConf
import tpen.hi_schema as H
from tpen.config_schema import ClosedSchemaError
import os
import sys

# The checkout this probe must be reading is named by TPEN_EXPECTED_ROOT, or by
# the working directory.  A shared venv in this project carries an editable tpen
# install pointing at ANOTHER worktree, so an unpinned probe silently measures
# somebody else's branch.
_EXPECTED_ROOT = os.path.realpath(os.environ.get("TPEN_EXPECTED_ROOT", os.getcwd()))
assert os.path.realpath(H.__file__).startswith(_EXPECTED_ROOT), (H.__file__, _EXPECTED_ROOT)
print("hi_schema source:", H.__file__, "| python:", sys.executable)

FAMILY = "tpen_he_importance"
carriers = {
    "rel+empty-key": {
        "experiment": {"name": "${.hi_name}", "hi_name": FAMILY},
        "": {"hi_name": "decoy"},
        "model": {"reference_energy": -2.903724377},
    },
    "ws+verbatim-key": {
        "experiment": {"name": "${ n }"},
        "n": FAMILY,
        " n ": "decoy",
        "model": {"reference_energy": -2.903724377},
    },
    "ws+padded-key": {
        "experiment": {"name": "${ names.hi }"},
        "names": {"hi": FAMILY},
        " names": {"hi ": "decoy"},
        "model": {"reference_energy": -2.903724377},
    },
    "schema-side rel+empty-key": {
        "schema": "${.sk}",  # top-level: relative == absolute... use nested? schema is top-level
        "sk": "tpen.hi.train.v1",
        "": {"sk": "decoy-schema"},
        "experiment": {"name": "other-exp"},
    },
}
for tag, tree in carriers.items():
    cfg = OmegaConf.create(tree)
    try:
        H.validate_hi_train_config(cfg, env={})
        outcome = "RETURNED CLEAN (zero enforcement)"
    except ClosedSchemaError as e:
        outcome = f"REFUSED {sorted({r.rule for r in e.rejections})}" if hasattr(e, "rejections") else f"REFUSED {e}"
    except Exception as e:
        outcome = f"RAISED {type(e).__name__}: {e}"
    try:
        resolved_name = OmegaConf.select(cfg, "experiment.name")
    except Exception as e:
        resolved_name = f"<{type(e).__name__}>"
    try:
        resolved_schema = OmegaConf.select(cfg, "schema")
    except Exception as e:
        resolved_schema = f"<{type(e).__name__}>"
    print(f"{tag:28s} | validate: {outcome}")
    print(f"{'':28s} | resolved name={resolved_name!r} schema={resolved_schema!r}")
