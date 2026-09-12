from omegaconf import OmegaConf
import tpen.hi_schema as H
from tpen.config_schema import ClosedSchemaError
assert "claude-hi-firewall-r2-review-tests" in H.__file__, H.__file__

# Simulate the grammar-normal repair direction for carrier 4 by writing the
# schema literally: the question is whether refusal happens at all.
trees = {
    "c4-literal-schema": {
        "schema": "tpen.hi.train.v1",
        "sk": "tpen.hi.train.v1",
        "": {"sk": "decoy-schema"},
        "experiment": {"name": "other-exp"},
    },
    "c4-literal-schema-plus-ref-energy": {
        "schema": "tpen.hi.train.v1",
        "sk": "tpen.hi.train.v1",
        "": {"sk": "decoy-schema"},
        "experiment": {"name": "other-exp"},
        "model": {"reference_energy": -2.903724377},
    },
}
for tag, tree in trees.items():
    cfg = OmegaConf.create(tree)
    try:
        H.validate_hi_train_config(cfg, env={})
        print(f"{tag}: RETURNED CLEAN")
    except ClosedSchemaError as e:
        print(f"{tag}: REFUSED {sorted({r.rule for r in e.rejections})}")
