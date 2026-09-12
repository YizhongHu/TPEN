"""Prove the Group-2 witness assertion can now FAIL.

Two arms.  The first shows the decoy is reachable at all: resolving the whole
configuration runs it.  The second installs a DEFECTIVE follower that
establishes identity by resolving the configuration -- exactly the defect the
witness half exists to catch -- and shows the arm's assertion goes red.
"""

import os

from omegaconf import OmegaConf

import tpen.config as config_module
import tpen.hi_schema as H
from tpen.config_schema import ClosedSchemaError

assert "claude-hi-firewall-r2-review-tests" in H.__file__, H.__file__

RESOLVER = config_module.BASIS_FEATURE_DIM_RESOLVER
FAMILY = H.HI_EXPERIMENT_NAME
TREE = {
    "experiment": {"name": "${ names.hi }"},
    "names": {"hi": FAMILY},
    "decoy": f"${{{RESOLVER}:x}}",
}

calls = []
original = config_module.basis_feature_dim
OmegaConf.register_new_resolver(RESOLVER, lambda a: calls.append(a) or 1, replace=True)
try:
    # ARM 1 -- the decoy is reachable: a whole-tree resolve runs it.
    OmegaConf.to_container(OmegaConf.create(TREE), resolve=True)
    print("arm 1, whole-tree resolve: witness calls =", len(calls))

    # ARM 2 -- a follower that resolves to find out what the config is.
    real_identity = H.identity_without_execution

    def resolving_identity(cfg, path):
        OmegaConf.to_container(cfg, resolve=True)  # the defect
        return real_identity(cfg, path)

    H.identity_without_execution = resolving_identity
    calls.clear()
    try:
        H.validate_hi_train_config(OmegaConf.create(TREE), env={})
        outcome = "RETURNED CLEAN"
    except ClosedSchemaError as error:
        outcome = f"REFUSED {sorted({r.rule for r in error.rejections})}"
    finally:
        H.identity_without_execution = real_identity
    print("arm 2, defective resolving follower:", outcome)
    print("arm 2 witness calls =", len(calls), "-> the arm's calls == [] assertion",
          "FAILS" if calls else "still passes (the decoy is not load-bearing)")

    # ARM 3 -- the shipped follower, same tree: the count must stay zero.
    calls.clear()
    try:
        H.validate_hi_train_config(OmegaConf.create(TREE), env={})
        outcome = "RETURNED CLEAN"
    except ClosedSchemaError as error:
        outcome = f"REFUSED {sorted({r.rule for r in error.rejections})}"
    print("arm 3, shipped follower:", outcome, "| witness calls =", len(calls))
finally:
    OmegaConf.register_new_resolver(RESOLVER, original, replace=True)
