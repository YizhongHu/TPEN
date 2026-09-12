"""Delegation spike: can OmegaConf's OWN resolution do the identity lookup
with ZERO possibility of resolver execution?

Mechanism under test: snapshot and EMPTY the process-global resolver registry,
call OmegaConf.select on the identity path, restore registry in finally.
- node references (any spelling) resolve with OmegaConf's own path semantics
- ANY resolver-call interpolation raises (unsupported type) = fail closed,
  and the callable cannot run because the registry no longer holds it.
Witness resolver registered BEFORE the guarded call; must record ZERO calls.
"""
import omegaconf
from omegaconf import OmegaConf
from omegaconf.base import Container

print("omegaconf", omegaconf.__version__)

# locate the registry the way the spike must: introspect, do not guess
import omegaconf.basecontainer as bc
REG = bc.BaseContainer._resolvers
print("registry object:", type(REG).__name__, "| entries pre-witness:", sorted(REG.keys()))

WITNESS = []
OmegaConf.register_new_resolver("spike_witness", lambda *a: WITNESS.append(a) or 1)
assert "spike_witness" in REG

FAMILY = "tpen_he_importance"


def guarded_select(cfg, path):
    """OmegaConf-native lookup with resolver execution made impossible."""
    snapshot = dict(bc.BaseContainer._resolvers)
    bc.BaseContainer._resolvers.clear()
    try:
        try:
            value = OmegaConf.select(cfg, path)
            return ("DETERMINED", value)
        except Exception as e:
            return ("REFUSED", f"{type(e).__name__}: {e}")
    finally:
        bc.BaseContainer._resolvers.clear()
        bc.BaseContainer._resolvers.update(snapshot)


cases = {
    # the four collision carriers -- must come back DETERMINED with the FAMILY,
    # i.e. OmegaConf semantics, no divergence possible
    "rel+empty-key": ({"experiment": {"name": "${.hi_name}", "hi_name": FAMILY},
                       "": {"hi_name": "decoy"}}, "experiment.name", "DETERMINED", FAMILY),
    "ws+verbatim-key": ({"experiment": {"name": "${ n }"}, "n": FAMILY,
                         " n ": "decoy"}, "experiment.name", "DETERMINED", FAMILY),
    "ws+padded-key": ({"experiment": {"name": "${ names.hi }"}, "names": {"hi": FAMILY},
                       " names": {"hi ": "decoy"}}, "experiment.name", "DETERMINED", FAMILY),
    "schema-side": ({"schema": "${.sk}", "sk": "tpen.hi.train.v1",
                     "": {"sk": "decoy"}}, "schema", "DETERMINED", "tpen.hi.train.v1"),
    # follow axes C already handles -- chain, path interpolation, mid-segment
    "chained": ({"experiment": {"name": "${a}"}, "a": "${b}", "b": FAMILY},
                "experiment.name", "DETERMINED", FAMILY),
    "path-interp": ({"experiment": {"name": "${choices.${slot}.basis}"},
                     "slot": "s1", "choices": {"s1": {"basis": FAMILY}}},
                    "experiment.name", "DETERMINED", FAMILY),
    "mid-segment": ({"experiment": {"name": "${a.b.c}"}, "a": {"b": "${x}"},
                     "x": {"c": FAMILY}}, "experiment.name", "DETERMINED", FAMILY),
    # resolver carriers -- must REFUSE with zero witness calls
    "custom-resolver": ({"experiment": {"name": "${spike_witness:x}"}},
                        "experiment.name", "REFUSED", None),
    "resolver-in-chain": ({"experiment": {"name": "${a}"}, "a": "${spike_witness:x}"},
                          "experiment.name", "REFUSED", None),
    "resolver-in-path": ({"experiment": {"name": "${choices.${spike_witness:x}.b}"},
                          "choices": {}}, "experiment.name", "REFUSED", None),
    "builtin-oc-env": ({"experiment": {"name": "${oc.env:HOME}"}},
                       "experiment.name", "REFUSED", None),
    "schema-trampoline": ({"schema": "tpen.hi.train.v${spike_witness:x}"},
                          "schema", "REFUSED", None),
    # boundary shapes
    "dangling": ({"experiment": {"name": "${nowhere}"}}, "experiment.name", "REFUSED", None),
    "cycle": ({"experiment": {"name": "${a}"}, "a": "${experiment.name}"},
              "experiment.name", "REFUSED", None),
    "missing-sentinel": ({"experiment": {"name": "???"}}, "experiment.name", None, None),
    "absent": ({"other": 1}, "experiment.name", "DETERMINED", None),
    "literal": ({"experiment": {"name": FAMILY}}, "experiment.name", "DETERMINED", FAMILY),
}

failures = 0
for tag, (tree, path, want_kind, want_value) in cases.items():
    cfg = OmegaConf.create(tree)
    kind, value = guarded_select(cfg, path)
    ok = True
    if want_kind is not None:
        ok = kind == want_kind and (kind == "REFUSED" or value == want_value)
    status = "OK " if ok else "MISMATCH"
    if not ok:
        failures += 1
    print(f"{status} {tag:20s} -> {kind}: {value!r}")

print("witness calls TOTAL (must be 0):", len(WITNESS))
print("registry restored:", "spike_witness" in bc.BaseContainer._resolvers)
print("failures:", failures)
