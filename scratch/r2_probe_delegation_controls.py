"""Controls for the delegation spike. Green arm first."""
from omegaconf import OmegaConf
import omegaconf.basecontainer as bc

WITNESS = []
OmegaConf.register_new_resolver("spike_witness", lambda *a: WITNESS.append(a) or 7)

carrier = OmegaConf.create({"experiment": {"name": "${spike_witness:x}"}})

# CONTROL 1 (green arm): UNGUARDED select must execute the witness.
value = OmegaConf.select(carrier, "experiment.name")
print("control-1 unguarded: value =", value, "| witness calls =", len(WITNESS))
assert len(WITNESS) == 1 and value == 7

# CONTROL 2: guarded refusal, then registry restored, then normal path unaffected.
snapshot = dict(bc.BaseContainer._resolvers)
bc.BaseContainer._resolvers.clear()
try:
    try:
        OmegaConf.select(carrier, "experiment.name")
        print("control-2 guarded: UNEXPECTED SUCCESS")
    except Exception as e:
        print("control-2 guarded: refused with", type(e).__name__, "| witness calls =", len(WITNESS))
finally:
    bc.BaseContainer._resolvers.clear()
    bc.BaseContainer._resolvers.update(snapshot)
assert len(WITNESS) == 1  # guard added zero calls

# fresh cfg: identical carrier resolved post-restore must fire the witness again
carrier2 = OmegaConf.create({"experiment": {"name": "${spike_witness:x}"}})
value2 = OmegaConf.select(carrier2, "experiment.name")
print("control-2 post-restore fresh cfg: value =", value2, "| witness calls =", len(WITNESS))
assert len(WITNESS) == 2 and value2 == 7

# and the SAME cfg object that was refused under guard: no poisoned cache
value3 = OmegaConf.select(carrier, "experiment.name")
print("control-2 post-restore same cfg: value =", value3, "| witness calls =", len(WITNESS))
assert len(WITNESS) == 3 and value3 == 7
print("ALL CONTROLS OK")
