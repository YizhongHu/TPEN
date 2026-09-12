"""Round-2 reviewer probe: follower path-spelling semantics vs OmegaConf grammar.

Read-only. No resolvers registered except a witness that must never fire.
Run with the omegaconf-2.3.0 venv python, cwd = review worktree.
"""
import sys

import omegaconf
from omegaconf import OmegaConf

import tpen.hi_schema as H

import os
import sys

# The checkout this probe must be reading is named by TPEN_EXPECTED_ROOT, or by
# the working directory.  A shared venv in this project carries an editable tpen
# install pointing at ANOTHER worktree, so an unpinned probe silently measures
# somebody else's branch.
_EXPECTED_ROOT = os.path.realpath(os.environ.get("TPEN_EXPECTED_ROOT", os.getcwd()))
assert os.path.realpath(H.__file__).startswith(_EXPECTED_ROOT), (H.__file__, _EXPECTED_ROOT)
print("hi_schema source:", H.__file__, "| python:", sys.executable)
print("python", sys.version.split()[0], "| omegaconf", omegaconf.__version__)
print("hi_schema", H.__file__)

WITNESS_CALLS = []
OmegaConf.register_new_resolver(
    "r2_witness", lambda *a: WITNESS_CALLS.append(a) or "WITNESS-RAN"
)

FAMILY = "tpen_he_importance"


def show(tag, tree, path="experiment.name"):
    try:
        cfg = OmegaConf.create(tree)
    except Exception as e:
        print(f"{tag:28s} | CREATE FAILED: {type(e).__name__}: {e}")
        return
    ident = H.identity_without_execution(cfg, path)
    try:
        resolved = OmegaConf.select(cfg, path)
    except Exception as e:
        resolved = f"<{type(e).__name__}: {e}>"
    follower = (
        f"DETERMINED {ident.value!r}" if ident.determined else f"REFUSED ({ident.reason})"
    )
    danger = ""
    if ident.determined and isinstance(resolved, str):
        if (resolved == FAMILY) != (ident.value == FAMILY):
            danger = "  <<< FAMILY DECISION FLIPS: FAIL-OPEN"
    print(f"{tag:28s} | follower: {follower}")
    print(f"{'':28s} | omegaconf: {resolved!r}{danger}")


# --- axis: whitespace inside the interpolation ---
show("ws-both", {"experiment": {"name": "${ names.hi }"}, "names": {"hi": FAMILY}})
show("ws-right", {"experiment": {"name": "${names.hi }"}, "names": {"hi": FAMILY}})
show("ws-left", {"experiment": {"name": "${ names.hi}"}, "names": {"hi": FAMILY}})

# --- axis: relative (leading-dot) reference ---
show("rel-sibling", {"experiment": {"name": "${.base}", "base": FAMILY}})
show("rel-grandparent", {"experiment": {"name": "${..top}"}, "top": FAMILY})

# --- COLLISION 1: relative ref + empty-string top-level key ---
show(
    "rel+empty-key COLLISION",
    {
        "experiment": {"name": "${.hi_name}", "hi_name": FAMILY},
        "": {"hi_name": "not-the-family"},
    },
)

# --- COLLISION 2: whitespace ref + literal whitespace-padded key ---
show(
    "ws+padded-key COLLISION",
    {
        "experiment": {"name": "${ names.hi }"},
        "names": {"hi": FAMILY},
        " names": {"hi ": "not-the-family"},  # only matches if body kept verbatim
    },
)
show(
    "ws+verbatim-key COLLISION",
    {
        "experiment": {"name": "${ n }"},
        "n": FAMILY,
        " n ": "not-the-family",
    },
)

# --- axis: interpolation-valued INTERMEDIATE segment of the target path ---
show(
    "mid-segment-interp",
    {"experiment": {"name": "${a.b.c}"}, "a": {"b": "${x}"}, "x": {"c": FAMILY}},
)

# --- axis: non-str key on the walked path ---
show("int-key", {"experiment": {"name": "${a.1}"}, "a": {1: FAMILY}})

# --- control: plain node ref must still be followed ---
show("control-plain-ref", {"experiment": {"name": "${names.hi}"}, "names": {"hi": FAMILY}})

print("witness resolver invocations:", len(WITNESS_CALLS))
