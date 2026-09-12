"""Audit the refusal-member census for substring shadowing and mention counts.

A member name that is a substring of another member's name makes a
mention-census over the contract suite report false coverage: every mention
of the longer name also counts as a mention of the shorter one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1] / "experiments" / "atomistic" / "he-importance"
_TARGET = _HERE / "inference_packet.py"
_SUITE = _HERE / "test_inference_packet.py"

_spec = importlib.util.spec_from_file_location("l4a_r3_census_target", _TARGET)
assert _spec is not None and _spec.loader is not None
ip = importlib.util.module_from_spec(_spec)
sys.modules["l4a_r3_census_target"] = ip
_spec.loader.exec_module(ip)

members = list(ip.PacketRefusal)
print(f"member count: {len(members)}")

shadowing = [
    (a.name, b.name)
    for a in members
    for b in members
    if a is not b and a.name in b.name
]
print(f"substring-shadowing ordered pairs: {len(shadowing)}")
for pair in shadowing:
    print(f"  SHADOWED {pair[0]} is a substring of {pair[1]}")

suite = _SUITE.read_text(encoding="utf-8")
print("mention counts in test_inference_packet.py:")
for member in members:
    print(f"  {member.name}: {suite.count(member.name)}")
