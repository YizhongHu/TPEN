"""Fit per-step rates from a legacy Polaris ``train.log`` probe.

The parser reads ``R/<tag>/train.log`` lines shaped
``<ISO timestamp>\t...Step N:...`` for the hardcoded arms ``a1``, ``a2``,
``a4``, and ``a1r``. It does not apply to current ferminet-codebase runs:
those runs write only ``train_stats.csv``, with no ``train.log`` and no time
column, and short rungs write no checkpoints. Consequently there are no
timestamps for this script to fit on those runs; an empty result means the
input format is unsupported, not that the run produced nothing. Current-run
rates come from per-row ``elapsed_sec`` in ``attempt_status.json`` compared
across two step counts instead.

``PRIOR`` is a hardcoded comparison constant from one specific prior Polaris
measurement (RAMP-D arm n-g1), not a general reference rate.
"""
import re, os, sys
from datetime import datetime
R = sys.argv[1]
STEP = re.compile(r"Step (\d+):")
def parse(tag):
    p = os.path.join(R, tag, "train.log")
    if not os.path.exists(p): return None
    pts = []
    for line in open(p, errors="replace"):
        if "\t" not in line: continue
        ts, rest = line.split("\t", 1)
        m = STEP.search(rest)
        if not m: continue
        try: t = datetime.strptime(ts.strip(), "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
        except Exception: continue
        pts.append((int(m.group(1)), t))
    return pts
def fit(pts, cut):
    p = [(s, t) for s, t in pts if s >= cut]
    if len(p) < 2: return None
    n = len(p); ms = sum(s for s, _ in p)/n; mt = sum(t for _, t in p)/n
    den = sum((s-ms)**2 for s, _ in p)
    return None if den == 0 else sum((s-ms)*(t-mt) for s, t in p)/den
res = {}
print("%-6s %6s %11s %11s %11s" % ("arm", "n", "cut50", "cut100", "cut200"))
for tag in ("a1", "a2", "a4", "a1r"):
    pts = parse(tag)
    if not pts:
        print("%-6s (not finished)" % tag); continue
    row = [fit(pts, c) for c in (50, 100, 200)]
    res[tag] = row[1]
    print("%-6s %6d %11.6f %11.6f %11.6f" % (tag, len(pts), *[r if r else float("nan") for r in row]))
print()
PRIOR = 0.4012836   # RAMP-D arm n-g1, independent Polaris measurement
if "a1" in res and res["a1"]:
    a1 = res["a1"]
    print("CONTROL: a1 vs the independent RAMP-D Polaris N measurement")
    print("  a1        %.6f s/step" % a1)
    print("  RAMP-D    %.6f s/step   -> difference %.2f%%" % (PRIOR, 100*abs(a1-PRIOR)/PRIOR))
    if "a1r" in res and res["a1r"]:
        print("  order control a1r %.6f -> %.3fx" % (res["a1r"], a1/res["a1r"]))
    print()
    for g, tag in ((2, "a2"), (4, "a4")):
        v = res.get(tag)
        if not v: continue
        eff = a1/(g*v)
        print("  %d GPU  %.6f s/step  speedup %.3fx  efficiency %.1f%%  ->  200k wall %.2f h"
              % (g, v, a1/v, 100*eff, 200000*v/3600))
    if res.get("a4"):
        w = 200000*res["a4"]/3600
        print()
        print("  POLARIS N per-row wall at 4 GPU: %.2f h" % w)
        print("  FRONTIER N per-row wall at 8 GCD: 8.09 h  (0.145678 s/step, 61.1%%)")
        print("  -> %s" % ("FRONTIER faster per row" if 8.09 < w else "POLARIS faster per row"))
