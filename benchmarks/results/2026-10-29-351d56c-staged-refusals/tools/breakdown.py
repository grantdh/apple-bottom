#!/usr/bin/env python3
"""breakdown.py <trace-dir> [<err.txt>]: the staged path's outcomes in one traced run, by routine
and reason (AB_TRACE staged=), then by call shape (npw, the plane waves of a k-point, shown as
'npw'), and the host stages the calls that left the staged path paid; with err.txt, the run's
AB_REPORT staged lines."""
import collections, re, sys
from pathlib import Path
d = Path(sys.argv[1]); f = sorted(d.glob("trace.*"))[0]
rows = []
for line in f.read_text().splitlines():
    if line.startswith("#"): continue
    t = dict(x.split("=", 1) for x in line.split() if "=" in x)
    if "staged" not in t: continue
    rows.append(t)
by = collections.Counter((r["routine"], r["staged"], r["path"]) for r in rows)
print(f"{f}: {len(rows)} calls reached the staged path")
for k, n in sorted(by.items()): print(f"  {k[0]} staged={k[1]:<12} path={k[2]:<20} {n}")
def cls(v):
    v = int(v)
    return "npw" if v > 17000 else ("nkb" if v == 1152 else ("nb" if v == 160 else str(v)))
shape = collections.Counter()
for r in rows:
    shape[(r["routine"], r["ta"], r["tb"], cls(r["M"]), cls(r["N"]), cls(r["K"]), r["staged"])] += 1
print("by shape (TRANS, M N K; npw: > 17000, nkb: 1152, nb: 160):")
for k, n in sorted(shape.items(), key=lambda kv: (kv[0][:6], -kv[1])): print("  ", *k, n)
ms = collections.defaultdict(float); cnt = collections.Counter()
for r in rows:
    if r["path"] != "gpu": continue
    g = "kept" if r["staged"].startswith("kept") else "host path"
    cnt[g] += 1
    for s in ("pack", "upload", "kernel", "download", "epilogue", "other"):
        ms[(g, s)] += int(r.get(s + "_ns", 0)) / 1e9
print("GPU calls' stages, s:")
for g in ("kept", "host path"):
    print(f"  {g:<10} {cnt[g]:5d} calls " + " ".join(f"{s}={ms[(g, s)]:.2f}" for s in ("pack", "upload", "kernel", "download", "epilogue", "other")))
if len(sys.argv) > 2:
    for line in open(sys.argv[2], errors="replace"):
        if re.search(r"AB_REPORT\[\d+\] \w+ (staged|calls=|gpu_stages_ms|fallbacks)", line): print(line.rstrip())
