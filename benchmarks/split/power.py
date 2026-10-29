#!/usr/bin/env python3
"""Wall power per engine from bench_split's SMC samples, by block instead of by call.

usage: power.py <out prefix ...>   e.g. .../sq1x8/Z8192.np1.d0

The fixed block runs each config as a run of consecutive calls. SMC's PDTR lags the load by
about a second, so each run is averaged from 1 s after its first call starts to its last call's
end; runs shorter than 1.5 s after trimming are skipped. Reports mean PDTR (DC input = wall side on
AC), the GPU rails (PC10+PC12+PC20+PC22) and PPBR (battery; ~0 means the adapter carries the load).
"""
import sys
from collections import defaultdict

LAG, MINLEN = 1.0, 1.5
for pre in sys.argv[1:]:
    rows = [l.rstrip("\n").split("\t") for l in open(pre + ".raw.tsv")]
    h = rows[0]; recs = [dict(zip(h, r)) for r in rows[1:]]
    recs = [r for r in recs if r["proc"] == "0" and r["phase"] == "1"]
    lines = [l.rstrip("\n").split("\t") for l in open(pre + ".power.tsv")]
    col = {k: i for i, k in enumerate(lines[0])}
    rails = [col[k] for k in ("PC10", "PC12", "PC20", "PC22")]
    smp = [(float(s[col["t"]]), float(s[col["PDTR"]]), float(s[col["PPBR"]]), sum(float(s[i]) for i in rails)) for s in lines[1:]]
    runs, cur = [], None
    for r in recs:
        key = (r["round"], r["cfg"])
        if cur and cur[0] == key:
            cur[2] = float(r["t1"])
        else:
            cur = [key, float(r["t0"]), float(r["t1"])]
            runs.append(cur)
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0.0])
    for (rnd, cfg), t0, t1 in runs:
        a, b = t0 + LAG, t1
        if b - a < MINLEN:
            continue
        s = [x for x in smp if a <= x[0] <= b]
        if not s:
            continue
        name = {"0": "ACC", "1": "GPU"}.get(cfg, "split")
        v = acc[name]
        v[0] += sum(x[1] for x in s); v[1] += sum(x[2] for x in s); v[2] += sum(x[3] for x in s); v[3] += len(s); v[4] += b - a
    print(pre.split("/")[-1] + ":", "  ".join(
        f"{k} PDTR {v[0]/v[3]:.1f} W (GPU rails {v[2]/v[3]:.1f}, battery {v[1]/v[3]:+.1f}; {v[3]} samples, {v[4]:.1f} s)" for k, v in sorted(acc.items()) if v[3]) or "no run long enough")
