#!/usr/bin/env python3
"""Wall power per engine from bench_split's SMC samples, by block instead of by call.

usage: power.py <out prefix ...>   e.g. .../sq1x8/Z8192.np1.d0

The fixed block runs each config as a run of consecutive calls. Configs are labelled by raw.tsv's
name column: ACC, GPU, split (the user split S<g>), split2 (its twin S<g>b), SEAM, HEAD, TAIL;
files written before that column have ACC (cfg 0), GPU (cfg 1) and split (any other). SMC's PDTR lags the load by
about a second, so each run is averaged from 1 s after its first call starts to its last call's
end; runs shorter than 1.5 s after trimming are skipped. Reports mean PDTR (DC input = wall side on
AC), the GPU rails (PC10+PC12+PC20+PC22) and PPBR (battery; ~0 means the adapter carries the load).

With the arms (bench_split's default) the trio SEAM, S<g>, S<g>b alternates call by call, so its calls
form no runs and get no power figure: only ACC, GPU, HEAD and TAIL can, when their repeats last long
enough. For the split's wall power use SPLIT_ARMS=0, the power4x2 suite or the sustained runs.
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # rawtsv.py, also under python3 -I
from rawtsv import read  # noqa: E402

LAG, MINLEN = 1.0, 1.5
LABEL = {"SPLIT": "split", "SPLIT2": "split2"}   # the other roles print as they are
for pre in sys.argv[1:]:
    recs = [r for r in read(pre + ".raw.tsv") if r["proc"] == "0" and r["phase"] == "1"]
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
            cur = [key, float(r["t0"]), float(r["t1"]), LABEL.get(r["role"], r["role"])]
            runs.append(cur)
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0.0])
    for _, t0, t1, name in runs:
        a, b = t0 + LAG, t1
        if b - a < MINLEN:
            continue
        s = [x for x in smp if a <= x[0] <= b]
        if not s:
            continue
        v = acc[name]
        v[0] += sum(x[1] for x in s); v[1] += sum(x[2] for x in s); v[2] += sum(x[3] for x in s); v[3] += len(s); v[4] += b - a
    trio = " (the arms' trio alternates call by call, so it has no runs: see the docstring)" if any(r["role"] == "SEAM" for r in recs) else ""
    print(pre.split("/")[-1] + ":", ("  ".join(
        f"{k} PDTR {v[0]/v[3]:.1f} W (GPU rails {v[2]/v[3]:.1f}, battery {v[1]/v[3]:+.1f}; {v[3]} samples, {v[4]:.1f} s)" for k, v in sorted(acc.items()) if v[3]) or "no run long enough") + trio)
