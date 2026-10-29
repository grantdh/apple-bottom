#!/usr/bin/env python3
"""Classify every GPU-routed BLAS-ABI call in bench_split's AB_TRACE files by the path that computed C.

usage: classify_trace.py <dir> [prefix]   reads <dir>/<case>.np<N>.d<D>.trace.<pid>

  staged    path=gpu, epilogue_ns == 0 (GPU-side conversion and epilogue; pack = the range check)
  host      path=gpu, epilogue_ns > 0  (host pack/conversion/epilogue: the staged result did not
            stand, or the call went straight to the host path)
  fallback  path=fallback:<reason>     (C computed by cblas on the GPU slice's thread)
The rule holds for bench_split's calls, all with alpha = 1 and beta 0 or 1: a staged ZGEMM with other
scalars applies them on the host and records an epilogue, so it would count as host here.
Calls are grouped by shape: the full call (GPU alone) and each slice width (split).
Stage times are medians in ms.
"""
import glob, os, re, statistics, sys
from collections import defaultdict

d = sys.argv[1]
pre = sys.argv[2] if len(sys.argv) > 2 else ""
kv = re.compile(r"(\w+)=(\S+)")
files = sorted(glob.glob(os.path.join(d, pre + "*.trace.*")))
runs = defaultdict(list)
for f in files:
    run = os.path.basename(f).split(".trace.")[0]
    for line in open(f):
        if "path=" not in line or line.startswith("#"):
            continue
        runs[run].append(dict(kv.findall(line)))
for run, calls in runs.items():
    groups = defaultdict(list)
    for c in calls:
        groups[(c["routine"], c["ta"], int(c["M"]), int(c["N"]), int(c["K"]))].append(c)
    full = max(groups, key=lambda k: k[2] * k[3] * k[4])
    tot = defaultdict(int)
    print(f"{run}: {len(calls)} GPU-routed calls")
    for key in sorted(groups, key=lambda k: -k[2] * k[3] * k[4]):
        cs = groups[key]
        n = defaultdict(int)
        for c in cs:
            p = c["path"]
            cls = p if p.startswith("fallback") else ("staged" if int(c.get("epilogue_ns", 0)) == 0 else "host")
            n[cls] += 1
            tot[cls] += 1
        st = {}
        gpu = [c for c in cs if c["path"] == "gpu"]
        for s in ("pack", "upload", "kernel", "download", "epilogue", "other"):
            v = [int(c[s + "_ns"]) / 1e6 for c in gpu if s + "_ns" in c]
            st[s] = statistics.median(v) if v else float("nan")
        wall = statistics.median([(int(c["t1_ns"]) - int(c["t0_ns"])) / 1e6 for c in cs])
        tag = "full" if key == full else "slice"
        print(f"  {tag:5s} {key[0]} {key[1]} M={key[2]} N={key[3]} K={key[4]}: n={len(cs)} " + " ".join(f"{k}={v}" for k, v in sorted(n.items()))
              + f" | wall {wall:.2f} pack {st['pack']:.3f} upload {st['upload']:.3f} kernel {st['kernel']:.2f} download {st['download']:.3f} epilogue {st['epilogue']:.3f} other {st['other']:.3f}")
    print("  TOTAL " + " ".join(f"{k}={v}" for k, v in sorted(tot.items())))
