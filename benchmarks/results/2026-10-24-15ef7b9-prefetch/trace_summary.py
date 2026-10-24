"""Summarize AB_TRACE files trace_{off,pf}_<round>.<pid> of bench_rect_dgemm runs.

usage: python3 trace_summary.py <dir with the trace files>
Per dgemm_ shape: the median kernel stage (command-buffer GPU time), wall time and pack stage per call,
with AB_PREFETCH=0 (off) and the default (pf).
"""
import glob, os, re, statistics as st, sys, collections
d = sys.argv[1]
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(os.path.join(d, "trace_*")):
    m = re.match(r"trace_(off|pf)_(\d)\.\d+$", os.path.basename(f))
    if not m: continue
    mode = m.group(1)
    for line in open(f):
        if line.startswith("#") or "routine=dgemm" not in line: continue
        kv = dict(x.split("=", 1) for x in line.split() if "=" in x)
        key = f'{kv["M"]}x{kv["N"]}x{kv["K"]}'
        agg[key][mode + "_wall"].append((int(kv["t1_ns"]) - int(kv["t0_ns"])) / 1e6)
        agg[key][mode + "_kernel"].append(int(kv["kernel_ns"]) / 1e6)
        agg[key][mode + "_pack"].append(int(kv["pack_ns"]) / 1e6)
print(f"{'dgemm_ shape':>16} {'calls':>5} {'kernel off':>10} {'kernel pf':>10} {'off/pf':>7} {'wall off':>9} {'wall pf':>8} {'off/pf':>7} {'pack':>6}")
for key, v in sorted(agg.items()):
    k0, k1 = st.median(v["off_kernel"]), st.median(v["pf_kernel"])
    w0, w1 = st.median(v["off_wall"]), st.median(v["pf_wall"])
    print(f"{key:>16} {len(v['pf_wall']):5d} {k0:10.3f} {k1:10.3f} {k0/k1:7.3f} {w0:9.2f} {w1:8.2f} {w0/w1:7.3f} {st.median(v['pf_pack']):6.1f}")
