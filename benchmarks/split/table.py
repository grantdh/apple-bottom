#!/usr/bin/env python3
"""Results tables from bench_split's output: run.sh's <outdir> or a results folder.

usage: table.py <dir> [suite ...]

For each case: the fixed block's lockstep times (the slowest process per repeat), median [min-max]
over every repeat, for Accelerate alone, the GPU alone and the split at the sweep's best share; the
split's speedups (ratios of medians); its throughput as a fraction of the two engines' summed; bit
repeats; the largest error in u^2 = 2^-48; and whether the case started and ended on AC power.
Then the sustained runs' throughput, wall power and GPU clock.
"""
import glob, os, re, statistics, sys
from collections import defaultdict

CLASS = {
    "D2048": "DGEMM 2048²", "D4096": "DGEMM 4096²", "D8192": "DGEMM 8192²",
    "Z2048": "ZGEMM 2048²", "Z4096": "ZGEMM 4096²", "Z8192": "ZGEMM 8192²",
    "UN1000": "U-N update 18277×500×1000", "UN512": "U-N update 18277×500×512",
    "UN1000L": "U-N update 18277×500×1000, shares 0.06–0.32", "UN512L": "U-N update 18277×500×512, shares 0.06–0.32",
    "UM160": "U-M update 18277×160×160", "UM320": "U-M update 18277×160×320", "UM1000": "U-M update 18277×160×1000",
    "RK500": "R-K overlap 500×500×18277", "RK160": "R-K overlap 160×160×18432",
}


def parse(txt, tag):
    """the key=value fields of the report's <tag> line (SUMMARY or SUSTAIN), or None. Reports written
    before stdout was line-buffered can have the exit-time AB_REPORT inside that line, cutting it short:
    a token without '=' is skipped, and the fields after the cut are missing."""
    m = re.search(rf"^{tag} (.*)$", txt, re.M)
    return dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv) if m else None


def on_ac(txt):
    return all(re.search(rf"^{w}: Now drawing from 'AC Power'", txt, re.M) for w in ("start", "end"))


def fixed_block(raw):
    """lockstep times (ms) per config of the fixed block (phase 1): {cfg: [one per repeat]}"""
    rows = [l.rstrip("\n").split("\t") for l in open(raw)]
    worst = defaultdict(float)
    for r in rows[1:]:
        d = dict(zip(rows[0], r))
        if d["phase"] == "1":
            k = (d["cfg"], d["round"], d["rep"])
            worst[k] = max(worst[k], float(d["total"]) * 1e3)
    out = defaultdict(list)
    for (cfg, _, _), v in worst.items():
        out[cfg].append(v)
    return out


def mr(v):
    return f"{statistics.median(v):.2f} [{min(v):.2f}–{max(v):.2f}]"


root = sys.argv[1]
subs = sys.argv[2:] or sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d != "sustain")
print("| suite | shape class | layout | data | Accelerate alone ms | GPU alone ms | split ms (f) | vs Accelerate | vs GPU "
      "| split ÷ summed GF/s | bits repeat | max err (u²) | power |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for s in subs:
    for f in sorted(glob.glob(os.path.join(root, s, "*.out"))):
        txt = open(f).read()
        v = parse(txt, "SUMMARY")
        if not v:
            continue
        threads = re.search(r"VECLIB_MAXIMUM_THREADS=(\d+)", txt).group(1)
        fb = fixed_block(f[:-4] + ".raw.tsv")
        split_cfg = max(fb, key=int)                     # the best split's config index (>= 2)
        la, lg, ls = (statistics.median(fb[k]) for k in ("0", "1", split_cfg))
        data = ("QE-like" if v["data"] == "1" else "uniform") + (", copied" if v.get("reg", "4") == "0" else "")
        print(f"| {s} | {CLASS.get(v['case'], v['case'])} | {v['np']}×{threads} | {data} | {mr(fb['0'])} | {mr(fb['1'])} | "
              f"{mr(fb[split_cfg])} ({float(v['f']):.2f}) | {la / ls:.2f}× | {lg / ls:.2f}× | {la * lg / (ls * (la + lg)):.2f} | "
              f"{v['bits_repeat']} ({v['split_reps']}) | {float(v['err_u2']):.2f} | {'AC' if on_ac(txt) else 'NOT AC'} |")

sus = sorted(glob.glob(os.path.join(root, "sustain", "*.sustain.out")))
if sus:
    print()
    print("| case | run | GPU part (f) | burst GF/s (first 10 s) | steady GF/s (from 20 s) | wall W | GPU MHz | power |")
    print("|---|---|---|---|---|---|---|---|")
    for f in sus:
        txt = open(f).read()
        v = parse(txt, "SUSTAIN")
        if not v:
            continue
        m = re.search(r"^case \S+ \w+ M=(\d+) N=(\d+) K=(\d+), split axis (\w)", txt, re.M)
        length = int(m.group({"M": 1, "N": 2, "K": 3}[m.group(4)]))
        st, bu, w, mhz = (v[k].split("/") for k in ("steady_gfs", "burst_gfs", "wall_w", "mhz"))
        gs = [v["g"], None, None, v["g_bal"]]
        for i, name in enumerate(("split", "GPU alone", "Accelerate alone", "split, rebalanced")):
            share = f"{gs[i]} ({int(gs[i]) / length:.3f})" if gs[i] else "–"
            print(f"| {v['case']} | {name} | {share} | {bu[i]} | {st[i]} | {w[i]} | {mhz[i]} | {'AC' if on_ac(txt) else 'NOT AC'} |")
        print(f"| {v['case']} | steady split ÷ GPU alone, ÷ Accelerate alone | | | {v['sp_gpu']}×, {v['sp_acc']}× "
              f"(rebalanced {v['sp_gpu_bal']}×, {v['sp_acc_bal']}×) | | | |")
