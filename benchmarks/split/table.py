#!/usr/bin/env python3
"""Results tables from bench_split's output: run.sh's <outdir> or a results folder.

usage: table.py <dir> [suite ...]

For each case: the fixed block's lockstep times (the slowest process per repeat), median [min-max]
over every repeat, for Accelerate alone, the GPU alone and the split at the sweep's best share; the
split's speedups (ratios of medians); its throughput as a fraction of the two engines' summed; bit
repeats; the largest error in u^2 = 2^-48; and whether the case started and ended on AC power.
Cases run with the arms (raw.tsv's name column has SEAM) get a second table: the library's seam and
the user split's twin against the user split (medians of the ratios paired by round and repeat), the
slices alone, and the seam's outcomes and bits. Then the sustained runs' throughput, wall power and
GPU clock.
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
    "A8_UN512": "A zUN.k1 9139×500×512", "A8_RK512": "A zRK.t1 512×500×9139", "A8_RK1000": "A zRK.t2 500×1000×9176",
    "B8_UM1152": "B zUM.k2 8927×160×1152", "B8_RK1152": "B zRK.t1 1152×160×8988",
}


def parse(txt, tag):
    """the key=value fields of the report's <tag> line (SUMMARY or SUSTAIN), or None. Reports written
    before stdout was line-buffered can have the exit-time AB_REPORT inside that line, cutting it short:
    a token without '=' is skipped, and the fields after the cut are missing."""
    m = re.search(rf"^{tag} (.*)$", txt, re.M)
    return dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv) if m else None


def on_ac(txt):
    return all(re.search(rf"^{w}: Now drawing from 'AC Power'", txt, re.M) for w in ("start", "end"))


def role(name):
    """a config's role from raw.tsv's name: ACC, GPU, SPLIT (S<g>), SPLIT2 (S<g>b), SEAM, HEAD, TAIL"""
    return "SPLIT" if re.fullmatch(r"S\d+", name) else "SPLIT2" if re.fullmatch(r"S\d+b", name) else name


def fixed_pairs(raw):
    """lockstep times (ms) of the fixed block (phase 1): {role: {(round, rep): ms}}. Files written
    before raw.tsv had a name column: cfg 0 is ACC, 1 GPU and the highest index the split."""
    rows = [l.rstrip("\n").split("\t") for l in open(raw)]
    h = rows[0]
    recs = [dict(zip(h, r)) for r in rows[1:]]
    recs = [d for d in recs if d["phase"] == "1"]
    if "name" in h:
        key = lambda d: role(d["name"])
    else:
        top = max(int(d["cfg"]) for d in recs)
        key = lambda d: {"0": "ACC", "1": "GPU"}.get(d["cfg"], "SPLIT" if int(d["cfg"]) == top else d["cfg"])
    out = defaultdict(dict)
    for d in recs:
        k = (d["round"], d["rep"])
        out[key(d)][k] = max(out[key(d)].get(k, 0.0), float(d["total"]) * 1e3)
    return out


def fixed_block(raw):
    """lockstep times (ms) per role of the fixed block: {role: [one per repeat]}"""
    return {k: list(v.values()) for k, v in fixed_pairs(raw).items()}


def paired(fp, a, b):
    """median over (round, rep) of a's lockstep time over b's"""
    r = [fp[a][k] / fp[b][k] for k in fp[a] if k in fp[b]]
    return (statistics.median(r), len(r)) if r else (float("nan"), 0)


def mr(v):
    return f"{statistics.median(v):.2f} [{min(v):.2f}–{max(v):.2f}]"


root = sys.argv[1]
arms = []
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
        fp = fixed_pairs(f[:-4] + ".raw.tsv")
        fb = {k: list(x.values()) for k, x in fp.items()}
        la, lg, ls = (statistics.median(fb[k]) for k in ("ACC", "GPU", "SPLIT"))
        data = ("QE-like" if v["data"] == "1" else "uniform") + (", copied" if v.get("reg", "4") == "0" else "")
        print(f"| {s} | {CLASS.get(v['case'], v['case'])} | {v['np']}×{threads} | {data} | {mr(fb['ACC'])} | {mr(fb['GPU'])} | "
              f"{mr(fb['SPLIT'])} ({float(v['f']):.2f}) | {la / ls:.2f}× | {lg / ls:.2f}× | {la * lg / (ls * (la + lg)):.2f} | "
              f"{v['bits_repeat']} ({v['split_reps']}) | {float(v['err_u2']):.2f} | {'AC' if on_ac(txt) else 'NOT AC'} |")
        if "SEAM" in fp:
            staging = re.search(r"AB_STAGING_MB=(\S+)", txt).group(1)
            arms.append((s, v, threads, data, staging, fp))

if arms:
    print()
    print("| suite | shape class | layout | data | staging MB | split ms | SEAM ms | SEAM ÷ split (pairs) | twin ÷ split (pairs) "
          "| HEAD ms | TAIL ms | (HEAD + TAIL) ÷ split | seam outcomes | seam bits: repeats, vs split |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s, v, threads, data, staging, fp in arms:
        md = lambda r: statistics.median(fp[r].values())
        rs, ns = paired(fp, "SEAM", "SPLIT")
        rn, nn = paired(fp, "SPLIT2", "SPLIT")
        print(f"| {s} | {CLASS.get(v['case'], v['case'])} | {v['np']}×{threads} | {data} | {staging} | {md('SPLIT'):.2f} | "
              f"{md('SEAM'):.2f} | {rs:.3f} ({ns}) | {rn:.3f} ({nn}) | {md('HEAD'):.2f} | {md('TAIL'):.2f} | "
              f"{(md('HEAD') + md('TAIL')) / md('SPLIT'):.2f} | {v.get('seam_out', '?')} | {v.get('seam_rep_diff', '?')}, {v.get('seam_vs_split', '?')} |")

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
        if "head_ms" in v:   # runs with the slices alone and the second control
            print(f"| {v['case']} | Accelerate alone, again | – | – | {v['acc_end_gfs']} | {v['acc_end_w']} | – | {'AC' if on_ac(txt) else 'NOT AC'} |")
            print(f"| {v['case']} | slices alone (steady ms per call): HEAD {v['head_ms']}, TAIL {v['tail_ms']}; split {v['split_steady_ms']} | | | "
                  f"(HEAD + TAIL) ÷ split {v['overlap']}; cpu slice {v['cpu_x']}×, gpu slice {v['gpu_x']}× their time alone | | | order {v['order']} |")
