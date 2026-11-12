#!/usr/bin/env python3
"""G3, the library's split against the user code's: paired intervals from bench_split's raw.tsv.

usage: g3.py [--level 0.95] [--resamples 9999] [--seed 1] [--max-fcpu 1.5] [--max-gpu-busy 0.25]
             [--tsv FILE] <dir> ...

Reads every <case>.np<N>.d<data>.raw.tsv below the directories (run.sh's <outdir> or its suites)
whose fixed block has the SEAM arm, with the .out report beside it. Left out, and listed at the end:
cases that run.sh set aside (a path with an incomplete- component) and reports without a SUMMARY
line or whose start and end lines are not both on AC power. Only rows under G3's conditions get a
verdict (copied operands, uniform data, 1x8 or 4x2, AB_STAGING_MB 2048 or 8192); the others read
"not G3". For each case and layout:

- Pairs by (round, rep) of the fixed block's lockstep times (the slowest process per repeat):
  SEAM with the user split S<g>, and the twin S<g>b with S<g> (the null run: the user split against
  itself).
- For both ratios: the median of the paired ratios; a BCa bootstrap interval of that median
  (scipy.stats.bootstrap, paired); and the Hodges-Lehmann estimate of the ratio (on log ratios) with
  its exact Wilcoxon signed-rank interval. Two-sided intervals at --level.
- G3's verdict (design section 12): the upper bound of SEAM / S<g> against max(1.03, the upper bound
  of S<g>b / S<g>); above it: "fix first". The gate uses the BCa bounds; the Hodges-Lehmann verdict is
  a check. At n pairs the exact interval cannot reach the level when n < 6 (it is then the range of
  the Walsh averages, with its own confidence).
- Outside load, from the .rounds.tsv beside the report: the largest over the rounds of other
  processes' CPU (cores) and of the GPU's busy fraction while this run left the GPU idle; a row with
  either over --max-fcpu or --max-gpu-busy is flagged.
- The seam's timeline per call, summarised over every process's SEAM calls, median [min-max] in ms:
  init (thread created - cut) and start (GPU part's start - thread created), beside the user split's
  GPU start (its GPU thread's start after the call's); the CPU part and the GPU part, each beside the
  user split's; the wait for the GPU (join - CPU part's end) beside the user split's; after the join
  (the K axis's sum or a redo); and the GPU part's submit (commit to GPUStartTime), GPU execution
  (GPUStartTime to GPUEndTime) and wake (GPUEndTime to waitUntilCompleted's return).

BCa needs scipy (on this Mac: /Library/Frameworks/Python.framework/Versions/3.13/bin/python3);
without it those columns read n/a and the verdict uses the Hodges-Lehmann bounds.
"""
import argparse, glob, math, os, re, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # rawtsv.py, also under python3 -I
from rawtsv import env, fixed_pairs, on_ac, parse, read  # noqa: E402

try:
    import numpy as np
    from scipy import stats as sps
except ImportError:
    np = sps = None


def walsh(d):
    return sorted((d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d)))


def signed_rank_cdf(n):
    """P(T <= t), t = 0..n(n+1)/2, for the Wilcoxon signed-rank statistic T with n untied pairs"""
    N = n * (n + 1) // 2
    c = [1] + [0] * N
    for k in range(1, n + 1):
        for t in range(N, k - 1, -1):
            c[t] += c[t - k]
    tot, acc, cdf = 2 ** n, 0, []
    for t in range(N + 1):
        acc += c[t]
        cdf.append(acc / tot)
    return cdf


def hodges_lehmann(d, level):
    """HL estimate of the location of d and its exact Wilcoxon interval: [A_(c), A_(N+1-c)] of the
    sorted Walsh averages A, c the largest with P(T <= c - 1) <= (1 - level) / 2; the achieved level"""
    n = len(d)
    A = walsh(d)
    est = statistics.median(A)
    if n == 0:
        return est, float("nan"), float("nan"), float("nan")
    cdf = signed_rank_cdf(n)
    alpha = (1 - level) / 2
    c = 0
    while c + 1 <= len(cdf) and cdf[c] <= alpha:   # cdf[c] = P(T <= c) = P(T <= (c + 1) - 1)
        c += 1
    if c == 0:   # n too small to reach the level: the full range, at its own confidence
        return est, A[0], A[-1], 1 - 2 / 2 ** n
    return est, A[c - 1], A[len(A) - c], 1 - 2 * cdf[c - 1]


def bca(x, y, level, resamples, seed):
    """BCa interval of median(x / y) over pairs; nan if scipy is missing or the ratios do not vary"""
    if sps is None or len(x) < 3 or len(set(a / b for a, b in zip(x, y))) < 2:
        return float("nan"), float("nan")
    res = sps.bootstrap((np.asarray(x), np.asarray(y)), lambda a, b, axis=-1: np.median(a / b, axis=axis), paired=True,
                        vectorized=True, confidence_level=level, n_resamples=resamples, method="BCa",
                        random_state=np.random.default_rng(seed))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def paired(L, a, b, level, resamples, seed):
    keys = sorted(k for k in L.get(a, {}) if k in L.get(b, {}))
    x = [L[a][k] for k in keys]
    y = [L[b][k] for k in keys]
    r = [p / q for p, q in zip(x, y)]
    out = {"n": len(r), "median": statistics.median(r) if r else float("nan")}
    out["bca_lo"], out["bca_hi"] = bca(x, y, level, resamples, seed)
    hl, lo, hi, conf = hodges_lehmann([math.log(v) for v in r], level)
    out["hl"], out["hl_lo"], out["hl_hi"], out["hl_conf"] = math.exp(hl), math.exp(lo), math.exp(hi), conf
    return out


def verdict(seam_hi, null_hi):
    if math.isnan(seam_hi):
        return "n/a"
    bound = max(1.03, null_hi) if not math.isnan(null_hi) else 1.03
    return ("pass" if seam_hi <= bound else "fix first") + f" ({seam_hi:.3f} vs {bound:.3f})"


# The timeline table's columns, in order: (name, role, columns). SEAM spans are b - a of two raw.tsv ns
# columns (left out when either is 0), or one ns column; the user split's (SPLIT) are its s columns.
TIMELINE = [
    ("init", "SEAM", ("at_cut_ns", "at_create_ns")), ("start", "SEAM", ("at_create_ns", "at_gpu0_ns")),
    ("user gpu start", "SPLIT", ("wake",)),
    ("cpu part", "SEAM", ("at_cpu0_ns", "at_cpu1_ns")), ("user cpu", "SPLIT", ("cpu",)),
    ("gpu part", "SEAM", ("at_gpu0_ns", "at_gpu1_ns")), ("user gpu", "SPLIT", ("gpu",)),
    ("wait for gpu", "SEAM", ("at_cpu1_ns", "at_join_ns")), ("user wait", "SPLIT", ("join",)),
    ("after join", "SEAM", ("at_join_ns", "at_end_ns")),
    ("submit", "SEAM", ("gpu_submit_ns",)), ("gpu exec", "SEAM", ("gpu_start_ns", "gpu_end_ns")),
    ("wake", "SEAM", ("gpu_wake_ns",)),
]


def timeline(recs):
    """{column: [ms per call]} over every process's fixed-block calls of the column's role"""
    out = defaultdict(list)
    for d in recs:
        if d["phase"] != "1":
            continue
        for name, role, cols in TIMELINE:
            if d["role"] != role:
                continue
            if role == "SPLIT":
                out[name].append(float(d[cols[0]]) * 1e3)
                continue
            v = [int(d[c]) for c in cols]
            if len(v) == 1:
                out[name].append(v[0] * 1e-6)
            elif v[0] and v[1]:
                out[name].append((v[1] - v[0]) * 1e-6)
    return out


def outside_load(path):
    """the largest over the rounds of other processes' CPU (cores, over the GPU-idle stretch, else the
    round) and of the GPU's busy fraction over that stretch, from .rounds.tsv; (nan, nan) without it"""
    if not os.path.exists(path):
        return float("nan"), float("nan")
    rows = [l.rstrip("\n").split("\t") for l in open(path)]
    recs = [dict(zip(rows[0], r)) for r in rows[1:]]
    big = lambda k: max((float(d[k]) for d in recs if d[k] != "nan"), default=float("nan"))
    fc = big("fcpu_idle")
    return (fc if not math.isnan(fc) else big("fcpu_round")), big("gpu_busy_idle")


def g3_conditions(v, layout, staging):
    """G3's conditions (design section 12): copied operands, uniform data, 1x8 or 4x2, staging 2048 or 8192 MB"""
    return v.get("reg") == "0" and v.get("data") == "0" and layout in ("1x8", "4x2") and staging in ("2048", "8192")


def mr(v):
    return f"{statistics.median(v):.3f} [{min(v):.3f}-{max(v):.3f}]" if v else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--resamples", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-fcpu", type=float, default=1.5, help="flag a row whose other processes used more cores (as the AB_SPLIT pilot)")
    ap.add_argument("--max-gpu-busy", type=float, default=0.25, help="flag a row whose GPU was busier while this run left it idle")
    ap.add_argument("--tsv", help="also write one row per case here")
    a = ap.parse_args()
    files = sorted({f for d in a.dirs for f in glob.glob(os.path.join(d, "**", "*.raw.tsv"), recursive=True)})
    rows, skipped = [], []
    for f in files:
        if any(part.startswith("incomplete-") for part in f.split(os.sep)):
            skipped.append((f, "set aside by run.sh"))
            continue
        recs = read(f)
        if not any(d["role"] == "SEAM" for d in recs):
            continue
        txt = open(f[:-8] + ".out").read() if os.path.exists(f[:-8] + ".out") else ""
        v = parse(txt, "SUMMARY")
        if not v or not on_ac(txt):
            skipped.append((f, "no SUMMARY line" if not v else "did not start and end on AC power"))
            continue
        vl, st = env(txt, "VECLIB_MAXIMUM_THREADS"), env(txt, "AB_STAGING_MB")
        layout, staging = f"{v.get('np', '?')}x{vl or '?'}", st or "?"
        L = fixed_pairs(recs)
        seam = paired(L, "SEAM", "SPLIT", a.level, a.resamples, a.seed)
        null = paired(L, "SPLIT2", "SPLIT", a.level, a.resamples, a.seed)
        g3 = g3_conditions(v, layout, staging)
        fcpu, gbusy = outside_load(f[:-8] + ".rounds.tsv")
        flag = "-" if math.isnan(fcpu) and math.isnan(gbusy) else \
            f"{'FLAG ' if fcpu > a.max_fcpu or gbusy > a.max_gpu_busy else ''}{fcpu:.2f} cores, GPU {gbusy:.2f}"
        rows.append({
            "suite": os.path.basename(os.path.dirname(f)), "case": v.get("case", os.path.basename(f).split(".")[0]),
            "layout": layout, "staging": staging, "copied": v.get("reg") == "0", "g": v.get("g", "?"), "seam": seam, "null": null,
            "gate": (verdict(seam["bca_hi"], null["bca_hi"]) if sps is not None else verdict(seam["hl_hi"], null["hl_hi"])) if g3 else "not G3",
            "check": verdict(seam["hl_hi"], null["hl_hi"]) if g3 else "not G3", "out": v.get("seam_out", "?"), "load": flag,
            "bits": f"{v.get('seam_rep_diff', '?')}, {v.get('seam_vs_split', '?')}", "tl": timeline(recs),
        })
    if not rows:
        sys.exit("no raw.tsv with SEAM calls under " + " ".join(a.dirs))
    pct = f"{a.level:.0%}"
    print(f"## G3: SEAM / S<g> against max(1.03, S<g>b / S<g>), {pct} intervals"
          + ("" if sps is not None else " (no scipy: BCa n/a, gate on Hodges-Lehmann)") + "\n")
    print("| suite | case | layout | staging MB | g | pairs | SEAM/S median | BCa | HL [exact] | S_b/S median | BCa | HL [exact] "
          "| G3 (BCa) | check (HL) | outside load (max over rounds) | seam outcomes | seam bits: repeats, vs S |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s, n = r["seam"], r["null"]
        f3 = lambda x: "n/a" if math.isnan(x) else f"{x:.3f}"
        hl = lambda p: f"{p['hl']:.3f} [{p['hl_lo']:.3f}, {p['hl_hi']:.3f}]" + ("" if p["hl_conf"] >= a.level - 1e-9 else f" ({p['hl_conf']:.0%})")
        print(f"| {r['suite']} | {r['case']} | {r['layout']} | {r['staging']}{' copied' if r['copied'] else ''} | {r['g']} | {s['n']} | "
              f"{s['median']:.3f} | [{f3(s['bca_lo'])}, {f3(s['bca_hi'])}] | {hl(s)} | {n['median']:.3f} | [{f3(n['bca_lo'])}, {f3(n['bca_hi'])}] | "
              f"{hl(n)} | {r['gate']} | {r['check']} | {r['load']} | {r['out']} | {r['bits']} |")
    print("\n## The seam's timeline per call, ms, median [min-max] over every process's SEAM calls; the user split's parts beside\n")
    names = [n for n, _, _ in TIMELINE]
    print("| suite | case | layout | staging MB | " + " | ".join(names) + " |")
    print("|" + "---|" * (4 + len(names)))
    for r in rows:
        print(f"| {r['suite']} | {r['case']} | {r['layout']} | {r['staging']} | " + " | ".join(mr(r["tl"][n]) for n in names) + " |")
    if skipped:
        print("\nLeft out:\n" + "\n".join(f"- {f}: {why}" for f, why in skipped))
    if a.tsv:
        with open(a.tsv, "w") as fh:
            cols = ["suite", "case", "layout", "staging", "copied", "g"]
            stat = ["n", "median", "bca_lo", "bca_hi", "hl", "hl_lo", "hl_hi", "hl_conf"]
            fh.write("\t".join(cols + [f"seam_{k}" for k in stat] + [f"null_{k}" for k in stat] + ["gate", "check", "load", "seam_out"]
                               + [f"tl_{n.replace(' ', '_')}_med" for n in names]) + "\n")
            for r in rows:
                vals = [str(r[c]) for c in cols] + [f"{r['seam'][k]:.6g}" for k in stat] + [f"{r['null'][k]:.6g}" for k in stat]
                vals += [r["gate"], r["check"], r["load"], r["out"]] + [f"{statistics.median(r['tl'][n]):.6g}" if r["tl"][n] else "nan" for n in names]
                fh.write("\t".join(vals) + "\n")


if __name__ == "__main__":
    main()
