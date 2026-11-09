#!/usr/bin/env python3
"""G3, the library's split against the user code's: paired intervals from bench_split's raw.tsv.

usage: g3.py [--level 0.95] [--resamples 9999] [--seed 1] [--tsv FILE] <dir> ...

Reads every <case>.np<N>.d<data>.raw.tsv below the directories (run.sh's <outdir> or its suites)
whose fixed block has the SEAM arm, with the .out report beside it. For each case and layout:

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
- The seam's timeline per call, summarised over every process's SEAM calls, median [min-max] in ms:
  init (thread created - cut), start (GPU part's start - thread created), CPU part, GPU part, wait for
  the GPU (join - CPU part's end), after the join (the K axis's sum or a redo), and the GPU part's
  submit (commit to GPUStartTime), GPU execution (GPUStartTime to GPUEndTime) and wake (GPUEndTime to
  waitUntilCompleted's return); with the user split's own cpu, gpu, join and wake beside them.

BCa needs scipy (on this Mac: /Library/Frameworks/Python.framework/Versions/3.13/bin/python3);
without it those columns read n/a and the verdict uses the Hodges-Lehmann bounds.
"""
import argparse, glob, math, os, re, statistics, sys
from collections import defaultdict

try:
    import numpy as np
    from scipy import stats as sps
except ImportError:
    np = sps = None


def role(name):
    return "SPLIT" if re.fullmatch(r"S\d+", name) else "SPLIT2" if re.fullmatch(r"S\d+b", name) else name


def read_raw(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path)]
    return [dict(zip(rows[0], r)) for r in rows[1:]]


def lockstep(recs):
    """{role: {(round, rep): seconds}} of the fixed block, the slowest process per repeat"""
    out = defaultdict(dict)
    for d in recs:
        if d["phase"] != "1":
            continue
        k = (int(d["round"]), int(d["rep"]))
        r = role(d["name"])
        out[r][k] = max(out[r].get(k, 0.0), float(d["total"]))
    return out


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


SPANS = [  # name, raw.tsv columns (a, b) for b - a, or one column of ns
    ("init", ("at_cut_ns", "at_create_ns")), ("start", ("at_create_ns", "at_gpu0_ns")),
    ("cpu part", ("at_cpu0_ns", "at_cpu1_ns")), ("gpu part", ("at_gpu0_ns", "at_gpu1_ns")),
    ("wait for gpu", ("at_cpu1_ns", "at_join_ns")), ("after join", ("at_join_ns", "at_end_ns")),
    ("submit", ("gpu_submit_ns",)), ("gpu exec", ("gpu_start_ns", "gpu_end_ns")), ("wake", ("gpu_wake_ns",)),
]


def timeline(recs):
    """{span: [ms per SEAM call]} over every process's fixed-block SEAM calls; a span with a 0 end is left out"""
    out = defaultdict(list)
    for d in recs:
        if d["phase"] != "1" or d["name"] != "SEAM":
            continue
        for name, cols in SPANS:
            v = [int(d[c]) for c in cols]
            if len(v) == 1:
                out[name].append(v[0] * 1e-6)
            elif v[0] and v[1]:
                out[name].append((v[1] - v[0]) * 1e-6)
    return out


def user_parts(recs):
    out = defaultdict(list)
    for d in recs:
        if d["phase"] == "1" and role(d["name"]) == "SPLIT":
            for c in ("cpu", "gpu", "join", "wake"):
                out[c].append(float(d[c]) * 1e3)
    return out


def mr(v):
    return f"{statistics.median(v):.3f} [{min(v):.3f}-{max(v):.3f}]" if v else "-"


def summary(txt):
    m = re.search(r"^SUMMARY (.*)$", txt, re.M)
    return dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv) if m else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--resamples", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tsv", help="also write one row per case here")
    a = ap.parse_args()
    files = sorted({f for d in a.dirs for f in glob.glob(os.path.join(d, "**", "*.raw.tsv"), recursive=True)})
    rows = []
    for f in files:
        recs = read_raw(f)
        if not recs or "name" not in recs[0] or not any(d["name"] == "SEAM" for d in recs):
            continue
        txt = open(f[:-8] + ".out").read() if os.path.exists(f[:-8] + ".out") else ""
        v = summary(txt)
        vl = re.search(r"VECLIB_MAXIMUM_THREADS=(\S+)", txt)
        st = re.search(r"AB_STAGING_MB=(\S+)", txt)
        L = lockstep(recs)
        seam = paired(L, "SEAM", "SPLIT", a.level, a.resamples, a.seed)
        null = paired(L, "SPLIT2", "SPLIT", a.level, a.resamples, a.seed)
        rows.append({
            "suite": os.path.basename(os.path.dirname(f)), "case": v.get("case", os.path.basename(f).split(".")[0]),
            "layout": f"{v.get('np', '?')}x{vl.group(1) if vl else '?'}", "staging": st.group(1) if st else "?",
            "copied": v.get("reg") == "0", "g": v.get("g", "?"), "seam": seam, "null": null,
            "gate": verdict(seam["bca_hi"], null["bca_hi"]) if sps is not None else verdict(seam["hl_hi"], null["hl_hi"]),
            "check": verdict(seam["hl_hi"], null["hl_hi"]), "out": v.get("seam_out", "?"),
            "bits": f"{v.get('seam_rep_diff', '?')}, {v.get('seam_vs_split', '?')}", "tl": timeline(recs), "user": user_parts(recs),
        })
    if not rows:
        sys.exit("no raw.tsv with SEAM calls under " + " ".join(a.dirs))
    pct = f"{a.level:.0%}"
    print(f"## G3: SEAM / S<g> against max(1.03, S<g>b / S<g>), {pct} intervals"
          + ("" if sps is not None else " (no scipy: BCa n/a, gate on Hodges-Lehmann)") + "\n")
    print("| suite | case | layout | staging MB | g | pairs | SEAM/S median | BCa | HL [exact] | S_b/S median | BCa | HL [exact] "
          "| G3 (BCa) | check (HL) | seam outcomes | seam bits: repeats, vs S |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s, n = r["seam"], r["null"]
        f3 = lambda x: "n/a" if math.isnan(x) else f"{x:.3f}"
        hl = lambda p: f"{p['hl']:.3f} [{p['hl_lo']:.3f}, {p['hl_hi']:.3f}]" + ("" if p["hl_conf"] >= a.level - 1e-9 else f" ({p['hl_conf']:.0%})")
        print(f"| {r['suite']} | {r['case']} | {r['layout']} | {r['staging']}{' copied' if r['copied'] else ''} | {r['g']} | {s['n']} | "
              f"{s['median']:.3f} | [{f3(s['bca_lo'])}, {f3(s['bca_hi'])}] | {hl(s)} | {n['median']:.3f} | [{f3(n['bca_lo'])}, {f3(n['bca_hi'])}] | "
              f"{hl(n)} | {r['gate']} | {r['check']} | {r['out']} | {r['bits']} |")
    print("\n## The seam's timeline per call, ms, median [min-max] over every process's SEAM calls; the user split's parts beside\n")
    names = [n for n, _ in SPANS]
    print("| suite | case | layout | staging MB | " + " | ".join(names) + " | user cpu | user gpu | user join | user wake |")
    print("|" + "---|" * (4 + len(names) + 4))
    for r in rows:
        print(f"| {r['suite']} | {r['case']} | {r['layout']} | {r['staging']} | " + " | ".join(mr(r["tl"][n]) for n in names)
              + " | " + " | ".join(mr(r["user"][c]) for c in ("cpu", "gpu", "join", "wake")) + " |")
    if a.tsv:
        with open(a.tsv, "w") as fh:
            cols = ["suite", "case", "layout", "staging", "copied", "g"]
            stat = ["n", "median", "bca_lo", "bca_hi", "hl", "hl_lo", "hl_hi", "hl_conf"]
            fh.write("\t".join(cols + [f"seam_{k}" for k in stat] + [f"null_{k}" for k in stat] + ["gate", "check", "seam_out"]
                               + [f"tl_{n.replace(' ', '_')}_med" for n in names]) + "\n")
            for r in rows:
                vals = [str(r[c]) for c in cols] + [f"{r['seam'][k]:.6g}" for k in stat] + [f"{r['null'][k]:.6g}" for k in stat]
                vals += [r["gate"], r["check"], r["out"]] + [f"{statistics.median(r['tl'][n]):.6g}" if r["tl"][n] else "nan" for n in names]
                fh.write("\t".join(vals) + "\n")


if __name__ == "__main__":
    main()
