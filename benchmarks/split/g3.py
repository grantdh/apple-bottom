#!/usr/bin/env python3
"""G3, the library's split against the user code's: paired intervals from bench_split's raw.tsv.

usage: g3.py [--level 0.95] [--t-level 0.95] [--resamples 9999] [--seed 1] [--max-fcpu 1.5]
             [--max-gpu-busy 0.25] [--tsv FILE] <dir> ...
       g3.py --simulate [--sims 400] [--seed 1]

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
- G3's verdict, two readings (design section 12's line, "upper bound over max(1.03, null spread): fix
  first", leaves the null spread to be defined; choosing between them is Grant's):
  (a) as designed: the upper bound of SEAM / S<g> against max(1.03, the upper bound of S<g>b / S<g>),
      on the pair intervals above; above it: "fix first". BCa bounds, the Hodges-Lehmann ones as a
      check. At n pairs the exact interval cannot reach the level when n < 6 (it is then the range of
      the Walsh averages, with its own confidence).
  (b) proposed, needs Grant: rounds as the unit, as Grant chose for G5. Per round, the mean over its
      repeats of log(SEAM / S<g>) and of log(S<g>b / S<g>); over the rounds, a one-sided --t-level t
      upper bound on each mean, exponentiated. "pass" if SEAM's bound is 1.03 or less; else
      "inconclusive: add rounds" if the null's own bound is above 1.03 (the user split against itself
      is too noisy to tell); else "fix first". Noise can still give "fix first", when SEAM's bound
      passes 1.03 by chance and the null's does not; less often than (a) (see --simulate).
  With each row, SEAM / S<g>'s one-sided lower bound at the same level, and the null spread by round:
  the sd of the null's per-round mean log ratios and the range of the per-round ratios.
- --simulate estimates how often each reading says "fix first" when the seam is exactly as fast as
  the user split (and how often it does when the seam is 5% slower), for 6 rounds of 2 repeats, per-
  call noise of 1-5% and per-round offsets of each arm of 0-2%; and for (b) also how often it would
  say "fix first" if that required SEAM / S<g>'s lower bound above 1.00 as well (a variant, not a
  reading g3.py applies).
- Outside load, from the report's SUMMARY (the largest over the rounds, .rounds.tsv has each): other
  processes' CPU in cores over the GPU-idle stretches (over whole rounds without the arms) and the
  GPU's busy fraction over those stretches; a row with either over --max-fcpu or --max-gpu-busy is
  flagged.
- The seam's timeline per call, summarised over every process's SEAM calls, median [min-max] in ms:
  init (thread created - cut) and start (GPU part's start - thread created), beside the user split's
  GPU start (its GPU thread's start after the call's); the CPU part and the GPU part, each beside the
  user split's; the wait for the GPU (join - CPU part's end) beside the user split's; after the join
  (the K axis's sum or a redo); and the GPU part's submit (commit to GPUStartTime), GPU execution
  (GPUStartTime to GPUEndTime) and wake (GPUEndTime to waitUntilCompleted's return).

BCa and reading (b)'s t quantile need scipy (on this Mac:
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3); without it those read n/a and (a)'s
verdict uses the Hodges-Lehmann bounds.
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


NAN = float("nan")


def by_round(L, a, b, q):
    """rounds as the unit: per round, the mean over its repeats of log(a / b), paired by (round, rep);
    over the rounds their geometric mean, the exp of one-sided q t upper and lower bounds on their
    mean (nan with fewer than 2 rounds or without scipy), the sd of the log means and their range"""
    acc = defaultdict(list)
    for k in L.get(a, {}):
        if k in L.get(b, {}):
            acc[k[0]].append(math.log(L[a][k] / L[b][k]))
    m = [statistics.fmean(acc[r]) for r in sorted(acc)]
    out = {"n": len(m), "mean": NAN, "t_hi": NAN, "t_lo": NAN, "sd": NAN, "lo": NAN, "hi": NAN}
    mu = statistics.fmean(m) if m else NAN
    if m:
        out.update(mean=math.exp(mu), lo=math.exp(min(m)), hi=math.exp(max(m)))
    if len(m) > 1:
        out["sd"] = statistics.stdev(m)
        if sps is not None:
            half = float(sps.t.ppf(q, len(m) - 1)) * out["sd"] / math.sqrt(len(m))
            out["t_hi"], out["t_lo"] = math.exp(mu + half), math.exp(mu - half)
    return out


def designed(seam_hi, null_hi):
    """reading (a): fix first above max(1.03, the null's own upper bound); (category, bound)"""
    bound = max(1.03, null_hi) if not math.isnan(null_hi) else 1.03
    return ("n/a" if math.isnan(seam_hi) else "pass" if seam_hi <= bound else "fix first"), bound


def proposed(seam_hi, null_hi):
    """reading (b): pass at 1.03, inconclusive when the null's own bound is above 1.03, else fix first"""
    if math.isnan(seam_hi):
        return "n/a"
    return "pass" if seam_hi <= 1.03 else "inconclusive: add rounds" if math.isnan(null_hi) or null_hi > 1.03 else "fix first"


# The timeline table's columns, in order: (name, role, columns). SEAM spans are b - a of two raw.tsv ns
# columns (left out when either is 0), or one ns column; the user split's (SPLIT) are its s columns.
TO_MS = {"SEAM": 1e-6, "SPLIT": 1e3}
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
            v = [float(d[c]) for c in cols]
            if len(v) == 1:
                out[name].append(v[0] * TO_MS[role])
            elif v[0] and v[1]:
                out[name].append((v[1] - v[0]) * TO_MS[role])
    return out


def g3_conditions(v, layout, staging):
    """G3's conditions (design section 12): copied operands, uniform data, 1x8 or 4x2, staging 2048 or 8192 MB"""
    return v.get("reg") == "0" and v.get("data") == "0" and layout in ("1x8", "4x2") and staging in ("2048", "8192")


def mr(v):
    return f"{statistics.median(v):.3f} [{min(v):.3f}-{max(v):.3f}]" if v else "-"


def simulate(sims, seed, level, tq, resamples):
    """false "fix first" rates of readings (a) and (b), and (b)'s "inconclusive" rate, for a seam as fast as
    the user split (ratio 1.00) and one 5% slower: 6 rounds x 2 repeats; each call's log time has noise
    of sd s; each arm in each round an offset of sd o (a round x arm effect, as position and drift give)"""
    import random
    rnd = random.Random(seed)
    print(f"## Simulated G3 readings: {sims} rows per line, 6 rounds of 2 repeats; (a) at two-sided {level:.0%} "
          f"pair intervals (BCa with {resamples} resamples), (b) at a one-sided {tq:.0%} t bound over rounds\n")
    print("| true SEAM/S | call noise sd | round x arm sd | (a) BCa: fix first | (a) HL: fix first | (b): fix first | (b): inconclusive "
          "| (b), fix first only if SEAM/S's lower bound > 1.00 |")
    print("|---|---|---|---|---|---|---|---|")
    for ratio in (1.00, 1.05):
        for sd in (0.01, 0.02, 0.03, 0.05):
            for o in (0.0, 0.01, 0.02):
                n_a = n_h = n_b = n_i = n_v = 0
                for _ in range(sims):
                    L = {"SPLIT": {}, "SEAM": {}, "SPLIT2": {}}
                    for r in range(6):
                        off = {c: rnd.gauss(0, o) for c in L}
                        for rep_ in range(2):
                            for c in L:
                                L[c][(r, rep_)] = math.exp(off[c] + rnd.gauss(0, sd) + (math.log(ratio) if c == "SEAM" else 0))
                    ps, pn = paired(L, "SEAM", "SPLIT", level, resamples, rnd.randrange(1 << 30)), paired(L, "SPLIT2", "SPLIT", level, resamples, rnd.randrange(1 << 30))
                    n_a += designed(ps["bca_hi"], pn["bca_hi"])[0] == "fix first"
                    n_h += designed(ps["hl_hi"], pn["hl_hi"])[0] == "fix first"
                    sr = by_round(L, "SEAM", "SPLIT", tq)
                    vb = proposed(sr["t_hi"], by_round(L, "SPLIT2", "SPLIT", tq)["t_hi"])
                    n_b += vb == "fix first"; n_i += vb == "inconclusive: add rounds"; n_v += vb == "fix first" and sr["t_lo"] > 1.0
                bca = f"{n_a / sims:.1%}" if sps is not None else "n/a"
                print(f"| {ratio:.2f} | {sd:.2f} | {o:.2f} | {bca} | {n_h / sims:.1%} | {n_b / sims:.1%} | {n_i / sims:.1%} | {n_v / sims:.1%} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--level", type=float, default=0.95, help="two-sided level of the pair intervals")
    ap.add_argument("--t-level", type=float, default=0.95, help="one-sided level of reading (b)'s t bound")
    ap.add_argument("--simulate", action="store_true", help="false-alarm rates of the two readings")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--resamples", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-fcpu", type=float, default=1.5, help="flag a row whose other processes used more cores (as the AB_SPLIT pilot)")
    ap.add_argument("--max-gpu-busy", type=float, default=0.25, help="flag a row whose GPU was busier while this run left it idle")
    ap.add_argument("--tsv", help="also write one row per case here")
    a = ap.parse_args()
    if a.simulate:
        return simulate(a.sims, a.seed, a.level, a.t_level, a.resamples)
    if not a.dirs:
        ap.error("give run.sh's <outdir> or suite directories")
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
        seam_r, null_r = by_round(L, "SEAM", "SPLIT", a.t_level), by_round(L, "SPLIT2", "SPLIT", a.t_level)
        fcpu = float(v.get("fcpu_max", "nan"))
        if math.isnan(fcpu):   # runs without the arms have no GPU-idle stretch
            fcpu = float(v.get("fcpu_round_max", "nan"))
        gk = "bca_hi" if sps is not None else "hl_hi"
        if g3_conditions(v, layout, staging):
            gate, check = designed(seam[gk], null[gk]), designed(seam["hl_hi"], null["hl_hi"])
            prop = proposed(seam_r["t_hi"], null_r["t_hi"])
        else:
            gate = check = ("not G3", NAN); prop = "not G3"
        rows.append({
            "suite": os.path.basename(os.path.dirname(f)), "case": v.get("case", os.path.basename(f).split(".")[0]),
            "layout": layout, "staging": staging, "copied": v.get("reg") == "0", "g": v.get("g", "?"), "seam": seam, "null": null,
            "gate": gate, "check": check, "proposed": prop, "seam_r": seam_r, "null_r": null_r,
            "fcpu": fcpu, "gbusy": float(v.get("gpu_idle_busy_max", "nan")), "out": v.get("seam_out", "?"),
            "bits": f"{v.get('seam_rep_diff', '?')}, {v.get('seam_vs_split', '?')}", "tl": timeline(recs),
        })
    if not rows:
        sys.exit("no raw.tsv with SEAM calls under " + " ".join(a.dirs))
    pct = f"{a.level:.0%}"
    vtext = lambda r, k, hi: r[k][0] if r[k][0] in ("not G3", "n/a") else f"{r[k][0]} ({r['seam'][hi]:.3f} vs {r[k][1]:.3f})"
    load = lambda r: "-" if math.isnan(r["fcpu"]) and math.isnan(r["gbusy"]) else \
        f"{'FLAG ' if r['fcpu'] > a.max_fcpu or r['gbusy'] > a.max_gpu_busy else ''}{r['fcpu']:.2f} cores, GPU {r['gbusy']:.2f}"
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
              f"{hl(n)} | {vtext(r, 'gate', 'bca_hi' if sps is not None else 'hl_hi')} | {vtext(r, 'check', 'hl_hi')} | {load(r)} | {r['out']} | {r['bits']} |")
    print("\n## The seam's timeline per call, ms, median [min-max] over every process's SEAM calls; the user split's parts beside\n")
    names = [n for n, _, _ in TIMELINE]
    print("| suite | case | layout | staging MB | " + " | ".join(names) + " |")
    print("|" + "---|" * (4 + len(names)))
    for r in rows:
        print(f"| {r['suite']} | {r['case']} | {r['layout']} | {r['staging']} | " + " | ".join(mr(r["tl"][n]) for n in names) + " |")
    print(f"\n## G3 reading (b), proposed, needs Grant: rounds as the unit, a one-sided {a.t_level:.0%} t bound on per-round mean "
          f"log ratios; pass at 1.03, inconclusive when the null's own bound is above 1.03 ((a), as designed, is in the table above)"
          + ("" if sps is not None else " (no scipy: n/a)") + "\n")
    print("| suite | case | layout | staging MB | rounds | SEAM/S by round [lower, upper t bound] | S_b/S by round [upper t bound] "
          "| null spread by round: sd of log, range | (b) proposed, needs Grant |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        sr, nr = r["seam_r"], r["null_r"]
        print(f"| {r['suite']} | {r['case']} | {r['layout']} | {r['staging']} | {sr['n']} | {sr['mean']:.3f} [{sr['t_lo']:.3f}, {sr['t_hi']:.3f}] | "
              f"{nr['mean']:.3f} [{nr['t_hi']:.3f}] | {nr['sd']:.4f}, {nr['lo']:.3f}-{nr['hi']:.3f} | {r['proposed']} |")
    if skipped:
        print("\nLeft out:\n" + "\n".join(f"- {f}: {why}" for f, why in skipped))
    if a.tsv:
        with open(a.tsv, "w") as fh:
            cols = ["suite", "case", "layout", "staging", "copied", "g"]
            stat = ["n", "median", "bca_lo", "bca_hi", "hl", "hl_lo", "hl_hi", "hl_conf"]
            fh.write("\t".join(cols + [f"seam_{k}" for k in stat] + [f"null_{k}" for k in stat]
                               + ["gate", "check", "rounds", "t_seam", "t_seam_lo", "t_null", "null_sd", "proposed", "fcpu_max", "gpu_busy_max", "seam_out"]
                               + [f"tl_{n.replace(' ', '_')}_med" for n in names]) + "\n")
            for r in rows:
                vals = [str(r[c]) for c in cols] + [f"{r['seam'][k]:.6g}" for k in stat] + [f"{r['null'][k]:.6g}" for k in stat]
                vals += [r["gate"][0], r["check"][0], str(r["seam_r"]["n"]), f"{r['seam_r']['t_hi']:.6g}", f"{r['seam_r']['t_lo']:.6g}", f"{r['null_r']['t_hi']:.6g}",
                         f"{r['null_r']['sd']:.6g}", r["proposed"], f"{r['fcpu']:.6g}", f"{r['gbusy']:.6g}", r["out"]] + [f"{statistics.median(r['tl'][n]):.6g}" if r["tl"][n] else "nan" for n in names]
                fh.write("\t".join(vals) + "\n")


if __name__ == "__main__":
    main()
