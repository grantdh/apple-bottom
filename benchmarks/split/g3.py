#!/usr/bin/env python3
"""G3, the library's split against the user code's: paired intervals from bench_split's raw.tsv.

usage: g3.py [--level 0.95] [--t-level 0.95] [--resamples 9999] [--seed 1] [--max-fcpu 1.5]
             [--max-gpu-busy 0.05] [--tsv FILE] <dir> ...
       g3.py --simulate [--sims 400] [--rule-sims 20000] [--seed 1]
       g3.py --self-test

Give a G3 run's directory and those of its reruns, in any order (below, G3 by row). Reads every <case>.np<N>.d<data>.raw.tsv below the directories (run.sh's <outdir> or its suites)
whose fixed block has the SEAM arm, with the .out report beside it. Left out, and listed at the end:
cases that run.sh set aside (a path with an incomplete- component), reports without a SUMMARY line,
and reports whose power does not meet run.sh's conditions (AC power from an adapter of 90 W or more on
the start and end lines, and at the start a battery of 95% or more, neither charging nor finishing
its charge, when printed; a G3 row's report left out for power counts as a withheld run of that row).
Only rows under G3's conditions get a verdict: copied operands, uniform
data, 1x8 or 4x2, AB_STAGING_MB 2048 or 8192 (else "not G3"), and G3's protocol, AB_TRACE and
AB_REPORT off, a positive multiple of 6 rounds and the user split's GPU slice on its process's GPU
thread (gpu_fresh=0, not SPLIT_GPU_FRESH=1) (else "not G3 (protocol)"). For each case and layout:

- Pairs by (round, rep) of the fixed block's lockstep times (the slowest process per repeat):
  SEAM with the user split S<g>, and the twin S<g>b with S<g> (the null run: the user split against
  itself).
- For both ratios: the median of the paired ratios; a BCa bootstrap interval of that median
  (scipy.stats.bootstrap, paired); and the Hodges-Lehmann estimate of the ratio (on log ratios) with
  its exact Wilcoxon signed-rank interval. Two-sided intervals at --level.
- Three readings of each row (design section 12's line, "upper bound over max(1.03, null spread):
  fix first", leaves the null spread to be defined). Grant chose (c) for G3 on 2026-11-22; (a) and (b)
  are printed for comparison:
  (a) as designed, not chosen: the upper bound of SEAM / S<g> against max(1.03, the upper bound of
      S<g>b / S<g>), on the pair intervals above; above it: "fix first". BCa bounds, the
      Hodges-Lehmann ones as a check. At n pairs the exact interval cannot reach the level when n < 6
      (it is then the range of the Walsh averages, with its own confidence).
  (b) proposed, not chosen: rounds as the unit, as Grant chose for G5. Per round, the mean over its
      repeats of log(SEAM / S<g>) and of log(S<g>b / S<g>); over the rounds, a one-sided --t-level t
      upper bound on each mean, exponentiated. "pass" if SEAM's bound is 1.03 or less; else
      "inconclusive: add rounds" if the null's own bound is above 1.03 (the user split against itself
      is too noisy to tell); else "fix first". Noise can still give "fix first", when SEAM's bound
      passes 1.03 by chance and the null's does not; less often than (a) (see --simulate).
      (b) also never says "fix first" while the null's bound is above 1.03, however slow the seam.
  (c) G3's reading: the same per-round t bounds, the seam's alone: "pass" if its upper bound is 1.03
      or less; else "fix first" if its one-sided lower bound is above 1.00 (the seam is slower than
      the user split); else "inconclusive: add rounds".
  With each row, SEAM / S<g>'s one-sided lower bound at the same level, and the null spread by round:
  the sd of the null's per-round mean log ratios and the range of the per-round ratios.
- --simulate, for a seam exactly as fast as the user split and for one 5%, 10% and 20% slower, at
  per-call noise of 1-5% and per-round offsets of each arm of 0-2%: per line, the share of rows withheld
  for a failed null and the share of each reading's "pass", "fix first" and "inconclusive" (of all
  rows) at 6 rounds of 2 repeats (--sims rows); and G3 by row (--rule-sims rows), each row run through
  standing() and rerun as it says, up to 8 runs: the share of rows whose "pass" or "fix first" stands
  or that go to Grant, the runs per row, and over 50 independent rows the chance that some row's "fix
  first" stands and the reruns expected.
- Verdicts are withheld, in every reading, from a row flagged for outside load (below): "flagged:
  rerun on a quiet machine"; from a row whose null failed, the two identical user splits differing
  (a two-sided 95% t interval of S<g>b / S<g> over rounds that excludes 1.00): "null failed: rerun";
  and from every row when that check cannot be made: "null check needs scipy: no verdict". So a
  loaded run, or one whose instrument is off, can neither pass nor fail G3. The null check is at 95%
  per row: an instrument that is fine still fails it in 5% of rows (rerun the case; a second failure
  by chance is again 5%), and in at least one of a run's 50 rows with probability 1 - 0.95^50 = 92%.
  Grant kept that level on 2026-11-22.
- G3 by row: each of G3's 50 rows (G3_CASES, read from run.sh's G3 suites) over its runs, in the order
  run.sh finished them (the last "after <case> ... rc=0" line of the power.log beside each report).
  The last run with a verdict decides, except that "fix first" stands only when the run with a verdict
  before it also read "fix first"; a row read "fix first" once is rerun to confirm it, at that run's
  rounds. A withheld row is rerun at its rounds; an inconclusive one at 12 rounds, and one still
  inconclusive at 12 or more is Grant's call (these three rules proposed with Grant's choices of
  2026-11-22, for him to confirm). Each open row gets its run.sh command (SPLIT_CASES, SPLIT_G3_ROUNDS). G3 says "fix first" when
  a row's "fix first" stands, "pass" when all 50 rows pass, else "open", with the build the reports
  name; it gives no verdict from runs of two builds, or of a build named "-dirty" or "unknown", which
  does not tell its source. A row with two runs, one of them without a finish time (not run by run.sh,
  or its power.log missing) or two with the same one, cannot be ordered and gets no verdict; the table
  names the report to leave out.
- Outside load, from the report's SUMMARY (the largest over the rounds, .rounds.tsv has each): other
  processes' CPU in cores over the GPU-idle stretches and the GPU's busy fraction over those
  stretches; a row with either over --max-fcpu or --max-gpu-busy, with a round whose load was not
  measured (load_unmeasured > 0; SPLIT_SAMPLER=0 leaves the GPU's unmeasured), or from a report older
  than that count (unknown, not 0), is flagged. The GPU limit, 0.05, is set on this measure: 0.000-0.005 per round in
  the quiet round-2 smoke runs, 0.12-0.17 with another session's GPU work.
- The seam's timeline per call, summarised over every process's SEAM calls, median [min-max] in ms:
  init (GPU part handed over - cut) and start (GPU part's start - handed over: the split worker's
  wake-up, a thread's start in runs before the worker), beside the user split's
  GPU start (its GPU thread's start after the call's); the CPU part and the GPU part, each beside the
  user split's; the wait for the GPU (join - CPU part's end) beside the user split's; after the join
  (the K axis's sum or a redo); and the GPU part's submit (commit to GPUStartTime), GPU execution
  (GPUStartTime to GPUEndTime) and wake (GPUEndTime to waitUntilCompleted's return).

A verdict, in any reading, needs scipy (on this Mac:
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3): the null check's t quantile, as well
as BCa and the t bounds of (b) and (c). Without it every row's verdict is withheld ("null check needs
scipy: no verdict"), BCa and the t bounds read n/a, and the Hodges-Lehmann intervals are still
printed; --simulate needs it too.
"""
import argparse, functools, glob, math, os, re, statistics, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # rawtsv.py, also under python3 -I
from rawtsv import env, fixed_pairs, gpu_fresh, parse, power_problem, read  # noqa: E402

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
            half = t_ppf(q, len(m) - 1) * out["sd"] / math.sqrt(len(m))
            out["t_hi"], out["t_lo"] = math.exp(mu + half), math.exp(mu - half)
    return out


@functools.lru_cache(maxsize=None)
def t_ppf(q, df):
    return float(sps.t.ppf(q, df))


NULL_Q = 0.975   # the null check's t interval: two-sided 95%


def null_check(null95):
    """"" if the two identical user splits agree (null95: by_round(L, "SPLIT2", "SPLIT", NULL_Q), whose t_lo
    and t_hi are a two-sided 95% t interval of S<g>b / S<g> over rounds; it holds 1.00), else why no
    verdict is given"""
    if sps is None:
        return "null check needs scipy: no verdict"
    if math.isnan(null95["t_lo"]):
        return "null check needs 2 rounds: no verdict"
    return NULL_FAILED if null95["t_lo"] > 1.0 or null95["t_hi"] < 1.0 else ""


def fx(x, p=3):
    return "n/a" if math.isnan(x) else f"{x:.{p}f}"


def iv(*xs):
    """bounds in brackets, or n/a when one is missing"""
    return "n/a" if any(math.isnan(x) for x in xs) else "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"


PASS, FIX, INCONCLUSIVE = "pass", "fix first", "inconclusive: add rounds"


def reading_c(seam):
    """reading (c), G3's: pass at 1.03, fix first when the seam's lower bound is above 1.00, else inconclusive"""
    if math.isnan(seam["t_hi"]):
        return "n/a"
    return PASS if seam["t_hi"] <= 1.03 else FIX if seam["t_lo"] > 1.0 else INCONCLUSIVE


def reading_a(seam_hi, null_hi):
    """reading (a), as designed: fix first above max(1.03, the null's own upper bound); (category, bound)"""
    bound = max(1.03, null_hi) if not math.isnan(null_hi) else 1.03
    return ("n/a" if math.isnan(seam_hi) else PASS if seam_hi <= bound else FIX), bound


def reading_b(seam_hi, null_hi):
    """reading (b), proposed: pass at 1.03, inconclusive when the null's own bound is above 1.03, else fix first"""
    if math.isnan(seam_hi):
        return "n/a"
    return PASS if seam_hi <= 1.03 else INCONCLUSIVE if math.isnan(null_hi) or null_hi > 1.03 else FIX


def readings(seam, null, seam_r, null_r):
    """the verdicts on a row that passed the null check, from paired() and by_round() of SEAM / S<g> and
    S<g>b / S<g>: (a) on BCa, (a)'s check on Hodges-Lehmann, (b) and (c)"""
    return (reading_a(seam["bca_hi"], null["bca_hi"]), reading_a(seam["hl_hi"], null["hl_hi"]),
            reading_b(seam_r["t_hi"], null_r["t_hi"]), reading_c(seam_r))


G3_ROUNDS, G3_MORE_ROUNDS = 6, 12   # a G3 run's rounds; an inconclusive row's rerun
G3_STAGING = ("2048", "8192")
FLAGGED, NULL_FAILED = "flagged: rerun on a quiet machine", "null failed: rerun"
RUN_SH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.sh")


def sh_words(text, name):
    """the words of a shell script's NAME="..." line, or [] without one"""
    m = re.search(rf'^{name}="([^"]*)"', text, re.M)
    return m[1].split() if m else []


def g3_cases():
    """{layout: cases} of run.sh's G3 suites: its SQ and QE cases at 1x8, its SQ4X2 and QE cases at 4x2
    (--self-test checks that the suites still take these lists)"""
    sh = open(RUN_SH).read()
    return {"1x8": sh_words(sh, "SQ") + sh_words(sh, "QE"), "4x2": sh_words(sh, "SQ4X2") + sh_words(sh, "QE")}


G3_CASES = g3_cases()
G3_ROWS = [(c, lay, st) for lay, cs in G3_CASES.items() for c in cs for st in G3_STAGING]   # G3's 50 rows


def standing(runs):
    """G3's verdict on one row over its runs in run order, each (reading (c) or why it was withheld,
    rounds): (verdict, next step); the step is (what, rounds) for run.sh, (what, None) for anything else,
    or None once the verdict stands. The last run with a verdict decides, but "fix first" stands only
    when the run with a verdict before it read "fix first" too"""
    if not runs:
        return "not run", ("run", G3_ROUNDS)
    use = [(v, n) for v, n in runs if v in (PASS, FIX, INCONCLUSIVE)]
    if not use:
        return "no verdict: " + runs[-1][0], ("rerun", runs[-1][1])
    v, n = use[-1]
    if v == PASS:
        return PASS, None
    if v == FIX:
        return (FIX, None) if len(use) > 1 and use[-2][0] == FIX else ("fix first once", ("confirm", n))
    if n >= G3_MORE_ROUNDS:
        return f"inconclusive at {n} rounds", ("Grant's call", None)
    return "inconclusive", ("rerun", G3_MORE_ROUNDS)


def by_row(runs):
    """G3 over its 50 rows, from their runs: dicts with key (case, layout, staging), c (reading (c) or why
    the run was withheld), rounds, label, time (when run.sh logged the case finished, or None) and build.
    Returns ([(key, its runs in run order, verdict, next step)], G3's line). A row whose runs cannot be
    put in order (two or more, one without a time or two with one time) gets no verdict, and G3 none from
    runs of two builds, or of a build whose name does not tell its source ("-dirty", "unknown")"""
    of = defaultdict(list)
    for r in runs:
        of[r["key"]].append(r)
    table = []
    for k in G3_ROWS:
        rs = of[k]
        times = [r["time"] for r in rs]
        if len(rs) > 1 and (None in times or len(set(times)) < len(rs)):
            untimed = [r["path"] for r in rs if r["time"] is None]
            what = (f"leave out {', '.join(untimed)}: no run.sh finish time (power.log) beside it" if untimed else
                    f"leave out one of {', '.join(r['path'] for r in rs if times.count(r['time']) > 1)}: one finish time")
            verdict, step = "runs not ordered", (what, None)
        else:
            rs = sorted(rs, key=lambda r: r["time"] or "")
            verdict, step = standing([(r["c"], r["rounds"]) for r in rs])
        table.append((k, rs, verdict, step))
    counts = Counter(v for _, _, v, _ in table)
    builds = sorted({r["build"] for _, rs, _, _ in table for r in rs})
    fix = [" ".join(k) for k, _, v, _ in table if v == FIX]
    if len(builds) > 1:
        return table, f"G3: no verdict: runs of {len(builds)} builds ({', '.join(builds)})"
    if builds and (builds[0].endswith("-dirty") or builds[0].endswith(" unknown")):
        return table, f"G3: no verdict: build {builds[0]} does not name its source (build a commit, as with AB_BUILD_SHA)"
    if fix:
        line = f"G3: fix first ({len(fix)} of {len(table)} rows: {', '.join(fix)})"
    elif counts[PASS] == len(table):
        line = f"G3: pass (all {len(table)} rows)"
    else:
        line = "G3: open (" + ", ".join(f"{v}: {n}" for v, n in counts.most_common()) + ")"
    return table, line + (f", build {builds[0]}" if builds else "")


def next_step(key, step):
    """the next step for a row: run.sh's command for a run or rerun, or what else to do"""
    if step is None:
        return "-"
    what, rounds = step
    if rounds is None:
        return what
    case, layout, staging = key
    if what == "run":   # run.sh skips the cases already complete in <outdir>
        return f"run: `run.sh <outdir> g3-{layout}-s{staging}`"
    return f"{what}: `SPLIT_CASES={case} SPLIT_G3_ROUNDS={rounds} run.sh <new outdir> g3-{layout}-s{staging}`"


def self_test():
    """run.sh's G3 suites against G3_CASES, reading_c(), standing() and by_row() on fixed inputs: the failures"""
    fails = []
    sh = open(RUN_SH).read()
    for pat in (r'g3-1x8-s8192\|g3-1x8-s2048\|g3-4x2-s8192\|g3-4x2-s2048\)',
                r'g3-1x8-\*\) np=1 vl=8 cs="\$SQ \$QE" ;; \*\) np=4 vl=2 cs="\$SQ4X2 \$QE"'):
        if not re.search(pat, sh):
            fails.append(f"run.sh's G3 suites have changed (no match for {pat})")
    if len(G3_ROWS) != 50 or len(set(G3_ROWS)) != 50:   # the README's count
        fails.append(f"G3 has {len(set(G3_ROWS))} distinct rows of {len(G3_ROWS)}, not 50: update the README")
    t = lambda lo, hi: {"t_lo": lo, "t_hi": hi}
    for seam, want in ((t(0.99, 1.03), PASS), (t(1.001, 1.031), FIX), (t(1.0, 1.05), INCONCLUSIVE), (t(NAN, NAN), "n/a")):
        if (got := reading_c(seam)) != want:
            fails.append(f"reading_c({seam}) = {got!r}, not {want!r}")
    for runs, want in (
        ([], ("not run", ("run", 6))),
        ([(PASS, 6)], (PASS, None)),
        ([(FIX, 6)], ("fix first once", ("confirm", 6))),
        ([(FIX, 6), (FIX, 6)], (FIX, None)),
        ([(FIX, 6), (PASS, 6)], (PASS, None)),
        ([(PASS, 6), (FIX, 6)], ("fix first once", ("confirm", 6))),
        ([(FIX, 6), (NULL_FAILED, 6)], ("fix first once", ("confirm", 6))),
        ([(FIX, 6), (NULL_FAILED, 6), (FIX, 6)], (FIX, None)),
        ([(FIX, 6), (INCONCLUSIVE, 6), (FIX, 12)], ("fix first once", ("confirm", 12))),
        ([(FLAGGED, 6)], ("no verdict: " + FLAGGED, ("rerun", 6))),
        ([(FLAGGED, 12)], ("no verdict: " + FLAGGED, ("rerun", 12))),
        ([(INCONCLUSIVE, 6)], ("inconclusive", ("rerun", 12))),
        ([(INCONCLUSIVE, 6), (FIX, 12)], ("fix first once", ("confirm", 12))),
        ([(INCONCLUSIVE, 6), (INCONCLUSIVE, 12)], ("inconclusive at 12 rounds", ("Grant's call", None))),
        ([(INCONCLUSIVE, 6), (NULL_FAILED, 12)], ("inconclusive", ("rerun", 12))),
    ):
        if (got := standing(runs)) != want:
            fails.append(f"standing({runs}) = {got}, not {want}")
    run = lambda k, v, t, b="1.6.0 x": {"key": k, "c": v, "rounds": 6, "label": "r", "path": "p", "time": t, "build": b}
    every = [run(k, PASS, "2026-11-22 10:00:00") for k in G3_ROWS]
    k0 = G3_ROWS[0]
    for runs, want in (
        (every, "G3: pass (all 50 rows), build 1.6.0 x"),
        (every[1:], "G3: open (pass: 49, not run: 1), build 1.6.0 x"),
        (every + [run(k0, FIX, "2026-11-22 11:00:00")], "G3: open (pass: 49, fix first once: 1), build 1.6.0 x"),
        (every + [run(k0, FIX, "2026-11-22 11:00:00"), run(k0, FIX, "2026-11-22 12:00:00")],
         f"G3: fix first (1 of 50 rows: {' '.join(k0)}), build 1.6.0 x"),
        # the order is the runs' times, not the order given
        ([run(k0, FIX, "2026-11-22 09:00:00")] + every, "G3: pass (all 50 rows), build 1.6.0 x"),
        (every + [run(k0, FIX, "2026-11-22 09:00:00")], "G3: pass (all 50 rows), build 1.6.0 x"),
        (every + [run(k0, PASS, None)], "G3: open (pass: 49, runs not ordered: 1), build 1.6.0 x"),
        (every + [run(k0, PASS, "2026-11-22 10:00:00")], "G3: open (pass: 49, runs not ordered: 1), build 1.6.0 x"),
        (every + [run(k0, PASS, "2026-11-22 11:00:00", "1.6.0 y")], "G3: no verdict: runs of 2 builds (1.6.0 x, 1.6.0 y)"),
        ([{**r, "build": "1.6.0 x-dirty"} for r in every], "G3: no verdict: build 1.6.0 x-dirty does not name its source "
         "(build a commit, as with AB_BUILD_SHA)"),
        ([{**r, "build": "1.6.0 unknown"} for r in every], "G3: no verdict: build 1.6.0 unknown does not name its source "
         "(build a commit, as with AB_BUILD_SHA)"),
        (every + [run(("X1", "1x8", "2048"), INCONCLUSIVE, "2026-11-22 11:00:00")], "G3: pass (all 50 rows), build 1.6.0 x"),
    ):
        if (got := by_row(runs)[1]) != want:
            fails.append(f"by_row: {got!r}, not {want!r}")
    return fails


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


def not_g3(v, txt, layout, staging, rounds):
    """why the row is not a G3 row, or "" if it is: G3's conditions (design section 12: copied operands,
    uniform data, 1x8 or 4x2, staging 2048 or 8192 MB), then the G3 suites' protocol (AB_TRACE and
    AB_REPORT off, which would profile only the user split; a positive multiple of 6 rounds, over which
    the trio's order is balanced; the user split's own GPU thread, not SPLIT_GPU_FRESH's thread per call)"""
    if not (v.get("reg") == "0" and v.get("data") == "0" and layout in G3_CASES and staging in G3_STAGING):
        return "not G3"
    if env(txt, "AB_TRACE") != "off" or env(txt, "AB_REPORT") != "off" or rounds <= 0 or rounds % 6 or gpu_fresh(v):
        return "not G3 (protocol)"
    return ""


def mr(v):
    return f"{statistics.median(v):.3f} [{min(v):.3f}-{max(v):.3f}]" if v else "-"


def finished(f, v):
    """when run.sh logged the case of report f (SUMMARY v) as finished: the last "after <case> np=..
    data=.. rc=0" line of the power.log beside it (run.sh keeps only that run's report), or None"""
    log = os.path.join(os.path.dirname(f), "power.log")
    if not os.path.exists(log):
        return None
    pat = rf"^(\d{{4}}-\d\d-\d\d \d\d:\d\d:\d\d) after {re.escape(v.get('case', '?'))} np={v.get('np')} veclib=\S+ data={v.get('data')} rc=0:"
    times = re.findall(pat, open(log).read(), re.M)
    return times[-1] if times else None


def build(txt):
    """the report's library version and build, from its first line"""
    m = re.match(r"apple-bottom (\S+) \(([^)]*)\)", txt)
    return f"{m[1]} {m[2]}" if m else "unknown"


SIM_LINES = [(ratio, sd, o) for ratio in (1.00, 1.05, 1.10, 1.20) for sd in (0.01, 0.02, 0.03, 0.05) for o in (0.0, 0.01, 0.02)]
SIM_CAP = 8   # runs per simulated row


def sim_block(rnd, ratio, sd, o, rounds):
    """one simulated fixed block, {arm: {(round, repeat): time}}: rounds x 2 repeats; each call's log time has
    noise of sd sd, each arm in each round an offset of sd o (a round x arm effect, as position and drift
    give), and SEAM is ratio times the user split"""
    L = {"SPLIT": {}, "SEAM": {}, "SPLIT2": {}}
    for r in range(rounds):
        off = {c: rnd.gauss(0, o) for c in L}
        for rep in range(2):
            for c in L:
                L[c][r, rep] = math.exp(off[c] + rnd.gauss(0, sd) + (math.log(ratio) if c == "SEAM" else 0))
    return L


def simulate(sims, rule_sims, seed, level, tq, resamples):
    """two tables over SIM_LINES, for a seam as fast as the user split (ratio 1.00: "fix first" is a false
    alarm) and 5, 10 and 20% slower. Readings (sims rows per line, at G3_ROUNDS): the share of rows withheld
    for a failed null and of each reading's verdicts, of all rows. G3 by row (rule_sims rows per line):
    each row run through standing() from its first run, rerun as it says, up to SIM_CAP runs; the share of
    rows ending at each verdict, the runs per row, and over 50 independent rows the chance that some row's
    "fix first" stands and the reruns expected"""
    if sps is None:
        sys.exit("--simulate needs scipy: without it no row gets a verdict")
    import random
    rnd = random.Random(seed)
    if sims:
        print(f"## Simulated G3 readings: {sims} rows per line, {G3_ROUNDS} rounds of 2 repeats; (a) at two-sided {level:.0%} "
              f"pair intervals (BCa with {resamples} resamples), (b) and (c) at one-sided {tq:.0%} t bounds over rounds; "
              f"rows whose null fails (two-sided 95% t over rounds) withheld; shares of all rows\n")
        print("| true SEAM/S | call noise sd | round x arm sd | withheld: null failed | (a) BCa: pass / fix first "
              "| (a) HL: pass / fix first | (b): pass / fix first / inconclusive | (c): pass / fix first / inconclusive |")
        print("|---|---|---|---|---|---|---|---|")
        pc = lambda k, *vs: " / ".join(f"{n[k, v] / sims:.1%}" for v in vs)   # n: the current line's counts
        three = (PASS, FIX, INCONCLUSIVE)
        for ratio, sd, o in SIM_LINES:
            n = defaultdict(int)
            for _ in range(sims):
                L = sim_block(rnd, ratio, sd, o, G3_ROUNDS)
                if null_check(by_round(L, "SPLIT2", "SPLIT", NULL_Q)):
                    n["withheld"] += 1
                    continue
                ps, pn = paired(L, "SEAM", "SPLIT", level, resamples, rnd.randrange(1 << 30)), paired(L, "SPLIT2", "SPLIT", level, resamples, rnd.randrange(1 << 30))
                va, vh, vb, vc = readings(ps, pn, by_round(L, "SEAM", "SPLIT", tq), by_round(L, "SPLIT2", "SPLIT", tq))
                for k, verdict in (("a", va[0]), ("h", vh[0]), ("b", vb), ("c", vc)):
                    n[k, verdict] += 1
            print(f"| {ratio:.2f} | {sd:.2f} | {o:.2f} | {n['withheld'] / sims:.1%} | {pc('a', PASS, FIX)} | "
                  f"{pc('h', PASS, FIX)} | {pc('b', *three)} | {pc('c', *three)} |")
    if not rule_sims:
        return
    print(f"\n## Simulated G3 by row: {rule_sims} rows per line, each from a first run at {G3_ROUNDS} rounds, rerun as "
          f"standing() says (a withheld run or a first \"fix first\" at its rounds, an inconclusive one at "
          f"{G3_MORE_ROUNDS}), up to {SIM_CAP} runs; reading (c) at one-sided {tq:.0%}; shares of rows\n")
    print("| true SEAM/S | call noise sd | round x arm sd | pass | fix first | inconclusive at 12: Grant's call "
          "| runs per row | 50 rows: some row fix first | 50 rows: reruns |")
    print("|---|---|---|---|---|---|---|---|---|")
    for ratio, sd, o in SIM_LINES:
        n, runs_n = Counter(), 0
        for _ in range(rule_sims):
            runs = []
            verdict, step = standing(runs)
            while step is not None and step[1] is not None and len(runs) < SIM_CAP:
                L = sim_block(rnd, ratio, sd, o, step[1])
                runs.append((null_check(by_round(L, "SPLIT2", "SPLIT", NULL_Q)) or reading_c(by_round(L, "SEAM", "SPLIT", tq)), step[1]))
                verdict, step = standing(runs)
            n[verdict if step is None else "grant" if step[1] is None else "cap"] += 1
            runs_n += len(runs)
        f = n[FIX] / rule_sims
        cap = f" (at the cap: {n['cap'] / rule_sims:.2%})" if n["cap"] else ""
        print(f"| {ratio:.2f} | {sd:.2f} | {o:.2f} | {n[PASS] / rule_sims:.2%} | {f:.2%} | {n['grant'] / rule_sims:.2%}{cap} | "
              f"{runs_n / rule_sims:.2f} | {1 - (1 - f) ** 50:.1%} | {50 * (runs_n / rule_sims - 1):.1f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--level", type=float, default=0.95, help="two-sided level of the pair intervals")
    ap.add_argument("--t-level", type=float, default=0.95, help="one-sided level of readings (b) and (c)'s t bounds")
    ap.add_argument("--simulate", action="store_true", help="false-alarm and detection rates of the readings")
    ap.add_argument("--sims", type=int, default=400, help="rows per line of the readings table (0: none)")
    ap.add_argument("--rule-sims", type=int, default=20000, help="rows per line of the G3-by-row table (0: none)")
    ap.add_argument("--resamples", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-fcpu", type=float, default=1.5, help="flag a row whose other processes used more cores (as the AB_SPLIT pilot)")
    ap.add_argument("--max-gpu-busy", type=float, default=0.05,
                    help="flag a row whose GPU was busier while this run left it idle (quiet smoke runs: up to 0.005)")
    ap.add_argument("--tsv", help="also write one row per case here")
    ap.add_argument("--self-test", action="store_true", help="check G3_CASES against run.sh, and standing() and by_row()")
    a = ap.parse_args()
    if a.self_test:
        fails = self_test()
        print("\n".join(fails) if fails else "g3: run.sh's G3 suites, reading_c(), standing() and by_row() on fixed inputs: all agree")
        return 1 if fails else 0
    if a.simulate:
        return simulate(a.sims, a.rule_sims, a.seed, a.level, a.t_level, a.resamples)
    if not a.dirs:
        ap.error("give run.sh's <outdir> or suite directories")
    # each file with its label: the given directory's name and the path below it, so that rows of two
    # outdirs (two days, say) stay apart
    files = {}
    for d in a.dirs:
        base = os.path.basename(os.path.normpath(d))
        for f in glob.glob(os.path.join(d, "**", "*.raw.tsv"), recursive=True):
            files.setdefault(os.path.realpath(f), os.path.normpath(os.path.join(base, os.path.relpath(os.path.dirname(f), d))))
    rows, skipped, g3_runs = [], [], []   # g3_runs: every run of a G3 row, reports left out for power included
    for f, label in sorted(files.items()):
        if any(part.startswith("incomplete-") for part in f.split(os.sep)):
            skipped.append((f, "set aside by run.sh"))
            continue
        recs = read(f)
        if not any(d["role"] == "SEAM" for d in recs):
            continue
        txt = open(f[:-8] + ".out").read() if os.path.exists(f[:-8] + ".out") else ""
        v = parse(txt, "SUMMARY")
        if not v:
            skipped.append((f, "no SUMMARY line"))
            continue
        vl, st = env(txt, "VECLIB_MAXIMUM_THREADS"), env(txt, "AB_STAGING_MB")
        layout, staging, rounds = f"{v.get('np', '?')}x{vl or '?'}", st or "?", int(v.get("rounds", "0"))
        case = v.get("case", os.path.basename(f).split(".")[0])
        g3 = not not_g3(v, txt, layout, staging, rounds)
        run = {"key": (case, layout, staging), "rounds": rounds, "label": label, "path": f[:-8] + ".out", "time": finished(f, v), "build": build(txt)}
        why = power_problem(txt)
        if why:   # a G3 row's report left out for power is a withheld run of that row: rerun it
            skipped.append((f, why))
            if g3:
                g3_runs.append({**run, "c": "left out: " + why})
            continue
        L = fixed_pairs(recs)
        seam = paired(L, "SEAM", "SPLIT", a.level, a.resamples, a.seed)
        null = paired(L, "SPLIT2", "SPLIT", a.level, a.resamples, a.seed)
        seam_r, null_r = by_round(L, "SEAM", "SPLIT", a.t_level), by_round(L, "SPLIT2", "SPLIT", a.t_level)
        fcpu, gbusy = float(v.get("fcpu_max", "nan")), float(v.get("gpu_idle_busy_max", "nan"))
        # rounds whose outside load was not measured; nan for a report older than the count: unknown
        unmeasured = float(v.get("load_unmeasured", "nan"))
        null95 = by_round(L, "SPLIT2", "SPLIT", NULL_Q)
        # no verdict, in any reading, for a row that is not G3's, a loaded run (a value or a round not
        # measured counts as loaded), or a failed or impossible null check
        quiet = fcpu <= a.max_fcpu and gbusy <= a.max_gpu_busy and unmeasured == 0
        withhold = not_g3(v, txt, layout, staging, rounds) or ("" if quiet else FLAGGED) or null_check(null95)
        if not withhold:
            ra, ra_hl, rb, rc = readings(seam, null, seam_r, null_r)
        else:
            ra = ra_hl = (withhold, NAN); rb = rc = withhold
        if g3:
            g3_runs.append({**run, "c": rc})
        rows.append({
            "suite": label, "case": case,
            "layout": layout, "staging": staging, "copied": v.get("reg") == "0", "g": v.get("g", "?"), "seam": seam, "null": null,
            "a": ra, "a_hl": ra_hl, "b": rb, "c": rc, "seam_r": seam_r, "null_r": null_r,
            "fcpu": fcpu, "gbusy": gbusy, "unmeasured": unmeasured, "quiet": quiet, "null95": null95, "out": v.get("seam_out", "?"),
            "bits": f"{v.get('seam_rep_diff', '?')}, {v.get('seam_vs_split', '?')}", "tl": timeline(recs),
        })
    if not rows and not skipped:
        sys.exit("no raw.tsv with SEAM calls under " + " ".join(a.dirs))
    pct = f"{a.level:.0%}"
    vtext = lambda r, k, hi: f"{r[k][0]} ({r['seam'][hi]:.3f} vs {r[k][1]:.3f})" if r[k][0] in (PASS, FIX) else r[k][0]
    def load(r):   # a value or a round not measured is unknown, not quiet: it flags too
        cpu = "CPU not measured" if math.isnan(r["fcpu"]) else f"{r['fcpu']:.2f} cores"
        gpu = "GPU not measured" if math.isnan(r["gbusy"]) else f"GPU {r['gbusy']:.3f}"
        u = r["unmeasured"]
        lu = "" if u == 0 else ", rounds not measured: " + ("unknown" if math.isnan(u) else f"{u:g}")
        return f"{'' if r['quiet'] else 'FLAG '}{cpu}, {gpu}{lu}"
    print(f"## Pairs: SEAM / S<g> and S<g>b / S<g>, {pct} intervals; reading (a), for comparison: SEAM / S<g>'s upper "
          f"bound against max(1.03, S<g>b / S<g>'s)"
          + ("" if sps is not None else " (no scipy: BCa, the t bounds and the null check n/a, so no verdict)") + "\n")
    print("| suite | case | layout | staging MB | g | pairs | SEAM/S median | BCa | HL [exact] | S_b/S median | BCa | HL [exact] "
          "| (a) BCa | (a) HL | outside load (max over rounds) | seam outcomes | seam bits: repeats, vs S |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s, n = r["seam"], r["null"]
        hl = lambda p: f"{fx(p['hl'])} {iv(p['hl_lo'], p['hl_hi'])}" + ("" if p["hl_conf"] >= a.level - 1e-9 else f" ({p['hl_conf']:.0%})")
        print(f"| {r['suite']} | {r['case']} | {r['layout']} | {r['staging']}{' copied' if r['copied'] else ''} | {r['g']} | {s['n']} | "
              f"{fx(s['median'])} | {iv(s['bca_lo'], s['bca_hi'])} | {hl(s)} | {fx(n['median'])} | {iv(n['bca_lo'], n['bca_hi'])} | "
              f"{hl(n)} | {vtext(r, 'a', 'bca_hi')} | {vtext(r, 'a_hl', 'hl_hi')} | {load(r)} | {r['out']} | {r['bits']} |")
    print("\n## The seam's timeline per call, ms, median [min-max] over every process's SEAM calls; the user split's parts beside\n")
    names = [n for n, _, _ in TIMELINE]
    print("| suite | case | layout | staging MB | " + " | ".join(names) + " |")
    print("|" + "---|" * (4 + len(names)))
    for r in rows:
        print(f"| {r['suite']} | {r['case']} | {r['layout']} | {r['staging']} | " + " | ".join(mr(r["tl"][n]) for n in names) + " |")
    print(f"\n## Readings (c), G3's, and (b), for comparison: rounds as the unit, one-sided {a.t_level:.0%} t bounds on "
          f"per-round mean log ratios; (c) passes at 1.03, fix first when the seam's lower bound is above 1.00, else "
          f"inconclusive; (b) passes at 1.03, inconclusive when the null's own bound is above 1.03"
          + ("" if sps is not None else " (no scipy: bounds n/a, no verdict)") + "\n")
    print("| suite | case | layout | staging MB | rounds | SEAM/S by round [lower, upper t bound] | S_b/S by round [upper t bound] "
          "| S_b/S two-sided 95% by round | null spread by round: sd of log, range | (b) | (c), G3's |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        sr, nr = r["seam_r"], r["null_r"]
        print(f"| {r['suite']} | {r['case']} | {r['layout']} | {r['staging']} | {sr['n']} | {fx(sr['mean'])} {iv(sr['t_lo'], sr['t_hi'])} | "
              f"{fx(nr['mean'])} {iv(nr['t_hi'])} | {iv(r['null95']['t_lo'], r['null95']['t_hi'])} | "
              f"{fx(nr['sd'], 4)}, {fx(nr['lo'])}-{fx(nr['hi'])} | {r['b']} | {r['c']} |")
    if sps is None:
        print("\n## G3 by row: no verdict without scipy")
    else:
        table, line = by_row(g3_runs)
        print("\n## G3 by row: reading (c) by run, in the order run.sh finished them (power.log); \"fix first\" stands "
              "when two runs with a verdict in a row read it, an inconclusive row is rerun at 12 rounds and then is Grant's "
              "call (these three rules proposed 2026-11-22, for Grant to confirm)\n")
        print("| case | layout | staging MB | (c) by run | G3 | next |")
        print("|---|---|---|---|---|---|")
        for k, rs, verdict, step in table:
            runs = "; ".join(f"{r['label']} ({r['time'] or 'no time'}): {r['c']}" + ("" if r["rounds"] == G3_ROUNDS else f" ({r['rounds']} rounds)")
                             for r in rs) or "-"
            print(f"| {' | '.join(k)} | {runs} | {verdict} | {next_step(k, step)} |")
        print(f"\n{line}")
    if skipped:
        print("\nLeft out:\n" + "\n".join(f"- {f}: {why}" for f, why in skipped))
    if a.tsv:
        with open(a.tsv, "w") as fh:
            cols = ["suite", "case", "layout", "staging", "copied", "g"]
            stat = ["n", "median", "bca_lo", "bca_hi", "hl", "hl_lo", "hl_hi", "hl_conf"]
            fh.write("\t".join(cols + [f"seam_{k}" for k in stat] + [f"null_{k}" for k in stat]
                               + ["reading_a", "reading_a_hl", "rounds", "t_seam", "t_seam_lo", "t_null", "null95_lo", "null95_hi", "null_sd", "reading_b", "reading_c", "fcpu_max", "gpu_busy_max", "load_unmeasured", "seam_out"]
                               + [f"tl_{n.replace(' ', '_')}_med" for n in names]) + "\n")
            for r in rows:
                vals = [str(r[c]) for c in cols] + [f"{r['seam'][k]:.6g}" for k in stat] + [f"{r['null'][k]:.6g}" for k in stat]
                vals += [r["a"][0], r["a_hl"][0], str(r["seam_r"]["n"]), f"{r['seam_r']['t_hi']:.6g}", f"{r['seam_r']['t_lo']:.6g}", f"{r['null_r']['t_hi']:.6g}", f"{r['null95']['t_lo']:.6g}", f"{r['null95']['t_hi']:.6g}",
                         f"{r['null_r']['sd']:.6g}", r["b"], r["c"], f"{r['fcpu']:.6g}", f"{r['gbusy']:.6g}", f"{r['unmeasured']:.6g}", r["out"]] + [f"{statistics.median(r['tl'][n]):.6g}" if r["tl"][n] else "nan" for n in names]
                fh.write("\t".join(vals) + "\n")


if __name__ == "__main__":
    sys.exit(main())
