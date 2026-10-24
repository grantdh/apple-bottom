"""Summarize e2e/: <stage>.r<round>.rect{d,z}.csv from bench_rect_dgemm / bench_rect_zgemm.

usage: python3 e2e_summary.py e2e > e2e_summary.txt
Per shape: the median over rounds of each stage's median wall time per call, and the per-round paired
ratios of two stages (median [min-max]). Stages: base (release/v1.5.0 e3f09ca), off (this branch with
AB_PREFETCH=0), pf (this branch, default).
"""
import csv, glob, statistics as st, sys, collections, os
B = sys.argv[1]
# stage -> routine -> tag -> round -> median wall_s over the 5 runs
data = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
shape = {}
for f in glob.glob(os.path.join(B, "*.r*.rect?.csv")):
    base = os.path.basename(f)
    stage, rnd, kind = base.split(".")[0], int(base.split(".")[1][1:]), base.split(".")[2]
    rows = list(csv.DictReader(open(f)))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["tag"]].append(float(r["wall_s"]))
        shape[r["tag"]] = (int(r["M"]), int(r["N"]), int(r["K"]))
    for tag, w in by.items():
        data[stage][kind][tag][rnd] = st.median(w)
order = ["ovlp_160_18432", "qe_si64_ovlp", "ovlp_256_18432", "ovlp_384_18432", "qe_si64_500b_ovlp", "ovlp_512_18432",
         "ovlp_512_4608", "ovlp_640_9216", "square_2048", "qe_si64_rot"]
for kind, name in (("rectd", "dgemm_ (BLAS ABI, end to end)"), ("rectz", "zgemm_ (BLAS ABI, end to end)")):
    print(f"\n{name}: median wall ms over rounds; paired per-round ratios median [min-max]")
    print(f"{'shape':>18} {'base':>9} {'off':>9} {'pf':>9}  {'base/pf':>22} {'off/pf':>22} {'base/off':>22}")
    for tag in order:
        M, N, K = shape[tag]
        rounds = sorted(set(data["pf"][kind][tag]) & set(data["base"][kind][tag]) & set(data["off"][kind][tag]))
        def med(s): return st.median(data[s][kind][tag][r] for r in rounds) * 1e3
        def ratio(a, b):
            x = [data[a][kind][tag][r] / data[b][kind][tag][r] for r in rounds]
            return f"{st.median(x):.3f} [{min(x):.3f}-{max(x):.3f}]"
        print(f"{M:>5}x{N:>5}x{K:>6} {med('base'):9.2f} {med('off'):9.2f} {med('pf'):9.2f}  {ratio('base','pf'):>22} {ratio('off','pf'):>22} {ratio('base','off'):>22}")
