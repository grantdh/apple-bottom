#!/usr/bin/env python3
"""summary.py <campaign-dir>: per configuration (input, layout, library, mode) of the timing runs,
the wall times and, summed over MPI ranks and averaged over the repetitions, the BLAS-ABI totals
from AB_REPORT: ZGEMM + DGEMM time, the GPU calls' kernel stage and host stages (pack, upload,
download, epilogue), and the staged path's outcomes."""
import csv, re, sys, statistics as st, collections
from pathlib import Path
out = Path(sys.argv[1])
def report(d):
    z = collections.defaultdict(float)
    for m in re.finditer(r"AB_REPORT\[\d+\] (zgemm|dgemm) (calls=.*|gpu_stages_ms .*|staged .*)", (out / d / "err.txt").read_text(errors="replace")):
        rt, rest = m[1], m[2]
        pre = rt + ("_st_" if rest.startswith("gpu_stages_ms") else "_sp_" if rest.startswith("staged") else "_")
        for k, v in re.findall(r"(\w+)=([-0-9.]+)", rest): z[pre + k] += float(v)
    return z
groups = collections.defaultdict(list)
for r in csv.DictReader(open(out / "results.csv")):
    if r["label"].startswith(("time-", "cause-", "t6-")):
        groups[(r["input"][0], r["np"], r["label"], r["mode"])].append(r)
print("| input | ranks | run | mode | n | wall s | BLAS s | GPU calls | kernel s | host stages s | staged: kept / kept_4m / other | energy Ry, iterations |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for (i, np_, lab, mode), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), kv[0][3], kv[0][2])):
    zs = [report(r["run_dir"]) for r in rs]
    mean = lambda f: st.mean(f(z) for z in zs)
    blas = mean(lambda z: z["zgemm_total_ms"] + z["dgemm_total_ms"]) / 1e3
    gpu = mean(lambda z: z["zgemm_gpu_calls"] + z["dgemm_gpu_calls"])
    kern = mean(lambda z: z["zgemm_st_kernel"] + z["dgemm_st_kernel"]) / 1e3
    host = mean(lambda z: sum(z[f"{rt}_st_{s}"] for rt in ("zgemm", "dgemm") for s in ("pack", "upload", "download", "epilogue"))) / 1e3
    kept = mean(lambda z: z["zgemm_sp_kept"] + z["dgemm_sp_kept"]); k4 = mean(lambda z: z["zgemm_sp_kept_4m"] + z["dgemm_sp_kept_4m"])
    oth = mean(lambda z: sum(v for k, v in z.items() if "_sp_" in k and not k.endswith(("_kept", "_kept_4m"))))
    sp = f"{kept:.0f} / {k4:.0f} / {oth:.0f}" if kept + k4 + oth else "-"
    walls = ", ".join(r["wall_s"] for r in rs); en = sorted({(r["energy_Ry"], r["scf_iters"]) for r in rs})
    print(f"| {i} | {np_} | {lab} | {mode} | {len(rs)} | {walls} | {blas:.1f} | {gpu:.0f} | {kern:.1f} | {host:.1f} | {sp} | {', '.join(f'{e} ({n})' for e, n in en)} |")
