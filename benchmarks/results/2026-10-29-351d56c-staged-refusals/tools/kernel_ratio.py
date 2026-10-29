#!/usr/bin/env python3
"""kernel_ratio.py <out-dir> <input> <np> [label ...]: per label, the GPU-routed ZGEMMs' kernel time
(AB_REPORT gpu_stages_ms kernel, summed over ranks) in AB_MODE=gpu runs, against Accelerate's
time for the same calls: the cpu-mode runs' zgemm total minus the gpu-mode run's own cpu_ms (the
calls the routing kept on Accelerate there). The cpu runs are pooled over every label (one QE
build and input). Also wall time, power and the staged outcomes."""
import csv, re, statistics as st, sys, collections
from pathlib import Path
out, inp, np_ = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
want = sys.argv[4:]
rows = [r for r in csv.DictReader(open(out / "results.csv")) if r["input"] == inp and r["np"] == np_ and "AB_TRACE" not in r["args"]]
def report(r):
    t = (out / r["run_dir"] / "err.txt").read_text(errors="replace")
    z = collections.defaultdict(float)
    for m in re.finditer(r"AB_REPORT\[\d+\] zgemm (calls=.*|gpu_stages_ms .*|staged .*)", t):
        rest = m[1]; pre = "st_" if rest.startswith("gpu_stages_ms") else "sp_" if rest.startswith("staged") else ""
        for k, v in re.findall(r"(\w+)=([-0-9.]+)", rest): z[pre + k] += float(v)
    return z
cpu = [report(r)["total_ms"] / 1e3 for r in rows if r["mode"] == "cpu"]
acc_total = st.median(cpu) if cpu else float("nan")
print(f"{inp} np={np_}: Accelerate zgemm total (cpu mode, n={len(cpu)}): {acc_total:.2f} s (runs {', '.join(f'{x:.2f}' for x in cpu)})")
labels = want or sorted({r["label"] for r in rows if r["mode"] == "gpu"})
for lab in labels:
    rs = [r for r in rows if r["label"] == lab and r["mode"] == "gpu"]
    if not rs: continue
    ks, ratios, walls = [], [], []
    for r in rs:
        z = report(r)
        k = z["st_kernel"] / 1e3; acc = acc_total - z["cpu_ms"] / 1e3
        ks.append(k); ratios.append(k / acc); walls.append(float(r["wall_s"]))
    z = report(rs[-1]); sp = {k[3:]: int(v) for k, v in z.items() if k.startswith("sp_")}
    print(f"{lab:<28} n={len(rs)} kernel {', '.join(f'{k:.1f}' for k in ks)} s; kernel/Accelerate "
          f"{', '.join(f'{x:.3f}' for x in ratios)}; wall {', '.join(f'{w:.1f}' for w in walls)} s; "
          f"power {'/'.join(sorted({r['power'] for r in rs}))}; gpu_calls {int(z['gpu_calls'])}"
          + (f"; staged {sp}" if sp else ""))
