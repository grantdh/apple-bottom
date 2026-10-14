#!/usr/bin/env python3
"""Markdown tables for VAL-1 from a run_campaign.sh directory.

Usage: python3 tables.py <out-dir> <label> [baseline-label]

Prints the accuracy, routing, run-time and GPU-stage tables for the runs of
<label>, and the run time of [baseline-label] for comparison. Medians over
repetitions; ranges in brackets.
"""
import csv, re, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import report  # noqa: E402  (per-run AB_REPORT parser)

C, LABEL = Path(sys.argv[1]), sys.argv[2]
BASE = sys.argv[3] if len(sys.argv) > 3 else None
rows = list(csv.DictReader(open(C / "results.csv")))
for r in rows:
    r.setdefault("libtag", r.get("label", ""))
    r["label"] = r["libtag"]

def run_dir(r):
    return C / "runs" / r["label"] / r["input"] / f'{r["mode"]}_r{r["rep"]}'

def med(xs):
    return st.median(xs) if xs else float("nan")

def rng(xs, nd=1):
    return f"{med(xs):.{nd}f} [{min(xs):.{nd}f}–{max(xs):.{nd}f}]" if len(xs) > 1 else f"{xs[0]:.{nd}f}"

inputs = sorted({r["input"] for r in rows if r["label"] == LABEL})

def runs(label, inp, mode):
    return [r for r in rows if r["label"] == label and r["input"] == inp and r["mode"] == mode]

def eig(out):
    t = out.read_text(errors="replace")
    tail = t.split("End of self-consistent calculation")[-1].split("highest occupied")[0]
    vals = []
    for blk in re.split(r"\n\s+k =", tail)[1:]:
        body = blk.split("bands (ev):")[-1].split("occupation numbers")[0]
        vals += [float(x) for x in re.findall(r"-?\d+\.\d+", body)]
    its = [float(x) for x in re.findall(r"^\s+total energy\s+=\s+(-?[0-9.]+) Ry", t, re.M)]
    return vals, its

print("### Accuracy\n")
print("| Input | Total energy, `cpu` (Ry) | Largest \\|ΔE\\|, `gpu` vs `cpu`: final / any iteration (Ry) | SCF iterations, `cpu` / `gpu` | Band energies per run | Largest \\|Δε\\|, any run vs `cpu` (eV) |")
print("|---|---|---|---|---|---|")
for inp in inputs:
    cpu, gpu = runs(LABEL, inp, "cpu"), runs(LABEL, inp, "gpu")
    if not cpu or not gpu: continue
    e_cpu = {r["energy_Ry"] for r in cpu}; e_gpu = {r["energy_Ry"] for r in gpu}
    ref_v, ref_i = eig(run_dir(cpu[0]) / "out.txt")
    dmax, n, dimax = 0.0, len(ref_v), 0.0
    for r in gpu + cpu[1:] + runs(LABEL, inp, "auto"):
        v, i = eig(run_dir(r) / "out.txt")
        if len(v) != n: dmax = float("inf")
        else: dmax = max([dmax] + [abs(a - b) for a, b in zip(v, ref_v)])
        dimax = max([dimax] + [abs(a - b) for a, b in zip(i, ref_i)])
    de = max(abs(float(a) - float(b)) for a in e_gpu for b in e_cpu)
    it = f'{"/".join(sorted({r["scf_iters"] for r in cpu}))} / {"/".join(sorted({r["scf_iters"] for r in gpu}))}'
    z = lambda x, f: "0" if x == 0 else format(x, f)
    print(f"| {inp} | {sorted(e_cpu)[0]} | {z(de, '.1e')} / {z(dimax, '.1e')} | {it} | {n} | {z(dmax, '.4f')} |")

print("\n### Routing (AB_REPORT, first repetition)\n")
print("| Input | Mode | Routine | Calls | On the GPU | GPU share of the routine's time | Fallbacks |")
print("|---|---|---|---:|---:|---:|---:|")
for inp in inputs:
    for mode in ("auto", "gpu"):
        rs = runs(LABEL, inp, mode)
        if not rs: continue
        rep = report(run_dir(rs[0]))
        for name in ("zgemm", "dgemm"):
            z = rep.get(name)
            if not z: continue
            fb = sum(v for k, v in z.items() if k.startswith("fb_")) + z.get("fallbacks", 0)
            print(f'| {inp} | `{mode}` | {name.upper()} | {z["calls"]:.0f} | {z["gpu_calls"]:.0f} | {z["gpu_time%"]:.1f}% | {fb:.0f} |')

print("\n### Run time (s, median [min–max])\n")
print("| Input | `cpu` | `auto` | `gpu` | `gpu` / `cpu` | ZGEMM+DGEMM time, `cpu` | same, `gpu` |" + (f" `auto`, {BASE} |" if BASE else ""))
print("|---|---|---|---|---:|---:|---:|" + ("---|" if BASE else ""))
for inp in inputs:
    w = {m: [float(r["wall_s"]) for r in runs(LABEL, inp, m)] for m in ("cpu", "auto", "gpu")}
    blas = {}
    for m in ("cpu", "gpu"):
        blas[m] = [sum(report(run_dir(r)).get(n, {}).get("total_ms", 0) for n in ("zgemm", "dgemm")) / 1e3
                   for r in runs(LABEL, inp, m)]
    line = f'| {inp} | {rng(w["cpu"])} | {rng(w["auto"])} | {rng(w["gpu"])} | {med(w["gpu"]) / med(w["cpu"]):.3f} | {med(blas["cpu"]):.1f} | {med(blas["gpu"]):.1f} |'
    if BASE:
        b = [float(r["wall_s"]) for r in runs(BASE, inp, "auto")]
        line += f" {rng(b) if b else '–'} |"
    print(line)

print("\n### GPU path stages, `gpu` mode (s, median over repetitions)\n")
stages = ["init", "alloc", "pack", "upload", "kernel", "download", "epilogue", "other"]
print("| Input | Routine | GPU calls | " + " | ".join(stages) + " | total | Accelerate, same routine (`cpu`) | kernel / Accelerate |")
print("|---|---|---:|" + "---:|" * (len(stages) + 3))
for inp in inputs:
    for name in ("zgemm", "dgemm"):
        reps = [report(run_dir(r)).get(name, {}) for r in runs(LABEL, inp, "gpu")]
        reps = [z for z in reps if z.get("gpu_calls")]
        if not reps: continue
        cpu_t = med([report(run_dir(r)).get(name, {}).get("total_ms", 0) / 1e3 for r in runs(LABEL, inp, "cpu")])
        vals = [med([z.get("st_" + s, 0) / 1e3 for z in reps]) for s in stages]
        tot = med([z["gpu_ms"] / 1e3 for z in reps])
        kern = vals[stages.index("kernel")]
        print(f'| {inp} | {name.upper()} | {med([z["gpu_calls"] for z in reps]):.0f} | ' + " | ".join(f"{v:.2f}" for v in vals)
              + f" | {tot:.2f} | {cpu_t:.2f} | {kern / cpu_t:.2f} |")
