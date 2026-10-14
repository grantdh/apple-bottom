#!/usr/bin/env python3
"""Markdown tables for VAL-1 from a run_campaign.sh directory.

Usage: python3 tables.py <out-dir> <label> [baseline-label]

Prints the accuracy, routing, run-time and GPU-stage tables for the runs of
<label>, and the run time of [baseline-label] for comparison. Medians over
repetitions; ranges in brackets.
"""
import re, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load, report, run_path, same_calls, wall  # noqa: E402

C, LABEL = Path(sys.argv[1]), sys.argv[2]
BASE = sys.argv[3] if len(sys.argv) > 3 else None
rows = load(C)
inputs = sorted({r["input"] for r in rows if r["label"] == LABEL})


def runs(label, inp, mode):
    return [r for r in rows if r["label"] == label and r["input"] == inp and r["mode"] == mode]


def walls(rs):
    return [x for x in map(wall, rs) if x is not None]


def rng(xs, nd=1):
    if not xs:
        return "–"
    return f"{st.median(xs):.{nd}f} [{min(xs):.{nd}f}–{max(xs):.{nd}f}]" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


def scf(out):
    """(band energies of the final iteration, total energy of each iteration)."""
    t = out.read_text(errors="replace")
    tail = t.split("End of self-consistent calculation")[-1].split("highest occupied")[0]
    vals = []
    for blk in re.split(r"\n\s+k =", tail)[1:]:
        body = blk.split("bands (ev):")[-1].split("occupation numbers")[0]
        vals += [float(x) for x in re.findall(r"-?\d+\.\d+", body)]
    its = [float(x) for x in re.findall(r"^\s+total energy\s+=\s+(-?[0-9.]+) Ry", t, re.M)]
    return vals, its


def num(x, fmt):
    return "0" if x == 0 else format(x, fmt)


print("### Accuracy\n")
print("| Input | Total energy, `cpu` (Ry) | Largest \\|ΔE\\|, `gpu` vs `cpu`: final / any iteration (Ry) "
      "| SCF iterations, `cpu` / `gpu` | Band energies per run | Largest \\|Δε\\|, any run vs `cpu` (eV) |")
print("|---|---|---|---|---|---|")
for inp in inputs:
    cpu, gpu = runs(LABEL, inp, "cpu"), runs(LABEL, inp, "gpu")
    if not cpu or not gpu:
        continue
    ref_v, ref_i = scf(run_path(C, cpu[0]) / "out.txt")
    de = max(abs(float(g["energy_Ry"]) - float(c["energy_Ry"])) for g in gpu for c in cpu)
    di = 0.0
    for r in gpu:
        _, i = scf(run_path(C, r) / "out.txt")
        di = max([di] + [abs(a - b) for a, b in zip(i, ref_i)])
    dv = 0.0
    for r in gpu + cpu[1:] + runs(LABEL, inp, "auto"):
        v, _ = scf(run_path(C, r) / "out.txt")
        dv = float("inf") if len(v) != len(ref_v) else max([dv] + [abs(a - b) for a, b in zip(v, ref_v)])
    it = f'{"/".join(sorted({r["scf_iters"] for r in cpu}))} / {"/".join(sorted({r["scf_iters"] for r in gpu}))}'
    print(f"| {inp} | {cpu[0]['energy_Ry']} | {num(de, '.1e')} / {num(di, '.1e')} | {it} | {len(ref_v)} | {num(dv, '.4f')} |")

print("\n### Routing (AB_REPORT, first repetition)\n")
print("| Input | Mode | Routine | Calls | On the GPU | GPU share of the routine's time | Fallbacks |")
print("|---|---|---|---:|---:|---:|---:|")
for inp in inputs:
    for mode in ("auto", "gpu"):
        rs = runs(LABEL, inp, mode)
        if not rs:
            continue
        rep = report(run_path(C, rs[0]))
        for name in ("zgemm", "dgemm"):
            z = rep.get(name)
            if z:
                print(f'| {inp} | `{mode}` | {name.upper()} | {z["calls"]:.0f} | {z["gpu_calls"]:.0f} '
                      f'| {z["gpu_time%"]:.1f}% | {z.get("fallbacks", 0):.0f} |')

print("\n### Run time (s, median [min–max])\n")
head = "| Input | `cpu` | `auto` | `gpu` | `gpu` / `cpu` | ZGEMM+DGEMM time, `cpu` | same, `gpu` |"
print(head + (f" `auto`, {BASE} |" if BASE else ""))
print("|---|---|---|---|---:|---:|---:|" + ("---|" if BASE else ""))
for inp in inputs:
    w = {m: walls(runs(LABEL, inp, m)) for m in ("cpu", "auto", "gpu")}
    blas = {m: [sum(report(run_path(C, r)).get(n, {}).get("total_ms", 0) for n in ("zgemm", "dgemm")) / 1e3
                for r in runs(LABEL, inp, m)] for m in ("cpu", "gpu")}
    ratio = f'{st.median(w["gpu"]) / st.median(w["cpu"]):.3f}' if w["gpu"] and w["cpu"] else "–"
    med = lambda xs: f"{st.median(xs):.1f}" if xs else "–"
    line = f'| {inp} | {rng(w["cpu"])} | {rng(w["auto"])} | {rng(w["gpu"])} | {ratio} | {med(blas["cpu"])} | {med(blas["gpu"])} |'
    if BASE:
        line += f" {rng(walls(runs(BASE, inp, 'auto')))} |"
    print(line)

print("\n### GPU path stages, `gpu` mode (s, median over repetitions)\n")
stages = ["init", "alloc", "pack", "upload", "kernel", "download", "epilogue", "other"]
print("| Input | Routine | GPU calls | " + " | ".join(stages) + " | total | Accelerate, same calls (`cpu`) | kernel / Accelerate |")
print("|---|---|---:|" + "---:|" * (len(stages) + 3))
for inp in inputs:
    cpu_reps = [report(run_path(C, r)) for r in runs(LABEL, inp, "cpu")]
    gpu_reps = [report(run_path(C, r)) for r in runs(LABEL, inp, "gpu")]
    for name in ("zgemm", "dgemm"):
        reps = [z[name] for z in gpu_reps if z.get(name, {}).get("gpu_calls")]
        sc = same_calls(cpu_reps, gpu_reps, name)
        if not reps or not sc:
            continue
        vals = [st.median(z.get("st_" + s, 0) / 1e3 for z in reps) for s in stages]
        tot = st.median(z["gpu_ms"] / 1e3 for z in reps)
        print(f'| {inp} | {name.upper()} | {st.median(z["gpu_calls"] for z in reps):.0f} | '
              + " | ".join(f"{v:.2f}" for v in vals) + f" | {tot:.2f} | {sc[1]:.2f} | {sc[0] / sc[1]:.2f} |")
