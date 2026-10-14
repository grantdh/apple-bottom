#!/usr/bin/env python3
"""Compare every run of a run_campaign.sh directory with the cpu-mode run of
the same input and label: per-iteration total energies, the final energy and
every printed band energy (verbosity = 'high').

Usage: python3 compare.py <out-dir> [reference-label]
"""
import re, sys
from pathlib import Path

C = Path(sys.argv[1])
REF_LABEL = sys.argv[2] if len(sys.argv) > 2 else None

def parse(out):
    t = out.read_text(errors="replace")
    iters = [float(x) for x in re.findall(r"^\s+total energy\s+=\s+(-?[0-9.]+) Ry", t, re.M)]
    acc = [float(x) for x in re.findall(r"estimated scf accuracy\s+<\s+([0-9.Ee+-]+) Ry", t)]
    final = re.findall(r"^!\s+total energy\s+=\s+(-?[0-9.]+) Ry", t, re.M)
    # band energies of the final iteration: blocks after "End of self-consistent calculation"
    tail = t.split("End of self-consistent calculation")[-1].split("highest occupied")[0]
    eig = []
    for blk in re.split(r"\n\s+k =", tail)[1:]:
        body = blk.split("bands (ev):")[-1].split("occupation numbers")[0]
        eig += [float(x) for x in re.findall(r"-?\d+\.\d+", body)]
    homo = re.findall(r"highest occupied(?:, lowest unoccupied)? level[^:]*:\s+([-0-9. ]+)", t)
    return dict(iters=iters, acc=acc, final=float(final[-1]) if final else None, eig=eig,
                homo=[float(x) for x in homo[-1].split()] if homo else [])

runs = sorted(p for p in (C / "runs").glob("*/*/*/out.txt"))
by = {}
for p in runs:
    lib, inp, name = p.parts[-4], p.parts[-3], p.parts[-2]
    by.setdefault((lib, inp), {})[name] = parse(p)

print("lib      input              run        final_Ry          dE_final    max|dE_iter|  n_iter  n_eig  max|d_eig| eV  homo/lumo")
for (lib, inp), rs in sorted(by.items()):
    ref_name = next((n for n in sorted(rs) if n.startswith("cpu")), None)
    ref = rs[ref_name] if ref_name else None
    # labels without a cpu run are compared with the cpu run of REF_LABEL
    if ref is None:
        alt = by.get((REF_LABEL, inp), {})
        ref_name = next((n for n in sorted(alt) if n.startswith("cpu")), None)
        ref = alt.get(ref_name) if ref_name else None
    for n, r in sorted(rs.items()):
        if not r["final"] or not ref or not ref["final"]:
            print(f"{lib:8} {inp:18} {n:10} (incomplete)"); continue
        k = min(len(r["iters"]), len(ref["iters"]))
        di = max((abs(a - b) for a, b in zip(r["iters"][:k], ref["iters"][:k])), default=float("nan"))
        de = None
        if len(r["eig"]) == len(ref["eig"]) and r["eig"]:
            de = max(abs(a - b) for a, b in zip(r["eig"], ref["eig"]))
        print(f"{lib:8} {inp:18} {n:10} {r['final']:.8f}  {r['final'] - ref['final']:+.2e}   {di:.2e}      "
              f"{len(r['iters']):>3}/{len(ref['iters']):<3} {len(r['eig']):>5}  {('-' if de is None else f'{de:.4f}'):>8}       {r['homo']}")
