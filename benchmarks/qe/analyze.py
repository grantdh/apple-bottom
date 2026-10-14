#!/usr/bin/env python3
"""Summarize a run_campaign.sh directory: results.csv plus each run's AB_REPORT.

Usage: python3 analyze.py <out-dir>
"""
import csv, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path


def report(run_dir):
    """AB_REPORT fields per routine from the run's stderr."""
    out = {}
    err = Path(run_dir) / "err.txt"
    if not err.exists():
        return out
    for line in err.read_text(errors="replace").splitlines():
        m = re.match(r"AB_REPORT\[\d+\] (\w+) (calls=.*)", line)
        if m:
            out.setdefault(m[1], {}).update({k: float(v) for k, v in re.findall(r"(\w+%?)=([-0-9.]+)", m[2])})
            continue
        m = re.match(r"AB_REPORT\[\d+\] (\w+) gpu_stages_ms (.*)", line)
        if m:
            out.setdefault(m[1], {}).update({"st_" + k: float(v) for k, v in re.findall(r"(\w+)=([-0-9.]+)", m[2])})
            continue
        m = re.match(r"AB_REPORT\[\d+\] (\w+) fallbacks (.*)", line)
        if m:
            out.setdefault(m[1], {}).update({"fb_" + k: float(v) for k, v in re.findall(r"(\w+)=([-0-9.]+)", m[2])})
    return out

def main():
    C = Path(sys.argv[1])
    rows = list(csv.DictReader(open(C / "results.csv")))
    for r in rows:  # run_campaign.sh writes "label"; early campaigns wrote "libtag"
        r.setdefault("libtag", r.get("label", ""))

    def run_dir(r):
        d = C / "runs" / r["libtag"] / r["input"] / f'{r["mode"]}_r{r["rep"]}'
        return d

    per = []
    for r in rows:
        rep = report(run_dir(r))
        z, d = rep.get("zgemm", {}), rep.get("dgemm", {})
        per.append(dict(r, z=z, d=d))

    def f(x, nd=1):
        return "-" if x is None else f"{x:.{nd}f}"

    print("== per run")
    print("lib      input              mode rep   wall_s  iters  energy_Ry          zgemm_s  z_gpu%  z_kern_s z_other_s  dgemm_s  d_gpu%  fb  load")
    for p in per:
        z, d = p["z"], p["d"]
        fb = sum(v for k, v in {**z, **d}.items() if k.startswith("fb_"))
        print(f'{p["libtag"]:8} {p["input"]:18} {p["mode"]:4} {p["rep"]:>3} {float(p["wall_s"] or "nan"):8.1f} {p["scf_iters"]:>5}  {p["energy_Ry"]:18} '
              f'{z.get("total_ms",0)/1e3:8.2f} {z.get("gpu_time%",0):6.1f} {z.get("st_kernel",0)/1e3:8.2f} {z.get("st_other",0)/1e3:9.2f} '
              f'{d.get("total_ms",0)/1e3:8.2f} {d.get("gpu_time%",0):6.1f} {fb:3.0f}  {p["load1"]}')

    print("\n== per configuration (median [min-max])")
    g = defaultdict(list)
    for p in per:
        g[(p["libtag"], p["input"], p["mode"])].append(p)
    summ = {}
    for k, ps in sorted(g.items()):
        w = [float(p["wall_s"]) for p in ps if p["wall_s"]]
        zs = [p["z"].get("total_ms", 0) / 1e3 for p in ps]
        kern = [p["z"].get("st_kernel", 0) / 1e3 for p in ps]
        ds = [p["d"].get("total_ms", 0) / 1e3 for p in ps]
        dk = [p["d"].get("st_kernel", 0) / 1e3 for p in ps]
        en = sorted({p["energy_Ry"] for p in ps}); it = sorted({p["scf_iters"] for p in ps})
        summ[k] = dict(wall=st.median(w) if w else None, z=st.median(zs), kern=st.median(kern), d=st.median(ds), dk=st.median(dk))
        print(f"{k[0]:8} {k[1]:18} {k[2]:4} n={len(ps)} wall {f(st.median(w))} [{f(min(w))}-{f(max(w))}]  zgemm {f(st.median(zs),2)}  z_kernel {f(st.median(kern),2)}  dgemm {f(st.median(ds),2)}  d_kernel {f(st.median(dk),2)}  E={en} iters={it}")

    print("\n== decision metric: GPU kernel time alone (gpu mode) / Accelerate time of the same routine (cpu mode)")
    for (lib, inp, mode), s in sorted(summ.items()):
        if mode != "gpu": continue
        cpu = summ.get((lib, inp, "cpu"))
        if not cpu: continue
        for name, kk, cc in (("zgemm", s["kern"], cpu["z"]), ("dgemm", s["dk"], cpu["d"])):
            if cc > 0.05 and kk > 0:
                print(f"{lib} {inp} {name}: kernel {kk:.2f} s / Accelerate {cc:.2f} s = {kk/cc:.2f}")
        if cpu["wall"] and s["wall"]:
            print(f"{lib} {inp} wall gpu/cpu = {s['wall']/cpu['wall']:.3f}; auto/cpu = "
                  f"{(summ.get((lib, inp, 'auto'), {}).get('wall') or float('nan'))/cpu['wall']:.3f}")


if __name__ == "__main__":
    main()
