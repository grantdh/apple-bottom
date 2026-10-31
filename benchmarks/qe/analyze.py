#!/usr/bin/env python3
"""Summarize a run_campaign.sh directory: results.csv plus each run's AB_REPORT.

Usage: python3 analyze.py <out-dir>

Prints every run, every configuration (median and range over repetitions),
and for each input the ratio of the GPU kernel time (gpu mode) to the time
Accelerate needs for the same calls (cpu mode).
"""
import csv, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path

COUNTS = ("calls", "gpu_calls", "cpu_calls", "total_ms", "gpu_ms", "cpu_ms", "fallbacks")
# Searched, not matched: in merged multi-rank stderr another rank's partial line can precede one.
AB_LINE = re.compile(r"AB_REPORT\[(\d+)\] (\w+) (calls=.*|gpu_stages_ms .*|fallbacks .*|staged .*)")
AB_HEADER = re.compile(r"AB_REPORT\[(\d+)\] apple-bottom ")
# The key prefix of each AB_REPORT line's fields, by the line's first word.
FIELD_PREFIX = {"gpu_stages_ms": "st_", "fallbacks": "fb_", "staged": "sp_"}


def report(run_dir):
    """AB_REPORT fields per routine from the run's stderr, summed over MPI ranks.

    Keys: the AB_REPORT counters, st_<stage> (GPU stage ms), fb_<reason>,
    sp_<outcome> (staged path: kept, or why not), ranks, and gpu_time% /
    gpu_calls% recomputed from the sums.
    """
    err = Path(run_dir) / "err.txt"
    if not err.exists():
        return {}
    per_pid = defaultdict(dict)                       # (pid, routine) -> fields
    headers = set()                                   # pids that printed an AB_REPORT header
    for line in err.read_text(errors="replace").splitlines():
        h = AB_HEADER.search(line)
        if h:
            headers.add(h[1])
        m = AB_LINE.search(line)
        if not m:
            continue
        pid, name, rest = m[1], m[2], m[3]
        prefix = FIELD_PREFIX.get(rest.split(" ", 1)[0], "")
        for k, v in re.findall(r"(\w+%?)=([-0-9.]+)", rest):
            if not k.endswith("%"):
                per_pid[(pid, name)][prefix + k] = float(v)
    out = {}
    for (pid, name), fields in per_pid.items():
        r = out.setdefault(name, {"ranks": 0})
        r["ranks"] += 1
        for k, v in fields.items():
            r[k] = r.get(k, 0.0) + v
    for name, r in out.items():
        if r["ranks"] < len(headers):
            print(f"analyze.py: {err}: {name} reported by {r['ranks']} of {len(headers)} ranks", file=sys.stderr)
        r["gpu_time%"] = 100.0 * r.get("gpu_ms", 0) / r["total_ms"] if r.get("total_ms") else 0.0
        r["gpu_calls%"] = 100.0 * r.get("gpu_calls", 0) / r["calls"] if r.get("calls") else 0.0
    return out


def load(out_dir):
    """results.csv rows, with "label" set (early campaigns wrote "libtag")."""
    rows = list(csv.DictReader(open(Path(out_dir) / "results.csv")))
    for r in rows:
        r["label"] = r.get("label") or r.get("libtag", "")
    return rows


def run_path(out_dir, r):
    """The run's directory: the recorded run_dir, else the default name."""
    if r.get("run_dir"):
        return Path(out_dir) / r["run_dir"]
    return Path(out_dir) / "runs" / r["label"] / r["input"] / f'{r["mode"]}_r{r["rep"]}'


def wall(r):
    return float(r["wall_s"]) if r.get("wall_s") else None


def same_calls(cpu_reports, gpu_reports, name):
    """(kernel s, Accelerate s for the same calls): the GPU-routed calls'
    kernel time in gpu mode, and their time on Accelerate in cpu mode (the
    routine's cpu-mode time minus the calls gpu mode also left on the CPU)."""
    g = [z[name] for z in gpu_reports if z.get(name, {}).get("gpu_calls")]
    c = [z[name] for z in cpu_reports if name in z]
    if not g or not c:
        return None
    kernel = st.median(z.get("st_kernel", 0) for z in g) / 1e3
    accel = (st.median(z["total_ms"] for z in c) - st.median(z.get("cpu_ms", 0) for z in g)) / 1e3
    return kernel, accel


def main():
    C = Path(sys.argv[1])
    rows = load(C)

    def f(x, nd=1):
        return "-" if x is None else f"{x:.{nd}f}"

    print("== per run")
    print("label    input              mode rep   wall_s  iters  energy_Ry          zgemm_s  z_gpu%  z_kern_s z_other_s  dgemm_s  d_gpu%  fb  load")
    reports = {}
    for r in rows:
        rep = reports[id(r)] = report(run_path(C, r))
        z, d = rep.get("zgemm", {}), rep.get("dgemm", {})
        fb = z.get("fallbacks", 0) + d.get("fallbacks", 0)
        print(f'{r["label"]:8} {r["input"]:18} {r["mode"]:4} {r["rep"]:>3} {f(wall(r)):>8} {r["scf_iters"]:>5}  {r["energy_Ry"]:18} '
              f'{z.get("total_ms", 0)/1e3:8.2f} {z.get("gpu_time%", 0):6.1f} {z.get("st_kernel", 0)/1e3:8.2f} {z.get("st_other", 0)/1e3:9.2f} '
              f'{d.get("total_ms", 0)/1e3:8.2f} {d.get("gpu_time%", 0):6.1f} {fb:3.0f}  {r["load1"]}')

    print("\n== per configuration (median [min-max])")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["input"], r["mode"])].append(r)
    for (label, inp, mode), rs in sorted(groups.items()):
        w = [x for x in map(wall, rs) if x is not None]
        wtxt = f"{f(st.median(w))} [{f(min(w))}-{f(max(w))}]" if w else "-"
        zs = [reports[id(r)].get("zgemm", {}).get("total_ms", 0) / 1e3 for r in rs]
        ds = [reports[id(r)].get("dgemm", {}).get("total_ms", 0) / 1e3 for r in rs]
        en = sorted({r["energy_Ry"] for r in rs}); it = sorted({r["scf_iters"] for r in rs})
        print(f"{label:8} {inp:18} {mode:4} n={len(rs)} wall {wtxt}  zgemm {f(st.median(zs), 2)}  dgemm {f(st.median(ds), 2)}  E={en} iters={it}")

    print("\n== GPU kernel time (gpu mode) / Accelerate time for the same calls (cpu mode)")
    for label, inp in sorted({(r["label"], r["input"]) for r in rows}):
        cpu = [reports[id(r)] for r in groups.get((label, inp, "cpu"), [])]
        gpu = [reports[id(r)] for r in groups.get((label, inp, "gpu"), [])]
        for name in ("zgemm", "dgemm"):
            sc = same_calls(cpu, gpu, name)
            if sc and sc[1] > 0:
                print(f"{label} {inp} {name}: kernel {sc[0]:.2f} s / Accelerate {sc[1]:.2f} s = {sc[0] / sc[1]:.2f}")
        wc = [x for x in map(wall, groups.get((label, inp, "cpu"), [])) if x]
        for mode in ("auto", "gpu"):
            wm = [x for x in map(wall, groups.get((label, inp, mode), [])) if x]
            if wc and wm:
                print(f"{label} {inp} wall {mode}/cpu = {st.median(wm) / st.median(wc):.3f}")


if __name__ == "__main__":
    main()
