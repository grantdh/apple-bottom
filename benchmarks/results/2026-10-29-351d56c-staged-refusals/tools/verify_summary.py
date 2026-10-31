#!/usr/bin/env python3
"""verify_summary.py <campaign-dir> <label>...: the AB_VERIFY_STAGED=1 runs of each label, per run and
routine, summed over ranks: the calls the staged path kept and how many of them the host-path redo
did not reproduce bit for bit; with tools/verify_more.py, each such call (which path the redo took,
the differing real and imaginary parts, both results' normwise error against FP64 cblas) and the
Metal command buffers that failed, by error; and the staged calls counted gpu_failed. Lines are
searched, not anchored: ranks' lines interleave in stderr."""
import collections, csv, re, sys
from pathlib import Path

camp = Path(sys.argv[1])
energy = {r["run_dir"]: (r["energy_Ry"], r["scf_iters"], r["wall_s"], r["power"])
          for r in csv.DictReader(open(camp / "results.csv"))}
for label in sys.argv[2:]:
    for d in sorted((camp / "runs" / label).glob("*/gpu_r*")):
        run = str(d.relative_to(camp))
        t = (d / "err.txt").read_text(errors="replace")
        e, it, wall, power = energy.get(run, ("?", "?", "?", "?"))
        print(f"{run}: E={e} Ry ({it} iterations), {power}")
        tot = collections.defaultdict(lambda: [0, 0, 0])
        for m in re.finditer(r"AB_VERIFY_STAGED\[\d+\] (\w+) kept_calls=(\d+) differ_calls=(\d+)", t):
            v = tot[m[1]]
            v[0] += 1; v[1] += int(m[2]); v[2] += int(m[3])
        for rt, (ranks, kept, bad) in sorted(tot.items()):
            print(f"  {rt}: ranks={ranks} kept_calls={kept} differ_calls={bad}")
        for m in re.finditer(r"AB_VERIFY_STAGED \w+ [A-Z]{2} M=.*?staged_record=-?\d+", t):
            print(f"    {m[0]}")
        errs = collections.Counter(m[1] for m in re.finditer(
            r"AB_CMDBUF_ERROR pid=\d+ (code=\d+ [^\n]*?\))", t))
        for msg, n in errs.items():
            print(f"    {n} failed command buffers: {msg}")
        gf = sum(int(m[1]) for m in re.finditer(r"AB_REPORT\[\d+\] [dz]gemm staged [^\n]*?gpu_failed=(\d+)", t))
        fb = sum(int(m[1]) for m in re.finditer(r"AB_REPORT\[\d+\] [dz]gemm fallbacks [^\n]*?gpu_failed=(\d+)", t))
        print(f"  staged gpu_failed={gf}, fallbacks gpu_failed={fb}")
