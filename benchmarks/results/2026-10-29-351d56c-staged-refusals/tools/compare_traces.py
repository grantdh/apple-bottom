#!/usr/bin/env python3
"""compare_traces.py <trace-dir-a> <trace-dir-b>: whether two AB_TRACE_HASH=1 runs of the same
pw.x input made the same BLAS calls on the same data, in the same order: per call the routine,
TRANS, dimensions, leading dimensions, alpha, beta and the hashes of op(A), op(B) and C on entry.
Since every call's inputs come from the results of the calls before it, equal sequences mean every
GEMM result the run read back was bit for bit the same. Prints the first difference, if any."""
import sys
from pathlib import Path
KEYS = ("routine", "ta", "tb", "M", "N", "K", "lda", "ldb", "ldc", "alpha", "beta", "hash_a", "hash_b", "hash_c")
def calls(d):
    files = sorted(Path(d).glob("trace.*"))
    if len(files) != 1: sys.exit(f"{d}: {len(files)} trace files, expected 1")
    out = []
    for line in files[0].read_text().splitlines():
        if line.startswith("#"): continue
        f = dict(t.split("=", 1) for t in line.split() if "=" in t)
        out.append((tuple(f.get(k, "-") for k in KEYS), f.get("path"), f.get("staged", "")))
    return out
a, b = calls(sys.argv[1]), calls(sys.argv[2])
print(f"{len(a)} calls vs {len(b)}")
for i, (x, y) in enumerate(zip(a, b)):
    if x[0] != y[0]:
        print(f"first difference at call {i + 1}:\n  {dict(zip(KEYS, x[0]))}\n  {dict(zip(KEYS, y[0]))}")
        sys.exit(1)
if len(a) != len(b): sys.exit("different call counts")
gpu = [(p1, p2) for (_, p1, _), (_, p2, _) in zip(a, b) if p1 != "cpu" or p2 != "cpu"]
print(f"identical: every call's arguments and input hashes; GPU-routed calls {len(gpu)}, "
      f"paths equal in {sum(p1 == p2 for p1, p2 in gpu)}")
