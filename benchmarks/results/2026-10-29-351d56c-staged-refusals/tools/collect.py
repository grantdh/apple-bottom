#!/usr/bin/env python3
"""collect.py <campaign-dir> <results-dir> <label-prefix>...: copies the campaign's runs whose label
starts with one of the prefixes into the results folder: their results.csv rows (frozen library
directories named as in LIBS) and reports/<label>_<input>_<rep>.txt, the run's header and its AB_REPORT
lines. The lines are searched, not anchored: in merged multi-rank stderr another rank's partial line
can precede one. A run already in results.csv is left as it is."""
import csv, re, sys
from pathlib import Path

LIBS = {"755a46613bf2": "<1.5.0>", "78eb74ff7cb4": "<counter>", "edd6aa2db786": "<fix>",
        "95a90d9c9134": "<serial3m>", "3ead2d56bc84": "<refactor>"}
camp, res, prefixes = Path(sys.argv[1]), Path(sys.argv[2]), tuple(sys.argv[3:])


def lib_name(path):
    for h, name in LIBS.items():
        if h in path:
            return f"{name}/libapplebottom.dylib"
    return path


rows = [r for r in csv.DictReader(open(camp / "results.csv")) if r["label"].startswith(prefixes)]
with open(res / "results.csv") as f:
    fields = next(csv.reader(f))
    have = {r["run_dir"] for r in csv.DictReader(open(res / "results.csv"))}
with open(res / "results.csv", "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    for r in rows:
        if r["run_dir"] in have:
            continue
        r["lib_loaded"] = lib_name(r["lib_loaded"])
        w.writerow({k: r.get(k, "") for k in fields})
        d = camp / r["run_dir"]
        _, label, inp, rep = r["run_dir"].split("/")
        head = (f"# {r['run_dir']}: {label} {inp} np={r['np']} mode={r['mode']} wall={r['wall_s']} s "
                f"power={r['power']} E={r['energy_Ry']} iters={r['scf_iters']} build={r['build']} "
                f"lib={r['lib_loaded']}")
        lines = re.findall(r"AB_REPORT\[\d+\] [^\n]*", (d / "err.txt").read_text(errors="replace"))
        (res / "reports" / f"{label}_{inp}_{rep}.txt").write_text("\n".join([head, *lines]) + "\n")
        print("collected", r["run_dir"])
