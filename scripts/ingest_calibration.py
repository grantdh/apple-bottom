#!/usr/bin/env python3
"""Files a `make calibrate` result as a chip's built-in routing table and regenerates src/dispatch_table.h.

    scripts/ingest_calibration.py DIR [DIR ...] [--force] [--allow-partial] [--any-cores]
    scripts/ingest_calibration.py --regenerate [--check]

DIR is a calibration results directory: the dispatch_table.txt and calibrate_summary.csv (and
calibrate_raw.csv, if present) that benchmarks/calibrate.c wrote, on this machine (`make calibrate`
runs this script on its own results) or on another one (the stages/calibrate/ directory of a field
report, or a directory someone sends). The pair is checked against the rule calibrate.c derives the
table by: every row is a measured >= 1.10x win of its routine and TRANS class, no measured non-win is
covered by a row, auto_route agrees with the rows, and the whole grid was measured (a partial grid,
from --only, is refused: it cannot stand for the chip). It is then filed under
benchmarks/calibration/<slug>/, the slug being the chip without "Apple " plus the GPU core count
(M2-Max-38c), with the identity keys (# chip:, # gpu_cores:, # cpu_matrix:) added to the table's
header when an older calibrate.c left them out (a table with no GPU core count is refused unless
--any-cores says it should stand for every variant of the chip), and src/dispatch_table.h is regenerated from every
filed directory, in slug order, one table each. Commit both.

--regenerate rebuilds the header from the filed directories alone; with --check it rebuilds in memory
and exits 1 if the committed header differs (make dispatch-table-check, run by ci-local and CI).
Python 3.9, standard library only.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION = os.path.join(REPO, "benchmarks", "calibration")
HEADER = os.path.join(REPO, "src", "dispatch_table.h")

WIN_SPEEDUP = 1.10          # benchmarks/calibrate.c WIN_SPEEDUP
MIN_DIM = 32                # AB_DISPATCH_DEFAULT_MIN_DIM: auto_route was derived with it
FULL_GRID = {"qe-paw", "qe-davidson", "qe-k222", "square", "square-trans"}   # calibrate.c build_grid()

KEY_RE = re.compile(r"^\s*#\s*([a-z][a-z0-9_]*):\s*([^#]*?)\s*(?:#.*)?$")      # "# chip: Apple M2 Max  # note"
ROW_RE = re.compile(r"^\s*(dgemm|zgemm)\s+([NTCntc])\s+([NTCntc])\s+(\d+)\s+(\d+)\s+(\d+)\s*(?:#.*)?$")
MACHINE_RE = re.compile(r"^\s*#\s*Machine:\s*(Apple [^,]+?)(?:,\s*(\d+)-core GPU)?(?:,|\s*$)")


class Ingest(Exception):
    pass


def trans_class(t):
    return "N" if t.upper() == "N" else "T"


def chip_slug(chip, cores):
    """As dispatch.c's ab_dispatch_chip_slug() and the field kit's chip_slug()."""
    s = re.sub(r"^Apple\s*", "", chip or "", flags=re.I)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-") or "unknown"
    return f"{s}-{cores}c" if cores and s != "unknown" else s


def infer_cpu_matrix(chip):
    """The matrix unit Accelerate runs on: AMX up to the M3, SME from the M4 (FEAT_SME)."""
    m = re.search(r"\bM(\d+)\b", chip or "")
    if not m:
        return "unknown"
    return "AMX" if int(m.group(1)) <= 3 else "SME"


class Table:
    """One dispatch_table.txt with its calibrate_summary.csv."""

    def __init__(self, directory):
        self.dir = directory
        self.txt = os.path.join(directory, "dispatch_table.txt")
        self.csv = os.path.join(directory, "calibrate_summary.csv")
        for p in (self.txt, self.csv):
            if not os.path.isfile(p):
                raise Ingest(f"{directory}: no {os.path.basename(p)}")
        self.keys, self.rows, self.lines = {}, [], []
        machine = None
        with open(self.txt, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.rstrip("\n")
                self.lines.append(line)
                m = KEY_RE.match(line)
                if m:
                    self.keys[m.group(1)] = m.group(2)
                    continue
                m = MACHINE_RE.match(line)
                if m and not machine:
                    machine = m
                m = ROW_RE.match(line)
                if m:
                    self.rows.append((m.group(1)[0], trans_class(m.group(2)), trans_class(m.group(3)),
                                      int(m.group(4)), int(m.group(5)), int(m.group(6))))
                elif line.strip() and not line.lstrip().startswith("#"):
                    raise Ingest(f"{self.txt}:{lineno}: not a row or a comment: {line!r}")
        # Identity: the keys calibrate.c writes, or the legacy "# Machine:" line.
        self.chip = self.keys.get("chip") or (machine.group(1).strip() if machine else "")
        if not self.chip:
            raise Ingest(f"{self.txt}: no '# chip:' key and no '# Machine:' line")
        cores = self.keys.get("gpu_cores") or (machine.group(2) if machine and machine.group(2) else "0")
        self.gpu_cores = int(cores) if cores.isdigit() else 0
        self.cpu_matrix = self.keys.get("cpu_matrix") or infer_cpu_matrix(self.chip)
        self.inferred = "cpu_matrix" not in self.keys
        self.slug = chip_slug(self.chip, self.gpu_cores)
        with open(self.csv, encoding="utf-8", newline="") as f:
            self.shapes = list(csv.DictReader(f))
        need = {"routine", "transA", "transB", "M", "N", "K", "source", "speedup", "gpu_path", "win", "auto_route"}
        if not self.shapes or need - set(self.shapes[0]):
            raise Ingest(f"{self.csv}: not a calibrate_summary.csv (columns {sorted(need)})")

    def covered(self, s):
        M, N, K = int(s["M"]), int(s["N"]), int(s["K"])
        return any(r[0] == s["routine"][0] and r[1] == trans_class(s["transA"]) and
                   r[2] == trans_class(s["transB"]) and M >= r[3] and N >= r[4] and K >= r[5]
                   for r in self.rows)

    def measured(self, row):
        for s in self.shapes:
            if (s["routine"][0], trans_class(s["transA"]), trans_class(s["transB"]),
                    int(s["M"]), int(s["N"]), int(s["K"])) == row:
                return s
        return None

    def validate(self, allow_partial):
        """The rule of benchmarks/calibrate.c derive_rule(), re-checked from the CSV."""
        for r in self.rows:
            s = self.measured(r)
            name = f"{r[0]}gemm {r[1]} {r[2]} {r[3]}x{r[4]}x{r[5]}"
            if s is None:
                raise Ingest(f"{self.txt}: row {name} is not a measured shape of {os.path.basename(self.csv)}")
            if s["win"] != "1" or s["gpu_path"] != "1" or float(s["speedup"]) < WIN_SPEEDUP:
                raise Ingest(f"{self.txt}: row {name} was not a GPU win (speedup {s['speedup']}, "
                             f"gpu_path {s['gpu_path']}, win {s['win']})")
        for s in self.shapes:
            name = f"{s['routine']} {s['transA']} {s['transB']} {s['M']}x{s['N']}x{s['K']}"
            gpu = self.covered(s) and min(int(s["M"]), int(s["N"]), int(s["K"])) > MIN_DIM
            if s["win"] != "1" and gpu:
                raise Ingest(f"{self.txt}: a row covers the measured non-win {name} ({s['speedup']}x)")
            if (s["auto_route"] == "gpu") != gpu:
                raise Ingest(f"{self.csv}: {name} says auto_route={s['auto_route']}, the rows say "
                             f"{'gpu' if gpu else 'cpu'}")
        missing = FULL_GRID - {s["source"] for s in self.shapes}
        if missing and not allow_partial:
            raise Ingest(f"{self.dir}: a partial grid (no {', '.join(sorted(missing))} shapes), from "
                         f"--only; it cannot stand for the chip. Use it directly with "
                         f"AB_DISPATCH_TABLE={self.txt}, or pass --allow-partial if you mean it")

    def speedup_of(self, row):
        s = self.measured(row)
        return f"{float(s['speedup']):.2f}x" if s else ""

    def provenance(self):
        """The header's comment lines that describe the calibration (not the rule or the keys)."""
        out, started = [], False
        for line in self.lines:
            if not line.lstrip().startswith("#"):
                break
            body = line.lstrip()[1:]
            body = body[1:] if body.startswith(" ") else body
            if not started:
                started = body.startswith("Date:")
                if not started:
                    continue
            if body.startswith("A call goes to the GPU"):
                break
            if KEY_RE.match(line) or body.startswith("routine transA"):
                continue
            out.append(body.rstrip())
        while out and not out[-1]:
            out.pop()
        return out

    def header_text_with_keys(self, source):
        """The table file as filed: identity keys present, and where it was filed from."""
        lines = list(self.lines)
        if "chip" not in self.keys or "gpu_cores" not in self.keys or "cpu_matrix" not in self.keys:
            keys = []
            if "chip" not in self.keys: keys.append(f"# chip: {self.chip}")
            if "gpu_cores" not in self.keys: keys.append(f"# gpu_cores: {self.gpu_cores}")
            if "cpu_matrix" not in self.keys:
                keys.append(f"# cpu_matrix: {self.cpu_matrix}" + ("   # inferred from the chip name" if self.inferred else ""))
            at = next((i + 1 for i, l in enumerate(lines) if l.startswith("# Written by")), 1)
            lines[at:at] = keys
        if "filed" not in self.keys:
            at = next((i + 1 for i, l in enumerate(lines) if l.startswith("# cpu_matrix")), 1)
            lines.insert(at, f"# filed: {datetime.date.today().isoformat()} from {source} by scripts/ingest_calibration.py")
        return "\n".join(lines) + "\n"


def filed_tables():
    out = []
    if os.path.isdir(CALIBRATION):
        for name in sorted(os.listdir(CALIBRATION)):
            d = os.path.join(CALIBRATION, name)
            if os.path.isfile(os.path.join(d, "dispatch_table.txt")):
                t = Table(d)
                if t.slug != name:
                    raise Ingest(f"{d}: directory name differs from the table's own slug {t.slug}")
                out.append(t)
    return out


def generate(tables):
    L = ["// =============================================================================",
         "// dispatch_table.h — built-in AB_MODE=auto tables, one per calibrated chip (dispatch.h)",
         "// =============================================================================",
         "// Generated by scripts/ingest_calibration.py from benchmarks/calibration/; do",
         "// not edit. `make calibrate` files this machine's calibration there and",
         "// regenerates this header; the script also files a calibration made on another",
         "// machine (a results directory with dispatch_table.txt and calibrate_summary.csv).",
         "// Commit this header with benchmarks/calibration/; `make dispatch-table-check`",
         "// verifies that the two agree.",
         "//",
         "// A call goes to the GPU iff M >= min_M, N >= min_N and K >= min_K for a row",
         "// of its routine and TRANS class ('T' rows also cover 'C') in the table of",
         "// this machine's chip (brand string and GPU core count; dispatch.h); no row,",
         "// or no table for the chip: Accelerate. Each row is a measured win (>= 1.10x)",
         "// with no slower shape of its class at least as large in M, N and K.",
         "//",
         "// Tables (slug: chip, GPU cores, Accelerate's matrix unit, rows):"]
    for t in tables:
        L.append(f"//   {t.slug}: {t.chip}, {t.gpu_cores or 'any'} cores, {t.cpu_matrix}, "
                 f"{len(t.rows)} row{'s' if len(t.rows) != 1 else ''}")
    if not tables:
        L.append("//   (none filed: every chip runs AB_MODE=auto calls on Accelerate)")
    L += ["// =============================================================================",
          "#ifndef AB_DISPATCH_TABLE_H", "#define AB_DISPATCH_TABLE_H", "", '#include "dispatch.h"', ""]
    for t in tables:
        ident = re.sub(r"[^A-Za-z0-9]", "_", t.slug).upper()
        L.append("// -----------------------------------------------------------------------------")
        L.append(f"// {t.slug}: {t.chip}, {t.gpu_cores or 'any'}-core GPU, Accelerate on {t.cpu_matrix}")
        L += [("// " + p).rstrip() for p in t.provenance()]
        L.append("// -----------------------------------------------------------------------------")
        L.append(f"static const ab_dispatch_row_t AB_DISPATCH_ROWS_{ident}[] = {{")
        for r in t.rows:
            L.append(f"    {{'{r[0]}', '{r[1]}', '{r[2]}', {r[3]}, {r[4]}, {r[5]}}},   // {t.speedup_of(r)}")
        L.append("    {0, 0, 0, 0, 0, 0}," + ("" if t.rows else "   // no rows: every call stays on Accelerate"))
        L += ["};", ""]
    L += ["// chip, gpu_cores, cpu_matrix, data (tests/test_dispatch.c checks each table",
          "// against its measurements, relative to the source root), rows",
          "static const ab_dispatch_table_t AB_DISPATCH_BUILTIN_TABLES[] = {"]
    for t in tables:
        ident = re.sub(r"[^A-Za-z0-9]", "_", t.slug).upper()
        data = f"benchmarks/calibration/{t.slug}/calibrate_summary.csv"
        L.append(f'    {{"{t.chip}", {t.gpu_cores}, "{t.cpu_matrix}", "{data}", AB_DISPATCH_ROWS_{ident}}},')
    L += ["    {NULL, 0, NULL, NULL, NULL},", "};", "", "#endif // AB_DISPATCH_TABLE_H", ""]
    return "\n".join(L)


def regenerate(check):
    tables = filed_tables()
    for t in tables:
        t.validate(allow_partial=True)      # filed tables were checked when filed; keep the rows honest
    text = generate(tables)
    old = open(HEADER, encoding="utf-8").read() if os.path.isfile(HEADER) else ""
    if check:
        if text != old:
            sys.exit(f"ingest_calibration: {os.path.relpath(HEADER, REPO)} differs from benchmarks/calibration/ "
                     f"({len(tables)} table{'s' if len(tables) != 1 else ''}); run `make dispatch-table` and commit it")
        print(f"{os.path.relpath(HEADER, REPO)} matches benchmarks/calibration/ ({len(tables)} table{'s' if len(tables) != 1 else ''})")
        return
    if text != old:
        with open(HEADER, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {os.path.relpath(HEADER, REPO)}: {len(tables)} table{'s' if len(tables) != 1 else ''}"
              f" ({', '.join(t.slug for t in tables) or 'none'})")
    else:
        print(f"{os.path.relpath(HEADER, REPO)} is up to date ({len(tables)} table{'s' if len(tables) != 1 else ''})")


def ingest(directory, force, allow_partial, any_cores=False):
    t = Table(directory)
    t.validate(allow_partial)
    if t.gpu_cores == 0 and not any_cores:
        raise Ingest(f"{t.txt}: no GPU core count in the header (calibrate.c reads it from the IORegistry; pass "
                     f"--gpu-cores to it on a machine where that fails). A table without one would apply to every "
                     f"core-count variant of {t.chip}; pass --any-cores if that is what you mean")
    dest = os.path.join(CALIBRATION, t.slug)
    if os.path.realpath(directory) == os.path.realpath(dest):
        raise Ingest(f"{os.path.relpath(dest, REPO)} is the filed directory itself; use --regenerate")
    if os.path.isdir(dest) and not force:
        raise Ingest(f"{os.path.relpath(dest, REPO)} exists (an earlier calibration of this chip); --force replaces it")
    # Build the replacement beside the old directory, then swap: a failure leaves the old one.
    final = dest
    dest = dest + ".new"
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    try:
        source = os.path.relpath(os.path.abspath(directory), REPO)
    except ValueError:
        source = os.path.abspath(directory)
    if source.startswith(".."):
        source = os.path.abspath(directory)
    with open(os.path.join(dest, "dispatch_table.txt"), "w", encoding="utf-8") as f:
        f.write(t.header_text_with_keys(source))
    shutil.copy(t.csv, os.path.join(dest, "calibrate_summary.csv"))
    raw = os.path.join(directory, "calibrate_raw.csv")
    if os.path.isfile(raw):
        shutil.copy(raw, os.path.join(dest, "calibrate_raw.csv"))
    if os.path.isdir(final):
        shutil.rmtree(final)
    os.rename(dest, final)
    dest = final
    wins = sum(1 for s in t.shapes if s["win"] == "1")
    print(f"filed {os.path.relpath(dest, REPO)}: {t.chip}, {t.gpu_cores or 'any'} GPU cores, Accelerate on "
          f"{t.cpu_matrix}{' (inferred)' if t.inferred else ''}; {len(t.shapes)} shapes, {wins} wins, "
          f"{len(t.rows)} row{'s' if len(t.rows) != 1 else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="calibration results directories to file")
    ap.add_argument("--force", action="store_true", help="replace a chip's earlier filed calibration")
    ap.add_argument("--allow-partial", action="store_true", help="file a table measured on part of the grid")
    ap.add_argument("--any-cores", action="store_true",
                    help="file a table with no GPU core count as the chip's table for every core count")
    ap.add_argument("--regenerate", action="store_true", help="only rebuild src/dispatch_table.h from benchmarks/calibration/")
    ap.add_argument("--check", action="store_true", help="with --regenerate: exit 1 if the header is stale, change nothing")
    args = ap.parse_args()
    if args.regenerate == bool(args.dirs):
        ap.error("give directories to file, or --regenerate")
    if args.check and not args.regenerate:
        ap.error("--check goes with --regenerate (it changes nothing)")
    try:
        for d in args.dirs:
            ingest(d, args.force, args.allow_partial, args.any_cores)
        regenerate(args.check and args.regenerate)
    except Ingest as e:
        sys.exit(f"ingest_calibration: {e}")


if __name__ == "__main__":
    main()
