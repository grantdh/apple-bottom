#!/usr/bin/env python3
"""Files an AB_SPLIT table with its evidence as a chip's built-in split table and regenerates src/split_table.h.

    scripts/ingest_split.py DIR [DIR ...] [--force]
    scripts/ingest_split.py --regenerate [--check]

DIR holds split_table.txt (the "# chip:", "# gpu_cores:", "# cpu_matrix:", "# library:", "# date:"
and "# method:" keys, then rows in the format of src/dispatch.h) and split_evidence.csv, one line per
measurement:

    row,source,input,layout,library,share,pairs,split_s,base_s,ratio,ratio_lo,ratio_hi,calls,kept,kept_4m,range,gpu_failed

row is the table row's index from 0; source is pw.x or bench_split; layout is ranks x threads (8x1);
split_s and base_s are the row's times split and on Accelerate, excluding each process's first split
call; ratio_lo and ratio_hi bound the paired interval of base_s / split_s. The rules (also checked by
tests/test_dispatch.c on the built-in tables):
  1. every row has evidence, all of it from the table's library and at the row's share;
  2. every evidence line has ratio_lo >= 1.00;
  3. each row's pooled ratio (sum of base_s / sum of split_s) is at least 1.10;
  4. a square row (a ratio= bound of at most 2) has bench_split evidence at 1x8 and 4x2 (well-scaled
     synthetic data, which the generated header labels as such), and pw.x evidence alone does not do
     for it; every other row has pw.x evidence for inputs A and B at 8x1, whatever else it has; a row
     with pw.x evidence has it for both A and B at 8x1, and one with bench_split evidence has both
     layouts; no other source counts;
  5. no row has scalars=a (or leaves scalars out).
Counts must be ASCII digits and every other number finite in NUMBER's grammar. Each line of the CSV is
split on every ',' (no quoting: a '"' is refused), ends in \n or \r\n (no other '\r'), is UTF-8 of at
most MAX_LINE bytes and has the 17 fields; tests/test_dispatch.c reads it the same way.
A table without rows passes. A table is filed under benchmarks/split/tables/<slug>/ (the chip without
"Apple " plus the GPU core count: M2-Max-38c), and src/split_table.h is regenerated from every filed
directory, in slug order. Commit both. A chip's earlier table is replaced only with --force; nothing
under benchmarks/calibration/ is touched.

--regenerate rebuilds the header from the filed directories alone; with --check it rebuilds in memory
and exits 1 if the committed header differs (make split-table-check, run by ci-local and CI). It also
warns, without failing, when src/ (split_table.h aside) changed since a table's library commit: its
shares were measured on other code (docs/RELEASING.md makes that a pre-tag check).
Python 3.9, standard library only.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(REPO, "benchmarks", "split", "tables")
HEADER = os.path.join(REPO, "src", "split_table.h")

MIN_LO = 1.00               # rule 2
MIN_POOLED = 1.10           # rule 3
SQUARE_RATIO = 2.0          # rule 4: a row with ratio <= this is square
KEYS = ("chip", "gpu_cores", "cpu_matrix", "library", "date", "method")
COLUMNS = ("row,source,input,layout,library,share,pairs,split_s,base_s,ratio,ratio_lo,ratio_hi,"
           "calls,kept,kept_4m,range,gpu_failed").split(",")
MAX_ROWS = 32               # AB_SPLIT_MAX_ROWS

KEY_RE = re.compile(r"^\s*#\s*([a-z][a-z0-9_]*):\s*([^#]*?)\s*(?:#.*)?$")      # "# chip: Apple M2 Max  # note"


class Ingest(Exception):
    pass


def chip_slug(chip, cores):
    """As dispatch.c's ab_dispatch_chip_slug() and scripts/ingest_calibration.py."""
    s = re.sub(r"^Apple\s*", "", chip or "", flags=re.I)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-") or "unknown"
    return f"{s}-{cores}c" if cores and s != "unknown" else s


def split_trans(routine, t):
    t = t.upper()
    return "T" if routine == "d" and t == "C" else t


DIGITS = re.compile(r"[0-9]+", re.ASCII)
# A number in a split row or the evidence: ASCII digits, an optional point, an optional exponent with
# its sign, and an optional leading '+'; finite. dispatch.c's parse_double() takes the same (strtod_l
# in the C locale over the characters 0-9 + - . e E, no leading '-') and tests/test_dispatch.c's
# checker the same characters; table values must also not be negative or subnormal (parse_number).
NUMBER = re.compile(r"\+?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?", re.ASCII)
MAX_LINE = 1000          # bytes of a CSV line, its end aside (tests/test_dispatch.c: the same)


def csv_fields(raw, where):
    """A line of split_evidence.csv (bytes, without its "\\n") split as tests/test_dispatch.c splits it:
    on every ','. A line end may be "\\r\\n"; any other '\\r', a '"' (no quoting), a line over MAX_LINE
    bytes or one that is not UTF-8 is refused."""
    if raw.endswith(b"\r"):
        raw = raw[:-1]
    if b"\r" in raw or b'"' in raw or len(raw) > MAX_LINE:
        raise Ingest(f"{where}: a quote, a bare carriage return or more than {MAX_LINE} bytes")
    return utf8(raw, where).split(",")


def utf8(raw, where):
    """raw (a line's bytes) as text; an Ingest error naming the line if it is not UTF-8."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise Ingest(f"{where}: not UTF-8") from None


def byte_lines(path):
    """path's lines as bytes, split after each "\\n" as fgets() splits them (the last may lack one)."""
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    return lines[:-1] if data.endswith(b"\n") else lines


def parse_int_range(v):
    lo, sep, hi = v.partition(":")
    if not sep or (lo and not DIGITS.fullmatch(lo)) or (hi and not DIGITS.fullmatch(hi)):
        raise ValueError(v)
    lo, hi = int(lo or 0), int(hi or 0)
    if lo > 2**31 - 1 or hi > 2**31 - 1 or (hi and hi < lo):
        raise ValueError(v)
    return lo, hi


def finite(v):
    """v as a finite float in the NUMBER grammar; ValueError otherwise."""
    if not NUMBER.fullmatch(v):
        raise ValueError(v)
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(v)
    return x


def parse_number(v):
    """A table value: finite, not negative, and 0 or at least DBL_MIN. dispatch.c's strtod_l sets ERANGE
    for a subnormal value and for nonzero digits that underflow to 0, and refuses both."""
    x = finite(v)
    if x < 0 or v.startswith("-"):
        raise ValueError(v)
    if 0 < x < sys.float_info.min or (x == 0 and re.search("[1-9]", re.split("[eE]", v)[0])):
        raise ValueError(v)
    return x


def parse_float_range(v):
    lo, sep, hi = v.partition(":")
    if not sep:
        raise ValueError(v)
    lo, hi = parse_number(lo) if lo else 0.0, parse_number(hi) if hi else 0.0
    if hi and hi < lo:
        raise ValueError(v)
    return lo, hi


def parse_row(line):
    """A row as dispatch.c's ab_split_parse_row() reads it: a dict, None for a blank or comment line;
    ValueError if malformed."""
    tok = line.split("#", 1)[0].split()
    if not tok:
        return None
    if len(tok) < 5 or tok[0].lower() not in ("dgemm", "zgemm"):
        raise ValueError(line)
    r = {"routine": tok[0][0].lower(), "m": (0, 0), "n": (0, 0), "k": (0, 0), "mn": (0.0, 0.0),
         "ratio": 0.0, "gf": 0.0, "scalars": "a"}
    if any(len(t) != 1 for t in tok[1:4]) or tok[1].upper() not in "NTC" or tok[2].upper() not in "NTC" \
            or tok[3].upper() not in "NMK":
        raise ValueError(line)
    r["ta"], r["tb"], r["axis"] = split_trans(r["routine"], tok[1]), split_trans(r["routine"], tok[2]), tok[3].upper()
    seen = set()
    for t in tok[4:]:
        key, eq, v = t.partition("=")
        if not eq or key in seen or key not in ("m", "n", "k", "mn", "ratio", "gf", "scalars", "f"):
            raise ValueError(line)
        seen.add(key)
        if key in ("m", "n", "k"):
            r[key] = parse_int_range(v)
        elif key == "mn":
            r[key] = parse_float_range(v)
        elif key == "ratio":
            r[key] = parse_number(v)
            if r[key] < 1:
                raise ValueError(line)
        elif key == "gf":
            r[key] = parse_number(v)
        elif key == "scalars":
            if v not in ("u", "z", "a"):
                raise ValueError(line)
            r[key] = v
        else:
            r["f"] = parse_number(v)
            if r["f"] > 1:
                raise ValueError(line)
    if "f" not in seen:
        raise ValueError(line)
    r["text"] = " ".join(tok)
    return r


def c_number(x):
    """A double as a C literal that reads back as the same value."""
    s = repr(float(x))
    return s if any(ch in s for ch in ".e") or s in ("inf", "nan") else s + ".0"


def c_row(r):
    return (f"{{'{r['routine']}', '{r['ta']}', '{r['tb']}', '{r['axis']}', {r['m'][0]}, {r['m'][1]}, "
            f"{r['n'][0]}, {r['n'][1]}, {r['k'][0]}, {r['k'][1]}, {c_number(r['mn'][0])}, "
            f"{c_number(r['mn'][1])}, {c_number(r['ratio'])}, {c_number(r['gf'])}, '{r['scalars']}', "
            f"{c_number(r['f'])}}}")


class Table:
    """One split_table.txt with its split_evidence.csv."""

    def __init__(self, directory):
        self.dir = directory
        self.txt = os.path.join(directory, "split_table.txt")
        self.csv = os.path.join(directory, "split_evidence.csv")
        for p in (self.txt, self.csv):
            if not os.path.isfile(p):
                raise Ingest(f"{directory}: no {os.path.basename(p)}")
        self.keys, self.rows = {}, []
        for lineno, raw in enumerate(byte_lines(self.txt), 1):
            line = utf8(raw, f"{self.txt}:{lineno}")
            m = KEY_RE.match(line)
            if m:
                self.keys[m.group(1)] = m.group(2)
                continue
            try:
                r = parse_row(line)
            except ValueError:
                raise Ingest(f"{self.txt}:{lineno}: not a split row or a comment: {line.rstrip()!r}")
            if r:
                self.rows.append(r)
        missing = [k for k in KEYS if not self.keys.get(k)]
        if missing:
            raise Ingest(f"{self.txt}: no {', '.join('# ' + k + ':' for k in missing)} key")
        if len(self.rows) > MAX_ROWS:
            raise Ingest(f"{self.txt}: {len(self.rows)} rows; the library takes at most {MAX_ROWS}")
        self.chip = self.keys["chip"]
        cores = self.keys["gpu_cores"]
        if not cores.isdigit() or int(cores) == 0:
            raise Ingest(f"{self.txt}: '# gpu_cores: {cores}' is not a core count; a split table is "
                         f"measured on one variant of a chip")
        self.gpu_cores = int(cores)
        self.cpu_matrix = self.keys["cpu_matrix"]
        self.library = self.keys["library"]
        if not re.fullmatch(r"[0-9a-f]{7,40}|none", self.library):
            raise Ingest(f"{self.txt}: '# library: {self.library}' is not a commit (or 'none' for a table "
                         f"without rows)")
        if self.library == "none" and self.rows:
            raise Ingest(f"{self.txt}: rows need the library commit they were measured on")
        self.slug = chip_slug(self.chip, self.gpu_cores)
        self.evidence = []
        for n, raw in enumerate(byte_lines(self.csv), 1):
            fields = csv_fields(raw, f"{self.csv}:{n}")
            if n == 1:
                if fields != COLUMNS:
                    raise Ingest(f"{self.csv}: the first line must be {','.join(COLUMNS)}")
                continue
            if len(fields) != len(COLUMNS):
                raise Ingest(f"{self.csv}:{n}: {len(fields)} fields, expected {len(COLUMNS)}")
            e = dict(zip(COLUMNS, fields))
            e["line"] = n
            self.evidence.append(e)

    def validate(self):
        """Rules 1-5 of the docstring."""
        by_row = {i: [] for i in range(len(self.rows))}
        self.pooled, self.label = {}, {}
        for e in self.evidence:
            where = f"{self.csv}:{e['line']}"
            for k in ("row", "pairs", "calls", "kept", "kept_4m", "range", "gpu_failed"):
                if not DIGITS.fullmatch(e[k]):
                    raise Ingest(f"{where}: {k} = {e[k]!r} is not a count (ASCII digits)")
            for k in ("share", "split_s", "base_s", "ratio", "ratio_lo", "ratio_hi"):
                try:
                    finite(e[k])
                except ValueError:
                    raise Ingest(f"{where}: {k} = {e[k]!r} is not a finite number")
            i = int(e["row"])
            share, lo = float(e["share"]), float(e["ratio_lo"])
            split_s, base_s = float(e["split_s"]), float(e["base_s"])
            if i not in by_row:
                raise Ingest(f"{where}: row {i}: the table has {len(self.rows)} rows")
            r = self.rows[i]
            if e["library"] != self.library:
                raise Ingest(f"{where}: library {e['library']}, the table's is {self.library} (rule 1)")
            if abs(share - r["f"]) > 1e-9:
                raise Ingest(f"{where}: share {share}, row {i} has f={r['f']} (rule 1)")
            if lo < MIN_LO:
                raise Ingest(f"{where}: ratio_lo {lo} < {MIN_LO:.2f} (rule 2)")
            if not (split_s > 0 and base_s > 0):
                raise Ingest(f"{where}: split_s and base_s must be positive")
            if e["source"] not in ("pw.x", "bench_split"):
                raise Ingest(f"{where}: source {e['source']!r} is neither pw.x nor bench_split (rule 4)")
            by_row[i].append(e)
        for i, r in enumerate(self.rows):
            lines, name = by_row[i], f"row {i} ({r['text']})"
            if not lines:
                raise Ingest(f"{self.txt}: {name} has no evidence (rule 1)")
            pooled = sum(float(e["base_s"]) for e in lines) / sum(float(e["split_s"]) for e in lines)
            if not (math.isfinite(pooled) and pooled >= MIN_POOLED):
                raise Ingest(f"{self.txt}: {name}: pooled ratio {pooled:.3f} < {MIN_POOLED:.2f} (rule 3)")
            have = {(e["source"], e["input"], e["layout"]) for e in lines}
            sources = {e["source"] for e in lines}
            square = 0 < r["ratio"] <= SQUARE_RATIO
            if "pw.x" in sources or not square:
                for inp in ("A", "B"):
                    if ("pw.x", inp, "8x1") not in have:
                        raise Ingest(f"{self.txt}: {name}: no pw.x evidence on input {inp} at 8x1 (rule 4)")
            if "bench_split" in sources or square:
                for layout in ("1x8", "4x2"):
                    if not any(s == "bench_split" and lay == layout for s, _, lay in have):
                        raise Ingest(f"{self.txt}: {name}: no bench_split evidence at {layout} (rule 4)")
            if r["scalars"] == "a":
                raise Ingest(f"{self.txt}: {name}: scalars=a (or none) is not allowed in a built-in table (rule 5)")
            self.pooled[i] = pooled
            self.label[i] = ("pw.x A/B 8x1" + (" and bench_split" if "bench_split" in sources else "")
                             if "pw.x" in sources else "bench_split only: well-scaled synthetic data")
        self.nlines = {i: len(by_row[i]) for i in by_row}


def filed_tables():
    out = []
    if os.path.isdir(TABLES):
        for name in sorted(os.listdir(TABLES)):
            d = os.path.join(TABLES, name)
            if os.path.isfile(os.path.join(d, "split_table.txt")):
                t = Table(d)
                if t.slug != name:
                    raise Ingest(f"{d}: directory name differs from the table's own slug {t.slug}")
                out.append(t)
    return out


def generate(tables):
    L = ["// =============================================================================",
         "// split_table.h — built-in AB_SPLIT tables, one per chip with filed evidence (dispatch.h)",
         "// =============================================================================",
         "// Generated by scripts/ingest_split.py from benchmarks/split/tables/; do not",
         "// edit. The script files a table (split_table.txt) with its measurements",
         "// (split_evidence.csv) after checking its rules, and regenerates this header;",
         "// commit the two together. `make split-table-check` verifies that they agree,",
         "// and tests/test_dispatch.c checks each table against its evidence.",
         "//",
         "// With AB_SPLIT=1 a dgemm_/zgemm_ call that matches a row is split between",
         "// Accelerate and the GPU (dispatch.h has the row format); AB_SPLIT_TABLE",
         "// replaces the table. A chip without a table splits nothing.",
         "//",
         "// Tables (slug: chip, GPU cores, Accelerate's matrix unit, library, rows):"]
    for t in tables:
        L.append(f"//   {t.slug}: {t.chip}, {t.gpu_cores} cores, {t.cpu_matrix}, {t.library}, "
                 f"{len(t.rows)} row{'s' if len(t.rows) != 1 else ''}")
    if not tables:
        L.append("//   (none filed: AB_SPLIT=1 splits only by AB_SPLIT_TABLE)")
    L += ["// =============================================================================",
          "#ifndef AB_SPLIT_TABLE_H", "#define AB_SPLIT_TABLE_H", "", '#include "dispatch.h"', ""]
    for t in tables:
        ident = re.sub(r"[^A-Za-z0-9]", "_", t.slug).upper()
        L.append("// -----------------------------------------------------------------------------")
        L.append(f"// {t.slug}: {t.chip}, {t.gpu_cores}-core GPU, Accelerate on {t.cpu_matrix}")
        L.append(f"// Library:  {t.library}")
        L.append(f"// Date:     {t.keys['date']}")
        L.append(f"// Method:   {t.keys['method']}")
        L.append("// -----------------------------------------------------------------------------")
        L.append(f"static const ab_split_row_t AB_SPLIT_ROWS_{ident}[] = {{")
        for i, r in enumerate(t.rows):
            L.append(f"    {c_row(r)},   // {r['text']}: {t.label[i]}, pooled {t.pooled[i]:.2f}x, "
                     f"{t.nlines[i]} line{'s' if t.nlines[i] != 1 else ''}")
        L.append("    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}," +
                 ("" if t.rows else "   // no rows: nothing is split"))
        L += ["};", ""]
    L += ["// chip, gpu_cores, cpu_matrix, library, data (tests/test_dispatch.c checks each",
          "// table against its evidence, relative to the source root), rows",
          "static const ab_split_table_t AB_SPLIT_BUILTIN_TABLES[] = {"]
    for t in tables:
        ident = re.sub(r"[^A-Za-z0-9]", "_", t.slug).upper()
        data = f"benchmarks/split/tables/{t.slug}/split_evidence.csv"
        L.append(f'    {{"{t.chip}", {t.gpu_cores}, "{t.cpu_matrix}", "{t.library}", "{data}", '
                 f"AB_SPLIT_ROWS_{ident}}},")
    L += ["    {NULL, 0, NULL, NULL, NULL, NULL},", "};", "", "#endif // AB_SPLIT_TABLE_H", ""]
    return "\n".join(L)


def stale_warnings(tables):
    """Tables with rows whose library commit is not the code in src/ now (warnings, not failures)."""
    for t in tables:
        if not t.rows:
            continue
        try:
            r = subprocess.run(["git", "-C", REPO, "diff", "--quiet", t.library, "HEAD", "--", "src",
                                ":!src/split_table.h"], capture_output=True)
        except OSError:
            print(f"ingest_split: warning: no git; cannot check {t.slug} against its library {t.library}")
            continue
        if r.returncode == 1:
            print(f"ingest_split: warning: src/ changed since {t.slug}'s library {t.library}; its shares were "
                  f"measured on other code")
        elif r.returncode != 0:
            why = (r.stderr.decode(errors="replace").strip().splitlines() or ["git diff failed"])[0]
            print(f"ingest_split: warning: cannot compare {t.slug}'s library {t.library} with HEAD ({why})")


def regenerate(check):
    tables = filed_tables()
    for t in tables:
        t.validate()
    text = generate(tables)
    old = open(HEADER, encoding="utf-8").read() if os.path.isfile(HEADER) else ""
    n = f"{len(tables)} table{'s' if len(tables) != 1 else ''}"
    if check:
        stale_warnings(tables)
        if text != old:
            sys.exit(f"ingest_split: {os.path.relpath(HEADER, REPO)} differs from benchmarks/split/tables/ ({n}); "
                     f"run `make split-table` and commit it")
        print(f"{os.path.relpath(HEADER, REPO)} matches benchmarks/split/tables/ ({n})")
        return
    if text != old:
        with open(HEADER, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {os.path.relpath(HEADER, REPO)}: {n} ({', '.join(t.slug for t in tables) or 'none'})")
    else:
        print(f"{os.path.relpath(HEADER, REPO)} is up to date ({n})")


def ingest(directory, force):
    t = Table(directory)
    t.validate()
    dest = os.path.join(TABLES, t.slug)
    if os.path.realpath(directory) == os.path.realpath(dest):
        raise Ingest(f"{os.path.relpath(dest, REPO)} is the filed directory itself; use --regenerate")
    if os.path.isdir(dest) and not force:
        raise Ingest(f"{os.path.relpath(dest, REPO)} exists (an earlier table of this chip); --force replaces it")
    # Build the replacement beside the old directory, then swap: a failure leaves the old one.
    staging = dest + ".new"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    shutil.copy(t.txt, os.path.join(staging, "split_table.txt"))
    shutil.copy(t.csv, os.path.join(staging, "split_evidence.csv"))
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.rename(staging, dest)
    print(f"filed {os.path.relpath(dest, REPO)}: {t.chip}, {t.gpu_cores} GPU cores, Accelerate on {t.cpu_matrix}, "
          f"library {t.library}; {len(t.rows)} row{'s' if len(t.rows) != 1 else ''}, "
          f"{len(t.evidence)} evidence line{'s' if len(t.evidence) != 1 else ''}")


def self_test():
    """Files made-up tables and checks each verdict: filed, or refused for the expected reason (a
    substring of the error). tests/test_dispatch.c's split_evidence_why() runs the same cases in C.
    Also checks generate()'s evidence labels. Returns the number of wrong verdicts."""
    def head(r0="scalars=u f=0.45", r1="ratio=2 scalars=z f=0.6"):
        return ("# chip: Apple M2 Max\n# gpu_cores: 38\n# cpu_matrix: AMX\n# library: abc1234\n# date: x\n"
                f"# method: self-test\nzgemm N N N m=8192: {r0}\nzgemm N N N m=2048: {r1}\n")
    pw = ("0,pw.x,A,8x1,abc1234,0.45,6,80,95,1.19,1.12,1.25,100,100,0,0,0\n"
          "0,pw.x,B,8x1,abc1234,0.45,6,70,80,1.14,1.08,1.20,100,99,0,1,0\n")
    sq = ("1,bench_split,sq,1x8,abc1234,0.6,20,1.0,1.3,1.3,1.2,1.4,20,20,0,0,0\n"
          "1,bench_split,sq,4x2,abc1234,0.6,20,1.0,1.2,1.2,1.1,1.3,20,20,0,0,0\n")
    pw1 = pw.replace("0,pw.x", "1,pw.x").replace("0.45", "0.6")      # pw.x evidence for row 1
    qe_bench = sq.replace("1,bench", "0,bench").replace("0.6", "0.45")
    a_line = pw.splitlines(True)[0]
    cases = [  # (what, table, evidence after the header line, None if filed, else the reason)
        ("pw.x for the QE row, bench_split for the square", head(), pw + sq, None),
        ("CRLF line ends", head(), (pw + sq).replace("\n", "\r\n"), None),
        ("the square also with pw.x", head(), pw + sq + pw1, None),
        ("rule 1: a row without evidence", head(), pw, "row 1"),
        ("rule 1: another library", head(), pw.replace("abc1234", "abc9999", 1) + sq, "library"),
        ("rule 1: another share", head(), pw.replace("0.45", "0.50", 1) + sq, "share"),
        ("rule 2: ratio_lo below 1", head(), pw.replace("1.12", "0.99") + sq, "rule 2"),
        ("rule 3: pooled below 1.10", head(), pw.replace(",80,95,", ",80,84,") + sq, "rule 3"),
        ("rule 3: pooled overflows", head(), pw.replace(",80,95,", ",80,1e308,").replace(",70,80,", ",70,1e308,") + sq,
         "rule 3"),
        ("rule 3: subnormal split_s", head(), pw.replace(",80,95,", ",1e-310,95,").replace(",70,80,", ",1e-310,80,") + sq,
         "rule 3"),
        ("rule 4: a QE-shaped row on bench_split alone", head(), qe_bench + sq, "no pw.x evidence on input A"),
        ("rule 4: a QE-shaped row with pw.x A only", head(), qe_bench + a_line + sq, "no pw.x evidence on input B"),
        ("rule 4: ratio=300 is not square", head(r1="ratio=300 scalars=z f=0.6"), pw + sq,
         "no pw.x evidence on input A"),
        ("rule 4: a square on pw.x alone", head(), pw + pw1, "no bench_split evidence at 1x8"),
        ("rule 4: another source", head(), pw.replace("pw.x,A", "replay,A") + sq, "source"),
        ("rule 5: scalars=a", head(r0="f=0.45"), pw + sq, "rule 5"),
        ("a NaN share", head(), pw.replace("0.45,6,80", "nan,6,80") + sq, "share"),
        ("a NaN split_s", head(), pw.replace(",80,95,", ",nan,95,") + sq, "split_s"),
        ("a NaN base_s", head(), pw.replace(",80,95,", ",80,nan,") + sq, "base_s"),
        ("a NaN ratio_lo", head(), pw.replace("1.12", "nan") + sq, "ratio_lo"),
        ("an infinite split_s", head(), pw.replace(",80,95,", ",inf,95,") + sq, "split_s"),
        ("an overflowing base_s", head(), pw.replace(",80,95,", ",80,1e999,") + sq, "base_s"),
        ("a hex float", head(), pw.replace(",80,95,", ",80,0x5f,") + sq, "base_s"),
        ("a negative base_s", head(), pw.replace(",80,95,", ",80,-95,") + sq, "base_s"),
        ("a non-ASCII digit in a number", head(), pw.replace(",80,95,", ",80,9٥,") + sq, "base_s"),
        ("a non-ASCII digit in a count", head(), pw.replace(",100,100,", ",100,1٠٠,", 1) + sq, "kept"),
        ("a count with a point", head(), pw.replace(",6,80,", ",6.0,80,") + sq, "pairs"),
        ("a row index beyond int", head(), pw.replace("0,pw.x,A", "2147483648,pw.x,A") + sq, "the table has"),
        ("a short line", head(), pw + sq.replace(",0,0,0\n", ",0,0\n", 1), "fields, expected 17"),
        ("a blank line", head(), "\n" + pw + sq, "fields, expected 17"),
        ("a quoted field", head(), pw.replace("pw.x,A", '"pw.x",A') + sq, "quote"),
        ("a bare carriage return", head(), pw.replace("pw.x,A", "pw.x\r,A") + sq, "carriage return"),
        ("a line over the limit", head(), pw.replace(",A,", "," + "A" * 1100 + ",") + sq, "bytes"),
        ("600 two-byte characters", head(), pw.replace(",A,", "," + "\u00e9" * 600 + ",") + sq, "bytes"),
        # a Latin-1 byte, an overlong '/', a surrogate (bytes written through surrogateescape)
        ("a Latin-1 byte in the evidence", head(), pw.replace("pw.x,A", "pw.x,A\udce9") + sq, "not UTF-8"),
        ("an overlong form", head(), pw.replace("pw.x,A", "pw.x,A\udcc0\udcaf") + sq, "not UTF-8"),
        ("a surrogate", head(), pw.replace("pw.x,A", "pw.x,A\udced\udca0\udc80") + sq, "not UTF-8"),
        ("a Latin-1 byte in the table", head().replace("# date: x", "# date: x\udce9"), pw + sq, "not UTF-8"),
    ]
    # generate()'s evidence labels for the rows of a filed case
    labels = {"pw.x for the QE row, bench_split for the square":
              ("pw.x A/B 8x1, pooled", "bench_split only: well-scaled synthetic data, pooled"),
              "the square also with pw.x": ("pw.x A/B 8x1, pooled", "pw.x A/B 8x1 and bench_split, pooled")}
    wrong = 0
    # the row grammar of dispatch.c's ab_split_parse_row(): signed exponents, no '-' value, no hex
    r = parse_row("zgemm N N N gf=1e-3 mn=2.5e-1:+4E+2 f=5e-1")
    if (r["gf"], r["mn"], r["f"]) != (1e-3, (0.25, 400.0), 0.5):
        wrong += 1
        print(f"ingest_split self-test: signed exponents read as {r}")
    # 0 and DBL_MIN, but no subnormal value or nonzero digits that underflow to 0 (strtod_l's ERANGE)
    r = parse_row("zgemm N N N gf=0e-400 mn=2.2250738585072014e-308: f=0")
    if (r["gf"], r["mn"]) != (0.0, (sys.float_info.min, 0.0)):
        wrong += 1
        print(f"ingest_split self-test: 0e-400 and DBL_MIN read as {r}")
    for bad in ("zgemm N N N f=-0", "zgemm N N N f=0x1p-1", "zgemm N N N f=nan", "zgemm N N N gf=1e999 f=0.5",
                "zgemm N N N gf=1e-310 f=0.5", "zgemm N N N mn=1e-400: f=0.5",
                "zgemm N N N gf=2.2250738585072e-308 f=0.5"):
        try:
            parse_row(bad)
            wrong += 1
            print(f"ingest_split self-test: '{bad}' parsed")
        except ValueError:
            pass
    with tempfile.TemporaryDirectory() as d:
        for what, table, body, reason in cases:
            with open(os.path.join(d, "split_table.txt"), "w", encoding="utf-8", errors="surrogateescape") as f:
                f.write(table)
            with open(os.path.join(d, "split_evidence.csv"), "w", encoding="utf-8", errors="surrogateescape",
                      newline="") as f:
                f.write(",".join(COLUMNS) + ("\r\n" if "\r\n" in body else "\n") + body)
            try:
                t = Table(d)
                t.validate()
                why = None
            except Ingest as e:
                why = str(e)
            if (reason is None) != (why is None) or (reason and reason not in why):
                wrong += 1
                print(f"ingest_split self-test: {what}: {'filed' if why is None else 'refused: ' + why}"
                      f" (expected {'filing' if reason is None else 'refusal: ' + reason})")
            elif why is None and what in labels:
                text = generate([t])
                for label in labels[what]:
                    if label not in text:
                        wrong += 1
                        print(f"ingest_split self-test: {what}: generate() does not label a row '{label}'")
    return wrong


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="directories with split_table.txt and split_evidence.csv to file")
    ap.add_argument("--force", action="store_true", help="replace a chip's earlier filed table")
    ap.add_argument("--regenerate", action="store_true", help="only rebuild src/split_table.h from benchmarks/split/tables/")
    ap.add_argument("--check", action="store_true", help="with --regenerate: exit 1 if the header is stale, change nothing")
    ap.add_argument("--self-test", action="store_true", help="check the filing rules on made-up tables, change nothing")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(1 if self_test() else 0)
    if args.regenerate == bool(args.dirs):
        ap.error("give directories to file, or --regenerate")
    if args.check and not args.regenerate:
        ap.error("--check goes with --regenerate (it changes nothing)")
    try:
        for d in args.dirs:
            ingest(d, args.force)
        regenerate(args.check and args.regenerate)
    except Ingest as e:
        sys.exit(f"ingest_split: {e}")


if __name__ == "__main__":
    main()
