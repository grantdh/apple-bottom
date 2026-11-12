#!/usr/bin/env python3
"""The next GPU share per row of an AB_SPLIT table, from the reports of runs made with it.

Each RUN is an err.txt file or a directory searched for them (set-aside .bad*/incomplete-* folders,
tmp and *.save are skipped; a file found twice counts once): the stderr of programs run with
AB_SPLIT=1 AB_SPLIT_TABLE=TABLE AB_REPORT=1, such as pw.x's every rank. Choose the runs to fit from
your campaign's ledger (failed or flagged runs left out); this script checks only that they belong
together. Its AB_REPORT headers name the table as split_table=file:PATH@DIGEST (the first 12 hex
digits of the SHA-256 of the bytes the library read): a file counts only if DIGEST is TABLE's now. A
header without a digest (an older library) is matched by PATH, read relative to the file's directory
(where the run started), with a note. Other files are skipped, with a note if they name TABLE's path.
The fit stops if a report's rows differ from TABLE's (count, routine, axis or share), or if the
reports come from more than one build of the library; a row reported by fewer ranks than printed
headers is noted.

For each row, its split_row lines (every rank of every run) give the seconds and the GFLOP of the
calls' CPU and GPU parts. Their ratios are the rates the two engines reached while running together
(under the shared power budget, with the other ranks' work): r_cpu = gf_cpu / cpu_s and r_gpu =
gf_gpu / gpu_s. The share at which both parts of a call would end together is f_bal = r_gpu /
(r_cpu + r_gpu). The next share moves the fraction D (default 0.7) of the way from the current one
to f_bal, rounded to 0.01; below F (default 0.08) the row goes to Accelerate (0), and a row at 0 stays
there. A row with fewer than N (default 20) split calls takes the pooled rates of the other non-square
rows with its routine and axis; a row with none keeps its share, and so does a square row (ratio <=
2, scripts/ingest_split.py's rule 4), whose share is fitted from bench_split's grid. This is
.dev/active/split/design.md section 8's rule (a port of Espressivo's calibrate.py).

The part sums include each process's first split call, whose GPU part carries the Metal start-up;
the evidence leaves it out (scripts/ingest_split.py). The library prints the parts of each row's
first split call in the process (first_cpu_ms, first_gpu_ms, first_gf_cpu, first_gf_gpu), and
--exclude-first (the default) leaves that call out of each rank line's rates and split count: for one
row of each process it is the start-up, for the others an ordinary call. Reports without the parts
(an older library) keep their first calls, with a note.

Columns: dfirst is how far leaving the first calls out moved f_bal; first/gpu is the first calls' GPU
parts (their wall times, for an older library) over the row's GPU-part time, so a row dominated by
them shows; wait g/c (every call) above 1 means the GPU part ended first more of the time (it waited
for the CPU's head), which f_bal > f also says.

Prints TABLE with the new shares on stdout (only changed rows rewritten; "# method:" and "# date:"
set, citing the build and TABLE's sha256), and the estimates on stderr. --cut ROW=L,... adds the GPU
part at the old and new shares for those axis lengths (src/split_slice.h's ab_split_cut, with the
alignments read from that header and AB_MIN_GPU_DIM's default from src/dispatch.h), since a share
that does not move the cut changes nothing. --self-test checks that cut against tests/test_dispatch.c.
Python 3.9, standard library only.
"""
import argparse
import datetime
import hashlib
import math
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "qe"))
from ingest_split import KEY_RE, SQUARE_RATIO, parse_row, table_line  # noqa: E402
from analyze import AB_HEADER, AB_SPLIT_ROW  # noqa: E402

KV = re.compile(r"(\w+)=(\S+)")
SUMS = ("calls", "split", "cpu_ms", "gpu_ms", "gf_cpu", "gf_gpu", "first_ms", "wait_cpu_ms", "wait_gpu_ms")
PARTS = ("cpu_ms", "gpu_ms", "gf_cpu", "gf_gpu")   # first_<part>: those of the row's first split call
SKIP_DIR = re.compile(r"(\.bad\d*|^incomplete-.*|^tmp|\.save)$")


def c_define(path, name):
    m = re.search(rf"^#define {name}\s+(\d+)", open(os.path.join(ROOT, path)).read(), re.M)
    if not m:
        sys.exit(f"fit_shares: no #define {name} in {path}")
    return int(m.group(1))


ALIGN_NM = c_define("src/split_slice.h", "AB_SPLIT_ALIGN_NM")
ALIGN_K = c_define("src/split_slice.h", "AB_SPLIT_ALIGN_K")
MIN_DIM = c_define("src/dispatch.h", "AB_DISPATCH_DEFAULT_MIN_DIM")


def cut(axis, L, f, min_dim):
    """ab_split_cut: the GPU's part of an axis of length L at share f (0: not cut)."""
    a = ALIGN_K if axis == "K" else ALIGN_NM
    if not f > 0 or L < 2 * a:
        return 0
    g = min(max(math.floor(f * L / a + 0.5) * a, a), L - a)   # lround(f * L / a) * a, f * L >= 0
    if g <= min_dim:
        g = (min_dim // a + 1) * a
        if L - g < a:
            return 0
    return g


def table_label(value):
    """(PATH, DIGEST) of a split_table=file: value; DIGEST None unless it ends in '@' and 12 hex digits."""
    m = re.fullmatch(r"(.*)@([0-9a-f]{12})", value)
    return m.groups() if m else (value, None)


def self_test():
    """cut() against tests/test_dispatch.c's test_split_cut vectors, and table_label() on labels."""
    src = open(os.path.join(ROOT, "tests", "test_dispatch.c")).read()
    body = src[src.index("test_split_cut(void)"):]
    vec = re.findall(r"\{'([NMK])', ([^,]+), ([-0-9.]+), ([^,]+), ([^}]+)\}", body[:body.index("begin();")])

    def ints(s):   # an int literal, INT_MAX, or INT_MAX - n
        s = s.strip()
        return 2**31 - 1 - int(s.split("-", 1)[1] if "-" in s else 0) if s.startswith("INT_MAX") else int(s)
    labels = {"/a/t.txt@0123456789ab": ("/a/t.txt", "0123456789ab"), "/a@b/t.txt": ("/a@b/t.txt", None),
              "/a/t@x@0123456789ab": ("/a/t@x", "0123456789ab"), "t@0123456789abc": ("t@0123456789abc", None),
              "t@0123456789AB": ("t@0123456789AB", None), "t": ("t", None)}

    def agree(what, n, bad):
        print(f"fit_shares: {what}: {'all agree' if n and not bad else f'{len(bad)} differ: {bad}'}")
        return bool(n and not bad)
    ok = agree(f"cut() on {len(vec)} vectors from tests/test_dispatch.c", len(vec),
               [v for v in vec if cut(v[0], ints(v[1]), float(v[2]), ints(v[3])) != ints(v[4])])
    ok &= agree(f"table_label() on {len(labels)} labels", len(labels),
                [v for v, want in labels.items() if table_label(v) != want])
    return 0 if ok else 1


def err_files(paths):
    seen = set()
    for p in paths:
        found = [p] if os.path.isfile(p) else []
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if not SKIP_DIR.search(d)]
            if "err.txt" in files:
                found.append(os.path.join(root, "err.txt"))
                dirs[:] = []
        for f in found:
            if os.path.realpath(f) not in seen:
                seen.add(os.path.realpath(f))
                yield f


def read_reports(path, table, digest):
    """(build, split_row fields per row and rank, ranks, matched by digest) of one stderr file whose
    headers name TABLE (module docstring); None if they name another table or none."""
    text = open(path, errors="replace").read()
    start = text.find("AB_REPORT[")
    if start < 0:
        return None
    builds, ranks, rows, named = set(), set(), defaultdict(list), set()
    for line in text[text.rfind("\n", 0, start) + 1:].splitlines():
        if h := AB_HEADER.search(line):
            ranks.add(h[1])
            if m := re.search(r"apple-bottom \S+ \((\w+)\)", line):
                builds.add(m[1])
            if m := re.search(r"split_table=file:(\S+)", line):
                named.add(m[1])
        elif m := AB_SPLIT_ROW.search(line):
            rows[int(m[2])].append(dict(KV.findall(m[3])))
    # (TABLE's path?, digest) per label the headers printed
    labels = {(os.path.realpath(os.path.join(os.path.dirname(path), p)) == table, d)
              for p, d in map(table_label, named)}
    digests = {d for _, d in labels}
    if digest in digests or (True, None) in labels:
        return builds, rows, ranks, digest in digests
    if old := sorted(d for same, d in labels if same and d):
        print(f"fit_shares: {path}: made with {table} at sha256 {', '.join(old)}, not {digest}; skipped",
              file=sys.stderr)
    return None


def less_first(c):
    """Sums c without the n_first first split calls summed into its first_<part> keys."""
    d = Counter(c)
    for k in PARTS:
        d[k] -= c["first_" + k]
    d["split"] -= c["n_first"]
    return d


def rates(t):
    if min(t["cpu_ms"], t["gpu_ms"], t["gf_cpu"], t["gf_gpu"]) <= 0:
        return None
    rc, rg = t["gf_cpu"] / (t["cpu_ms"] / 1e3), t["gf_gpu"] / (t["gpu_ms"] / 1e3)
    return rc, rg, rg / (rc + rg)


def cut_arg(s):
    i, eq, ls = s.partition("=")
    if not eq:
        raise argparse.ArgumentTypeError(f"{s!r}: want ROW=L[,L...]")
    return int(i), [int(x) for x in ls.split(",") if x]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table")
    ap.add_argument("--damp", type=float, default=0.7)
    ap.add_argument("--floor", type=float, default=0.08)
    ap.add_argument("--min-calls", type=int, default=20)
    ap.add_argument("--min-dim", type=int, default=MIN_DIM, help=f"AB_MIN_GPU_DIM of the runs (default {MIN_DIM})")
    ap.add_argument("--cut", type=cut_arg, action="append", default=[], metavar="ROW=L[,L...]")
    ap.add_argument("--exclude-first", action=argparse.BooleanOptionalAction, default=True,
                    help="leave each rank line's first split call per row out of the rates (default)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.table or not a.runs:
        ap.error("--table and at least one RUN are needed")
    table = os.path.realpath(a.table)
    with open(table, "rb") as f:
        data = f.read()
    digest = hashlib.sha256(data).hexdigest()[:12]      # as the library prints it
    raw = data.split(b"\n")                             # its lines, as ingest_split's byte_lines() reads them
    if data.endswith(b"\n"):
        raw.pop()
    lines = [table_line(r, f"{table}:{n + 1}") for n, r in enumerate(raw)]
    rows = []                                           # (line index, row)
    for n, l in enumerate(lines):
        try:
            if r := parse_row(l):
                rows.append((n, r))
        except ValueError:
            sys.exit(f"{table}:{n + 1}: not a split row or a comment: {l!r}")
    want = [(r["routine"] + "gemm", r["axis"], f"{r['f']:g}") for _, r in rows]

    tot, builds, used, skipped, by_path, absent = defaultdict(Counter), set(), 0, 0, 0, 0
    for path in err_files(a.runs):
        rep = read_reports(path, table, digest)
        if rep is None:
            skipped += 1
            continue
        b, got, ranks, by_digest = rep
        builds |= b
        by_path += not by_digest
        have = [sorted({(kv["routine"], kv["axis"], kv["f"]) for kv in v}) for _, v in sorted(got.items())]
        if have != [[w] for w in want]:
            sys.exit(f"{path}: its split rows {have} are not {table}'s {want}")
        for i, lines_i in got.items():
            if len(lines_i) < len(ranks):
                print(f"fit_shares: {path}: row {i} reported by {len(lines_i)} of {len(ranks)} ranks", file=sys.stderr)
            for kv in lines_i:
                v = {k: float(kv[k]) for k in SUMS}
                if v["split"] and all("first_" + k in kv for k in PARTS):
                    v.update({"first_" + k: float(kv["first_" + k]) for k in PARTS}, n_first=1)
                elif v["split"]:
                    absent += 1
                tot[i].update(v)
        used += 1
    if not used:
        sys.exit(f"no err.txt under {' '.join(a.runs)} names {table}")
    if len(builds) != 1:
        sys.exit(f"the reports come from {len(builds) or 'no'} builds ({', '.join(sorted(builds))}); fit one at a time")
    build = builds.pop()
    print(f"fit_shares: {used} reports used (build {build}), {skipped} skipped (another table or none)", file=sys.stderr)
    if by_path:
        print(f"fit_shares: {by_path} of them name the table by path alone (no digest in their headers): "
              f"an edit of {table} in place since they ran would go unseen", file=sys.stderr)
    view = less_first if a.exclude_first else Counter
    excluded = sum(c["n_first"] for c in tot.values()) if a.exclude_first else 0
    if excluded:
        print(f"fit_shares: --exclude-first: {excluded} first split calls left out (one per row per rank line)",
              file=sys.stderr)
    if absent and a.exclude_first:
        print(f"fit_shares: {absent} rank lines with split calls have no first-call parts (an older library): "
              f"their first calls stay in", file=sys.stderr)

    square = [0 < r["ratio"] <= SQUARE_RATIO for _, r in rows]
    pooled = defaultdict(Counter)
    for i, (_, r) in enumerate(rows):
        if not square[i]:
            pooled[(r["routine"], r["axis"])].update(tot[i])
    cuts = dict(a.cut)

    print(f"{'row':>3} {'axis':>4} {'f':>5} {'calls':>7} {'split':>7} {'r_cpu':>7} {'r_gpu':>7} {'f_bal':>6} "
          f"{'dfirst':>7} {'first/gpu':>9} {'wait g/c':>8} {'source':>7} {'next':>5}  cut old -> new", file=sys.stderr)
    for i, (n, r) in enumerate(rows):
        own, f = tot[i], r["f"]
        t, c, src = view(own), own, "own"
        if 0 < t["split"] < a.min_calls:
            c, src = pooled[(r["routine"], r["axis"])], "pooled"
        est, est_full = rates(view(c)), rates(c)
        if square[i] or f <= 0 or est is None:
            nxt, src = f, "square" if square[i] else "kept"
        else:
            nxt = f + a.damp * (est[2] - f)
            nxt = 0.0 if nxt < a.floor else round(nxt, 2)
            if nxt != f:
                lines[n] = re.sub(r"(?<![^ \t])f=[^ \t#]+", f"f={nxt:.2f}", lines[n], count=1)
        rc, rg, fb = est or (math.nan,) * 3
        moved = fb - est_full[2] if est_full else math.nan
        first = own["first_gpu_ms" if own["n_first"] else "first_ms"] / own["gpu_ms"] if own["gpu_ms"] else math.nan
        wait = own["wait_gpu_ms"] / own["wait_cpu_ms"] if own["wait_cpu_ms"] else math.nan
        cs = "  ".join(f"{L}: {cut(r['axis'], L, f, a.min_dim)} -> {cut(r['axis'], L, nxt, a.min_dim)}"
                       for L in cuts.get(i, []))
        print(f"{i:3d} {r['axis']:>4} {f:5.2f} {t['calls']:7.0f} {t['split']:7.0f} {rc:7.1f} {rg:7.1f} {fb:6.3f} "
              f"{moved:+7.3f} {first:9.3f} {wait:8.2f} {src:>7} {nxt:5.2f}  {cs}", file=sys.stderr)

    keys = {"method": f"fit_shares.py from {used} reports of build {build} made with {os.path.basename(table)} "
                      f"(sha256 {digest}); damp {a.damp:g}, floor {a.floor:g}, min-calls {a.min_calls}, "
                      f"first calls {'left out' if excluded else 'in'}",
            "date": datetime.date.today().isoformat()}
    for n, l in enumerate(lines):
        m = KEY_RE.match(l)
        if m and m.group(1) in keys:
            lines[n] = f"# {m.group(1)}: {keys.pop(m.group(1))}"
    first_row = rows[0][0] if rows else len(lines)
    lines[first_row:first_row] = [f"# {k}: {v}" for k, v in keys.items()]   # keys the table lacked
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
