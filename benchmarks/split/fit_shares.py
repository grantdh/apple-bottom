#!/usr/bin/env python3
"""The next GPU share per row of an AB_SPLIT table, from the reports of runs made with it.

Each RUN is an err.txt file or a directory searched for them (set-aside .bad*/incomplete-* folders,
tmp and *.save are skipped; a file found twice counts once): the stderr of programs run with
AB_SPLIT=1 AB_SPLIT_TABLE=TABLE AB_REPORT=1, such as pw.x's every rank. Choose the runs to fit from
your campaign's ledger (failed or flagged runs left out); this script checks only that they belong
together. Each rank's AB_REPORT header names its table as split_table=file:PATH@DIGEST, the last
field of the line (the first 12 hex digits of the SHA-256 of the bytes the library read): a rank's
split_row lines count only if DIGEST is TABLE's now. A header without a digest (an older library) is
matched by PATH, read relative to the file's directory (where the run started), with a note. Other
ranks' lines are left out: noted if the file is used, and also if a header names TABLE's path at
another digest. A file with no counting rank is skipped. The fit stops if a report's rows differ from
TABLE's (count, routine, axis or share), or if the ranks come from more than one build (the id in the
header's parentheses, "?" if none). A dirty or unknown build is noted, as it names no one library. A
row reported by fewer ranks than the file has counting headers is noted.

Ranks that share stderr can have their lines cut and joined; a record runs from "AB_REPORT[<pid>] " to
the next on its line or to the line's end. These damaged records are left out, with a note:
  - a split_row line whose fields are not the library's keys in its order (ROW_KEYS, or an older
    library's) with numbers for values;
  - a header whose digest was lost: cut, or missing where its build prints digests (its rank's lines
    go with it, its table unknown);
  - a record cut before its kind shows (a split_row line if its rank has a counting header, else that
    rank's header).
What the text cannot show is a join of two lines cut at the same field.

For each row, its split_row lines (every rank of every run) give the seconds and the GFLOP of the
calls' CPU and GPU parts. Their ratios are the rates the two engines reached while running together
(under the shared power budget, with the other ranks' work): r_cpu = gf_cpu / cpu_s and r_gpu =
gf_gpu / gpu_s. The share at which both parts of a call would end together is f_bal = r_gpu /
(r_cpu + r_gpu). The next share moves the fraction D (default 0.7) of the way from the current one
to f_bal, rounded to 0.01; below F (default 0.08) the row goes to Accelerate (0), and a row at 0 stays
there. A row with fewer than N (default 20) split calls takes the pooled rates of the non-square rows
with its routine and axis, itself included (as calibrate.py pools them); a row with none keeps its
share, and so does a square row (ratio <= 2, scripts/ingest_split.py's rule 4), whose share is fitted
from bench_split's grid. This is .dev/active/split/design.md section 8's rule (a port of Espressivo's
calibrate.py).

The part sums include each process's first split calls, which the evidence leaves out
(scripts/ingest_split.py); the Metal start-up is in the GPU part of the first whose tail ran. The
library prints the CPU part of each row's first split call in the process (first_cpu_ms,
first_gf_cpu) and the GPU part of its first split call whose tail ran (first_gpu_ms, first_gf_gpu),
and --exclude-first (the default) leaves those parts out of each rank line's rates, and one call out
of its split count: for one row of each process that is the start-up, for the others an ordinary
call. Reports without the parts (an older library) keep their first calls, with a note.

Columns: dfirst is how far leaving the first calls out moved f_bal; first/gpu is the first calls' GPU
parts (their wall times, for an older library) over the row's GPU-part time, so a row dominated by
them shows; wait g/c (every call) above 1 means the GPU part ended first more of the time (it waited
for the CPU's head), which f_bal > f also says.

Prints TABLE with the new shares on stdout (only changed rows rewritten; "# method:" and "# date:"
set, citing the build and TABLE's sha256, and how many reports had no digest to check it by), and
the estimates on stderr. --cut ROW=L,... adds the GPU part at the old and new shares for those axis
lengths (src/split_slice.h's ab_split_cut, with the alignments read from that header and
AB_MIN_GPU_DIM's default from src/dispatch.h), since a share that does not move the cut changes
nothing. --self-test checks that cut against tests/test_dispatch.c.
Python 3.9, standard library only.
"""
import argparse
import contextlib
import datetime
import hashlib
import io
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ingest_split import KEY_RE, SQUARE_RATIO, parse_row, table_line  # noqa: E402

RECORD = re.compile(r"AB_REPORT\[(\d+)\] ")   # the library's prefix of each AB_REPORT line
BUILD = re.compile(r"apple-bottom \S+ \(([^)]*)\)")
LABEL = re.compile(r"split_table=file:(.*)$")   # the header's last field, spaces and all
# A split_row line's keys after i=, in the library's order; without the first-call parts before them.
ROW_KEYS = ("routine", "axis", "f", "calls", "split", "wall_ms", "first_ms", "first_cpu_ms", "first_gpu_ms",
            "first_gf_cpu", "first_gf_gpu", "cpu_ms", "gpu_ms", "wait_cpu_ms", "wait_gpu_ms", "redo_ms",
            "sum_ms", "gf", "gf_cpu", "gf_gpu", "kept", "kept_4m", "range", "gpu_failed")
OLD_ROW_KEYS = tuple(k for k in ROW_KEYS if k == "first_ms" or not k.startswith("first_"))
PRINTED = re.compile(r"[0-9]+(?:\.[0-9]*)?(?:e[-+]?[0-9]+)?")   # a number as %g, %.3f and %llu print it
SUMS = ("calls", "split", "cpu_ms", "gpu_ms", "gf_cpu", "gf_gpu", "first_ms", "wait_cpu_ms", "wait_gpu_ms")
# first_<part>: the first-call parts (module docstring)
PARTS = tuple(k[len("first_"):] for k in ROW_KEYS if k.startswith("first_") and k != "first_ms")
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


def report_case(lines, table="/t/t.txt", digest="0123456789ab"):
    """read_reports on a stderr file of lines: (counting pids, rank lines per row, notes)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "err.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        notes = io.StringIO()
        with contextlib.redirect_stderr(notes):
            rep = read_reports(path, table, digest) or (None, {}, (), None)
    return sorted(rep[2]), {i: len(v) for i, v in rep[1].items()}, notes.getvalue()


def self_test():
    """cut() against tests/test_dispatch.c's test_split_cut vectors, table_label() on labels, and
    read_reports() on lines of ranks cut and joined."""
    src = open(os.path.join(ROOT, "tests", "test_dispatch.c")).read()
    body = src[src.index("test_split_cut(void)"):]
    vec = re.findall(r"\{'([NMK])', ([^,]+), ([-0-9.]+), ([^,]+), ([^}]+)\}", body[:body.index("begin();")])

    def ints(s):   # an int literal, INT_MAX, or INT_MAX - n
        s = s.strip()
        return 2**31 - 1 - int(s.split("-", 1)[1] if "-" in s else 0) if s.startswith("INT_MAX") else int(s)
    labels = {"/a/t.txt@0123456789ab": ("/a/t.txt", "0123456789ab"), "/a@b/t.txt": ("/a@b/t.txt", None),
              "/a b/t.txt@0123456789ab": ("/a b/t.txt", "0123456789ab"),
              "/a/t@x@0123456789ab": ("/a/t@x", "0123456789ab"), "t@0123456789abc": ("t@0123456789abc", None),
              "t@0123456789AB": ("t@0123456789AB", None), "t": ("t", None)}

    def agree(what, n, bad):
        print(f"fit_shares: {what}: {'all agree' if n and not bad else f'{len(bad)} differ: {bad}'}")
        return bool(n and not bad)
    ok = agree(f"cut() on {len(vec)} vectors from tests/test_dispatch.c", len(vec),
               [v for v in vec if cut(v[0], ints(v[1]), float(v[2]), ints(v[3])) != ints(v[4])])
    ok &= agree(f"table_label() on {len(labels)} labels", len(labels),
                [v for v, want in labels.items() if table_label(v) != want])

    def head(pid, label="/t/t.txt@0123456789ab"):
        return (f"AB_REPORT[{pid}] apple-bottom 1.6.0 (abc) AB_MODE=auto chip=\"Apple M2 Max\" gpu_cores=38 "
                f"table=builtin:M2-Max-38c AB_SPLIT=1 split_table=file:{label}")

    def row(pid):
        return f"AB_REPORT[{pid}] split_row i=0 routine=zgemm axis=N f=0.5 " + " ".join(
            f"{k}={i + 1}.000" for i, k in enumerate(ROW_KEYS[3:]))
    r22, r33 = row(22), row(33)
    in_digest, in_label = head(11)[:-9], head(11)[:-25]    # 11's header cut inside its digest, its label
    cases = {   # name: (lines, the counting pids, rank lines per row, a note that must be there)
        "a header cut by another's": ([in_digest + head(22), row(11), r22], ["22"], {0: 1}, "ranks 11 were"),
        "a row joined to another's leftover from another field": (
            [head(22), head(33), r33, r22.partition(" gpu_ms=")[0] + " gpu_ms=" + r33.partition(" first_gpu_ms=")[2]],
            ["22", "33"], {0: 1}, "1 damaged"),
        "a header joined to a row's head, its tail after it": (
            [head(22), r22, in_label + r22.partition(" cpu_ms=")[0] + head(11)[len(in_label):], row(11)], ["22"],
            {0: 1}, "ranks 11 were"),
        "records cut before their kind shows": (   # "AB_REPORT[11] apple-bott", "AB_REPORT[22] split_ro"
            [head(11)[:24] + head(22), r22[:22] + head(33), r22, r33, row(11)], ["22", "33"], {0: 2},
            "1 damaged"),
        "a table path holding AB_REPORT[": (
            [head(22, "/t/AB_REPORT[x]/t.txt@0123456789ab"), r22], ["22"], {0: 1}, ""),
    }
    for name, (lines, pids, rows, note) in cases.items():
        got = report_case(lines)
        ok &= agree(f"read_reports() on {name}", 1,
                    [] if got[:2] == (pids, rows) and note in got[2] else [got])
    # ROW_KEYS as src/profiling/blas_profiler.c prints a split_row line now
    c = open(os.path.join(ROOT, "src", "profiling", "blas_profiler.c")).read()
    c = c[c.index("static void report_split_rows(report_line_t* l, FILE* out, const char* prefix) {"):]
    keys = tuple(re.findall(r"(\w+)=%", "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', c[:c.index("line_emit(")]))))
    ok &= agree("ROW_KEYS against blas_profiler.c's split_row line", 1, [] if keys == ("i",) + ROW_KEYS else [keys])
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


def records(line):
    """The AB_REPORT records on a line as (pid, text after the prefix, cut), each running to the next
    "AB_REPORT[<pid>] " or the line's end (before the first: another rank's partial line); cut if another
    follows, written into it, so that its end may be lost."""
    starts = list(RECORD.finditer(line))
    ends = [m.start() for m in starts[1:]] + [len(line)]
    return [(m[1], line[m.end():e], e < len(line)) for m, e in zip(starts, ends)]


def row_fields(body):
    """(row, fields) of a split_row record's text (after its prefix) whose fields are the library's keys
    in its order with numbers for values; None for a damaged one."""
    m = re.fullmatch(r"split_row i=([0-9]+)((?: \w+=\S+)+)", body)
    if not m:
        return None
    kv = [t.split("=", 1) for t in m[2].split()]
    if tuple(k for k, _ in kv) not in (ROW_KEYS, OLD_ROW_KEYS) or not all(PRINTED.fullmatch(v) for _, v in kv[2:]):
        return None
    f = dict(kv)
    return (int(m[1]), f) if f["routine"] in ("dgemm", "zgemm") and f["axis"] in ("N", "M", "K") else None


def read_reports(path, table, digest):
    """Of one stderr file, the ranks whose header names TABLE (module docstring): (their builds, their
    split_row fields per row, their pids, whether one matched by path only); None if no line counts."""
    text = open(path, errors="replace").read()
    start = text.find("AB_REPORT[")
    if start < 0:
        return None
    heads, lines, kindless, damaged = {}, [], [], 0   # pid: (build, PATH, DIGEST, cut); (pid, row, fields)
    for line in text[text.rfind("\n", 0, start) + 1:].splitlines():
        for pid, body, was_cut in records(line):
            if body.startswith("apple-bottom "):
                b, lab = BUILD.match(body), LABEL.search(body)
                p, d = table_label(lab[1]) if lab else (None, None)
                heads[pid] = (b[1] if b else "?", p, d, was_cut)
            elif body.startswith("split_row"):
                if not was_cut and (r := row_fields(body)):
                    lines.append((pid, *r))
                else:
                    damaged += 1
            elif was_cut and ("apple-bottom ".startswith(body) or "split_row".startswith(body)):
                kindless.append(pid)
    digest_builds = {b for b, _, d, _ in heads.values() if d}
    ours, old, cut = {}, set(), set()   # pid: (build, matched by path only); stale digests; tables unknown
    for pid, (build, p, d, was_cut) in heads.items():
        if d is None and (was_cut or (p is not None and build in digest_builds)):   # its digest lost
            cut.add(pid)
            continue
        same = p is not None and os.path.realpath(os.path.join(os.path.dirname(path), p)) == table
        if d == digest or (same and d is None):
            ours[pid] = (build, d is None)
        elif same:
            old.add(d)
    for pid in kindless:   # a split_row line of a rank with a counting header, else that rank's header
        if pid in ours:
            damaged += 1
        elif pid not in heads:
            cut.add(pid)
    if old:
        print(f"fit_shares: {path}: ranks made with {table} at sha256 {', '.join(sorted(old))}, not {digest}; "
              f"left out", file=sys.stderr)
    if cut:
        print(f"fit_shares: {path}: the headers of ranks {', '.join(sorted(cut))} were damaged (cut, or no digest), "
              f"their tables unknown; their lines left out", file=sys.stderr)
    if damaged:
        print(f"fit_shares: {path}: {damaged} damaged split_row lines (cut or joined to another rank's) left out",
              file=sys.stderr)
    rows, dropped = defaultdict(list), 0
    for pid, i, kv in lines:
        if pid in ours:
            rows[i].append(kv)
        elif pid not in cut:
            dropped += 1
    if not rows:
        return None
    if dropped:
        print(f"fit_shares: {path}: {dropped} split_row lines of ranks that name another table or none left out",
              file=sys.stderr)
    return {b for b, _ in ours.values()}, rows, ours, any(po for _, po in ours.values())


def less_first(c):
    """c with each first_<part> taken off its part (PARTS) and n_first off split."""
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
                    help="leave each rank line's first-call parts per row out of the rates (default)")
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
        b, got, ranks, path_only = rep
        builds |= b
        by_path += path_only
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
    if build in ("?", "unknown") or build.endswith("-dirty"):
        print(f"fit_shares: build {build} names no one library (a dirty tree, or no commit)", file=sys.stderr)
    if by_path:
        print(f"fit_shares: {by_path} of them name the table by path only (no digest in their headers): "
              f"an edit of {table} in place since they ran would go unseen", file=sys.stderr)
    view = less_first if a.exclude_first else Counter
    excluded = sum(c["n_first"] for c in tot.values()) if a.exclude_first else 0
    if excluded:
        print(f"fit_shares: --exclude-first: the first-call parts of {excluded} rank lines left out", file=sys.stderr)
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

    unchecked = f", unchecked for {by_path} of them, whose headers carry none" if by_path else ""
    keys = {"method": f"fit_shares.py from {used} reports of build {build} made with {os.path.basename(table)} "
                      f"(sha256 {digest}{unchecked}); damp {a.damp:g}, floor {a.floor:g}, min-calls {a.min_calls}, "
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
