#!/usr/bin/env python3
"""Tabulate traced Quantum ESPRESSO runs and their replays, and compare them.

Usage: python3 benchmarks/replay_summary.py <dir> [<dir> ...]

Each <dir> is a traced pw.x run (its AB_TRACE files, trace.<pid>) or a
replay of one (benchmarks/replay.c output, with replay.txt). The tables and
the comparisons are described in "Replay" in benchmarks/qe/README.md.
Standard library only (Python 3.9+).
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "qe"))
from analyze import load, run_path, wall  # noqa: E402

STAGES = ["init", "alloc", "pack", "upload", "kernel", "download", "epilogue", "other"]
SPLIT = ["temps", "encode", "submit", "wake", "rest"]
NP1_TOL = 0.15      # replay kernel and Accelerate sums vs the traced run's
NP8_TOL = 0.30      # replay gpu - cpu wall difference vs the traced run's


def fields(line):
    """The key=value tokens of a line, as a dict."""
    return dict(tok.split("=", 1) for tok in line.split() if "=" in tok)


def read_trace(path):
    """(header, calls) of one AB_TRACE file: header holds the key=value tokens
    of its '#' lines (the first of each key), calls iterates over its call
    lines as they are read, each a dict of its fields."""
    f = open(path)
    header, first = {}, None
    for line in f:
        if not line.startswith("#"):
            first = line
            break
        for k, v in fields(line[1:]).items():
            header.setdefault(k, v)

    def calls():
        with f:
            if first is not None and first.strip():
                yield fields(first)
            for line in f:
                if line.strip():
                    yield fields(line)
    return header, calls()


def read_replay_txt(d):
    """(settings, ranks) of a replay's replay.txt: the key=value tokens of its
    settings line, and of each rank's line (a dict per rank, in rank order)."""
    settings, ranks = {}, []
    for line in (Path(d) / "replay.txt").read_text().splitlines():
        if line.startswith("#"):
            continue
        kv = fields(line)
        if "rank" in kv:
            ranks.append(kv)
        else:
            settings.update(kv)
    return settings, ranks


def pw_wall(run_dir):
    """pw.x's wall time from the results.csv of run_campaign.sh, if one lists run_dir."""
    for parent in run_dir.parents:
        if (parent / "results.csv").exists():
            return next((wall(r) for r in load(parent)
                         if run_path(parent, r).resolve() == run_dir), None)
    return None


class Source:
    """One traced run or one replay, summed over its trace files as they are read."""

    def __init__(self, d):
        self.dir = Path(d).resolve()
        self.replay = (self.dir / "replay.txt").exists()
        self.pw_wall = self.barrier_s = None
        if self.replay:
            settings, ranks = read_replay_txt(self.dir)
            files = [self.dir / f"trace.{r['pid']}" for r in ranks]
            self.mode = settings["mode"]
            source = Path(settings["source"])
            self.group = source.parent
            self.label = f"replay {self.mode} of {source.name}"
            if float(settings.get("gap_scale", "1")) != 1.0:
                self.label += f" (gaps x{settings['gap_scale']})"
            self.barrier_s = sum(float(r["barrier_s"]) for r in ranks)
        else:
            files = [f for f in sorted(self.dir.glob("trace.*")) if f.name[6:].isdigit()]
            self.mode = None
            self.group = self.dir.parent
            self.pw_wall = pw_wall(self.dir)
        if not files:
            sys.exit(f"replay_summary: no trace.<pid> files in {self.dir}")
        self.ranks = len(files)
        self.calls = self.gpu_calls = 0
        self.wall = self.calls_s = self.acc_s = self.gpu_s = 0.0
        self.stage = {k: 0.0 for k in STAGES + SPLIT}
        self.kernels, spans = [], []
        for f in files:
            header, calls = read_trace(f)
            self.mode = self.mode or header.get("AB_MODE", "unset")
            self.add_rank(calls, spans)
        self.busy = union_length(spans) / 1e9
        if not self.replay:
            self.label = f"QE {self.dir.name} ({self.mode})"

    def add_rank(self, calls, spans):
        """Adds one rank's calls; their GPU intervals go to spans."""
        first = last = None
        for c in calls:
            t0, t1 = int(c["t0_ns"]), int(c["t1_ns"])
            first = t0 if first is None else min(first, t0)
            last = t1 if last is None else max(last, t1)
            s = (t1 - t0) / 1e9
            self.calls += 1
            self.calls_s += s
            if c["path"] != "gpu":
                self.acc_s += s
                continue
            self.gpu_calls += 1
            self.gpu_s += s
            for k in STAGES + SPLIT:
                self.stage[k] += int(c[k + "_ns"]) / 1e9
            self.kernels.append(int(c["kernel_ns"]) / 1e6)
            g0, g1 = int(c["gpu_start_ns"]), int(c["gpu_end_ns"])
            if g0 and g1 > g0:
                spans.append((g0, g1))
        if first is not None:
            self.wall = max(self.wall, (last - first) / 1e9)

    def kernel_stats(self):
        if not self.kernels:
            return "–"
        k = sorted(self.kernels)
        p90 = k[min(len(k) - 1, int(0.9 * len(k)))]
        return f"{statistics.median(k):.2f} / {p90:.2f} / {k[-1]:.2f}"


def union_length(spans):
    """Total length of the union of [start, end) intervals."""
    total, cur0, cur1 = 0, None, None
    for a, b in sorted(spans):
        if cur1 is None or a > cur1:
            if cur1 is not None:
                total += cur1 - cur0
            cur0, cur1 = a, b
        else:
            cur1 = max(cur1, b)
    if cur1 is not None:
        total += cur1 - cur0
    return total


def fmt(x, digits=2):
    return "–" if x is None else f"{x:.{digits}f}"


def ratio(a, b):
    return a / b if b else None


def check(r, tol):
    return "ok" if r is not None and abs(r - 1.0) <= tol else "OUT"


def print_group(group, sources):
    print(f"### {group}\n")
    print("| source | ranks | calls | wall s | Σ calls s | Σ Accelerate s | GPU calls | "
          "Σ GPU calls s | Σ barrier s | GPU busy s | kernel ms/call (median / p90 / max) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in sources:
        print(f"| {s.label} | {s.ranks} | {s.calls} | {fmt(s.wall)} | {fmt(s.calls_s)} | "
              f"{fmt(s.acc_s)} | {s.gpu_calls} | {fmt(s.gpu_s)} | {fmt(s.barrier_s)} | "
              f"{fmt(s.busy)} | {s.kernel_stats()} |")
    print()
    gpu_sources = [s for s in sources if s.gpu_calls]
    if gpu_sources:
        print("GPU calls, stages summed over ranks (s); other = temps + encode + submit + wake "
              "+ rest:\n")
        print("| source | " + " | ".join(STAGES + SPLIT) + " |")
        print("|---|" + "---:|" * len(STAGES + SPLIT))
        for s in gpu_sources:
            print(f"| {s.label} | " + " | ".join(fmt(s.stage[k]) for k in STAGES + SPLIT) + " |")
        print()
    for s in sources:
        if s.pw_wall is not None:
            print(f"- {s.label}: pw.x wall {s.pw_wall:.2f} s (results.csv)")

    traced = {s.mode: s for s in sources if not s.replay}
    replays = [s for s in sources if s.replay]
    for r in replays:
        ref = traced.get("cpu" if r.mode == "cpu" else "gpu")
        if not ref:
            continue
        parts = []
        k = ratio(r.stage["kernel"], ref.stage["kernel"])
        if k is not None:
            parts.append(f"Σ kernel {r.stage['kernel']:.2f} / {ref.stage['kernel']:.2f} s = "
                         f"{k:.3f}× ({check(k, NP1_TOL)})")
        a = ratio(r.acc_s, ref.acc_s)
        if a is not None:
            parts.append(f"Σ Accelerate {r.acc_s:.2f} / {ref.acc_s:.2f} s = {a:.3f}× "
                         f"({check(a, NP1_TOL)})")
        if parts:
            print(f"- {r.label} vs {ref.label}, within ±{NP1_TOL:.0%}: " + "; ".join(parts))
    rep = {}
    for r in replays:
        rep.setdefault(r.mode, r)
    if "cpu" in rep and "gpu" in rep and "cpu" in traced and "gpu" in traced:
        d_rep = rep["gpu"].wall - rep["cpu"].wall
        d_qe = traced["gpu"].wall - traced["cpu"].wall
        q = ratio(d_rep, d_qe)
        line = (f"- gpu − cpu wall: replay {d_rep:+.2f} s, traced {d_qe:+.2f} s, "
                f"{fmt(q, 3)}× (within ±{NP8_TOL:.0%}: {check(q, NP8_TOL)})")
        if traced["gpu"].pw_wall is not None and traced["cpu"].pw_wall is not None:
            line += f"; pw.x {traced['gpu'].pw_wall - traced['cpu'].pw_wall:+.2f} s"
        print(line)
    if "nohost" in rep and "cpu" in rep:
        n, c = rep["nohost"].wall, rep["cpu"].wall
        line = f"- nohost / cpu replay wall: {n:.2f} / {c:.2f} s = {n / c:.3f}×"
        if "gpu" in rep:
            line += f"; gpu / cpu replay wall: {rep['gpu'].wall / c:.3f}×"
        print(line)
    print()


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    sources = [Source(d) for d in argv[1:]]
    groups = {}
    for s in sources:
        groups.setdefault(s.group, []).append(s)
    for group, members in groups.items():
        members.sort(key=lambda s: (s.replay, ["cpu", "gpu", "nohost"].index(s.mode)
                                    if s.mode in ("cpu", "gpu", "nohost") else 3, s.label))
        try:
            name = str(group.relative_to(Path.cwd()))
        except ValueError:
            name = str(group)
        print_group(name, members)


if __name__ == "__main__":
    main(sys.argv)
