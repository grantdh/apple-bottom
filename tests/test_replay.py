#!/usr/bin/env python3
"""
Self-test of the QE call-mix replay (benchmarks/replay.c) on synthetic
traces; no Quantum ESPRESSO needed. Run with `make test-replay`, or

    python3 -I tests/test_replay.py build/replay

It starts GPU work in several processes: on a shared machine, run it under
the machine's benchmark lock.

The synthetic run: 2 MPI ranks in one pool, 24 calls each in the AB_TRACE
line format, taken with AB_MODE=gpu: DGEMMs and ZGEMMs of every kind the
replay treats differently (GPU-routed; M, N or K <= 32 or alpha = 0, so on
Accelerate; transA C, c or T with transB N, followed by a pool barrier),
with a few ms of CPU time between calls, and after each barrier call the
wait of the earlier rank for the later one, as mp_sum would. One such call
is not reduced (like PAW's per-atom DGEMMs): one rank goes on before the
other has made it. Checks:

  cpu, gpu, nohost   each finishes (no barrier deadlock) and replays every
                     call; its own traces, which replay.txt maps to ranks,
                     repeat each call's arguments and show the path the mode
                     implies (nohost: GPU calls with every host stage zero);
                     per rank, wall = calls + gaps + barrier waits, the gaps
                     are the traced ones less the traced pool waits, and
                     replay.txt counts the barriers, the unreduced calls left
                     without one and the removed waits
  --gap-scale 0.5    halves the gaps; --no-gaps drops them
  AB_MIN_GPU_DIM     nohost routes with the value in the trace header, not
                     the caller's
  round trip         one rank of the cpu replay's own traces replays again
  uneven pool        a reduction one rank lacks gets no barrier, the others do
  --pools 2          pools of one rank: no barriers
  worker killed      the other is killed, the replay fails naming the rank
  worker stopped     its partner times out at the barrier (--timeout 2), the
                     parent kills the stopped worker, the replay fails
  replay_summary.py  reads the synthetic run and the three replays
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "benchmarks" / "replay_summary.py"
sys.path.insert(0, str(SUMMARY.parent))
from replay_summary import read_replay_txt, read_trace  # noqa: E402

MIN_DIM = 32            # AB_MIN_GPU_DIM default: dimensions <= this stay on Accelerate
T_START = 10**12        # ns; rank 1 starts 3 ms later
CALLS_PER_RANK = 24
RUN_TIMEOUT_S = 300     # a replay still running then counts as deadlocked

# routine, ta, tb, M, N, K, alpha, beta, ld padding
CYCLE = [
    ("zgemm", "C", "N", 96, 80, 200, (1.0, 0.0), (0.0, 0.0), 2),
    ("zgemm", "N", "N", 200, 80, 96, (1.0, 0.0), (1.0, 0.0), 0),
    ("zgemm", "N", "N", 8, 80, 8, (1.0, 0.0), (0.0, 0.0), 0),
    ("dgemm", "T", "N", 40, 36, 300, (2.0, 0.0), (0.0, 0.0), 1),
    ("dgemm", "N", "N", 300, 36, 40, (1.0, 0.0), (1.0, 0.0), 3),
    ("dgemm", "N", "T", 16, 16, 16, (1.0, 0.0), (0.0, 0.0), 0),
    ("zgemm", "N", "N", 64, 48, 40, (0.0, 0.0), (0.5, -0.25), 0),
    ("dgemm", "C", "N", 8, 8, 1000, (1.0, 0.0), (0.0, 0.0), 0),
    ("zgemm", "c", "n", 48, 64, 96, (1.0, -0.5), (-1.0, 0.0), 5),
    ("zgemm", "T", "C", 64, 48, 40, (1.0, 0.0), (0.0, 0.0), 0),
]
# Like PAW's per-atom DGEMMs: transA C, transB N, but not summed over the pool;
# in the trace, rank 0 goes on before rank 1 has made the call.
UNREDUCED = ("dgemm", "C", "N", 8, 8, 1000)

passed = failed = 0


def check(cond, what):
    global passed, failed
    if cond:
        passed += 1
        print(f"✓ PASS  {what}")
    else:
        failed += 1
        print(f"✗ FAIL  {what}")
    return cond


def notrans(t):
    return t in "Nn"


def gpu_routed(c, min_dim=MIN_DIM):
    routine, ta, tb, m, n, k, alpha, beta, pad = c
    return alpha != (0.0, 0.0) and min(m, n, k) > min_dim


def barrier_call(c):
    return not notrans(c[1]) and notrans(c[2])


def scalar(routine, v):
    return f"{v[0].hex()},{v[1].hex()}" if routine == "zgemm" else v[0].hex()


def call_fields(c):
    """The argument fields of c's trace line, as a dict of strings."""
    routine, ta, tb, m, n, k, alpha, beta, pad = c
    return {"routine": routine, "ta": ta, "tb": tb, "M": str(m), "N": str(n), "K": str(k),
            "lda": str((m if notrans(ta) else k) + pad),
            "ldb": str((k if notrans(tb) else n) + pad), "ldc": str(m + pad),
            "alpha": scalar(routine, alpha), "beta": scalar(routine, beta)}


def call_args(c):
    return " ".join(f"{k}={v}" for k, v in call_fields(c).items())


def same_args(line, c):
    """True if a trace line (dict) holds exactly c's arguments; alpha and beta
    compared as numbers (C's %a and Python's float.hex() spell them differently)."""
    want = call_fields(c)
    for k, v in want.items():
        if k in ("alpha", "beta"):
            if [float.fromhex(x) for x in line[k].split(",")] != \
                    [float.fromhex(x) for x in v.split(",")]:
                return False
        elif line.get(k) != v:
            return False
    return True


def reduced(c):
    """QE sums the result over the pool: the replay's barrier calls."""
    return barrier_call(c) and c[:6] != UNREDUCED


def synthetic_run(d):
    """Writes the 2-rank trace. Returns the calls and, per rank, the gaps the
    replay should wait (traced gaps less the traced pool waits), the waits
    removed and the start offset."""
    calls = [CYCLE[i % len(CYCLE)] for i in range(CALLS_PER_RANK)]
    cur = [T_START, T_START + 3_000_000]
    lines = [[], []]
    times = [[], []]
    early = None            # after an unreduced call: rank 0 goes on at once
    for i, c in enumerate(calls):
        t0t1 = []
        for r in range(2):
            work = (2 + (i + r) % 3) * 1_000_000 if i else 0          # 2-4 ms of CPU time
            if early is not None and r == 0:
                work = 500_000
            dur = 300_000 + 50_000 * r if gpu_routed(c) else 40_000
            t0 = cur[r] + work
            if r == 1 and barrier_call(c) and not reduced(c):
                t0 = max(t0, t0t1[0][1] + 1_000_000)                  # after rank 0's end
            t0t1.append((t0, t0 + dur))
        early = i if barrier_call(c) and not reduced(c) else None
        latest = max(t1 for _, t1 in t0t1)
        cur = [latest + 100_000] * 2 if reduced(c) else [t1 for _, t1 in t0t1]  # mp_sum 0.1 ms
        for r in range(2):
            t0, t1 = t0t1[r]
            times[r].append((t0, t1, latest - t1 if reduced(c) else 0))
            path = "gpu" if gpu_routed(c) else "cpu"
            extra = ""
            if path == "gpu":
                k = dur - 100_000
                extra = (f" init_ns=0 alloc_ns=10000 pack_ns=20000 upload_ns=20000 kernel_ns={k}"
                         f" download_ns=10000 epilogue_ns=10000 other_ns=30000 temps_ns=5000"
                         f" encode_ns=5000 submit_ns=10000 wake_ns=9000 rest_ns=1000"
                         f" gpu_start_ns={t0 + 60_000} gpu_end_ns={t0 + 60_000 + k}")
            lines[r].append(f"seq={i + 1} t0_ns={t0} t1_ns={t1} {call_args(c)} path={path}{extra}")
    expect = []
    for r in range(2):
        with open(d / f"trace.{1000 + r}", "w") as f:
            f.write(f"# apple-bottom 1.4.0 (synthetic) AB_TRACE pid={1000 + r}\n"
                    f"# AB_MODE=gpu AB_TRACE_HASH=0 AB_MIN_GPU_DIM={MIN_DIM}\n"
                    f"# mpi_rank={r} mpi_size=2\n"
                    "# line: seq t0_ns t1_ns routine ta tb M N K lda ldb ldc alpha beta path\n")
            f.write("\n".join(lines[r]) + "\n")
        gaps, removed = 0, 0
        for i in range(1, CALLS_PER_RANK):
            wait = times[r][i - 1][2]
            gaps += times[r][i][0] - times[r][i - 1][1] - wait
            removed += wait
        expect.append({"gaps_ns": gaps, "removed_ns": removed,
                       "offset_ns": times[r][0][0] - T_START})
    return calls, expect


def env():
    e = {k: v for k, v in os.environ.items() if not k.startswith("AB_")}
    e.pop("REPLAY_TEST_SIGNAL", None)
    return e


def run(replay, args, extra_env=None, timeout=RUN_TIMEOUT_S):
    e = env()
    e.update(extra_env or {})
    t0 = time.time()
    try:
        p = subprocess.run([replay] + args, env=e, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, time.time() - t0
    return p, time.time() - t0


def read_out_traces(d):
    """rank -> call lines (dicts) of a replay's own trace files, which
    replay.txt maps to ranks by the workers' pids."""
    _, ranks = read_replay_txt(d)
    return {int(r["rank"]): list(read_trace(d / f"trace.{r['pid']}")[1]) for r in ranks}


def check_replay(replay, src, mode, calls, expect, out, scale=1.0):
    args = [str(src), "--mode", mode, "--out", str(out)]
    if scale == 0.0:
        args.append("--no-gaps")
    elif scale != 1.0:
        args += ["--gap-scale", str(scale)]
    tag = f"{mode}" + (" --no-gaps" if scale == 0.0 else f" --gap-scale {scale}" if scale != 1.0 else "")
    p, secs = run(replay, args)
    if not check(p is not None and p.returncode == 0,
                 f"replay {tag}: finishes ({secs:.1f} s)"):
        if p is not None:
            print(p.stdout + p.stderr)
        return
    check("warning" not in p.stderr, f"replay {tag}: no warning ({p.stderr.strip()[:200]})")
    settings, ranks = read_replay_txt(out)
    check(settings.get("mode") == mode and len(ranks) == 2 and
          all(int(r["calls"]) == CALLS_PER_RANK for r in ranks),
          f"replay {tag}: {CALLS_PER_RANK} calls on each of 2 ranks")
    n_barrier = sum(reduced(c) for c in calls)
    n_skipped = sum(barrier_call(c) for c in calls) - n_barrier
    check(all(int(r["barriers"]) == n_barrier and int(r["skipped"]) == n_skipped
              for r in ranks),
          f"replay {tag}: {n_barrier} pool barriers per rank, {n_skipped} calls the trace "
          "shows unreduced skipped")
    for r, (rk, e) in enumerate(zip(ranks, expect)):
        wall, cs, gs, bs = (float(rk[k]) for k in ("wall_s", "calls_s", "gaps_s", "barrier_s"))
        acc = cs + gs + bs
        check(abs(wall - acc) <= 1e-3 + 0.01 * wall,
              f"replay {tag}: rank {r} wall {wall * 1e3:.2f} ms = calls {cs * 1e3:.2f} + gaps "
              f"{gs * 1e3:.2f} + barriers {bs * 1e3:.2f} ms")
        want = e["gaps_ns"] * scale / 1e9
        check(abs(gs - want) <= 5e-4 + 0.02 * want,
              f"replay {tag}: rank {r} gaps {gs * 1e3:.2f} ms, traced less pool waits "
              f"{want * 1e3:.2f} ms")
        removed = float(rk["removed_s"])
        check(abs(removed - e["removed_ns"] / 1e9) < 1e-6,
              f"replay {tag}: rank {r} traced pool waits removed {removed * 1e3:.3f} ms")
        off = float(rk["offset_s"])
        want_off = e["offset_ns"] * scale / 1e9
        check(abs(off - want_off) <= 5e-4, f"replay {tag}: rank {r} starts {off * 1e3:.2f} ms "
                                           f"after the first rank ({want_off * 1e3:.2f})")
    traces = read_out_traces(out)
    check(sorted(traces) == [0, 1] and len(list(out.glob("trace.*"))) == 2,
          f"replay {tag}: one trace per worker, replay.txt maps them to ranks 0 and 1")
    for r, lines in sorted(traces.items()):
        args_ok = len(lines) == CALLS_PER_RANK and all(same_args(ln, c)
                                                         for ln, c in zip(lines, calls))
        check(args_ok, f"replay {tag}: rank {r} trace repeats each call's arguments in order")
        want = ["gpu" if mode != "cpu" and gpu_routed(c) else "cpu" for c in calls]
        got = [ln["path"] for ln in lines]
        check(got == want, f"replay {tag}: rank {r} paths {got.count('gpu')} gpu, "
                           f"{got.count('cpu')} cpu as {mode} mode implies")
        if mode == "nohost":
            host = ("init_ns", "alloc_ns", "pack_ns", "upload_ns", "download_ns", "epilogue_ns")
            gl = [ln for ln in lines if ln["path"] == "gpu"]
            check(all(all(ln[k] == "0" for k in host) and int(ln["kernel_ns"]) > 0 for ln in gl),
                  f"replay {tag}: rank {r} native GPU calls have no host stages, kernel > 0")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    replay = str(Path(sys.argv[1]).resolve())
    tmp = Path(tempfile.mkdtemp(prefix="test_replay."))
    src = tmp / "run"
    src.mkdir()
    calls, expect = synthetic_run(src)

    outs = {}
    for mode in ("cpu", "gpu", "nohost"):
        outs[mode] = tmp / f"replay_{mode}"
        check_replay(replay, src, mode, calls, expect, outs[mode])
    check_replay(replay, src, "cpu", calls, expect, tmp / "replay_half", scale=0.5)
    check_replay(replay, src, "cpu", calls, expect, tmp / "replay_nogaps", scale=0.0)

    # AB_MIN_GPU_DIM: the replay routes with the trace header's value, whatever
    # the caller's environment says.
    hdr = tmp / "min_dim_48"
    shutil.copytree(src, hdr)
    for f in hdr.glob("trace.*"):
        f.write_text(f.read_text().replace(f"AB_MIN_GPU_DIM={MIN_DIM}", "AB_MIN_GPU_DIM=48"))
    p, _ = run(replay, [str(hdr), "--mode", "nohost", "--no-gaps", "--out", str(tmp / "replay_48")],
               {"AB_MIN_GPU_DIM": "8"})
    ok = p is not None and p.returncode == 0 and "AB_MIN_GPU_DIM=48 (as traced)" in p.stdout
    if ok:
        want = ["gpu" if gpu_routed(c, 48) else "cpu" for c in calls]
        ok = all([ln["path"] for ln in lines] == want
                 for lines in read_out_traces(tmp / "replay_48").values())
    check(ok, "AB_MIN_GPU_DIM=48 in the trace header: nohost runs natively only the calls "
              "above 48, with AB_MIN_GPU_DIM=8 in the environment")

    # Round trip: one rank of the replay's own traces, written by the library,
    # replays again (alone in a directory: a worker's header names no MPI rank).
    again = tmp / "again"
    again.mkdir()
    _, ranks = read_replay_txt(outs["cpu"])
    shutil.copy(outs["cpu"] / f"trace.{ranks[0]['pid']}", again)
    p, _ = run(replay, [str(again), "--mode", "cpu", "--no-gaps",
                        "--out", str(tmp / "replay_again")])
    ok = p is not None and p.returncode == 0 and f"AB_MIN_GPU_DIM={MIN_DIM} (as traced)" in p.stdout
    if ok:
        _, ranks = read_replay_txt(tmp / "replay_again")
        ok = len(ranks) == 1 and int(ranks[0]["calls"]) == CALLS_PER_RANK
    check(ok, "round trip: rank 0 of the cpu replay's traces replays with the same calls")

    # Rank 1 without its first reduced call: that reduction gets no barrier,
    # the others still do.
    bad = tmp / "uneven"
    shutil.copytree(src, bad)
    lines = (bad / "trace.1001").read_text().splitlines(keepends=True)
    first = next(i for i, ln in enumerate(lines) if " ta=C tb=N " in ln)
    del lines[first]
    (bad / "trace.1001").write_text("".join(lines))
    p, _ = run(replay, [str(bad), "--mode", "cpu", "--out", str(tmp / "replay_uneven")])
    ok = p is not None and p.returncode == 0
    if ok:
        _, ranks = read_replay_txt(tmp / "replay_uneven")
        want = sum(reduced(c) for c in calls) - 1
        ok = [int(r["barriers"]) for r in ranks] == [want, want]
    check(ok, "uneven pool: a reduction one rank lacks gets no barrier, the others do")
    p, _ = run(replay, [str(src), "--mode", "cpu", "--pools", "2", "--no-gaps",
                        "--out", str(tmp / "replay_pools2")])
    ok = p is not None and p.returncode == 0
    if ok:
        _, ranks = read_replay_txt(tmp / "replay_pools2")
        ok = [int(r["barriers"]) for r in ranks] == [0, 0]
    check(ok, "--pools 2: pools of one rank, no barriers")

    # A worker that dies: the parent kills the other one, the replay fails at once.
    p, secs = run(replay, [str(src), "--mode", "cpu", "--out", str(tmp / "replay_kill")],
                  {"REPLAY_TEST_SIGNAL": "1:5:KILL"}, timeout=60)
    check(p is not None and p.returncode == 1 and "rank 1" in p.stderr and
          "signal 9" in p.stderr and secs < 30,
          f"worker killed: the replay fails in {secs:.1f} s naming it "
          f"({p.stderr.strip()[:160] if p else 'timeout'})")
    # A worker that hangs: its partner times out at the barrier, then the
    # parent kills the hung worker.
    p, secs = run(replay, [str(src), "--mode", "cpu", "--timeout", "2",
                           "--out", str(tmp / "replay_stop")],
                  {"REPLAY_TEST_SIGNAL": "1:5:STOP"}, timeout=90)
    check(p is not None and p.returncode == 1 and "did not arrive" in p.stderr and
          "rank 0" in p.stderr and "exit status 3" in p.stderr and secs < 30,
          f"worker stopped: timeout, kill and failure in {secs:.1f} s "
          f"({p.stderr.strip()[:200] if p else 'timeout'})")

    p = subprocess.run([sys.executable, "-I", str(SUMMARY), str(src)] +
                       [str(outs[m]) for m in ("cpu", "gpu", "nohost")],
                       capture_output=True, text=True)
    ok = p.returncode == 0 and all(f"replay {m} of run" in p.stdout
                                   for m in ("cpu", "gpu", "nohost"))
    check(ok and "QE run (gpu)" in p.stdout and "nohost / cpu replay wall" in p.stdout,
          "replay_summary.py: one row per source and the comparisons")
    if not ok:
        print(p.stdout + p.stderr)
    else:
        print("\n" + p.stdout)

    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        print(f"files kept in {tmp}")
    else:
        shutil.rmtree(tmp)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
