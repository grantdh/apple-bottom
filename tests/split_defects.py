#!/usr/bin/env python3
"""Plants each defect AB_SPLIT's tests must catch, one at a time, and checks that they do.

    tests/split_defects.py [--keep DIR] [NAME ...]

Copies the source tree (without build/ and .git) into a scratch directory, builds the library,
build/test_dispatch and build/test_split there, and checks that the catching tests pass on the clean
copy. Then for each defect (or those named) it applies the defect's edits, rebuilds, runs the tests
that should catch it, and restores the files: the defect counts as caught if one of those tests
fails. A build or test still running after RUN_LIMIT_S seconds is ended, with every process it
started, and counts as failing (a TIMEOUT line says so), so a defect that hangs a test is caught,
not waited on. Prints a line per defect and exits 1 if any defect went uncaught, or is stale: its
edit no longer matches the source, or the source no longer builds with it. Needs the GPU; about 3
minutes.
`make test-split-defects` runs it. Python 3.9, standard library only.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BW, SL, DC, BP = "src/blas_wrapper.c", "src/split_slice.h", "src/dispatch.c", "src/profiling/blas_profiler.c"
DISPATCH = ["build/test_dispatch"]
SPLIT = lambda *names: ["build/test_split", *names]   # noqa: E731
# A clean build or test takes under a minute, and test_split ends a test of its own after 180 s
# (TEST_TIMEOUT_S): this bound is the backstop.
RUN_LIMIT_S = 300

# name, the edits (file, old text, new text; old must occur once), the tests that must catch it.
DEFECTS = [
    # slice errors
    ("cut off by one", [(SL, "const size_t cut = (size_t)(N - g);", "const size_t cut = (size_t)(N - g + 1);")],
     [DISPATCH, SPLIT("test_split_composition")]),
    ("A offset wrong for ta 'C'", [(SL, "gpu->a = na ? cut : cut * (size_t)ldA;", "gpu->a = cut;")],
     [DISPATCH, SPLIT("test_split_composition")]),
    ("conjugation lost", [(BW, "    tail->K = tc.K;\n",
                           "    tail->K = tc.K;\n    if (tail->ta == CblasConjTrans) tail->ta = CblasTrans;\n")],
     [DISPATCH, SPLIT("test_split_composition")]),
    ("beta applied twice (K tail)", [(BW, "        tail->ldC = w->M;\n        tail->beta = 0.0;\n",
                                      "        tail->ldC = w->M;\n")],
     [DISPATCH, SPLIT("test_split_composition")]),
    # K axis errors
    ("P added when refused", [(BW, "    } else if (P && w->routine == 'z') {", "    } if (P && w->routine == 'z') {")],
     [SPLIT("test_split_exponent_range")]),
    ("K sum skipped", [(BW, "c[col * w->ldC + i] += pz[col * w->M + i];", "(void)pz[col * w->M + i];")],
     [DISPATCH, SPLIT("test_split_composition")]),
    ("K add before join", [
        (BW, "    if (!joined) {\n        split_join(&j, own);",
         "    if (P && !joined && w->routine == 'd') {\n"
         "        const double *pd0 = P;\n        double *c0 = w->C;\n"
         "        for (size_t col = 0; col < (size_t)w->N; col++)\n"
         "            for (size_t i = 0; i < (size_t)w->M; i++) c0[col * w->ldC + i] += pd0[col * w->M + i];\n"
         "    }\n    if (!joined) {\n        split_join(&j, own);"),
        (BW, "c[col * w->ldC + i] += pd[col * w->M + i];", "(void)pd[col * w->M + i];")],
     [DISPATCH, SPLIT("test_split_composition")]),
    ("K redo with beta 0", [(BW, "        tail->ldC = w->ldC;\n        tail->beta = 1.0;", "        tail->ldC = w->ldC;\n        tail->beta = 0.0;")],
     [SPLIT("test_split_exponent_range")]),
    # failure paths
    ("retry skipped", [(BW, "if (j->why == AB_PROF_STAGED_GPU_FAILED && t_stage_cb_failed) {",
                        "if (0 && j->why == AB_PROF_STAGED_GPU_FAILED && t_stage_cb_failed) {")],
     [SPLIT("test_split_scenarios")]),
    ("retry made off the staged path", [(BW, "        j->retries = 1;\n        j->why = run_staged(p, &j->g);",
                                         "        j->retries = 1;\n        part_cblas(&j->tail);\n"
                                         "        j->why = AB_PROF_STAGED_KEPT;")],
     [SPLIT("test_split_scenarios")]),
    ("slot leak on refusal", [(BW, "    ab_internal_stage_release(s);\n    prof_stage(p, AB_PROF_STAGE_ALLOC);\n    return why;",
                               "    if (staged_kept(why)) ab_internal_stage_release(s);\n"
                               "    prof_stage(p, AB_PROF_STAGE_ALLOC);\n    return why;")],
     [SPLIT("test_split_exponent_range")]),
    ("NaN made by the GPU taken for data", [(BW, "if (j.why == AB_PROF_STAGED_NONFINITE && stage_reads_refused(&j.g))",
                                             "if (j.why == AB_PROF_STAGED_NONFINITE)")],
     [SPLIT("test_split_scenarios")]),
    ("ZGEMM tail's overflowing product taken for a GPU failure",
     [(BW, "        return !ab_zgemm_product_fits(g->K, mx[0], mx[1], g->alpha[0], g->alpha[1]) ||\n",
       "        return mx[0] >= AB_HIWORD_NONFINITE || mx[1] >= AB_HIWORD_NONFINITE ||\n")],
     [SPLIT("test_split_exponent_range")]),
    ("ZGEMM tail's C out of range taken for a GPU failure",
     [(BW, "               !ab_zgemm_c_fits(g->beta[0], g->beta[1], mx[2]);", "               mx[2] >= AB_HIWORD_NONFINITE;")],
     [SPLIT("test_split_exponent_range")]),
    ("P leak", [(BW, "    free(P);\n    rec->staged", "    rec->staged")],
     [SPLIT("test_split_interactions")]),
    # routing
    ("share read from load", [(DC, "out->g = ab_split_cut(r->axis, out->L, r->share, c->min_dim);",
                               "double load[1] = {0};\n        getloadavg(load, 1);\n"
                               "        out->g = ab_split_cut(r->axis, out->L, r->share * (1.0 + 0.5 * fmod(load[0], 1.0)), "
                               "c->min_dim);")],
     [DISPATCH]),
    ("trylock inserted", [(BW, "static ab_split_outcome_t split_run(CallProf *p,",
                           "static pthread_mutex_t g_split_try = PTHREAD_MUTEX_INITIALIZER;\n"
                           "static ab_split_outcome_t split_run(CallProf *p,"),
                          (BW, "    const size_t e = w->routine == 'z' ? sizeof(double complex) : sizeof(double);\n    ab_split_part_t hc, tc;",
                           "    if (pthread_mutex_trylock(&g_split_try)) return split_declined(rec, AB_PROF_SPLIT_DECLINE_THREAD);\n"
                           "    const size_t e = w->routine == 'z' ? sizeof(double complex) : sizeof(double);\n    ab_split_part_t hc, tc;"),
                          (BW, "    if (run_tail) pthread_setcancelstate(cancel_state, NULL);\n",
                           "    if (run_tail) pthread_setcancelstate(cancel_state, NULL);\n"
                           "    pthread_mutex_unlock(&g_split_try);\n")],
     [SPLIT("test_split_determinism")]),
    ("Strassen on a slice", [(BW, "        j.g.classical = true;\n", "")],
     [SPLIT("test_split_scenarios")]),
    ("floor not applied to g", [(SL, "if (g <= (long)min_dim) {", "if (0) {")],
     [DISPATCH]),
    ("AB_MODE=cpu ignored", [(DC, "    if (ab_dispatch_mode_route(c, MN < K ? MN : K) == 0) return false;\n    for (int i = 0; i < c->split_nrows; i++) {",
                              "    (void)MN;\n    for (int i = 0; i < c->split_nrows; i++) {")],
     [DISPATCH, SPLIT("test_split_scenarios")]),
    ("C in place on M", [(BW, "        j.g.c_staged = axis == 'M';\n", "")],
     [SPLIT("test_split_registered_c")]),
    # reporting
    ("path left on the worker", [(BW, "    split_note_path(rec);\n    return true;\n}", "    return true;\n}")],
     [SPLIT("test_split_scenarios")]),
    ("trace stage missing", [(BP, '" init_ns=%llu cpu_part_ns=%llu join_ns=%llu redo_ns=%llu sum_ns=%llu other_ns=%llu",\n'
                                  "             LLU(s->init_ns), LLU(s->cpu_part_ns), LLU(s->join_ns), LLU(s->redo_ns), LLU(s->sum_ns),",
                              '" init_ns=%llu cpu_part_ns=%llu redo_ns=%llu sum_ns=%llu other_ns=%llu",\n'
                              "             LLU(s->init_ns), LLU(s->cpu_part_ns), LLU(s->redo_ns), LLU(s->sum_ns),")],
     [SPLIT("test_split_reports")]),
    ("table digest of the path, not the bytes",
     [(DC, "sha256_update(&sha, line, (size_t)len);", "sha256_update(&sha, path, strlen(path));")], [DISPATCH]),
    ("first split call's CPU part from the last", [(BP, "if (!r->split) {", "if (1) {")],
     [SPLIT("test_split_reports")]),
    ("first split call's GPU part from the call's wall time",
     [(BP, "r->first_gpu_ns = s->gpu_wall_ns;", "r->first_gpu_ns = elapsed_ns;")], [SPLIT("test_split_reports")]),
    # the split worker
    ("worker kept after the join", [(BW, "    atomic_store_explicit(&g_pool.state, POOL_IDLE, memory_order_release);\n", "")],
     [SPLIT("test_split_worker")]),
    ("worker used in a forked child", [(BW, "    if (!forks) forks = pthread_atfork(NULL, NULL, split_pool_child) == 0;\n",
                                        "    forks = true;\n")],
     [SPLIT("test_split_fork")]),
    ("worker started without a split", [(BW, "uint64_t ab_internal_split_worker(void) {",
                                         "__attribute__((constructor)) static void split_pool_eager(void) {\n"
                                         "    if (split_pool_take()) atomic_store(&g_pool.state, POOL_IDLE);\n}\n\n"
                                         "uint64_t ab_internal_split_worker(void) {")],
     [SPLIT("test_split_off")]),
    ("the join cancellable", [(BW, "    if (run_tail) pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cancel_state);\n", "")],
     [SPLIT("test_split_worker")]),
    ("callers wait for the worker",
     [(BW, "    if ((s != POOL_IDLE && s != POOL_NONE) ||\n"
           "        !atomic_compare_exchange_strong_explicit(&g_pool.state, &s, POOL_BUSY, memory_order_acquire,\n"
           "                                                 memory_order_relaxed))\n"
           "        return false;\n",
           "    for (;;) {\n"
           "        while (s == POOL_BUSY) {\n"
           "            pthread_yield_np();\n"
           "            s = atomic_load_explicit(&g_pool.state, memory_order_relaxed);\n"
           "        }\n"
           "        if (s != POOL_IDLE && s != POOL_NONE) return false;\n"
           "        if (atomic_compare_exchange_strong_explicit(&g_pool.state, &s, POOL_BUSY, memory_order_acquire,\n"
           "                                                    memory_order_relaxed))\n"
           "            break;\n"
           "    }\n")],
     [SPLIT("test_split_worker")]),
    ("worker takes the caller's signals", [(BW, "    sigfillset(&all);\n", "    sigemptyset(&all);\n")],
     [SPLIT("test_split_worker")]),
    ("worker blocks its own faults", [(BW, "    pthread_sigmask(SIG_UNBLOCK, &s, NULL);\n", "")],
     [SPLIT("test_split_worker")]),
    ("worker takes SIGPIPE", [(BW, "SIGSYS, SIGPROF};", "SIGSYS, SIGPROF, SIGPIPE};")],
     [SPLIT("test_split_worker")]),
    ("worker blocks SIGPROF", [(BW, "SIGSYS, SIGPROF};", "SIGSYS};")],
     [SPLIT("test_split_worker")]),
    ("no QoS loan in the join", [(BW, "q > QOS_CLASS_DEFAULT ?", "q > QOS_CLASS_USER_INTERACTIVE ?")],
     [SPLIT("test_split_worker")]),
    ("QoS loan never ended", [(BW, "    if (o) pthread_override_qos_class_end_np(o);\n", "")],
     [SPLIT("test_split_worker")]),
    ("worker spins", [(BW, "        dispatch_semaphore_wait(g_pool.go, DISPATCH_TIME_FOREVER);\n",
                       "        while (dispatch_semaphore_wait(g_pool.go, DISPATCH_TIME_NOW)) {}\n")],
     [SPLIT("test_split_worker")]),
]

BUILD = ["make", "-j8", "lib", "build/test_dispatch", "build/test_split"]


def run(cmd, cwd, log):
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        # a process group of its own, for one kill to end all it started
        p = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=RUN_LIMIT_S)
        except BaseException as e:
            # the time limit, or this script ended (the group misses the terminal's Ctrl-C)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            if not isinstance(e, subprocess.TimeoutExpired):
                raise
            f.write(f"split_defects: ended after {RUN_LIMIT_S} s\n")
            print(f"  TIMEOUT {' '.join(cmd)}: ended after {RUN_LIMIT_S} s")
            return 1


def main():
    sys.stdout.reconfigure(line_buffering=True)   # each defect's line reaches a log as it is made
    for sig in (signal.SIGTERM, signal.SIGHUP):    # as SystemExit, so run() ends its group
        signal.signal(sig, lambda n, _: sys.exit(128 + n))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="defects to plant (default: all)")
    ap.add_argument("--keep", help="build in this directory and keep it (default: a temporary one)")
    args = ap.parse_args()
    defects = [d for d in DEFECTS if not args.names or d[0] in args.names]
    if len(defects) != (len(args.names) or len(DEFECTS)):
        sys.exit(f"split_defects: unknown defect among {args.names}")
    work = args.keep or tempfile.mkdtemp(prefix="split_defects.")
    tree = os.path.join(work, "tree")
    log = os.path.join(work, "log.txt")
    if not os.path.isdir(tree):
        shutil.copytree(REPO, tree, ignore=shutil.ignore_patterns("build", ".git"))
    print(f"split_defects: building in {tree} (log {log})")
    if run(BUILD, tree, log):
        sys.exit("split_defects: the clean copy does not build")
    for cmd in sorted({tuple(c) for d in defects for c in d[2]}):
        if run(list(cmd), tree, log):
            sys.exit(f"split_defects: {' '.join(cmd)} fails on the clean copy")
    missed = 0
    for name, edits, tests in defects:
        saved = {}
        ok = True
        for path, old, new in edits:
            full = os.path.join(tree, path)
            text = saved.get(path) or open(full).read()
            saved.setdefault(path, text)
            cur = open(full).read()
            if cur.count(old) != 1:
                print(f"  STALE   {name}: {path} has {cur.count(old)} copies of the text to change")
                ok = False
                break
            with open(full, "w") as f:
                f.write(cur.replace(old, new))
        caught = []
        if ok and run(BUILD, tree, log):
            # a defect that does not build tests nothing: its edit has gone stale
            print(f"  STALE   {name}: does not build")
            ok = False
        elif ok:
            for cmd in tests:
                if run(cmd, tree, log):
                    caught.append(" ".join(cmd))
        for path, text in saved.items():
            with open(os.path.join(tree, path), "w") as f:
                f.write(text)
        if not ok:
            missed += 1
        elif caught:
            print(f"  caught  {name}: {', '.join(caught)}")
        else:
            missed += 1
            print(f"  MISSED  {name}: {', '.join(' '.join(c) for c in tests)} all passed")
    run(BUILD, tree, log)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print(f"split_defects: {len(defects) - missed} of {len(defects)} defects caught")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
