#!/usr/bin/env python3
"""Plants each defect AB_SPLIT's tests must catch, one at a time, and checks that they do.

    tests/split_defects.py [--keep DIR] [NAME ...]

Copies the source tree (without build/ and .git) into a scratch directory, builds the library,
build/test_dispatch and build/test_split there, and checks that the catching tests pass on the clean
copy. Then for each defect (or those named) it applies the defect's edits, rebuilds, runs the tests
that should catch it, and restores the files: the defect counts as caught if one of those tests
fails. Prints a line per defect and exits 1 if any defect went uncaught, or is stale: its edit no longer
matches the source, or the source no longer builds with it. Needs the GPU; about half an hour.
`make test-split-defects` runs it. Python 3.9, standard library only.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BW, SL, DC, BP = "src/blas_wrapper.c", "src/split_slice.h", "src/dispatch.c", "src/profiling/blas_profiler.c"
DISPATCH = ["build/test_dispatch"]
SPLIT = lambda *names: ["build/test_split", *names]   # noqa: E731

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
        (BW, "    if (!joined) {\n        pthread_join(worker, NULL);",
         "    if (P && !joined && w->routine == 'd') {\n"
         "        const double *pd0 = P;\n        double *c0 = w->C;\n"
         "        for (size_t col = 0; col < (size_t)w->N; col++)\n"
         "            for (size_t i = 0; i < (size_t)w->M; i++) c0[col * w->ldC + i] += pd0[col * w->M + i];\n"
         "    }\n    if (!joined) {\n        pthread_join(worker, NULL);"),
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
]

BUILD = ["make", "-j8", "lib", "build/test_dispatch", "build/test_split"]


def run(cmd, cwd, log):
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
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
