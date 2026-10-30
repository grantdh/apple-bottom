#!/usr/bin/env python3
"""EXPERIMENT ONLY (never for release): adds to a verify worktree (a commit plus verify-staged.patch)
   - which path the host-path redo of a kept call took (cblas = it fell back, so a difference is not
     the staged path's), the differing real and imaginary parts, and both results' normwise error
     against cblas_zgemm / cblas_dgemm from C on entry, and the call's staged record (raw value);
   - every failed Metal command buffer's error (commit_and_wait);
   - a line when the GPU's line scales are not the host's (staged_refusal(s)).
Usage: verify_more.py <worktree>. Idempotent: refuses to patch twice."""
import re, sys
from pathlib import Path

wt = Path(sys.argv[1])
bw = wt / "src/blas_wrapper.c"
t = bw.read_text()
assert "AB_VERIFY_MORE" not in t, "already patched"

for routine, z, ctype in (("dgemm", 0, "double"), ("zgemm", 1, "double complex")):
    old = re.search(
        r"    if \(orig && ab_internal_last_staged\(NULL\)\) \{\n"
        rf"        {re.escape(ctype)} \*st = malloc\(\(size_t\)M \* N \* sizeof \*st\);\n"
        r".*?        free\(st\);\n    \}\n", t, re.S)
    assert old, routine
    parts = 2 if z else 1
    trans = "true" if z else "false"
    alpha = "&alpha" if z else "alpha"
    beta = "&beta" if z else "beta"
    new = f"""    if (orig && ab_internal_last_staged(NULL)) {{  // AB_VERIFY_MORE
        size_t mn = (size_t)M * N;
        {ctype} *st = malloc(mn * sizeof *st), *ref = malloc(mn * sizeof *ref);
        verify_block(C, ldC, M, N, sizeof *C, st, true);          // the staged C
        verify_block(C, ldC, M, N, sizeof *C, orig, false);       // C as it was
        memcpy(ref, orig, mn * sizeof *ref);                      // C on entry, for the reference
        t_force_host = true;
        {routine}_blas(NULL, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        t_force_host = false;
        const char *host_path = _ab_last_path;                    // "cpu": the redo fell back to cblas
        verify_block(C, ldC, M, N, sizeof *C, orig, true);        // the host path's C
        unsigned long long bad = 0, bad_part[2] = {{0, 0}};
        for (size_t i = 0; i < mn; i++) {{
            bool d = false;
            for (int q = 0; q < {parts}; q++) {{
                bool dq = memcmp((const double *)&orig[i] + q, (const double *)&st[i] + q,
                                 sizeof(double)) != 0;
                bad_part[q] += dq;
                d = d || dq;
            }}
            bad += d;
        }}
        __atomic_fetch_add(&g_ver_calls[{z}], 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&g_ver_elems[{z}], bad, __ATOMIC_RELAXED);
        if (bad) {{
            __atomic_fetch_add(&g_ver_bad[{z}], 1, __ATOMIC_RELAXED);
            cblas_{routine}(CblasColMajor, ab_blas_trans(transA, {trans}), ab_blas_trans(transB, {trans}),
                        M, N, K, {alpha}, A, ldA, B, ldB, {beta}, ref, M);
            double nr = 0, es = 0, eh = 0;
            for (size_t i = 0; i < mn * {parts}; i++) {{
                double r = ((const double *)ref)[i];
                double ds = ((const double *)st)[i] - r, dh = ((const double *)orig)[i] - r;
                nr += r * r;
                es += ds * ds;
                eh += dh * dh;
            }}
            fprintf(stderr, "AB_VERIFY_STAGED {routine} %c%c M=%d N=%d K=%d: %llu elements differ "
                    "(re %llu, im %llu); host redo on %s; vs FP64 staged %.3e host %.3e; "
                    "staged_record=%d\\n", transA, transB, M, N, K, bad, bad_part[0], bad_part[1],
                    host_path, sqrt(es / nr), sqrt(eh / nr), p ? (int)p->staged : -2);
        }}
        free(st);
        free(ref);
    }}
"""
    t = t[:old.start()] + new + t[old.end():]

# The GPU's line scales against the host's.
m = re.search(r"(    if \(memcmp\(sa\.e, gpu, \(size_t\)g->M \* sizeof \*gpu\) \|\|\n"
              r"        memcmp\(sb\.e, gpu \+ g->M, \(size_t\)g->N \* sizeof \*gpu\)\))\n"
              r"(        return [^;]+;)\n", t)
assert m, "scales memcmp"
t = (t[:m.start()] + m.group(1) + " {\n"
     '        fprintf(stderr, "AB_VERIFY_SCALES %c%c%c M=%d N=%d K=%d: GPU line scales differ\\n",\n'
     "                g->routine, g->ta, g->tb, g->M, g->N, g->K);  // AB_VERIFY_MORE\n"
     "    " + m.group(2) + "\n    }\n" + t[m.end():])
bw.write_text(t)

am = wt / "src/apple_bottom.m"
s = am.read_text()
old = "    note_gpu_time(cmdBuf, commit_ns, host_ns());\n    return cmdBuf.status != MTLCommandBufferStatusError &&"
assert s.count(old) == 1, "commit_and_wait"
s = s.replace(old, """    note_gpu_time(cmdBuf, commit_ns, host_ns());
    if (cmdBuf.status == MTLCommandBufferStatusError)  // AB_VERIFY_MORE
        fprintf(stderr, "AB_CMDBUF_ERROR pid=%d code=%ld %s\\n", (int)getpid(), (long)cmdBuf.error.code,
                cmdBuf.error.localizedDescription.UTF8String ?: "");
    return cmdBuf.status != MTLCommandBufferStatusError &&""")
am.write_text(s)
print("patched", wt)
