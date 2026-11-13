#ifndef AB_TEST_KERNELS_H
#define AB_TEST_KERNELS_H

// Which GEMM kernels a call ran (test_correctness.c, test_precision.c,
// test_blas_abi.c), from the dispatches the library counts per kernel where it
// binds the pipeline (gemm_pipeline in src/apple_bottom.m). Kernels go by their
// Metal function names: dd_dgemm, dd_dgemm_narrow, dd_dgemm_ab,
// dd_dgemm_ab_narrow, dd_dgemm_ab_ts and dd_dsyrk, each with a register-
// prefetch instance <name>_pf. A coverage case names the kernel it is for and
// checks it, so a routing change that moves the case fails it.

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// src/apple_bottom.m: dispatches of kernel `kind`'s plain (pf 0) or _pf (pf 1)
// instance since the process started, and its name (NULL from the last kind on)
uint64_t ab_internal_kernel_dispatches(int kind, int pf);
const char *ab_internal_kernel_name(int kind);
uint64_t ab_internal_prefetch_dispatches(void);   // every _pf instance's

#define KERNEL_KINDS_MAX 16

// Dispatches of the library's kinds: n[kind][0] plain, n[kind][1] _pf
typedef struct {
    int kinds;
    uint64_t n[KERNEL_KINDS_MAX][2];
} KernelCounts;

// Every kind's dispatches so far
static inline KernelCounts kernel_counts(void) {
    KernelCounts c = {0};
    for (; c.kinds < KERNEL_KINDS_MAX && ab_internal_kernel_name(c.kinds); c.kinds++)
        for (int pf = 0; pf < 2; pf++)
            c.n[c.kinds][pf] = ab_internal_kernel_dispatches(c.kinds, pf);
    return c;
}

// The dispatches since `before` (kernel_counts)
static inline KernelCounts kernel_counts_since(const KernelCounts *before) {
    KernelCounts c = kernel_counts();
    for (int k = 0; k < c.kinds; k++)
        for (int pf = 0; pf < 2; pf++) c.n[k][pf] -= before->n[k][pf];
    return c;
}

// The kind of the kernel named `name`, or -1
static inline int kernel_kind(const KernelCounts *c, const char *name) {
    for (int k = 0; k < c->kinds; k++)
        if (!strcmp(ab_internal_kernel_name(k), name)) return k;
    return -1;
}

// Every dispatch in c
static inline uint64_t kernel_total(const KernelCounts *c) {
    uint64_t n = 0;
    for (int k = 0; k < c->kinds; k++) n += c->n[k][0] + c->n[k][1];
    return n;
}

// c as "dd_dgemm_narrow_pf x3, dd_dgemm x1", or "none", in buf
static inline const char *kernel_list(const KernelCounts *c, char *buf, size_t size) {
    size_t used = 0;
    buf[0] = '\0';
    for (int k = 0; k < c->kinds; k++)
        for (int pf = 0; pf < 2 && used < size; pf++)
            if (c->n[k][pf])
                used += (size_t)snprintf(buf + used, size - used, "%s%s%s x%llu",
                                         used ? ", " : "", ab_internal_kernel_name(k),
                                         pf ? "_pf" : "", (unsigned long long)c->n[k][pf]);
    if (!used) snprintf(buf, size, "none");
    return buf;
}

// Whether c is one or more dispatches of kernel `want` (either instance) and
// nothing else; buf gets "ran <kernel_list>", then ", not <want>" if not.
static inline bool kernel_check(const KernelCounts *c, const char *want, char *buf, size_t size) {
    int k = kernel_kind(c, want);
    uint64_t total = kernel_total(c);
    bool ok = k >= 0 && total > 0 && c->n[k][0] + c->n[k][1] == total;
    char list[160];
    snprintf(buf, size, "ran %s%s%s", kernel_list(c, list, sizeof list), ok ? "" : ", not ",
             ok ? "" : want);
    return ok;
}

// The first kind but `skip` (NULL: none) that c has no dispatch of (of its _pf
// instance if pf), or NULL: what a set of coverage cases misses
static inline const char *kernel_missing(const KernelCounts *c, bool pf, const char *skip) {
    for (int k = 0; k < c->kinds; k++) {
        const char *name = ab_internal_kernel_name(k);
        if ((!skip || strcmp(name, skip)) && !c->n[k][1] && (pf || !c->n[k][0])) return name;
    }
    return NULL;
}

#endif /* AB_TEST_KERNELS_H */
