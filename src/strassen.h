// =============================================================================
// strassen.h — routing to the opt-in Strassen-Winograd GEMM (strassen.m)
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
#ifndef AB_STRASSEN_H
#define AB_STRASSEN_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// True if AB_STRASSEN is set (and not 0) and M, N and K are all at least
// AB_STRASSEN_MIN (default 4096); both are read once per process.
// ab_dgemm, ab_dgemm_scaled and ab_zgemm / ab_zgemm_ex then call
// ab_dgemm_strassen / ab_zgemm_strassen.
bool ab_internal_strassen_wanted(int M, int N, int K);

// Frees the workspace kept between calls (ab_shutdown).
void ab_internal_strassen_release(void);

#ifdef __cplusplus
}
#endif

#endif // AB_STRASSEN_H
