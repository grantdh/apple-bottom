// =============================================================================
// blas_trans.h — BLAS TRANS argument parsing (internal)
// =============================================================================
//
// The single place where a Fortran/C BLAS 'TRANS' character is interpreted.
// Semantics follow reference BLAS LSAME: case-insensitive, and 'C' (conjugate
// transpose) is a plain transpose for real types. Any other character maps to
// AB_TRANS_INVALID, an out-of-range CBLAS value, so the cblas fallback rejects
// it through its own parameter check, as reference xerbla would.
// =============================================================================
#ifndef AB_BLAS_TRANS_H
#define AB_BLAS_TRANS_H

#include <Accelerate/Accelerate.h>
#include <stdbool.h>

#define AB_TRANS_INVALID ((enum CBLAS_TRANSPOSE)0)

static inline enum CBLAS_TRANSPOSE ab_blas_trans(char t, bool is_complex) {
    switch (t) {
    case 'N': case 'n': return CblasNoTrans;
    case 'T': case 't': return CblasTrans;
    case 'C': case 'c': return is_complex ? CblasConjTrans : CblasTrans;
    default:            return AB_TRANS_INVALID;
    }
}

#endif // AB_BLAS_TRANS_H
