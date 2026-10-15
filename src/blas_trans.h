// =============================================================================
// blas_trans.h — BLAS TRANS and argument checks (internal)
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

// Reference BLAS argument checks (xerbla): TRANS flags, M, N, K >= 0 and
// each leading dimension >= max(1, rows of the stored operand). The GPU path
// indexes the caller's arrays with these, so a call that fails a check goes
// to cblas instead, which rejects it before touching A, B or C.
static inline bool ab_blas_args_valid(enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb,
                                      int M, int N, int K, int ldA, int ldB, int ldC) {
    int rowsA = ta == CblasNoTrans ? M : K, rowsB = tb == CblasNoTrans ? K : N;
    return ta != AB_TRANS_INVALID && tb != AB_TRANS_INVALID &&
           M >= 0 && N >= 0 && K >= 0 &&
           ldA >= 1 && ldA >= rowsA && ldB >= 1 && ldB >= rowsB &&
           ldC >= 1 && ldC >= M;
}

#endif // AB_BLAS_TRANS_H
