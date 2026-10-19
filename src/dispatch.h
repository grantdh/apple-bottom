// =============================================================================
// dispatch.h — GPU-or-Accelerate routing of BLAS-ABI calls (internal)
// =============================================================================
//
// Where a raw-pointer BLAS-ABI call runs. GEMM (dgemm_, zgemm_, ab_dgemm_,
// ab_zgemm_, ab_dgemm_blas, ab_zgemm_blas) is routed here entirely; BLAS-2
// (dgemv_, zgemv_, zhemv_, ab_*gemv_blas, ab_zhemv_blas) shares steps 1-3
// through ab_dispatch_mode_route() and has its own auto rule (blas2_abi.c).
// The native ABMatrix and device APIs never come here. Calls with invalid
// arguments or alpha == 0 run on Accelerate before routing (blas_wrapper.c,
// blas2_abi.c).
//
// Precedence, per call; the settings are read once per process:
//   1. AB_MODE=cpu: Accelerate.
//   2. M, N or K <= AB_MIN_GPU_DIM (default 32): Accelerate, in every mode.
//   3. AB_MODE=gpu: the GPU.
//   4. AB_MODE=auto (the default; any value other than cpu/gpu) with a FLOP
//      override: the GPU iff the call's FLOPs (2·M·N·K real, 8·M·N·K
//      complex) reach AB_CROSSOVER_FLOPS_REAL (DGEMM) or AB_CROSSOVER_FLOPS
//      (ZGEMM, and DGEMM when AB_CROSSOVER_FLOPS_REAL is unset).
//   5. AB_MODE=auto: the dispatch table, read from the file AB_DISPATCH_TABLE
//      names or else the built-in calibration (dispatch_table.h, written by
//      `make calibrate`): the GPU iff a row of the call's routine and
//      transposition class has min_M <= M, min_N <= N and min_K <= K.
//
// Table file format (what `make calibrate` writes), one row per line, '#'
// starts a comment:
//   <dgemm|zgemm> <transA> <transB> <min_M> <min_N> <min_K>
// TRANS is a class: 'N', or 'T' for both 'T' and 'C' ('C' is accepted as
// 'T'). A file without rows keeps every auto call on Accelerate. A file that
// cannot be read or parsed is reported on stderr and the built-in table used.
// =============================================================================
#ifndef AB_DISPATCH_H
#define AB_DISPATCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum { AB_DISPATCH_AUTO, AB_DISPATCH_CPU, AB_DISPATCH_GPU } ab_dispatch_mode_t;

// A row sends calls of `routine` ('d' DGEMM, 'z' ZGEMM) whose TRANS flags
// fall in classes ta, tb ('N' or 'T') to the GPU when M >= m, N >= n, K >= k.
// The built-in table ends with a row whose routine is 0.
typedef struct {
    char routine, ta, tb;
    int m, n, k;
} ab_dispatch_row_t;

#define AB_DISPATCH_MAX_ROWS        64
#define AB_DISPATCH_DEFAULT_MIN_DIM 32

typedef struct {
    ab_dispatch_mode_t mode;                // AB_MODE
    int min_dim;                            // AB_MIN_GPU_DIM
    uint64_t crossover_flops;               // AB_CROSSOVER_FLOPS, 0 = unset
    uint64_t crossover_flops_real;          // AB_CROSSOVER_FLOPS_REAL, 0 = unset
    int nrows;
    ab_dispatch_row_t rows[AB_DISPATCH_MAX_ROWS];
} ab_dispatch_config_t;

// True if `row` covers a call: same routine and TRANS classes, and M, N, K
// at least the row's minimums.
bool ab_dispatch_row_covers(const ab_dispatch_row_t *row, char routine,
                            char transA, char transB, int M, int N, int K);

// Steps 1-3 above for a call whose smallest dimension is min_dim: 0 for
// Accelerate, 1 for the GPU, -1 if the mode leaves it to the auto rule.
int ab_dispatch_mode_route(const ab_dispatch_config_t *c, int min_dim);

// The routing decision for one call with valid arguments under config c:
// true for the GPU. Pure: reads no environment and touches no GPU.
bool ab_dispatch_gpu(const ab_dispatch_config_t *c, char routine,
                     char transA, char transB, int M, int N, int K);

// Parses one table line into *row: 1 for a row, 0 for a blank or comment
// line, -1 if malformed.
int ab_dispatch_parse_row(const char *line, ab_dispatch_row_t *row);

// Replaces c's rows with those of the table file at path. On failure c is
// unchanged and err says why (with the line number for a malformed line).
bool ab_dispatch_load_table(ab_dispatch_config_t *c, const char *path,
                            char *err, size_t err_len);

// Fills c from the environment variables above and the built-in table.
void ab_dispatch_config_from_env(ab_dispatch_config_t *c);

// The configuration read from the environment at the first call in this
// process (of this or ab_dispatch_route_gpu()).
const ab_dispatch_config_t *ab_dispatch_process_config(void);

// ab_dispatch_gpu() under ab_dispatch_process_config().
bool ab_dispatch_route_gpu(char routine, char transA, char transB, int M, int N, int K);

// The whole routing of a BLAS-ABI call, as dgemm_ and zgemm_ make it
// (blas_wrapper.c): valid arguments (ab_blas_args_valid), alpha != 0, then
// ab_dispatch_route_gpu(). A call sent to the GPU can still fall back to
// Accelerate (no device, operands beyond the ABMatrix limits).
bool ab_internal_blas_routes_gpu(char routine, char transA, char transB, int M, int N, int K,
                                 int ldA, int ldB, int ldC, bool alpha_nonzero);

// Records where the latest BLAS-ABI call on this thread computed its result,
// for ab_get_last_dispatch_path() (blas_wrapper.c): the BLAS-2 entry points
// (blas2_abi.c) report through this.
void ab_internal_set_last_path(bool gpu);

#endif // AB_DISPATCH_H
