// =============================================================================
// test_device_api.c — Smoke tests for the device-buffer layer
// =============================================================================
//
// Exercises the Week-1 surface:
//   1. alloc → size → free
//   2. H2D → D2H roundtrip, byte-identical
//   3. D2D copy with source/destination offsets
//   4. memset then D2H
//   5. NULL / out-of-range error paths
//   6. Stream create / sync / destroy doesn't crash
//
// Pass criteria: all assertions hold. Exits non-zero on first failure.
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "apple_bottom.h"
#include "apple_bottom_device.h"

#define N_ELEMS 1024
#define N_BYTES (N_ELEMS * sizeof(double))

static int failures = 0;

#define CHECK(cond, msg) do {                                        \
    if (!(cond)) {                                                   \
        fprintf(stderr, "  ✗ FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        failures++;                                                  \
    } else {                                                         \
        printf("  ✓ %s\n", msg);                                     \
    }                                                                \
} while (0)

static void test_lifecycle(void) {
    printf("[1] lifecycle\n");
    ab_dev_buffer_t b = ab_dev_malloc(N_BYTES);
    CHECK(b != NULL, "ab_dev_malloc succeeds");
    CHECK(ab_dev_buffer_size(b) == N_BYTES, "buffer_size matches request");
    ab_dev_free(b);
    ab_dev_free(NULL);  // must not crash
    CHECK(1, "ab_dev_free(NULL) safe");
}

static void test_roundtrip(void) {
    printf("[2] host ↔ device roundtrip\n");
    double* src = (double*)malloc(N_BYTES);
    double* dst = (double*)malloc(N_BYTES);
    assert(src && dst);
    for (int i = 0; i < N_ELEMS; i++) src[i] = (double)i * 0.5 - 3.14;

    ab_dev_buffer_t buf = ab_dev_malloc(N_BYTES);
    assert(buf);

    ABStatus s1 = ab_dev_memcpy_h2d(buf, 0, src, N_BYTES);
    CHECK(s1 == AB_OK, "h2d returns AB_OK");

    memset(dst, 0xAB, N_BYTES);
    ABStatus s2 = ab_dev_memcpy_d2h(dst, buf, 0, N_BYTES);
    CHECK(s2 == AB_OK, "d2h returns AB_OK");

    CHECK(memcmp(src, dst, N_BYTES) == 0, "byte-identical roundtrip");

    ab_dev_free(buf);
    free(src);
    free(dst);
}

static void test_d2d_with_offsets(void) {
    printf("[3] device → device with offsets\n");
    double* host = (double*)malloc(N_BYTES);
    double* back = (double*)calloc(N_ELEMS, sizeof(double));
    for (int i = 0; i < N_ELEMS; i++) host[i] = (double)(i + 1);

    ab_dev_buffer_t a = ab_dev_malloc(N_BYTES);
    ab_dev_buffer_t b = ab_dev_malloc(N_BYTES);
    assert(a && b);

    CHECK(ab_dev_memcpy_h2d(a, 0, host, N_BYTES) == AB_OK, "seed a");
    CHECK(ab_dev_memset(b, 0, 0, N_BYTES) == AB_OK, "zero b");

    // Copy second half of a → first half of b.
    size_t half = N_BYTES / 2;
    CHECK(ab_dev_memcpy_d2d(b, 0, a, half, half) == AB_OK, "d2d mid-slice");

    CHECK(ab_dev_memcpy_d2h(back, b, 0, N_BYTES) == AB_OK, "read b back");

    int ok = 1;
    for (int i = 0; i < N_ELEMS/2; i++) {
        if (back[i] != host[i + N_ELEMS/2]) { ok = 0; break; }
    }
    CHECK(ok, "first half of b matches second half of a");

    int zero_ok = 1;
    for (int i = N_ELEMS/2; i < N_ELEMS; i++) {
        if (back[i] != 0.0) { zero_ok = 0; break; }
    }
    CHECK(zero_ok, "second half of b still zero");

    ab_dev_free(a);
    ab_dev_free(b);
    free(host);
    free(back);
}

static void test_memset(void) {
    printf("[4] memset\n");
    ab_dev_buffer_t b = ab_dev_malloc(N_BYTES);
    assert(b);
    CHECK(ab_dev_memset(b, 0, 0x5A, N_BYTES) == AB_OK, "memset ok");

    unsigned char* host = (unsigned char*)malloc(N_BYTES);
    CHECK(ab_dev_memcpy_d2h(host, b, 0, N_BYTES) == AB_OK, "read back");

    int ok = 1;
    for (size_t i = 0; i < N_BYTES; i++) if (host[i] != 0x5A) { ok = 0; break; }
    CHECK(ok, "all bytes == 0x5A");

    ab_dev_free(b);
    free(host);
}

static void test_error_paths(void) {
    printf("[5] error paths\n");
    ab_dev_buffer_t b = ab_dev_malloc(64);
    assert(b);
    double tmp[1] = {0};
    CHECK(ab_dev_memcpy_h2d(NULL, 0, tmp, 8) == AB_ERROR_INVALID_ARG, "h2d NULL dst");
    CHECK(ab_dev_memcpy_h2d(b, 0, NULL, 8) == AB_ERROR_INVALID_ARG, "h2d NULL src");
    CHECK(ab_dev_memcpy_h2d(b, 60, tmp, 8) == AB_ERROR_INVALID_ARG, "h2d overflow");
    CHECK(ab_dev_memcpy_d2h(tmp, b, 60, 8) == AB_ERROR_INVALID_ARG, "d2h overflow");
    CHECK(ab_dev_malloc(0) == NULL, "malloc(0) returns NULL");
    CHECK(ab_dev_buffer_size(NULL) == 0, "size(NULL) == 0");
    ab_dev_free(b);
}

static void test_streams(void) {
    printf("[6] streams\n");
    ab_dev_stream_t s = ab_dev_stream_create();
    CHECK(s != NULL, "stream_create ok");
    CHECK(ab_dev_stream_sync(s) == AB_OK, "stream_sync ok");
    CHECK(ab_dev_stream_sync(NULL) == AB_OK, "default stream sync ok");
    ab_dev_stream_destroy(s);
    CHECK(1, "stream_destroy did not crash");
}

int main(void) {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("apple-bottom device-API smoke tests\n");
    printf("═══════════════════════════════════════════════════════════\n");

    if (ab_init() != AB_OK) {
        fprintf(stderr, "ab_init failed — no Metal device?\n");
        return 2;
    }

    test_lifecycle();
    test_roundtrip();
    test_d2d_with_offsets();
    test_memset();
    test_error_paths();
    test_streams();

    ab_shutdown();

    printf("═══════════════════════════════════════════════════════════\n");
    if (failures == 0) {
        printf("✓ All device-API tests passed\n");
        return 0;
    } else {
        printf("✗ %d device-API test(s) failed\n", failures);
        return 1;
    }
}
