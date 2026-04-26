// Meta-test for the registry per-test delta ghost-guard. Does NOT link
// against libapplebottom — it tests the runner pattern, not the library.
// The binary returns a distinguished non-zero code (42) when the guard
// fires correctly; the Makefile target wraps this inversion.

#include <stdio.h>
#include <stdlib.h>
#include "test_registry.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define PASS() do { tests_passed++; printf("PASS\n"); } while (0)
#define FAIL(msg) do { tests_failed++; printf("FAIL: %s\n", msg); } while (0)

static void test_normal_pass(void) {
    PASS();
}

static void test_ghost_no_record(void) {
    /* Deliberately falls off the end without PASS or FAIL.
     * The per-test delta in main() must catch this and exit 42. */
    volatile int x = 1 + 1;
    (void)x;
}

static const TestCase TESTS[] = {
    {"test_normal_pass",     test_normal_pass},
    {"test_ghost_no_record", test_ghost_no_record},
};
static const int N_TESTS = (int)TEST_REGISTRY_SIZE(TESTS);

int main(void) {
    for (int i = 0; i < N_TESTS; i++) {
        printf("[%d/%d] %s ... ", i + 1, N_TESTS, TESTS[i].name);
        fflush(stdout);
        int pre = tests_passed + tests_failed;
        TESTS[i].fn();
        int post = tests_passed + tests_failed;
        if (post == pre) {
            fprintf(stderr,
                "GHOST DETECTED (expected): test '%s' reported no result\n",
                TESTS[i].name);
            return 42;
        }
    }
    fprintf(stderr, "META-FAIL: ghost detector did not fire\n");
    return 0;
}
