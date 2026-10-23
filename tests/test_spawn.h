#ifndef AB_TEST_SPAWN_H
#define AB_TEST_SPAWN_H

// Running a test binary again as a child process, for tests that need a
// fresh environment (test_blas_abi.c, test_trace.c). posix_spawn, not a bare
// fork(): Metal cannot reach MTLCompilerService from a forked child that did
// not exec.

#include "test_blas_check.h"   // environ, is_ab_env()
#include <errno.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// The caller's environment minus every entry is_ab_env() matches and its
// MallocNanoZone (test_blas_abi's memory suite sets one), so a child sees
// only the settings a test gives it, with room for `extra` more entries and
// the terminating NULL. *n is the number of entries copied; the strings are
// the caller's, the array is to free().
static inline char **child_environ(size_t extra, size_t *n) {
    size_t ne = 0;
    while (environ[ne]) ne++;
    char **env = calloc(ne + extra + 1, sizeof *env);
    size_t k = 0;
    for (size_t i = 0; i < ne; i++)
        if (!is_ab_env(environ[i]) && strncmp(environ[i], "MallocNanoZone=", 15))
            env[k++] = environ[i];
    *n = k;
    return env;
}

// Spawns argv[0] with env, its stdout and stderr on a pipe. Returns the read
// end, or NULL with the failure in why.
static inline FILE *spawn_piped(char *const argv[], char *const env[], pid_t *pid,
                                char *why, size_t why_len) {
    int fds[2];
    if (pipe(fds) != 0) {
        snprintf(why, why_len, "pipe: %s", strerror(errno));
        return NULL;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_addclose(&fa, fds[1]);
    fflush(stdout);
    int rc = posix_spawn(pid, argv[0], &fa, NULL, argv, env);
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);
    if (rc != 0) {
        close(fds[0]);
        snprintf(why, why_len, "posix_spawn: %s", strerror(rc));
        return NULL;
    }
    return fdopen(fds[0], "r");
}

// Everything left to read from in, which is closed, as a string to free().
static inline char *slurp(FILE *in) {
    char *s = NULL, buf[4096];
    size_t len = 0;
    FILE *mem = open_memstream(&s, &len);
    for (size_t got; (got = fread(buf, 1, sizeof buf, in)) > 0; ) fwrite(buf, 1, got, mem);
    fclose(in);
    fclose(mem);
    return s;
}

#endif /* AB_TEST_SPAWN_H */
