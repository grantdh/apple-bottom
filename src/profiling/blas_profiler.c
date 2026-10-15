// =============================================================================
// blas_profiler.c — implementation
// =============================================================================
#include "blas_profiler.h"
#include "apple_bottom.h"

#include <mach/mach_time.h>
#include <os/lock.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// =============================================================================
// Build identity
// =============================================================================

// The Makefile passes the git commit, with "-dirty" if tracked files differed
// from it.
#ifndef AB_BUILD_SHA
#define AB_BUILD_SHA "unknown"
#endif

const char* ab_build_info(void) {
    return "apple-bottom " APPLE_BOTTOM_VERSION_STRING " (" AB_BUILD_SHA ")";
}

// =============================================================================
// State
// =============================================================================

static ab_prof_entry_t g_entries[AB_PROF_MAX_ROUTINES];
static int             g_entry_count = 0;
static os_unfair_lock  g_lock        = OS_UNFAIR_LOCK_INIT;
static pthread_once_t  g_env_once    = PTHREAD_ONCE_INIT;
static int             g_verbose     = 0;      // AB_PROFILE
static int             g_report      = 0;      // AB_REPORT
static uint64_t        g_threshold_ns = 0;     // AB_PROFILE_THRESHOLD_NS
static int             g_forked_child = 0;
static int             g_reported     = 0;     // exit reports written

static mach_timebase_info_data_t g_timebase = {0, 0};

const char* const ab_prof_stage_names[AB_PROF_STAGE_COUNT] = {
    "init", "alloc", "pack", "upload", "kernel", "download", "epilogue"
};

static const char* const fallback_names[AB_PROF_FB_COUNT] = {
    "no_device", "too_large", "gpu_failed", "other"
};

// =============================================================================
// Init, fork & env-var handling
// =============================================================================

// A fork()ed child inherits the parent's counters. It starts from zero, so
// its reports cover only its own calls and never repeat the parent's. The
// child has a single thread, so the lock is reset rather than taken.
static void ab_prof_atfork_child(void) {
    g_lock = OS_UNFAIR_LOCK_INIT;
    memset(g_entries, 0, sizeof(g_entries[0]) * (size_t)g_entry_count);
    g_entry_count = 0;
    g_forked_child = 1;
    g_reported = 0;
}

__attribute__((constructor))
static void ab_prof_ctor(void) {
    mach_timebase_info(&g_timebase);
    pthread_atfork(NULL, NULL, ab_prof_atfork_child);
}

static int env_flag(const char* name) {
    const char* e = getenv(name);
    return e && *e && *e != '0';
}

static void read_env(void) {
    g_verbose = env_flag("AB_PROFILE");
    g_report  = env_flag("AB_REPORT");
    const char* thresh = getenv("AB_PROFILE_THRESHOLD_NS");
    if (thresh) g_threshold_ns = strtoull(thresh, NULL, 10);
}

int ab_prof_enabled(void) {
    pthread_once(&g_env_once, read_env);
    return g_verbose || g_report;
}

int ab_prof_verbose(void) {
    pthread_once(&g_env_once, read_env);
    return g_verbose;
}

uint64_t ab_prof_now_ticks(void) {
    return mach_absolute_time();
}

uint64_t ab_prof_ticks_to_ns(uint64_t ticks) {
    if (g_timebase.denom == 0) mach_timebase_info(&g_timebase);
    return ticks * g_timebase.numer / g_timebase.denom;
}

// =============================================================================
// Record
// =============================================================================

static int dim_to_bin(size_t d) {
    if (d <=   32) return 0;
    if (d <=  128) return 1;
    if (d <=  512) return 2;
    if (d <= 1024) return 3;
    if (d <= 2048) return 4;
    if (d <= 4096) return 5;
    if (d <= 8192) return 6;
    return 7;
}

// Must be called with g_lock held.
static ab_prof_entry_t* find_or_create_locked(const char* name) {
    for (int i = 0; i < g_entry_count; i++) {
        if (strncmp(g_entries[i].name, name, AB_PROF_NAME_LEN) == 0) {
            return &g_entries[i];
        }
    }
    if (g_entry_count >= AB_PROF_MAX_ROUTINES) return NULL;
    ab_prof_entry_t* e = &g_entries[g_entry_count++];
    memset(e, 0, sizeof(*e));
    strncpy(e->name, name, AB_PROF_NAME_LEN - 1);
    e->name[AB_PROF_NAME_LEN - 1] = '\0';
    return e;
}

void ab_prof_record(const char* routine, uint64_t elapsed_ns,
                    size_t m, size_t n, size_t k, int used_gpu,
                    const uint64_t* stage_ns, ab_prof_fallback_t fallback) {
    if (!ab_prof_enabled()) return;
    if (elapsed_ns < g_threshold_ns) return;

    os_unfair_lock_lock(&g_lock);
    ab_prof_entry_t* e = find_or_create_locked(routine);
    if (e) {
        e->call_count++;
        e->total_ns += elapsed_ns;
        if (used_gpu) {
            e->gpu_calls++;
            e->gpu_ns += elapsed_ns;
            if (stage_ns)
                for (int s = 0; s < AB_PROF_STAGE_COUNT; s++) e->stage_ns[s] += stage_ns[s];
        } else {
            e->cpu_calls++;
            e->cpu_ns += elapsed_ns;
            if (fallback > AB_PROF_FB_NONE && fallback < AB_PROF_FB_COUNT)
                e->fallbacks[fallback]++;
        }

        size_t max_mn = m > n ? m : n;
        int bin = dim_to_bin(max_mn);
        e->dim_histogram[bin]++;
        e->ns_by_bin[bin]  += elapsed_ns;

        if (m > e->max_m) e->max_m = m;
        if (n > e->max_n) e->max_n = n;
        if (k > e->max_k) e->max_k = k;
        e->sum_m += m;
        e->sum_n += n;
        e->sum_k += k;
    }
    os_unfair_lock_unlock(&g_lock);
}

// =============================================================================
// Shared report helpers
// =============================================================================

static double pct(uint64_t part, uint64_t whole) {
    return whole ? 100.0 * (double)part / (double)whole : 0.0;
}

static uint64_t fallback_total(const ab_prof_entry_t* e) {
    uint64_t t = 0;
    for (int r = 0; r < AB_PROF_FB_COUNT; r++) t += e->fallbacks[r];
    return t;
}

// The GPU calls' wall time outside the named stages: mostly the kernel
// call's own cost besides GPU execution (encoding, submit and wait latency,
// ZGEMM's GPU-side temporaries).
static uint64_t stage_other(const ab_prof_entry_t* e) {
    uint64_t t = 0;
    for (int s = 0; s < AB_PROF_STAGE_COUNT; s++) t += e->stage_ns[s];
    return e->gpu_ns > t ? e->gpu_ns - t : 0;
}

// Entry indices by total_ns, descending.
static void sort_by_time(int idx[AB_PROF_MAX_ROUTINES]) {
    for (int i = 0; i < g_entry_count; i++) idx[i] = i;
    for (int i = 0; i < g_entry_count - 1; i++) {
        for (int j = i + 1; j < g_entry_count; j++) {
            if (g_entries[idx[j]].total_ns > g_entries[idx[i]].total_ns) {
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
        }
    }
}

// =============================================================================
// Human-readable dump (AB_PROFILE)
// =============================================================================

static const char* bin_labels[AB_PROF_HISTOGRAM_BINS] = {
    "1-32", "33-128", "129-512", "513-1K",
    "1K-2K", "2K-4K", "4K-8K",  "8K+"
};

void ab_prof_dump(FILE* out) {
    fprintf(out, "=== %s BLAS profile, pid %d", ab_build_info(), (int)getpid());
    if (g_threshold_ns)
        fprintf(out, ", calls under %llu ns not counted", (unsigned long long)g_threshold_ns);
    fprintf(out, " ===\n");
    if (g_entry_count == 0) {
        fprintf(out, "No BLAS-ABI calls recorded.\n\n");
        return;
    }
    fprintf(out, "gpu_calls%%: share of calls whose C the GPU computed; "
                 "gpu_time%%: share of wall time spent in those calls.\n");
    fprintf(out, "%-12s %10s %10s %11s %12s %12s %12s %10s %10s %10s %9s %9s\n",
            "routine", "calls", "gpu_calls", "gpu_calls%", "total_ms", "gpu_ms", "cpu_ms",
            "gpu_time%", "fallbacks", "avg_us", "max_dim", "hotbin");
    fprintf(out, "----------------------------------------------------------------"
                 "--------------------------------------------------------------\n");

    int idx[AB_PROF_MAX_ROUTINES];
    sort_by_time(idx);

    uint64_t grand_total_ns = 0, any_gpu = 0, any_fallback = 0;
    for (int i = 0; i < g_entry_count; i++) {
        grand_total_ns += g_entries[i].total_ns;
        any_gpu += g_entries[i].gpu_calls;
        any_fallback += fallback_total(&g_entries[i]);
    }

    for (int i = 0; i < g_entry_count; i++) {
        ab_prof_entry_t* e = &g_entries[idx[i]];
        if (e->call_count == 0) continue;

        double avg_us   = (e->total_ns / 1e3) / e->call_count;
        size_t max_dim  = e->max_m > e->max_n ? e->max_m : e->max_n;

        // Hottest bin by time
        int hot = 0;
        for (int b = 1; b < AB_PROF_HISTOGRAM_BINS; b++)
            if (e->ns_by_bin[b] > e->ns_by_bin[hot]) hot = b;

        fprintf(out, "%-12s %10llu %10llu %10.1f%% %12.3f %12.3f %12.3f %9.1f%% %10llu %10.1f %9zu %9s\n",
                e->name,
                (unsigned long long)e->call_count,
                (unsigned long long)e->gpu_calls,
                pct(e->gpu_calls, e->call_count),
                e->total_ns / 1e6,
                e->gpu_ns / 1e6,
                e->cpu_ns / 1e6,
                pct(e->gpu_ns, e->total_ns),
                (unsigned long long)fallback_total(e),
                avg_us,
                max_dim,
                bin_labels[hot]);
    }

    if (any_gpu) {
        fprintf(out, "\nGPU calls by stage, ms (kernel: command-buffer GPU time; "
                     "other: rest of gpu_ms):\n%-12s", "routine");
        for (int s = 0; s < AB_PROF_STAGE_COUNT; s++)
            fprintf(out, " %10s", ab_prof_stage_names[s]);
        fprintf(out, " %10s %12s\n", "other", "gpu_ms");
        for (int i = 0; i < g_entry_count; i++) {
            ab_prof_entry_t* e = &g_entries[idx[i]];
            if (e->gpu_calls == 0) continue;
            fprintf(out, "%-12s", e->name);
            for (int s = 0; s < AB_PROF_STAGE_COUNT; s++)
                fprintf(out, " %10.3f", e->stage_ns[s] / 1e6);
            fprintf(out, " %10.3f %12.3f\n", stage_other(e) / 1e6, e->gpu_ns / 1e6);
        }
    }

    if (any_fallback) {
        fprintf(out, "\nFallbacks: GPU-routed calls that ran on cblas (in cpu_ms):\n%-12s",
                "routine");
        for (int r = 0; r < AB_PROF_FB_COUNT; r++) fprintf(out, " %10s", fallback_names[r]);
        fprintf(out, "\n");
        for (int i = 0; i < g_entry_count; i++) {
            ab_prof_entry_t* e = &g_entries[idx[i]];
            if (fallback_total(e) == 0) continue;
            fprintf(out, "%-12s", e->name);
            for (int r = 0; r < AB_PROF_FB_COUNT; r++)
                fprintf(out, " %10llu", (unsigned long long)e->fallbacks[r]);
            fprintf(out, "\n");
        }
    }

    fprintf(out, "\nTotal BLAS time: %.1f ms across %d routines\n",
            grand_total_ns / 1e6, g_entry_count);

    // Dimension distribution table for top routine (most time)
    ab_prof_entry_t* top = &g_entries[idx[0]];
    if (top->call_count > 0) {
        fprintf(out, "\nDimension distribution for '%s' (by max(M,N)):\n", top->name);
        fprintf(out, "  bin         calls      time_ms   pct_time\n");
        for (int b = 0; b < AB_PROF_HISTOGRAM_BINS; b++) {
            if (top->dim_histogram[b] == 0) continue;
            fprintf(out, "  %-9s %10llu %12.1f %8.1f%%\n",
                    bin_labels[b],
                    (unsigned long long)top->dim_histogram[b],
                    top->ns_by_bin[b] / 1e6,
                    pct(top->ns_by_bin[b], top->total_ns));
        }
    }
    fprintf(out, "\n");
}

// =============================================================================
// Exit summary (AB_REPORT)
// =============================================================================

// One report line, built in a buffer and written with a single call, so the
// lines of MPI ranks that share stderr do not interleave mid-line.
typedef struct { char buf[512]; size_t n; } report_line_t;

static void line_add(report_line_t* l, const char* fmt, ...) {
    if (l->n >= sizeof l->buf - 1) return;
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(l->buf + l->n, sizeof l->buf - l->n, fmt, ap);
    va_end(ap);
    if (k > 0) l->n += (size_t)k;
    if (l->n > sizeof l->buf - 1) l->n = sizeof l->buf - 1;
}

static void line_emit(report_line_t* l, FILE* out) {
    fputs(l->buf, out);
    l->n = 0;
    l->buf[0] = '\0';
}

// Line-oriented key=value lines, each tagged with the pid so MPI ranks that
// share stderr stay apart.
void ab_prof_report(FILE* out) {
    const char* mode = getenv("AB_MODE");
    int pid = (int)getpid();
    report_line_t l = { .n = 0 };
    l.buf[0] = '\0';
    line_add(&l, "AB_REPORT[%d] %s AB_MODE=%s", pid, ab_build_info(),
             mode && *mode ? mode : "unset");
    if (g_threshold_ns)
        line_add(&l, " AB_PROFILE_THRESHOLD_NS=%llu", (unsigned long long)g_threshold_ns);
    line_add(&l, "\n");
    line_emit(&l, out);
    if (g_entry_count == 0) {
        line_add(&l, "AB_REPORT[%d] no BLAS-ABI calls recorded\n", pid);
        line_emit(&l, out);
        return;
    }

    int idx[AB_PROF_MAX_ROUTINES];
    sort_by_time(idx);
    for (int i = 0; i < g_entry_count; i++) {
        ab_prof_entry_t* e = &g_entries[idx[i]];
        if (e->call_count == 0) continue;
        line_add(&l, "AB_REPORT[%d] %s calls=%llu gpu_calls=%llu cpu_calls=%llu "
                     "gpu_calls%%=%.1f total_ms=%.3f gpu_ms=%.3f cpu_ms=%.3f "
                     "gpu_time%%=%.1f fallbacks=%llu\n",
                 pid, e->name,
                 (unsigned long long)e->call_count,
                 (unsigned long long)e->gpu_calls,
                 (unsigned long long)e->cpu_calls,
                 pct(e->gpu_calls, e->call_count),
                 e->total_ns / 1e6, e->gpu_ns / 1e6, e->cpu_ns / 1e6,
                 pct(e->gpu_ns, e->total_ns),
                 (unsigned long long)fallback_total(e));
        line_emit(&l, out);
        if (e->gpu_calls) {
            line_add(&l, "AB_REPORT[%d] %s gpu_stages_ms", pid, e->name);
            for (int s = 0; s < AB_PROF_STAGE_COUNT; s++)
                line_add(&l, " %s=%.3f", ab_prof_stage_names[s], e->stage_ns[s] / 1e6);
            line_add(&l, " other=%.3f\n", stage_other(e) / 1e6);
            line_emit(&l, out);
        }
        if (fallback_total(e)) {
            line_add(&l, "AB_REPORT[%d] %s fallbacks", pid, e->name);
            for (int r = 0; r < AB_PROF_FB_COUNT; r++)
                line_add(&l, " %s=%llu", fallback_names[r], (unsigned long long)e->fallbacks[r]);
            line_add(&l, "\n");
            line_emit(&l, out);
        }
    }
}

// =============================================================================
// JSON dump
// =============================================================================

void ab_prof_dump_json(const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "ab_prof: failed to open %s for JSON dump\n", path);
        return;
    }

    // Timestamp
    time_t now = time(NULL);
    struct tm tmv;
    gmtime_r(&now, &tmv);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tmv);

    fprintf(f, "{\n");
    fprintf(f, "  \"build\": \"%s\",\n", ab_build_info());
    fprintf(f, "  \"timestamp\": \"%s\",\n", ts);
    fprintf(f, "  \"pid\": %d,\n", (int)getpid());
    fprintf(f, "  \"threshold_ns\": %llu,\n", (unsigned long long)g_threshold_ns);
    fprintf(f, "  \"routines\": [\n");

    int first = 1;
    for (int i = 0; i < g_entry_count; i++) {
        ab_prof_entry_t* e = &g_entries[i];
        if (e->call_count == 0) continue;
        if (!first) fprintf(f, ",\n");
        first = 0;

        double avg_m = (double)e->sum_m / e->call_count;
        double avg_n = (double)e->sum_n / e->call_count;
        double avg_k = (double)e->sum_k / e->call_count;

        fprintf(f,
            "    {\n"
            "      \"name\": \"%s\",\n"
            "      \"calls\": %llu,\n"
            "      \"total_ns\": %llu,\n"
            "      \"gpu_calls\": %llu, \"cpu_calls\": %llu,\n"
            "      \"gpu_ns\": %llu, \"cpu_ns\": %llu,\n"
            "      \"gpu_stage_ns\": {",
            e->name,
            (unsigned long long)e->call_count,
            (unsigned long long)e->total_ns,
            (unsigned long long)e->gpu_calls,
            (unsigned long long)e->cpu_calls,
            (unsigned long long)e->gpu_ns,
            (unsigned long long)e->cpu_ns);
        for (int s = 0; s < AB_PROF_STAGE_COUNT; s++)
            fprintf(f, "\"%s\": %llu, ", ab_prof_stage_names[s],
                    (unsigned long long)e->stage_ns[s]);
        fprintf(f, "\"other\": %llu},\n      \"fallbacks\": {",
                (unsigned long long)stage_other(e));
        for (int r = 0; r < AB_PROF_FB_COUNT; r++)
            fprintf(f, "%s\"%s\": %llu", r ? ", " : "", fallback_names[r],
                    (unsigned long long)e->fallbacks[r]);
        fprintf(f,
            "},\n"
            "      \"max_m\": %zu, \"max_n\": %zu, \"max_k\": %zu,\n"
            "      \"avg_m\": %.1f, \"avg_n\": %.1f, \"avg_k\": %.1f,\n"
            "      \"dim_histogram\": [",
            e->max_m, e->max_n, e->max_k,
            avg_m, avg_n, avg_k);

        for (int b = 0; b < AB_PROF_HISTOGRAM_BINS; b++) {
            fprintf(f, "%s%llu", b ? ", " : "",
                    (unsigned long long)e->dim_histogram[b]);
        }
        fprintf(f, "],\n      \"ns_by_bin\": [");
        for (int b = 0; b < AB_PROF_HISTOGRAM_BINS; b++) {
            fprintf(f, "%s%llu", b ? ", " : "",
                    (unsigned long long)e->ns_by_bin[b]);
        }
        fprintf(f, "]\n    }");
    }

    fprintf(f, "\n  ]\n}\n");
    fclose(f);
}

// =============================================================================
// Exit hook
// =============================================================================

// Once per process: the AB_PROFILE table (and JSON), then the AB_REPORT
// summary. A fork()ed child reports only if it made calls of its own.
__attribute__((destructor))
static void ab_prof_atexit(void) {
    if (!ab_prof_enabled() || __atomic_exchange_n(&g_reported, 1, __ATOMIC_ACQ_REL)) return;
    if (g_forked_child && g_entry_count == 0) return;

    if (g_verbose) {
        ab_prof_dump(stderr);
        const char* json_path = getenv("AB_PROFILE_JSON");
        if (json_path && *json_path) {
            ab_prof_dump_json(json_path);
            fprintf(stderr, "ab_prof: JSON written to %s\n", json_path);
        }
    }
    if (g_report) ab_prof_report(stderr);
}
