// =============================================================================
// blas_profiler.c — implementation
// =============================================================================
#include "blas_profiler.h"

#include <mach/mach_time.h>
#include <os/lock.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// =============================================================================
// State
// =============================================================================

static ab_prof_entry_t g_entries[AB_PROF_MAX_ROUTINES];
static int             g_entry_count = 0;
static os_unfair_lock  g_lock        = OS_UNFAIR_LOCK_INIT;
static int             g_enabled     = -1;     // -1 = unchecked, 0/1 = cached
static uint64_t        g_threshold_ns = 0;     // AB_PROFILE_THRESHOLD_NS

static mach_timebase_info_data_t g_timebase = {0, 0};

// =============================================================================
// Init & env-var handling
// =============================================================================

__attribute__((constructor))
static void ab_prof_ctor(void) {
    mach_timebase_info(&g_timebase);
}

int ab_prof_enabled(void) {
    if (g_enabled == -1) {
        const char* e = getenv("AB_PROFILE");
        g_enabled = (e && *e && *e != '0') ? 1 : 0;

        const char* thresh = getenv("AB_PROFILE_THRESHOLD_NS");
        if (thresh) {
            g_threshold_ns = strtoull(thresh, NULL, 10);
        }
    }
    return g_enabled;
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
                    size_t m, size_t n, size_t k, int used_gpu) {
    if (!ab_prof_enabled()) return;
    if (elapsed_ns < g_threshold_ns) return;

    os_unfair_lock_lock(&g_lock);
    ab_prof_entry_t* e = find_or_create_locked(routine);
    if (e) {
        e->call_count++;
        e->total_ns += elapsed_ns;
        if (used_gpu) e->gpu_dispatch_count++;
        else          e->amx_fallback_count++;

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
// Human-readable dump
// =============================================================================

static const char* bin_labels[AB_PROF_HISTOGRAM_BINS] = {
    "1-32", "33-128", "129-512", "513-1K",
    "1K-2K", "2K-4K", "4K-8K",  "8K+"
};

void ab_prof_dump(FILE* out) {
    if (g_entry_count == 0) return;

    fprintf(out, "\n=== apple-bottom BLAS profile ===\n");
    fprintf(out, "%-12s %10s %12s %9s %9s %9s %9s\n",
            "routine", "calls", "total_ms", "gpu_pct", "avg_us", "max_dim", "hotbin");
    fprintf(out, "---------------------------------------------------------------------------------\n");

    // Sort indices by total_ns desc
    int idx[AB_PROF_MAX_ROUTINES];
    for (int i = 0; i < g_entry_count; i++) idx[i] = i;
    for (int i = 0; i < g_entry_count - 1; i++) {
        for (int j = i + 1; j < g_entry_count; j++) {
            if (g_entries[idx[j]].total_ns > g_entries[idx[i]].total_ns) {
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
        }
    }

    uint64_t grand_total_ns = 0;
    for (int i = 0; i < g_entry_count; i++)
        grand_total_ns += g_entries[i].total_ns;

    for (int i = 0; i < g_entry_count; i++) {
        ab_prof_entry_t* e = &g_entries[idx[i]];
        if (e->call_count == 0) continue;

        double total_ms = e->total_ns / 1e6;
        double gpu_pct  = 100.0 * e->gpu_dispatch_count / e->call_count;
        double avg_us   = (e->total_ns / 1e3) / e->call_count;
        size_t max_dim  = e->max_m > e->max_n ? e->max_m : e->max_n;

        // Hottest bin by time
        int hot = 0;
        for (int b = 1; b < AB_PROF_HISTOGRAM_BINS; b++)
            if (e->ns_by_bin[b] > e->ns_by_bin[hot]) hot = b;

        fprintf(out, "%-12s %10llu %12.1f %8.1f%% %9.1f %9zu %9s\n",
                e->name,
                (unsigned long long)e->call_count,
                total_ms,
                gpu_pct,
                avg_us,
                max_dim,
                bin_labels[hot]);
    }

    fprintf(out, "\nTotal BLAS time: %.1f ms across %d routines\n",
            grand_total_ns / 1e6, g_entry_count);

    // Dimension distribution table for top routine (most time)
    if (g_entry_count > 0) {
        ab_prof_entry_t* top = &g_entries[idx[0]];
        if (top->call_count > 0) {
            fprintf(out, "\nDimension distribution for '%s' (by max(M,N)):\n", top->name);
            fprintf(out, "  bin         calls      time_ms   pct_time\n");
            for (int b = 0; b < AB_PROF_HISTOGRAM_BINS; b++) {
                if (top->dim_histogram[b] == 0) continue;
                double pct = top->total_ns > 0
                    ? 100.0 * top->ns_by_bin[b] / top->total_ns : 0;
                fprintf(out, "  %-9s %10llu %12.1f %8.1f%%\n",
                        bin_labels[b],
                        (unsigned long long)top->dim_histogram[b],
                        top->ns_by_bin[b] / 1e6,
                        pct);
            }
        }
    }
    fprintf(out, "\n");
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
    fprintf(f, "  \"timestamp\": \"%s\",\n", ts);
    fprintf(f, "  \"pid\": %d,\n", (int)getpid());
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
            "      \"gpu_dispatch_count\": %llu,\n"
            "      \"amx_fallback_count\": %llu,\n"
            "      \"max_m\": %zu, \"max_n\": %zu, \"max_k\": %zu,\n"
            "      \"avg_m\": %.1f, \"avg_n\": %.1f, \"avg_k\": %.1f,\n"
            "      \"dim_histogram\": [",
            e->name,
            (unsigned long long)e->call_count,
            (unsigned long long)e->total_ns,
            (unsigned long long)e->gpu_dispatch_count,
            (unsigned long long)e->amx_fallback_count,
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
// Atexit hook
// =============================================================================

__attribute__((destructor))
static void ab_prof_atexit(void) {
    if (!ab_prof_enabled()) return;

    ab_prof_dump(stderr);

    const char* json_path = getenv("AB_PROFILE_JSON");
    if (json_path && *json_path) {
        ab_prof_dump_json(json_path);
        fprintf(stderr, "ab_prof: JSON written to %s\n", json_path);
    }
}
