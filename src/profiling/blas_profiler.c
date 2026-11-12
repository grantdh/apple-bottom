// =============================================================================
// blas_profiler.c — implementation
// =============================================================================
#include "blas_profiler.h"
#include "apple_bottom.h"
#include "dispatch.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach/mach_time.h>
#include <os/lock.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
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
static int             g_trace        = 0;     // AB_TRACE
static int             g_trace_hash   = 0;     // AB_TRACE_HASH, with AB_TRACE
static int             g_fail_stage   = -1;    // AB_TEST_FAIL_STAGE
static long            g_fail_only    = 0;     // AB_TEST_FAIL_ONLY; 0: every command buffer
static atomic_long     g_fail_seen;            // command buffers ab_test_fail_cmdbuf has seen
static int             g_host_pack    = 0;     // AB_HOST_PACK
static size_t          g_staging_cap  = (size_t)2048 << 20;   // AB_STAGING_MB, in bytes

// The trace file of this process, <AB_TRACE>.<pid>, opened at its first line.
// A fork()ed child closes the one it inherits (ab_prof_atfork_child).
static char            g_trace_path[PATH_MAX];
static os_unfair_lock  g_trace_lock   = OS_UNFAIR_LOCK_INIT;
static int             g_trace_fd     = -2;    // -2 not opened yet, -1 open failed
static uint64_t        g_trace_seq    = 0;     // lines this process wrote

static mach_timebase_info_data_t g_timebase = {0, 0};

const char* const ab_prof_stage_names[AB_PROF_STAGE_COUNT] = {
    "init", "alloc", "pack", "upload", "kernel", "download", "epilogue"
};

static const char* const fallback_names[AB_PROF_FB_COUNT] = {
    "no_device", "too_large", "gpu_failed", "other", "range"
};

const char* const ab_prof_staged_names[AB_PROF_STAGED_COUNT] = {
    "kept", "kept_4m", "host_pack", "fpcr", "scalar", "alloc", "gpu_failed", "nonfinite", "range"
};

const char* const ab_prof_split_decline_names[AB_PROF_SPLIT_DECLINE_COUNT] = {
    "no_device", "too_large", "host_pack", "fpcr", "scalar", "scratch", "thread"
};

// AB_SPLIT: per row of the split table, the calls it covered (also those it
// did not cut) and what became of them (AB_REPORT's split_row lines). first_ns
// is the wall time of the first call the row covered; first_cpu_ns and the
// other first_ fields are the parts of its first split call, whose GPU part
// carries Metal's start-up when it is the process's first, so that a fit of
// the shares can leave that call out.
typedef struct {
    uint64_t calls, split, kept, kept_4m, range, gpu_failed;
    uint64_t wall_ns, first_ns, cpu_ns, gpu_ns, wait_cpu_ns, wait_gpu_ns, redo_ns, sum_ns;
    uint64_t first_cpu_ns, first_gpu_ns;
    double gf, gf_cpu, gf_gpu, first_gf_cpu, first_gf_gpu;
} split_row_stats_t;

static split_row_stats_t g_split_rows[AB_SPLIT_MAX_ROWS];   // under g_lock

// =============================================================================
// Init, fork & env-var handling
// =============================================================================

// A fork()ed child inherits the parent's counters. It starts from zero, so
// its reports cover only its own calls and never repeat the parent's. The
// child has a single thread, so the locks are reset rather than taken. It
// closes the parent's trace file now, while the descriptor is certainly
// still that file, and its first trace line opens one of its own.
static void ab_prof_atfork_child(void) {
    g_lock = OS_UNFAIR_LOCK_INIT;
    g_trace_lock = OS_UNFAIR_LOCK_INIT;
    memset(g_entries, 0, sizeof(g_entries[0]) * (size_t)g_entry_count);
    g_entry_count = 0;
    memset(g_split_rows, 0, sizeof g_split_rows);
    g_forked_child = 1;
    g_reported = 0;
    if (g_trace_fd >= 0) close(g_trace_fd);
    g_trace_fd = -2;
    g_trace_seq = 0;
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

// AB_TEST_FAIL_STAGE names a GPU step that can fail: alloc or kernel, or one
// of the other failures of blas_profiler.h.
static int parse_fail_stage(const char* name) {
    static const struct { const char* name; int stage; } OTHER[] = {
        {"kernel_once", AB_PROF_FAIL_KERNEL_ONCE}, {"redo_4m", AB_PROF_FAIL_REDO_4M},
        {"scales", AB_PROF_FAIL_SCALES}, {"split_thread", AB_PROF_FAIL_SPLIT_THREAD},
        {"split_scratch", AB_PROF_FAIL_SPLIT_SCRATCH}, {"split_fpcr", AB_PROF_FAIL_SPLIT_FPCR},
        {"split_poison", AB_PROF_FAIL_SPLIT_POISON}, {"nonfinite", AB_PROF_FAIL_NONFINITE},
    };
    if (!name || !*name) return -1;
    if (!strcmp(name, ab_prof_stage_names[AB_PROF_STAGE_ALLOC])) return AB_PROF_STAGE_ALLOC;
    if (!strcmp(name, ab_prof_stage_names[AB_PROF_STAGE_KERNEL])) return AB_PROF_STAGE_KERNEL;
    for (size_t i = 0; i < sizeof OTHER / sizeof OTHER[0]; i++)
        if (!strcmp(name, OTHER[i].name)) return OTHER[i].stage;
    fprintf(stderr, "apple-bottom: AB_TEST_FAIL_STAGE=%s: expected alloc, kernel, kernel_once, "
                    "redo_4m, scales, split_thread, split_scratch, split_fpcr, split_poison or nonfinite; ignored\n", name);
    return -1;
}

static void read_env(void) {
    g_verbose = env_flag("AB_PROFILE");
    g_report  = env_flag("AB_REPORT");
    const char* thresh = getenv("AB_PROFILE_THRESHOLD_NS");
    if (thresh) g_threshold_ns = strtoull(thresh, NULL, 10);
    // The trace path is copied: the environment may change later. Room is
    // left for the ".<pid>" suffix.
    const char* trace = getenv("AB_TRACE");
    if (trace && *trace) {
        if (strlen(trace) + 16 < sizeof g_trace_path) {
            strcpy(g_trace_path, trace);
            g_trace = 1;
        } else {
            fprintf(stderr, "apple-bottom: AB_TRACE: path too long; no trace\n");
        }
    }
    g_trace_hash = g_trace && env_flag("AB_TRACE_HASH");
    g_fail_stage = parse_fail_stage(getenv("AB_TEST_FAIL_STAGE"));
    const char* only = getenv("AB_TEST_FAIL_ONLY");
    if (only) g_fail_only = atol(only);
    g_host_pack = env_flag("AB_HOST_PACK");
    const char* staging = getenv("AB_STAGING_MB");
    if (staging && *staging) {
        long long mb = atoll(staging);
        g_staging_cap = mb > 0 ? (size_t)mb << 20 : 0;
    }
}

int ab_prof_enabled(void) {
    pthread_once(&g_env_once, read_env);
    return g_verbose || g_report;
}

int ab_prof_verbose(void) {
    pthread_once(&g_env_once, read_env);
    return g_verbose;
}

int ab_prof_active(void) {
    pthread_once(&g_env_once, read_env);
    return g_verbose || g_report || g_trace;
}

int ab_trace_enabled(void) {
    pthread_once(&g_env_once, read_env);
    return g_trace;
}

int ab_trace_hash_enabled(void) {
    pthread_once(&g_env_once, read_env);
    return g_trace_hash;
}

int ab_test_fail_stage(void) {
    pthread_once(&g_env_once, read_env);
    return g_fail_stage;
}

int ab_test_fail_cmdbuf(void) {
    if (ab_test_fail_stage() != AB_PROF_STAGE_KERNEL) return 0;
    return !g_fail_only || atomic_fetch_add(&g_fail_seen, 1) + 1 == g_fail_only;
}

int ab_host_pack(void) {
    pthread_once(&g_env_once, read_env);
    return g_host_pack;
}

size_t ab_staging_cap_bytes(void) {
    pthread_once(&g_env_once, read_env);
    return g_staging_cap;
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
                    const uint64_t* stage_ns, ab_prof_fallback_t fallback,
                    ab_prof_staged_t staged) {
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
        if (staged > AB_PROF_STAGED_NONE && staged < AB_PROF_STAGED_COUNT) e->staged[staged]++;

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

void ab_prof_record_split(const char* routine, uint64_t elapsed_ns, size_t m, size_t n, size_t k,
                          const ab_prof_split_t* s) {
    if (!ab_prof_enabled() || s->row < 0 || s->row >= AB_SPLIT_MAX_ROWS) return;
    if (elapsed_ns < g_threshold_ns) return;
    const double gf = (routine[0] == 'z' ? 8.0 : 2.0) * (double)m * (double)n * (double)k / 1e9;
    os_unfair_lock_lock(&g_lock);
    split_row_stats_t* r = &g_split_rows[s->row];
    if (!r->calls) r->first_ns = elapsed_ns;
    r->calls++;
    r->wall_ns += elapsed_ns;
    r->gf += gf;
    ab_prof_entry_t* e = find_or_create_locked(routine);
    if (e && s->kept) {   // the call itself, as ab_prof_record counts a call
        e->call_count++;
        e->total_ns += elapsed_ns;
        e->split_calls++;
        e->split_ns += elapsed_ns;
        size_t max_mn = m > n ? m : n;
        int bin = dim_to_bin(max_mn);
        e->dim_histogram[bin]++;
        e->ns_by_bin[bin] += elapsed_ns;
        if (m > e->max_m) e->max_m = m;
        if (n > e->max_n) e->max_n = n;
        if (k > e->max_k) e->max_k = k;
        e->sum_m += m;
        e->sum_n += n;
        e->sum_k += k;
    }
    ab_prof_split_stats_t* x = e ? &e->split : NULL;
    if (x && (s->split || s->declined > AB_PROF_SPLIT_DECLINE_NONE)) x->calls++;
    if (x && s->declined > AB_PROF_SPLIT_DECLINE_NONE && s->declined < AB_PROF_SPLIT_DECLINE_COUNT)
        x->declined[s->declined]++;
    if (s->split) {
        bool k4 = s->staged == AB_PROF_STAGED_KEPT_4M, range = s->staged == AB_PROF_STAGED_RANGE;
        if (!r->split) {
            r->first_cpu_ns = s->cpu_part_ns;
            r->first_gpu_ns = s->gpu_wall_ns;
            r->first_gf_cpu = s->gf_cpu;
            r->first_gf_gpu = s->gf_gpu;
        }
        r->split++;
        r->kept += s->kept && !k4;
        r->kept_4m += s->kept && k4;
        r->range += !s->kept && range;
        r->gpu_failed += !s->kept && !range;
        r->cpu_ns += s->cpu_part_ns;
        r->gpu_ns += s->gpu_wall_ns;
        r->wait_cpu_ns += s->wait_cpu_ns;
        r->wait_gpu_ns += s->wait_gpu_ns;
        r->redo_ns += s->redo_ns;
        r->sum_ns += s->sum_ns;
        r->gf_cpu += s->gf_cpu;
        r->gf_gpu += s->gf_gpu;
    }
    if (x && s->split) {
        bool k4 = s->staged == AB_PROF_STAGED_KEPT_4M, range = s->staged == AB_PROF_STAGED_RANGE;
        x->kept += s->kept && !k4;
        x->kept_4m += s->kept && k4;
        x->range += !s->kept && range;
        x->gpu_failed += !s->kept && !range;
        x->scales += !s->kept && s->scales;
        x->alloc += !s->kept && s->staged == AB_PROF_STAGED_ALLOC;
        x->nonfinite += !s->kept && s->staged == AB_PROF_STAGED_NONFINITE;
        x->fpcr += !s->kept && s->staged == AB_PROF_STAGED_FPCR;
        x->retries += (uint64_t)s->retries;
        x->retry_kept += s->retry_kept;
        x->axis[s->axis == 'N' ? 0 : s->axis == 'M' ? 1 : 2]++;
        x->init_ns += s->init_ns;
        x->cpu_ns += s->cpu_part_ns;
        x->gpu_ns += s->gpu_wall_ns;
        x->wait_cpu_ns += s->wait_cpu_ns;
        x->wait_gpu_ns += s->wait_gpu_ns;
        x->redo_ns += s->redo_ns;
        x->sum_ns += s->sum_ns;
        if (!s->kept) x->wasted_gpu_ns += s->gpu_wall_ns;
        x->gf_cpu += s->gf_cpu;
        x->gf_gpu += s->gf_gpu;
    }
    os_unfair_lock_unlock(&g_lock);
}

// =============================================================================
// Shared report helpers
// =============================================================================

// " AB_SPLIT=1 split_table=<source>" for the report headers when AB_SPLIT is
// on, else nothing, so that default reports read as before.
static const char* split_token(void) {
    static char token[sizeof " AB_SPLIT=1 split_table=" + sizeof ((ab_dispatch_config_t*)0)->split_table];
    const ab_dispatch_config_t* dc = ab_dispatch_process_config();
    if (!dc->split_on) return "";
    snprintf(token, sizeof token, " AB_SPLIT=1 split_table=%s", dc->split_table);
    return token;
}

static bool split_on(void) { return ab_dispatch_process_config()->split_on; }

#define LLU(x) ((unsigned long long)(x))

// One report line (or a trace line, or the trace header), built in a buffer
// and written with a single call, so the lines of MPI ranks that share
// stderr do not interleave mid-line.
typedef struct { char buf[2048]; size_t n; } report_line_t;

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

static void report_split(report_line_t* l, FILE* out, const char* prefix, const ab_prof_entry_t* e);
static void report_split_rows(report_line_t* l, FILE* out, const char* prefix);

static double pct(uint64_t part, uint64_t whole) {
    return whole ? 100.0 * (double)part / (double)whole : 0.0;
}

// A routine's counts by reason: its fallbacks or its staged-path outcomes.
typedef struct {
    const char* const* names;
    int n;
    const uint64_t* (*of)(const ab_prof_entry_t* e);
} counts_t;

static const uint64_t* fallbacks_of(const ab_prof_entry_t* e) { return e->fallbacks; }
static const uint64_t* staged_of(const ab_prof_entry_t* e) { return e->staged; }

static const counts_t FALLBACKS = {fallback_names, AB_PROF_FB_COUNT, fallbacks_of};
static const counts_t STAGED = {ab_prof_staged_names, AB_PROF_STAGED_COUNT, staged_of};

static uint64_t counts_total(const counts_t* k, const ab_prof_entry_t* e) {
    uint64_t t = 0;
    for (int r = 0; r < k->n; r++) t += k->of(e)[r];
    return t;
}

static uint64_t fallback_total(const ab_prof_entry_t* e) { return counts_total(&FALLBACKS, e); }

// The AB_PROFILE table of one kind of counts, if any routine has some: a
// column per reason, a row per such routine, in idx order.
static void dump_counts(FILE* out, const char* title, const counts_t* k, const int* idx) {
    uint64_t any = 0;
    for (int i = 0; i < g_entry_count; i++) any += counts_total(k, &g_entries[i]);
    if (!any) return;
    fprintf(out, "\n%s:\n%-12s", title, "routine");
    for (int r = 0; r < k->n; r++) fprintf(out, " %10s", k->names[r]);
    fprintf(out, "\n");
    for (int i = 0; i < g_entry_count; i++) {
        ab_prof_entry_t* e = &g_entries[idx[i]];
        if (counts_total(k, e) == 0) continue;
        fprintf(out, "%-12s", e->name);
        for (int r = 0; r < k->n; r++) fprintf(out, " %10llu", (unsigned long long)k->of(e)[r]);
        fprintf(out, "\n");
    }
}

// The part of a GPU time outside its named stages, "other".
static uint64_t other_ns(uint64_t gpu_ns, const uint64_t stage_ns[AB_PROF_STAGE_COUNT]) {
    uint64_t t = 0;
    for (int s = 0; s < AB_PROF_STAGE_COUNT; s++) t += stage_ns[s];
    return gpu_ns > t ? gpu_ns - t : 0;
}

// The GPU calls' wall time outside the named stages: mostly the kernel
// call's own cost besides GPU execution (encoding, submit and wait latency,
// ZGEMM's GPU-side temporaries).
static uint64_t stage_other(const ab_prof_entry_t* e) {
    return other_ns(e->gpu_ns, e->stage_ns);
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

// " AB_PRECISION=fast" for the report headers in fast mode, else nothing, so
// default-mode reports read as before.
static const char* precision_token(void) {
    return strcmp(ab_precision_mode(), "fast") == 0 ? " AB_PRECISION=fast" : "";
}

void ab_prof_dump(FILE* out) {
    fprintf(out, "=== %s%s%s BLAS profile, pid %d", ab_build_info(), precision_token(), split_token(),
            (int)getpid());
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

    uint64_t grand_total_ns = 0, any_gpu = 0;
    for (int i = 0; i < g_entry_count; i++) {
        grand_total_ns += g_entries[i].total_ns;
        any_gpu += g_entries[i].gpu_calls;
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

    dump_counts(out, "Fallbacks: GPU-routed calls that ran on cblas (in cpu_ms)", &FALLBACKS, idx);
    dump_counts(out, "Staged path: GPU-routed calls that kept its result, else by why not",
                &STAGED, idx);
    if (split_on()) {
        // AB_SPLIT: the split calls are in neither gpu_calls nor cpu_calls.
        report_line_t l = {.n = 0};
        l.buf[0] = '\0';
        fprintf(out, "\nAB_SPLIT: calls whose GPU part stood (split_calls), the executor by routine, "
                     "the split table's rows:\n");
        for (int i = 0; i < g_entry_count; i++) {
            const ab_prof_entry_t* e = &g_entries[idx[i]];
            if (e->split_calls || e->split.calls)
                fprintf(out, "%s split_calls=%llu split_ms=%.3f\n", e->name, LLU(e->split_calls),
                        e->split_ns / 1e6);
            report_split(&l, out, "", e);
        }
        report_split_rows(&l, out, "");
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

// A routine's AB_REPORT line of one kind of counts, if it has any.
static void report_counts(report_line_t* l, FILE* out, int pid, const ab_prof_entry_t* e,
                          const char* what, const counts_t* k) {
    if (!counts_total(k, e)) return;
    line_add(l, "AB_REPORT[%d] %s %s", pid, e->name, what);
    for (int r = 0; r < k->n; r++)
        line_add(l, " %s=%llu", k->names[r], (unsigned long long)k->of(e)[r]);
    line_add(l, "\n");
    line_emit(l, out);
}

// A routine's AB_SPLIT line, if a row cut any of its calls, after prefix.
static void report_split(report_line_t* l, FILE* out, const char* prefix, const ab_prof_entry_t* e) {
    const ab_prof_split_stats_t* x = &e->split;
    if (!x->calls) return;
    line_add(l, "%s%s split calls=%llu kept=%llu kept_4m=%llu range=%llu gpu_failed=%llu scales=%llu "
                "alloc=%llu nonfinite=%llu fpcr=%llu retries=%llu retry_kept=%llu",
             prefix, e->name, LLU(x->calls), LLU(x->kept), LLU(x->kept_4m), LLU(x->range),
             LLU(x->gpu_failed), LLU(x->scales), LLU(x->alloc), LLU(x->nonfinite), LLU(x->fpcr),
             LLU(x->retries), LLU(x->retry_kept));
    for (int d = 0; d < AB_PROF_SPLIT_DECLINE_COUNT; d++)
        line_add(l, " declined_%s=%llu", ab_prof_split_decline_names[d], LLU(x->declined[d]));
    line_add(l, " axis_N=%llu axis_M=%llu axis_K=%llu init_ms=%.3f cpu_ms=%.3f gpu_ms=%.3f "
                "wait_cpu_ms=%.3f wait_gpu_ms=%.3f redo_ms=%.3f sum_ms=%.3f wasted_gpu_ms=%.3f "
                "gf_cpu=%.3f gf_gpu=%.3f\n",
             LLU(x->axis[0]), LLU(x->axis[1]), LLU(x->axis[2]), x->init_ns / 1e6, x->cpu_ns / 1e6,
             x->gpu_ns / 1e6, x->wait_cpu_ns / 1e6, x->wait_gpu_ns / 1e6, x->redo_ns / 1e6,
             x->sum_ns / 1e6, x->wasted_gpu_ns / 1e6, x->gf_cpu, x->gf_gpu);
    line_emit(l, out);
}

// A line per row of the split table, f = 0 rows and rows without calls
// included, so that a report names every share it ran with.
static void report_split_rows(report_line_t* l, FILE* out, const char* prefix) {
    const ab_dispatch_config_t* dc = ab_dispatch_process_config();
    for (int i = 0; i < dc->split_nrows && i < AB_SPLIT_MAX_ROWS; i++) {
        const ab_split_row_t* row = &dc->split_rows[i];
        const split_row_stats_t* r = &g_split_rows[i];
        line_add(l, "%ssplit_row i=%d routine=%cgemm axis=%c f=%g calls=%llu split=%llu wall_ms=%.3f "
                    "first_ms=%.3f first_cpu_ms=%.3f first_gpu_ms=%.3f first_gf_cpu=%.3f first_gf_gpu=%.3f "
                    "cpu_ms=%.3f gpu_ms=%.3f wait_cpu_ms=%.3f wait_gpu_ms=%.3f "
                    "redo_ms=%.3f sum_ms=%.3f gf=%.3f gf_cpu=%.3f gf_gpu=%.3f kept=%llu kept_4m=%llu "
                    "range=%llu gpu_failed=%llu\n",
                 prefix, i, row->routine, row->axis, row->share, LLU(r->calls), LLU(r->split),
                 r->wall_ns / 1e6, r->first_ns / 1e6, r->first_cpu_ns / 1e6, r->first_gpu_ns / 1e6,
                 r->first_gf_cpu, r->first_gf_gpu, r->cpu_ns / 1e6, r->gpu_ns / 1e6,
                 r->wait_cpu_ns / 1e6, r->wait_gpu_ns / 1e6, r->redo_ns / 1e6, r->sum_ns / 1e6, r->gf,
                 r->gf_cpu, r->gf_gpu, LLU(r->kept), LLU(r->kept_4m), LLU(r->range), LLU(r->gpu_failed));
        line_emit(l, out);
    }
}

// Line-oriented key=value lines, each tagged with the pid so MPI ranks that
// share stderr stay apart.
void ab_prof_report(FILE* out) {
    const char* mode = getenv("AB_MODE");
    int pid = (int)getpid();
    report_line_t l = { .n = 0 };
    l.buf[0] = '\0';
    line_add(&l, "AB_REPORT[%d] %s AB_MODE=%s%s", pid, ab_build_info(),
             mode && *mode ? mode : "unset", precision_token());
    if (g_threshold_ns)
        line_add(&l, " AB_PROFILE_THRESHOLD_NS=%llu", (unsigned long long)g_threshold_ns);
    // The chip and the table auto routed by (dispatch.h), so a report from
    // another machine says what routed it.
    const ab_dispatch_config_t* dc = ab_dispatch_process_config();
    line_add(&l, " chip=\"%s\" gpu_cores=%d table=%s%s", dc->chip[0] ? dc->chip : "unknown", dc->gpu_cores,
             dc->table, split_token());
    line_add(&l, "\n");
    line_emit(&l, out);
    if (g_entry_count == 0) {
        line_add(&l, "AB_REPORT[%d] no BLAS-ABI calls recorded\n", pid);
        line_emit(&l, out);
        return;
    }

    char prefix[32];
    snprintf(prefix, sizeof prefix, "AB_REPORT[%d] ", pid);
    int idx[AB_PROF_MAX_ROUTINES];
    sort_by_time(idx);
    for (int i = 0; i < g_entry_count; i++) {
        ab_prof_entry_t* e = &g_entries[idx[i]];
        if (e->call_count == 0) continue;
        line_add(&l, "AB_REPORT[%d] %s calls=%llu gpu_calls=%llu cpu_calls=%llu "
                     "gpu_calls%%=%.1f total_ms=%.3f gpu_ms=%.3f cpu_ms=%.3f "
                     "gpu_time%%=%.1f fallbacks=%llu",
                 pid, e->name,
                 (unsigned long long)e->call_count,
                 (unsigned long long)e->gpu_calls,
                 (unsigned long long)e->cpu_calls,
                 pct(e->gpu_calls, e->call_count),
                 e->total_ns / 1e6, e->gpu_ns / 1e6, e->cpu_ns / 1e6,
                 pct(e->gpu_ns, e->total_ns),
                 (unsigned long long)fallback_total(e));
        if (split_on())
            line_add(&l, " split_calls=%llu split_ms=%.3f", LLU(e->split_calls), e->split_ns / 1e6);
        line_add(&l, "\n");
        line_emit(&l, out);
        if (e->gpu_calls) {
            line_add(&l, "AB_REPORT[%d] %s gpu_stages_ms", pid, e->name);
            for (int s = 0; s < AB_PROF_STAGE_COUNT; s++)
                line_add(&l, " %s=%.3f", ab_prof_stage_names[s], e->stage_ns[s] / 1e6);
            line_add(&l, " other=%.3f\n", stage_other(e) / 1e6);
            line_emit(&l, out);
        }
        report_counts(&l, out, pid, e, "fallbacks", &FALLBACKS);
        report_counts(&l, out, pid, e, "staged", &STAGED);
        report_split(&l, out, prefix, e);
    }
    if (split_on()) report_split_rows(&l, out, prefix);
}

// =============================================================================
// Per-call trace (AB_TRACE)
// =============================================================================

uint64_t ab_prof_kernel_ns(uint64_t wall_ns, double gpu_seconds) {
    uint64_t gpu = (uint64_t)(gpu_seconds * 1e9);
    return gpu && gpu < wall_ns ? gpu : wall_ns;
}

uint64_t ab_prof_take_kernel_ns(uint64_t wall_ns, ab_prof_kernel_split_t* split) {
    ab_prof_kernel_split_t k;
    uint64_t ns = ab_prof_kernel_ns(wall_ns, ab_internal_take_gpu_seconds(&k));
    if (ns < wall_ns) {   // the GPU's time was used
        split->temps_ns += k.temps_ns;
        split->encode_ns += k.encode_ns;
        split->submit_ns += k.submit_ns;
        split->wake_ns += k.wake_ns;
    }
    if (!split->gpu_start_ns) split->gpu_start_ns = k.gpu_start_ns;
    if (k.gpu_end_ns) split->gpu_end_ns = k.gpu_end_ns;
    if (!split->commit_ns) split->commit_ns = k.commit_ns;
    if (k.done_ns) split->done_ns = k.done_ns;
    split->groups += k.groups;
    if (k.groups_max > split->groups_max) split->groups_max = k.groups_max;
    split->dispatches += k.dispatches;
    split->cmdbufs += k.cmdbufs;
    split->kinds |= k.kinds;
    if (!split->mtl_error) split->mtl_error = k.mtl_error;
    return ns;
}

void ab_prof_stage_mark(uint64_t* stage_ns, uint64_t* mark, ab_prof_stage_t s) {
    if (!stage_ns) return;
    uint64_t now = ab_prof_now_ticks();
    stage_ns[s] += ab_prof_ticks_to_ns(now - *mark);
    *mark = now;
}

void ab_prof_print_gpu_call(const char* routine, int m, int n, int k, uint64_t total_ns,
                            const uint64_t* stage_ns, ab_prof_staged_t staged) {
    if (!ab_prof_verbose()) return;
    // One write per line, so lines from concurrent callers do not interleave.
    char line[512];
    size_t len = (size_t)snprintf(line, sizeof line, "AB_PROFILE %s M=%d N=%d K=%d ", routine, m, n, k);
    uint64_t named = 0;
    for (int s = 0; s < AB_PROF_STAGE_COUNT && len < sizeof line; s++) {
        named += stage_ns[s];
        len += (size_t)snprintf(line + len, sizeof line - len, " %s=%.3f", ab_prof_stage_names[s],
                                stage_ns[s] / 1e6);
    }
    if (len < sizeof line)
        len += (size_t)snprintf(line + len, sizeof line - len, " other=%.3f  total=%.3f ms",
                                (total_ns > named ? total_ns - named : 0) / 1e6, total_ns / 1e6);
    if (len < sizeof line && staged > AB_PROF_STAGED_NONE)
        len += (size_t)snprintf(line + len, sizeof line - len, "  staged=%s",
                                ab_prof_staged_names[staged]);
    if (len < sizeof line) snprintf(line + len, sizeof line - len, "\n");
    fputs(line, stderr);
}

// The GPU part's thread time outside its named stages.
static uint64_t split_gpu_other(const ab_prof_split_t* s) {
    return other_ns(s->gpu_wall_ns, s->gpu_stage_ns);
}

// The caller's time outside its named split stages.
static uint64_t split_other(uint64_t total_ns, const ab_prof_split_t* s) {
    uint64_t t = s->init_ns + s->cpu_part_ns + s->join_ns + s->redo_ns + s->sum_ns;
    return total_ns > t ? total_ns - t : 0;
}

void ab_prof_print_split_call(const char* routine, int m, int n, int k, uint64_t total_ns,
                              const ab_prof_split_t* s) {
    if (!ab_prof_verbose()) return;
    report_line_t l = {.n = 0};
    l.buf[0] = '\0';
    line_add(&l, "AB_PROFILE %s M=%d N=%d K=%d  split row=%d axis=%c g=%d  init=%.3f cpu_part=%.3f "
                 "join=%.3f redo=%.3f sum=%.3f other=%.3f  total=%.3f ms  gpu_wall=%.3f",
             routine, m, n, k, s->row, s->axis, s->g, s->init_ns / 1e6, s->cpu_part_ns / 1e6,
             s->join_ns / 1e6, s->redo_ns / 1e6, s->sum_ns / 1e6, split_other(total_ns, s) / 1e6,
             total_ns / 1e6, s->gpu_wall_ns / 1e6);
    for (int st = 0; st < AB_PROF_STAGE_COUNT; st++)
        line_add(&l, " %s=%.3f", ab_prof_stage_names[st], s->gpu_stage_ns[st] / 1e6);
    line_add(&l, " other=%.3f  %s", split_gpu_other(s) / 1e6, s->kept ? "kept" : "made again on Accelerate");
    if (s->staged > AB_PROF_STAGED_NONE && s->staged < AB_PROF_STAGED_COUNT)
        line_add(&l, "  staged=%s", ab_prof_staged_names[s->staged]);
    if (s->retries) line_add(&l, "  retries=%d", s->retries);
    line_add(&l, "\n");
    fputs(l.buf, stderr);
}

void ab_prof_print_fallback(const char* routine, int m, int n, int k, const char* why) {
    if (ab_prof_verbose())
        fprintf(stderr, "AB_PROFILE %s M=%d N=%d K=%d  GPU fallback to cblas: %s\n", routine, m, n,
                k, why);
}

// The header of a new trace file: the build, the settings and the line format.
static void trace_header(int fd, pid_t pid) {
    const char* mode = getenv("AB_MODE");
    report_line_t l = { .n = 0 };
    l.buf[0] = '\0';
    line_add(&l, "# %s AB_TRACE pid=%d\n", ab_build_info(), (int)pid);
    // AB_MIN_GPU_DIM as the routing resolved it, so a replay routes as this run did.
    const ab_dispatch_config_t* dc = ab_dispatch_process_config();
    line_add(&l, "# AB_MODE=%s AB_TRACE_HASH=%d AB_MIN_GPU_DIM=%d%s\n", mode && *mode ? mode : "unset",
             g_trace_hash, dc->min_dim, precision_token());
    // The chip and the routing table auto used (dispatch.h): a trace from
    // another machine then says what routed it. A replay routes by its own
    // machine's table.
    line_add(&l, "# chip=\"%s\" gpu_cores=%d table=%s%s\n", dc->chip[0] ? dc->chip : "unknown", dc->gpu_cores,
             dc->table, split_token());
    // The MPI rank and size one launcher exported (Open MPI, else PMI), so a
    // replay can map the files of one run to its ranks; -1 when not under MPI.
    const char *rank = getenv("OMPI_COMM_WORLD_RANK"), *size = getenv("OMPI_COMM_WORLD_SIZE");
    if (!rank || !size) {
        rank = getenv("PMI_RANK");
        size = getenv("PMI_SIZE");
    }
    if (!rank || !size) rank = size = "-1";
    line_add(&l, "# mpi_rank=%s mpi_size=%s\n", rank, size);
    line_add(&l, "# clock: ns of mach_absolute_time (CLOCK_UPTIME_RAW), shared by every "
                 "process on this machine\n"
                 "# line: seq t0_ns t1_ns routine ta tb M N K lda ldb ldc alpha beta path\n"
                 "#   ta, tb as passed; alpha, beta as %%a hex floats, zgemm real,imaginary\n"
                 "#   path: cpu, gpu, or fallback:<reason> for a GPU-routed call run on cblas\n"
                 "# gpu lines add the stages");
    for (int s = 0; s < AB_PROF_STAGE_COUNT; s++)
        line_add(&l, " %s_ns", ab_prof_stage_names[s]);
    line_add(&l, " other_ns (sum: t1_ns - t0_ns),\n"
                 "#   other_ns = temps_ns + encode_ns + submit_ns + wake_ns + rest_ns,\n"
                 "#   gpu_start_ns gpu_end_ns: GPUStartTime, GPUEndTime (0: none reported)\n"
                 "# calls that reached the staged path add staged: kept (kept_4m: a ZGEMM by 4M\n"
                 "#   after its Gauss-3M run), or why it did not keep the call\n"
                 "# AB_TRACE_HASH=1 adds hash_a, hash_b (op(A), op(B) when read) and hash_c "
                 "(C on entry when beta != 0):\n"
                 "#   FNV-1a over the 64-bit words of the referenced elements, column by "
                 "column:\n"
                 "#   from the 64-bit offset basis, h = (h ^ word) * FNV prime per word\n");
    if (dc->split_on) {             // in a write of its own: the header fills the buffer
        (void)write(fd, l.buf, l.n);
        l.n = 0;
        l.buf[0] = '\0';
        line_add(&l, "# AB_SPLIT: a split call's path is split (the GPU's part stood) or fallback:<reason> "
                     "(made again on Accelerate),\n"
                     "#   then row axis g retries staged, the caller's stages init_ns cpu_part_ns join_ns "
                     "redo_ns sum_ns other_ns\n"
                     "#   (sum: t1_ns - t0_ns), and the GPU part's thread:");
        for (int s = 0; s < AB_PROF_STAGE_COUNT; s++) line_add(&l, " gpu_%s_ns", ab_prof_stage_names[s]);
        line_add(&l, " gpu_other_ns\n"
                     "#   (sum: gpu_wall_ns), gpu_wall_ns wait_cpu_ns wait_gpu_ns; a call a row covers "
                     "without splitting adds\n"
                     "#   row axis g (0: not cut) and declined=<why> when declined\n");
        // The timeline (split_times.h) and the settings that shape it.
        line_add(&l, "# AB_SPLIT timeline, ns on the clock above (0: did not happen): at_cut_ns at_create_ns "
                     "at_cpu0_ns at_cpu1_ns\n"
                     "#   at_gpu0_ns at_gpu1_ns at_join_ns at_end_ns; the GPU part's command buffers: "
                     "at_commit_ns (first commit)\n"
                     "#   gpu_start_ns gpu_end_ns at_done_ns (last completion), gpu_encode_ns gpu_submit_ns "
                     "gpu_wake_ns gpu_temps_ns\n"
                     "#   (summed), gpu_cmdbufs gpu_dispatches gpu_groups gpu_groups_max (GEMM-family "
                     "threadgroups), gpu_kinds\n"
                     "#   (bit 2*kind+prefetch), gpu_mtl_error (MTLCommandBufferError code; 0 none, -1 "
                     "no code, -2 a test hook)\n");
        // As passed, but the staging cap as in effect (AB_STAGING_MB, default 2048).
        const char *vl = getenv("VECLIB_MAXIMUM_THREADS"), *omp = getenv("OMP_NUM_THREADS"),
                   *pf = getenv("AB_PREFETCH");
        line_add(&l, "# env VECLIB_MAXIMUM_THREADS=%s OMP_NUM_THREADS=%s AB_PREFETCH=%s staging_mb=%zu\n",
                 vl ? vl : "unset", omp ? omp : "unset", pf ? pf : "unset", ab_staging_cap_bytes() >> 20);
    }
    (void)write(fd, l.buf, l.n);
}

// This process's trace file, created with its header; -1 if it cannot be opened.
static int trace_open(pid_t pid) {
    char path[PATH_MAX];
    snprintf(path, sizeof path, "%s.%d", g_trace_path, (int)pid);
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0)
        fprintf(stderr, "apple-bottom: AB_TRACE: cannot open %s: %s; no trace for pid %d\n",
                path, strerror(errno), (int)pid);
    else
        trace_header(fd, pid);
    return fd;
}

// Numbers the line and appends it to this process's trace file with one
// writev(2) from the stack, so nothing is buffered for a fork() to duplicate.
// The first line of each process opens its file. Lines are written under the
// lock, so the file holds them in seq order.
static void trace_write(const report_line_t* l) {
    os_unfair_lock_lock(&g_trace_lock);
    if (g_trace_fd == -2) g_trace_fd = trace_open(getpid());
    if (g_trace_fd >= 0) {
        char seq[32];
        int n = snprintf(seq, sizeof seq, "seq=%llu ", (unsigned long long)++g_trace_seq);
        struct iovec iov[2] = {{seq, (size_t)n}, {(void*)l->buf, l->n}};
        (void)writev(g_trace_fd, iov, 2);
    }
    os_unfair_lock_unlock(&g_trace_lock);
}

// A TRANS flag as passed; anything but a visible character in hex.
static const char* trans_text(char t, char out[8]) {
    if (t > ' ' && t < 0x7f && t != '=') snprintf(out, 8, "%c", t);
    else snprintf(out, 8, "\\x%02x", (unsigned char)t);
    return out;
}

// A split call's fields (ab_trace_record_split): path, row, the caller's
// stages, which sum to t1_ns - t0_ns, and the GPU part's thread, whose stages
// sum to gpu_wall_ns.
static void trace_split_fields(report_line_t* l, uint64_t elapsed_ns, const ab_prof_split_t* s) {
    if (s->kept) line_add(l, " path=split");
    else line_add(l, " path=fallback:%s", fallback_names[s->staged == AB_PROF_STAGED_RANGE
                                                         ? AB_PROF_FB_RANGE : AB_PROF_FB_GPU_FAILED]);
    line_add(l, " row=%d axis=%c g=%d retries=%d", s->row, s->axis, s->g, s->retries);
    if (s->staged > AB_PROF_STAGED_NONE && s->staged < AB_PROF_STAGED_COUNT)
        line_add(l, " staged=%s", ab_prof_staged_names[s->staged]);
    line_add(l, " init_ns=%llu cpu_part_ns=%llu join_ns=%llu redo_ns=%llu sum_ns=%llu other_ns=%llu",
             LLU(s->init_ns), LLU(s->cpu_part_ns), LLU(s->join_ns), LLU(s->redo_ns), LLU(s->sum_ns),
             LLU(split_other(elapsed_ns, s)));
    for (int st = 0; st < AB_PROF_STAGE_COUNT; st++)
        line_add(l, " gpu_%s_ns=%llu", ab_prof_stage_names[st], LLU(s->gpu_stage_ns[st]));
    line_add(l, " gpu_other_ns=%llu gpu_wall_ns=%llu wait_cpu_ns=%llu wait_gpu_ns=%llu",
             LLU(split_gpu_other(s)), LLU(s->gpu_wall_ns), LLU(s->wait_cpu_ns), LLU(s->wait_gpu_ns));
    const ab_split_times_t* t = &s->times;
    const ab_prof_kernel_split_t* k = &t->gpu;
    line_add(l, " at_cut_ns=%llu at_create_ns=%llu at_cpu0_ns=%llu at_cpu1_ns=%llu at_gpu0_ns=%llu "
                "at_gpu1_ns=%llu at_join_ns=%llu at_end_ns=%llu",
             LLU(t->cut_ns), LLU(t->create_ns), LLU(t->cpu0_ns), LLU(t->cpu1_ns), LLU(t->gpu0_ns),
             LLU(t->gpu1_ns), LLU(t->join_ns), LLU(t->end_ns));
    line_add(l, " at_commit_ns=%llu gpu_start_ns=%llu gpu_end_ns=%llu at_done_ns=%llu gpu_encode_ns=%llu "
                "gpu_submit_ns=%llu gpu_wake_ns=%llu gpu_temps_ns=%llu",
             LLU(k->commit_ns), LLU(k->gpu_start_ns), LLU(k->gpu_end_ns), LLU(k->done_ns),
             LLU(k->encode_ns), LLU(k->submit_ns), LLU(k->wake_ns), LLU(k->temps_ns));
    line_add(l, " gpu_cmdbufs=%u gpu_dispatches=%u gpu_groups=%llu gpu_groups_max=%u gpu_kinds=%u "
                "gpu_mtl_error=%d",
             (unsigned)k->cmdbufs, (unsigned)k->dispatches, LLU(k->groups), (unsigned)k->groups_max,
             (unsigned)k->kinds, (int)k->mtl_error);
}

// Fields as key=value, in the order of the header's legend. GPU calls add
// the stages, which sum to t1_ns - t0_ns as in AB_PROFILE (other_ns is the
// remainder), and split other_ns so that the parts sum to it exactly:
// rest_ns is what the kernel step's split leaves, and may be a few ns below
// zero from rounding between the host and GPU clocks.
static void trace_line(const ab_trace_call_t* c, uint64_t t0_ns, uint64_t elapsed_ns,
                       const uint64_t* stage_ns, ab_prof_fallback_t fallback,
                       ab_prof_staged_t staged, const ab_prof_split_t* split) {
    static const char* const hash_keys[3] = {"hash_a", "hash_b", "hash_c"};
    bool z = c->routine[0] == 'z';
    char ta[8], tb[8];
    report_line_t l = { .n = 0 };
    l.buf[0] = '\0';
    line_add(&l, "t0_ns=%llu t1_ns=%llu routine=%s ta=%s tb=%s M=%d N=%d K=%d "
                 "lda=%d ldb=%d ldc=%d alpha=%a",
             (unsigned long long)t0_ns, (unsigned long long)(t0_ns + elapsed_ns), c->routine,
             trans_text(c->ta, ta), trans_text(c->tb, tb), c->m, c->n, c->k,
             c->lda, c->ldb, c->ldc, c->alpha[0]);
    if (z) line_add(&l, ",%a", c->alpha[1]);
    line_add(&l, " beta=%a", c->beta[0]);
    if (z) line_add(&l, ",%a", c->beta[1]);

    if (split && split->split) {
        trace_split_fields(&l, elapsed_ns, split);
    } else if (stage_ns) {
        line_add(&l, " path=gpu");
        for (int s = 0; s < AB_PROF_STAGE_COUNT; s++)
            line_add(&l, " %s_ns=%llu", ab_prof_stage_names[s], (unsigned long long)stage_ns[s]);
        const ab_prof_kernel_split_t* k = &c->split;
        uint64_t other = other_ns(elapsed_ns, stage_ns);
        int64_t rest = (int64_t)other -
                       (int64_t)(k->temps_ns + k->encode_ns + k->submit_ns + k->wake_ns);
        line_add(&l, " other_ns=%llu temps_ns=%llu encode_ns=%llu submit_ns=%llu wake_ns=%llu "
                     "rest_ns=%lld gpu_start_ns=%llu gpu_end_ns=%llu",
                 (unsigned long long)other, (unsigned long long)k->temps_ns,
                 (unsigned long long)k->encode_ns, (unsigned long long)k->submit_ns,
                 (unsigned long long)k->wake_ns, (long long)rest,
                 (unsigned long long)k->gpu_start_ns, (unsigned long long)k->gpu_end_ns);
    } else if (fallback > AB_PROF_FB_NONE && fallback < AB_PROF_FB_COUNT) {
        line_add(&l, " path=fallback:%s", fallback_names[fallback]);
    } else {
        line_add(&l, " path=cpu");
    }
    if (!(split && split->split) && staged > AB_PROF_STAGED_NONE && staged < AB_PROF_STAGED_COUNT)
        line_add(&l, " staged=%s", ab_prof_staged_names[staged]);
    if (split && !split->split) {   // a call its row covered without splitting it
        line_add(&l, " row=%d axis=%c g=%d", split->row, split->axis, split->g);
        if (split->declined > AB_PROF_SPLIT_DECLINE_NONE && split->declined < AB_PROF_SPLIT_DECLINE_COUNT)
            line_add(&l, " declined=%s", ab_prof_split_decline_names[split->declined]);
    }
    for (int h = 0; h < 3; h++)
        if (c->hashed[h])
            line_add(&l, " %s=%016llx", hash_keys[h], (unsigned long long)c->hash[h]);
    line_add(&l, "\n");
    trace_write(&l);
}

void ab_trace_record(const ab_trace_call_t* c, uint64_t t0_ns, uint64_t elapsed_ns,
                     const uint64_t* stage_ns, ab_prof_fallback_t fallback,
                     ab_prof_staged_t staged) {
    trace_line(c, t0_ns, elapsed_ns, stage_ns, fallback, staged, NULL);
}

void ab_trace_record_split(const ab_trace_call_t* c, uint64_t t0_ns, uint64_t elapsed_ns,
                           const uint64_t* stage_ns, ab_prof_fallback_t fallback,
                           ab_prof_staged_t staged, const ab_prof_split_t* s) {
    trace_line(c, t0_ns, elapsed_ns, stage_ns, fallback, staged, s);
}

// =============================================================================
// JSON dump
// =============================================================================

// A routine's counts of one kind as JSON members.
static void json_counts(FILE* f, const ab_prof_entry_t* e, const counts_t* k) {
    for (int r = 0; r < k->n; r++)
        fprintf(f, "%s\"%s\": %llu", r ? ", " : "", k->names[r], (unsigned long long)k->of(e)[r]);
}

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
    if (*precision_token()) fprintf(f, "  \"precision\": \"fast\",\n");
    fprintf(f, "  \"timestamp\": \"%s\",\n", ts);
    fprintf(f, "  \"pid\": %d,\n", (int)getpid());
    fprintf(f, "  \"threshold_ns\": %llu,\n", (unsigned long long)g_threshold_ns);
    if (split_on()) {
        fprintf(f, "  \"split_table\": \"");
        for (const char* p = ab_dispatch_process_config()->split_table; *p; p++) {
            if (*p == '"' || *p == '\\') fputc('\\', f);
            fputc((unsigned char)*p < 0x20 ? '?' : *p, f);
        }
        fprintf(f, "\",\n");
    }
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
        json_counts(f, e, &FALLBACKS);
        fprintf(f, "},\n      \"staged\": {");
        json_counts(f, e, &STAGED);
        if (split_on()) {
            const ab_prof_split_stats_t* x = &e->split;
            fprintf(f, "},\n      \"split_calls\": %llu, \"split_ns\": %llu,\n      \"split\": {"
                       "\"calls\": %llu, \"kept\": %llu, \"kept_4m\": %llu, \"range\": %llu, "
                       "\"gpu_failed\": %llu, \"scales\": %llu, \"alloc\": %llu, \"nonfinite\": %llu, "
                       "\"fpcr\": %llu, \"retries\": %llu, \"retry_kept\": %llu",
                    LLU(e->split_calls), LLU(e->split_ns), LLU(x->calls), LLU(x->kept), LLU(x->kept_4m),
                    LLU(x->range), LLU(x->gpu_failed), LLU(x->scales), LLU(x->alloc), LLU(x->nonfinite),
                    LLU(x->fpcr), LLU(x->retries), LLU(x->retry_kept));
            for (int d = 0; d < AB_PROF_SPLIT_DECLINE_COUNT; d++)
                fprintf(f, ", \"declined_%s\": %llu", ab_prof_split_decline_names[d], LLU(x->declined[d]));
            fprintf(f, ", \"axis_N\": %llu, \"axis_M\": %llu, \"axis_K\": %llu, \"init_ns\": %llu, "
                       "\"cpu_ns\": %llu, \"gpu_ns\": %llu, \"wait_cpu_ns\": %llu, \"wait_gpu_ns\": %llu, "
                       "\"redo_ns\": %llu, \"sum_ns\": %llu, \"wasted_gpu_ns\": %llu, \"gf_cpu\": %.6f, "
                       "\"gf_gpu\": %.6f",
                    LLU(x->axis[0]), LLU(x->axis[1]), LLU(x->axis[2]), LLU(x->init_ns), LLU(x->cpu_ns),
                    LLU(x->gpu_ns), LLU(x->wait_cpu_ns), LLU(x->wait_gpu_ns), LLU(x->redo_ns),
                    LLU(x->sum_ns), LLU(x->wasted_gpu_ns), x->gf_cpu, x->gf_gpu);
        }
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

    fprintf(f, "\n  ]");
    if (split_on()) {
        const ab_dispatch_config_t* dc = ab_dispatch_process_config();
        fprintf(f, ",\n  \"split_rows\": [");
        for (int i = 0; i < dc->split_nrows && i < AB_SPLIT_MAX_ROWS; i++) {
            const ab_split_row_t* row = &dc->split_rows[i];
            const split_row_stats_t* r = &g_split_rows[i];
            fprintf(f, "%s\n    {\"i\": %d, \"routine\": \"%cgemm\", \"axis\": \"%c\", \"f\": %g, "
                       "\"calls\": %llu, \"split\": %llu, \"wall_ns\": %llu, \"first_ns\": %llu, "
                       "\"first_cpu_ns\": %llu, \"first_gpu_ns\": %llu, "
                       "\"first_gf_cpu\": %.6f, \"first_gf_gpu\": %.6f, "
                       "\"cpu_ns\": %llu, \"gpu_ns\": %llu, \"wait_cpu_ns\": %llu, \"wait_gpu_ns\": %llu, "
                       "\"redo_ns\": %llu, \"sum_ns\": %llu, \"gf\": %.6f, \"gf_cpu\": %.6f, "
                       "\"gf_gpu\": %.6f, \"kept\": %llu, \"kept_4m\": %llu, \"range\": %llu, "
                       "\"gpu_failed\": %llu}",
                    i ? "," : "", i, row->routine, row->axis, row->share, LLU(r->calls), LLU(r->split),
                    LLU(r->wall_ns), LLU(r->first_ns), LLU(r->first_cpu_ns), LLU(r->first_gpu_ns),
                    r->first_gf_cpu, r->first_gf_gpu, LLU(r->cpu_ns), LLU(r->gpu_ns), LLU(r->wait_cpu_ns),
                    LLU(r->wait_gpu_ns), LLU(r->redo_ns), LLU(r->sum_ns), r->gf, r->gf_cpu, r->gf_gpu,
                    LLU(r->kept), LLU(r->kept_4m), LLU(r->range), LLU(r->gpu_failed));
        }
        fprintf(f, "\n  ]");
    }
    fprintf(f, "\n}\n");
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
