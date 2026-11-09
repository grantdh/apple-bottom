// =============================================================================
// powerlog: power, energy and clock-state residency through a block, no root
// =============================================================================
// Streams one TSV file while a command runs (or until SIGINT/SIGTERM without
// one), every row stamped in ns of mach_absolute_time (CLOCK_UPTIME_RAW), the
// clock of AB_TRACE's t0_ns and of GPUStartTime, so rows line up with traces:
//
//   S  t_ns  <SMC keys>         SMC power in W, every -i ms (default 100)
//   E  t0_ns t1_ns group subgroup channel unit value
//                               an IOReport simple channel's change over
//                               [t0, t1], every -r S rows (default 5: 0.5 s)
//   R  t0_ns t1_ns group subgroup channel state=residency;...
//                               an IOReport state channel's residency over
//                               [t0, t1], by the states' own names
//   M  t_ns key=value...        the adapter's watts, the battery's charge and
//                               state, the thermal pressure level and load1,
//                               at the start, every -m s (default 60) and at
//                               the end
//
// The IOReport groups default to "Energy Model", "GPU Stats:GPU Performance
// States", "CPU Stats:CPU Complex Performance States" and "AMC Stats:Perf
// Counters" (bytes read and written by each agent: the CPU clusters, the GPU
// as GFX*, and DRAM as DCS); -g GROUP[:SUBGROUP] replaces them (repeat).
// E rows whose value is 0 are left out: unprivileged channels can read 0.
// The sampler used 3% of one core over a 10 s run at 100 ms (M2 Max). At
// utility QoS it took 2.5%, but timer coalescing halved its rate (45 S rows in
// 10 s, not 90), so it keeps the default.
//
// `powerlog --list [seconds]` subscribes to every channel, samples twice and
// prints each channel with its format and whether it changed, to find which
// channels give data without root (on another chip, for example).
//
// Usage: powerlog -o FILE.tsv [-i ms] [-r n] [-m s] [-g group[:subgroup]]... [-- cmd args...]
//        powerlog --list [seconds]
// With a command, the exit status is the command's.
// =============================================================================

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/ps/IOPSKeys.h>
#include <IOKit/ps/IOPowerSources.h>
#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <notify.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

// ---------------------------------------------------------------- SMC (no root)
typedef struct { char major, minor, build, reserved; UInt16 release; } SMCVers;
typedef struct { UInt16 version, length; UInt32 cpuPLimit, gpuPLimit, memPLimit; } SMCPLimit;
typedef struct { UInt32 dataSize; UInt32 dataType; char dataAttributes; } SMCKeyInfo;
typedef struct {
    UInt32 key; SMCVers vers; SMCPLimit pLimitData; SMCKeyInfo keyInfo;
    char result, status, data8; UInt32 data32; unsigned char bytes[32];
} SMCKeyData;
static io_connect_t g_smc;
static UInt32 k4(const char *s) { return ((UInt32)s[0] << 24) | ((UInt32)s[1] << 16) | ((UInt32)s[2] << 8) | (UInt32)s[3]; }
static int smc_call(SMCKeyData *in, SMCKeyData *out) {
    size_t sz = sizeof(SMCKeyData);
    return IOConnectCallStructMethod(g_smc, 2, in, sizeof(SMCKeyData), out, &sz) == KERN_SUCCESS && !out->result ? 0 : -1;
}
static int smc_info(UInt32 key, SMCKeyInfo *ki) {
    SMCKeyData in = {0}, out = {0};
    in.key = key;
    in.data8 = 9;
    if (smc_call(&in, &out)) return -1;
    *ki = out.keyInfo;
    return 0;
}
static int smc_flt(UInt32 key, SMCKeyInfo ki, float *f) {
    SMCKeyData in = {0}, out = {0};
    in.key = key;
    in.keyInfo.dataSize = ki.dataSize;
    in.data8 = 5;
    if (smc_call(&in, &out)) return -1;
    memcpy(f, out.bytes, 4);
    return 0;
}
// PDTR: DC input (the adapter's output on AC; includes battery charging). PPBR:
// battery. PC10+PC12+PC20+PC22: GPU rails (M2 Max). PSTR: SMC system total.
static const char *KEYS[] = {"PDTR", "PPBR", "PC10", "PC12", "PC20", "PC22", "PSTR"};
#define NK ((int)(sizeof KEYS / sizeof *KEYS))
static SMCKeyInfo g_ki[NK];
static bool g_kok[NK];
static bool smc_open(void) {
    io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"));
    if (!svc || IOServiceOpen(svc, mach_task_self(), 0, &g_smc) != KERN_SUCCESS) return false;
    bool any = false;
    for (int k = 0; k < NK; k++) any |= g_kok[k] = smc_info(k4(KEYS[k]), &g_ki[k]) == 0;
    return any;
}

// ---------------------------------------------------------------- IOReport (no root)
static struct {
    CFDictionaryRef (*all)(uint64_t, uint64_t);
    CFDictionaryRef (*group)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
    void (*merge)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
    void *(*sub)(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
    CFDictionaryRef (*samples)(void *, CFMutableDictionaryRef, CFTypeRef);
    CFDictionaryRef (*delta)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
    int64_t (*integer)(CFDictionaryRef, int32_t);
    CFStringRef (*gname)(CFDictionaryRef), (*sgname)(CFDictionaryRef), (*cname)(CFDictionaryRef),
        (*unit)(CFDictionaryRef);
    int32_t (*format)(CFDictionaryRef), (*scount)(CFDictionaryRef);
    CFStringRef (*sname)(CFDictionaryRef, int32_t);
    int64_t (*sres)(CFDictionaryRef, int32_t);
} ior;
enum { IOR_FORMAT_SIMPLE = 1, IOR_FORMAT_STATE = 2 };   // IOReportChannelGetFormat

static bool ior_load(void) {
    void *h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
    if (!h) return false;
#define SYM(field, name) ior.field = (__typeof__(ior.field))dlsym(h, name)
    SYM(all, "IOReportCopyAllChannels");
    SYM(group, "IOReportCopyChannelsInGroup");
    SYM(merge, "IOReportMergeChannels");
    SYM(sub, "IOReportCreateSubscription");
    SYM(samples, "IOReportCreateSamples");
    SYM(delta, "IOReportCreateSamplesDelta");
    SYM(integer, "IOReportSimpleGetIntegerValue");
    SYM(gname, "IOReportChannelGetGroup");
    SYM(sgname, "IOReportChannelGetSubGroup");
    SYM(cname, "IOReportChannelGetChannelName");
    SYM(unit, "IOReportChannelGetUnitLabel");
    SYM(format, "IOReportChannelGetFormat");
    SYM(scount, "IOReportStateGetCount");
    SYM(sname, "IOReportStateGetNameForIndex");
    SYM(sres, "IOReportStateGetResidency");
#undef SYM
    return ior.all && ior.group && ior.merge && ior.sub && ior.samples && ior.delta && ior.integer &&
           ior.gname && ior.sgname && ior.cname && ior.unit && ior.format && ior.scount && ior.sname && ior.sres;
}

// s as UTF-8 into buf, tabs and newlines replaced so the TSV stays whole; "-" if NULL.
static const char *cstr(CFStringRef s, char *buf, size_t n) {
    if (!s || !CFStringGetCString(s, buf, (CFIndex)n, kCFStringEncodingUTF8)) return "-";
    for (char *c = buf; *c; c++)
        if (*c == '\t' || *c == '\n' || *c == '\r') *c = ' ';
    return buf;
}

typedef struct { void *sub; CFMutableDictionaryRef subbed; } Subscription;

// A subscription to the groups "G[:S]" (all channels when n == 0).
static bool ior_subscribe(char **groups, int n, Subscription *out) {
    CFMutableDictionaryRef ch = NULL;
    if (n == 0) {
        CFDictionaryRef all = ior.all(0, 0);
        if (!all) return false;
        ch = CFDictionaryCreateMutableCopy(kCFAllocatorDefault, 0, all);
        CFRelease(all);
    }
    for (int i = 0; i < n; i++) {
        char g[256];
        snprintf(g, sizeof g, "%s", groups[i]);
        char *colon = strchr(g, ':');
        if (colon) *colon = '\0';
        CFStringRef gs = CFStringCreateWithCString(NULL, g, kCFStringEncodingUTF8);
        CFStringRef ss = colon ? CFStringCreateWithCString(NULL, colon + 1, kCFStringEncodingUTF8) : NULL;
        CFDictionaryRef d = ior.group(gs, ss, 0, 0, 0);
        CFRelease(gs);
        if (ss) CFRelease(ss);
        if (!d) {
            fprintf(stderr, "powerlog: IOReport group \"%s\": no channels\n", groups[i]);
            continue;
        }
        if (!ch) ch = CFDictionaryCreateMutableCopy(kCFAllocatorDefault, 0, d);
        else ior.merge(ch, d, NULL);
        CFRelease(d);
    }
    if (!ch) return false;
    out->sub = ior.sub(NULL, ch, &out->subbed, 0, NULL);
    CFRelease(ch);
    return out->sub != NULL;
}

// Each channel of the change from s0 to s1, passed to row().
static void ior_each(CFDictionaryRef s0, CFDictionaryRef s1, void (*row)(CFDictionaryRef, void *), void *arg) {
    CFDictionaryRef d = ior.delta(s0, s1, NULL);
    if (!d) return;
    CFArrayRef items = CFDictionaryGetValue(d, CFSTR("IOReportChannels"));
    for (CFIndex i = 0; items && i < CFArrayGetCount(items); i++) row(CFArrayGetValueAtIndex(items, i), arg);
    CFRelease(d);
}

typedef struct { FILE *f; uint64_t t0, t1; } RowOut;

// An E or R row; names are converted only for rows that are written.
static void log_row(CFDictionaryRef it, void *arg) {
    const RowOut *o = arg;
    char g[128], sg[128], c[128], u[64], st[64];
    int fmt = ior.format(it);
    if (fmt == IOR_FORMAT_SIMPLE) {
        int64_t v = ior.integer(it, 0);
        if (v)
            fprintf(o->f, "E\t%llu\t%llu\t%s\t%s\t%s\t%s\t%lld\n", (unsigned long long)o->t0,
                    (unsigned long long)o->t1, cstr(ior.gname(it), g, sizeof g), cstr(ior.sgname(it), sg, sizeof sg),
                    cstr(ior.cname(it), c, sizeof c), cstr(ior.unit(it), u, sizeof u), (long long)v);
    } else if (fmt == IOR_FORMAT_STATE) {
        fprintf(o->f, "R\t%llu\t%llu\t%s\t%s\t%s\t", (unsigned long long)o->t0, (unsigned long long)o->t1,
                cstr(ior.gname(it), g, sizeof g), cstr(ior.sgname(it), sg, sizeof sg), cstr(ior.cname(it), c, sizeof c));
        for (int k = 0, n = ior.scount(it); k < n; k++)
            fprintf(o->f, "%s%s=%lld", k ? ";" : "", cstr(ior.sname(it, k), st, sizeof st), (long long)ior.sres(it, k));
        fputc('\n', o->f);
    }
}

// --list: each channel, its format and whether it changed.
static void list_row(CFDictionaryRef it, void *arg) {
    (void)arg;
    char g[128], sg[128], c[128], u[64];
    const char *gn = cstr(ior.gname(it), g, sizeof g), *sgn = cstr(ior.sgname(it), sg, sizeof sg),
               *cn = cstr(ior.cname(it), c, sizeof c);
    int fmt = ior.format(it);
    if (fmt == IOR_FORMAT_SIMPLE) {
        printf("simple\t%s\t%s\t%s\t%s\t%s\n", gn, sgn, cn, cstr(ior.unit(it), u, sizeof u),
               ior.integer(it, 0) ? "changed" : "zero");
    } else if (fmt == IOR_FORMAT_STATE) {
        int64_t tot = 0;
        int n = ior.scount(it);
        for (int k = 0; k < n; k++) tot += ior.sres(it, k);
        printf("state\t%s\t%s\t%s\t%d states\t%s\n", gn, sgn, cn, n, tot ? "changed" : "zero");
    } else {
        printf("format %d\t%s\t%s\t%s\n", fmt, gn, sgn, cn);
    }
}

// A new sample: the rows of its change since *prev (taken at *tprev), which it replaces.
static void ior_step(FILE *f, const Subscription *s, CFDictionaryRef *prev, uint64_t *tprev) {
    CFDictionaryRef cur = ior.samples(s->sub, s->subbed, NULL);
    uint64_t t = now_ns();
    if (*prev && cur) {
        RowOut o = {f, *tprev, t};
        ior_each(*prev, cur, log_row, &o);
    }
    if (*prev) CFRelease(*prev);
    *prev = cur;
    *tprev = t;
    fflush(f);
}

// ---------------------------------------------------------------- machine state
// In-process, with no fork: the adapter's watts (-1 when none is attached), the
// battery's charge, whether it is charging, the power source, the thermal
// pressure level (com.apple.system.thermalpressurelevel: 0 nominal, higher
// worse) and load1.
static int dict_int(CFDictionaryRef d, CFStringRef key) {
    int v = -1;
    CFNumberRef n = d ? CFDictionaryGetValue(d, key) : NULL;
    if (n && CFGetTypeID(n) == CFNumberGetTypeID()) CFNumberGetValue(n, kCFNumberIntType, &v);
    return v;
}
static void machine_row(FILE *f) {
    CFDictionaryRef ad = IOPSCopyExternalPowerAdapterDetails();
    int watts = dict_int(ad, CFSTR(kIOPSPowerAdapterWattsKey));
    if (ad) CFRelease(ad);
    int charge = -1, max = -1, charging = -1;
    char source[32] = "-";
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    CFArrayRef list = info ? IOPSCopyPowerSourcesList(info) : NULL;
    for (CFIndex i = 0; list && i < CFArrayGetCount(list); i++) {
        CFDictionaryRef d = IOPSGetPowerSourceDescription(info, CFArrayGetValueAtIndex(list, i));
        if (!d) continue;
        charge = dict_int(d, CFSTR(kIOPSCurrentCapacityKey));
        max = dict_int(d, CFSTR(kIOPSMaxCapacityKey));
        CFBooleanRef ch = CFDictionaryGetValue(d, CFSTR(kIOPSIsChargingKey));
        if (ch) charging = CFBooleanGetValue(ch);
        char buf[32];
        snprintf(source, sizeof source, "%s", cstr(CFDictionaryGetValue(d, CFSTR(kIOPSPowerSourceStateKey)), buf,
                                                   sizeof buf));
    }
    if (list) CFRelease(list);
    if (info) CFRelease(info);
    static int token = -1;
    uint64_t thermal = UINT64_MAX;
    if (token == -1 && notify_register_check("com.apple.system.thermalpressurelevel", &token) != NOTIFY_STATUS_OK)
        token = -2;
    if (token >= 0) notify_get_state(token, &thermal);
    double la[3] = {0};
    getloadavg(la, 3);
    fprintf(f, "M\t%llu\tadapter_w=%d\tbattery=%d/%d\tcharging=%d\tsource=\"%s\"\tthermal=%lld\tload1=%.2f\n",
            (unsigned long long)now_ns(), watts, charge, max, charging, source,
            thermal == UINT64_MAX ? -1LL : (long long)thermal, la[0]);
    fflush(f);
}

// ---------------------------------------------------------------- main
static volatile sig_atomic_t g_stop;
static void on_signal(int s) { (void)s; g_stop = 1; }

static int usage(void) {
    fprintf(stderr, "usage: powerlog -o FILE.tsv [-i ms] [-r n] [-m s] [-g group[:subgroup]]... [-- cmd args...]\n"
                    "       powerlog --list [seconds]\n");
    return 2;
}

int main(int argc, char **argv) {
    const bool ior_ok = ior_load();
    if (!ior_ok) fprintf(stderr, "powerlog: IOReport unavailable; SMC only\n");
    if (argc >= 2 && !strcmp(argv[1], "--list")) {
        double secs = argc >= 3 ? atof(argv[2]) : 1.0;
        Subscription s;
        if (!ior_ok || !ior_subscribe(NULL, 0, &s)) {
            fprintf(stderr, "powerlog: no IOReport subscription\n");
            return 1;
        }
        CFDictionaryRef a = ior.samples(s.sub, s.subbed, NULL);
        usleep((useconds_t)(secs * 1e6));
        CFDictionaryRef b = ior.samples(s.sub, s.subbed, NULL);
        if (a && b) ior_each(a, b, list_row, NULL);
        return 0;
    }
    const char *out = NULL;
    int interval_ms = 100, ior_every = 5, machine_s = 60, ng = 0;
    char *groups[32];
    int i = 1;
    for (; i < argc; i++) {
        if (!strcmp(argv[i], "--")) { i++; break; }
        if (i + 1 >= argc) return usage();
        if (!strcmp(argv[i], "-o")) out = argv[++i];
        else if (!strcmp(argv[i], "-i")) interval_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r")) ior_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m")) machine_s = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g") && ng < 32) groups[ng++] = argv[++i];
        else return usage();
    }
    if (!out || interval_ms < 1 || ior_every < 1 || machine_s < 1) return usage();
    static char *defaults[] = {"Energy Model", "GPU Stats:GPU Performance States",
                               "CPU Stats:CPU Complex Performance States", "AMC Stats:Perf Counters"};
    if (!ng) {
        for (size_t k = 0; k < sizeof defaults / sizeof *defaults; k++) groups[ng++] = defaults[k];
    }
    FILE *f = fopen(out, "w");
    if (!f) {
        fprintf(stderr, "powerlog: %s: %s\n", out, strerror(errno));
        return 1;
    }
    bool smc = smc_open();
    Subscription sub = {0};
    bool have_ior = ior_ok && ior_subscribe(groups, ng, &sub);
    fprintf(f, "# powerlog: clock ns of mach_absolute_time (CLOCK_UPTIME_RAW); S every %d ms, IOReport every %d S "
               "rows, M every %d s\n", interval_ms, ior_every, machine_s);
    fprintf(f, "# S\tt_ns");
    for (int k = 0; k < NK; k++) fprintf(f, "\t%s%s", KEYS[k], g_kok[k] ? "" : "(absent)");
    fprintf(f, "\n# IOReport groups:");
    for (int k = 0; k < ng; k++) fprintf(f, " \"%s\"", groups[k]);
    fprintf(f, "%s\n", have_ior ? "" : " (no subscription)");
    if (!smc) fprintf(stderr, "powerlog: SMC unavailable\n");

    pid_t child = -1;
    if (i < argc) {
        child = fork();
        if (child == 0) {
            execvp(argv[i], argv + i);
            fprintf(stderr, "powerlog: %s: %s\n", argv[i], strerror(errno));
            _exit(127);
        }
        if (child < 0) {
            fprintf(stderr, "powerlog: fork: %s\n", strerror(errno));
            return 1;
        }
    }
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    machine_row(f);
    CFDictionaryRef prev = have_ior ? ior.samples(sub.sub, sub.subbed, NULL) : NULL;
    uint64_t tprev = now_ns(), tmachine = tprev;
    int status = 0;
    for (long tick = 1;; tick++) {
        if (child > 0 && waitpid(child, &status, WNOHANG) == child) break;
        if (g_stop) {
            if (child > 0) {
                kill(child, SIGTERM);
                waitpid(child, &status, 0);
            }
            break;
        }
        uint64_t t = now_ns();
        if (smc) {
            fprintf(f, "S\t%llu", (unsigned long long)t);
            for (int k = 0; k < NK; k++) {
                float v = NAN;
                if (g_kok[k]) smc_flt(k4(KEYS[k]), g_ki[k], &v);
                fprintf(f, "\t%.3f", v);
            }
            fputc('\n', f);
        }
        if (have_ior && tick % ior_every == 0) ior_step(f, &sub, &prev, &tprev);
        if (t - tmachine >= (uint64_t)machine_s * 1000000000ull) {
            machine_row(f);
            tmachine = t;
        }
        usleep((useconds_t)interval_ms * 1000);
    }
    if (have_ior) ior_step(f, &sub, &prev, &tprev);   // the last interval
    machine_row(f);
    fclose(f);
    if (child > 0) return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    return 0;
}
