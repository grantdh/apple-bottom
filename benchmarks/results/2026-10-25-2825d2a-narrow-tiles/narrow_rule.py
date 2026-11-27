#!/usr/bin/env python3
"""The narrow-tile rule (use_narrow_tiles in src/apple_bottom.m) before and after this change, scored on nrprobe /
pfprobe timings, and the grid sweeps it rests on.

  narrow_rule.py score FILE...   old vs new rule on 'time' outputs: per line and per file (ratio of mean GPU times)
  narrow_rule.py steps           the step boundaries of every one-row grid sweep (sweep_*.txt)
  narrow_rule.py predict FILE... each line's kernel times predicted from the sweeps (K scaled from 1024), against measured

For each line (M x N x K, np products) the candidates are the kernels the library routes to:
  M < 4N, or ab_dgemm whatever the shape: 64x64 (dd_dgemm_pf alone with <= 64 threadgroups, else dd_dgemm) or narrow
                                           (dd_dgemm_narrow_pf);
  M >= 4N through the scaled dispatch:     tall-skinny (dd_dgemm_ab_ts_pf up to 512 threadgroups, else dd_dgemm_ab_ts)
                                           or narrow (dd_dgemm_ab_narrow_pf).
The scaled 64x64 and narrow kernels track the plain ones within 2% (2026-10-24-15ef7b9-prefetch/kernel_scaled.txt)."""
import math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def cdiv(a, b): return (a + b - 1) // b
def grids(M, N): return cdiv(M, 64) * cdiv(N, 64), cdiv(M, 32) * cdiv(N, 32), cdiv(M, 128) * cdiv(N, 16)

def old_rule(M, N, ts, conc):
    """dce55b5: narrow for C <= 480^2; over tall-skinny only where 32 columns pad N no more than 16, or <= 16 ts groups"""
    if M * N > 480 * 480: return False
    if not ts: return True
    return cdiv(N, 32) * 32 == cdiv(N, 16) * 16 or grids(M, N)[2] <= 16

STEP = 33   # AB_GPU_STEP_GROUPS
def steps(g): return cdiv(g, STEP)

def new_rule(M, N, ts, conc):
    """this change; the constants are AB_NARROW_MAX_GROUPS(_CONCURRENT), AB_NARROW_MAX_GROUPS_VS_PF, AB_PREFETCH_MAX_GROUPS"""
    g64, gn, gts = grids(M, N)
    if gn > (441 if conc else 1024): return False
    if ts: return steps(gn) <= 2 * steps(gts)
    return g64 > (0 if conc else 64) or gn <= 239

def load_time(path):
    rows, hdr = [], None
    for ln in open(path):
        if ln.startswith('#') or not ln.strip() or 'skipped' in ln: continue
        f = ln.split('|')[0].split()
        if f[0] == 'M': hdr = f; continue
        if hdr is None or ln.strip().startswith('raw') or len(f) != len(hdr): continue
        rows.append({k: (int(v.rstrip('x')) if k in ('M', 'N', 'K', 'np', 't64', 't32', 'tts') else float(v))
                     for k, v in zip(hdr, f)})
    return rows

def candidates(r):
    """(tall-skinny dispatch?, {'w': 64x64 or tall-skinny ms, 'n': narrow ms}) or None"""
    M, N, np_ = r['M'], r['N'], r['np']
    g64, gn, gts = grids(M, N)
    if M >= 4 * N and 'tsp' in r and 'nwsp' in r:
        return True, {'w': r['tsp'] if gts <= 512 else r['ts'], 'n': r['nwsp']}
    if 'sq' in r and 'nwp' in r:
        return False, {'w': r['sqp'] if (np_ == 1 and g64 <= 64) else r['sq'], 'n': r['nwp']}
    return None

def score(paths):
    allg = []
    for p in paths:
        g, ro, rn, lines = [], [], [], []
        for r in load_time(p):
            c = candidates(r)
            if c is None: continue
            ts, t = c
            conc = r['np'] > 1
            o = 'n' if old_rule(r['M'], r['N'], ts, conc) else 'w'
            n = 'n' if new_rule(r['M'], r['N'], ts, conc) else 'w'
            best = min(t.values())
            g.append(t[o] / t[n]); ro.append(t[o] / best - 1); rn.append(t[n] / best - 1)
            lines.append("%6d %5d %6d %dx %-2s  %s %8.3f  narrow %8.3f  old %s new %s  old/new %.3f" % (
                r['M'], r['N'], r['K'], r['np'], 'ts' if ts else '', 'ts ' if ts else '64 ', t['w'], t['n'],
                o, n, t[o] / t[n]))
        print("\n".join(lines))
        gm = math.exp(sum(map(math.log, g)) / len(g))
        print("== %s: %d lines, %d rerouted; old/new geomean %.3f (min %.3f, max %.3f); regret against the faster "
              "kernel: old mean %.3f max %.3f, new mean %.3f max %.3f\n" % (
                  os.path.basename(p), len(g), sum(x != 1 for x in g), gm, min(g), max(g),
                  sum(ro) / len(ro), max(ro), sum(rn) / len(rn), max(rn)))
        allg += g

def load_sweep(path):
    rows, hdr = [], None
    for ln in open(path):
        if ln.startswith('#') or not ln.strip(): continue
        f = ln.split('|')[0].split()
        if f[0] == 'G': hdr = f; continue
        if hdr is None or len(f) != len(hdr): continue
        rows.append({k: (int(v.rstrip('x')) if k in ('G', 'K', 'np') else float(v)) for k, v in zip(hdr, f)})
    return rows

def sweep_steps():
    for fn in sorted(os.listdir(HERE)):
        if not fn.startswith('sweep_'): continue
        rows = load_sweep(os.path.join(HERE, fn))
        for k in [k for k in rows[0] if k not in ('G', 'K', 'np')]:
            out, prev = [], None
            for r in rows:
                if prev is not None and r[k] > prev * 1.06: out.append("%d" % r['G'])
                prev = r[k]
            print("%-22s %-5s first %.3f ms; steps up at G = %s" % (fn, k, rows[0][k], " ".join(out[:24])))

def table():
    T = {}
    for fn in os.listdir(HERE):
        if not fn.startswith('sweep_') or 'big' in fn: continue
        for r in load_sweep(os.path.join(HERE, fn)):
            for k in r:
                if k not in ('G', 'K', 'np'): T.setdefault((k, r['np']), {})[r['G']] = r[k]
    return T

def predict(paths):
    T = table()
    def t(k, np_, G):
        d = T.get((k, np_))
        if not d: return float('nan')
        gmax = max(d)
        if G <= gmax: return d[G]
        g0 = int(gmax * 0.6)
        return d[gmax] + (d[gmax] - d[g0]) / (gmax - g0) * (G - gmax)
    for p in paths:
        for r in load_time(p):
            c = candidates(r)
            if c is None or r['np'] not in (1, 3): continue
            ts, meas = c
            g64, gn, gts = grids(r['M'], r['N'])
            s = r['K'] / 1024
            pw = (t('tsp' if gts <= 512 else 'ts', r['np'], gts) if ts else
                  t('sqp' if (r['np'] == 1 and g64 <= 64) else 'sq', r['np'], g64)) * s
            pn = t('nwp', r['np'], gn) * s
            print("%6d %5d %6d %dx  %s meas %8.3f pred %8.3f   narrow meas %8.3f pred %8.3f" % (
                r['M'], r['N'], r['K'], r['np'], 'ts' if ts else '64', meas['w'], pw, meas['n'], pn))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'score': score(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == 'steps': sweep_steps()
    elif len(sys.argv) > 1 and sys.argv[1] == 'predict': predict(sys.argv[2:])
    else: print(__doc__)
