#!/usr/bin/env python3
"""
Analyze GEMM call profile from apple-bottom's AB_PROFILE_FILE output.

Usage:
    # First, run QE with profiling enabled:
    AB_PROFILE_FILE=gemm_trace.log pw.x < si64.in

    # Then analyze:
    python3 scripts/analyze_gemm_profile.py gemm_trace.log

Output includes:
    - Call count by function (dgemm vs zgemm)
    - Size distribution histogram
    - GPU vs CPU routing breakdown
    - Batching opportunity analysis (sequences of same-size calls)
"""

import sys
import collections
from pathlib import Path


def parse_profile(path):
    """Parse AB_PROFILE_FILE format: func M N K MNK gpu"""
    calls = []
    with open(path) as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                calls.append({
                    'func': parts[0],
                    'M': int(parts[1]),
                    'N': int(parts[2]),
                    'K': int(parts[3]),
                    'mnk': int(parts[4]),
                    'gpu': int(parts[5]),
                })
    return calls


def analyze(calls):
    print(f"Total GEMM calls: {len(calls)}")
    print()

    # Breakdown by function
    func_counts = collections.Counter(c['func'] for c in calls)
    print("Call counts by function:")
    for fn, count in func_counts.most_common():
        gpu_count = sum(1 for c in calls if c['func'] == fn and c['gpu'])
        print(f"  {fn}: {count} calls ({gpu_count} GPU, {count - gpu_count} CPU)")
    print()

    # Size distribution
    print("Size distribution (M×N×K):")
    size_buckets = collections.Counter()
    for c in calls:
        key = f"{c['M']}×{c['N']}×{c['K']}"
        size_buckets[key] += 1

    for size, count in size_buckets.most_common(20):
        c = next(x for x in calls if f"{x['M']}×{x['N']}×{x['K']}" == size)
        route = "GPU" if c['gpu'] else "CPU"
        flops = c['mnk'] * (8 if c['func'] == 'zgemm' else 2)
        print(f"  {size:>20s}: {count:>5d} calls  ({route})  {flops/1e6:.1f} MFLOP/call")
    if len(size_buckets) > 20:
        print(f"  ... and {len(size_buckets) - 20} more unique sizes")
    print()

    # GPU routing statistics
    gpu_calls = [c for c in calls if c['gpu']]
    cpu_calls = [c for c in calls if not c['gpu']]
    print(f"GPU routing: {len(gpu_calls)} GPU, {len(cpu_calls)} CPU")
    if gpu_calls:
        avg_gpu = sum(c['mnk'] for c in gpu_calls) / len(gpu_calls)
        print(f"  Avg GPU call size (MNK): {avg_gpu:.0f}")
    if cpu_calls:
        avg_cpu = sum(c['mnk'] for c in cpu_calls) / len(cpu_calls)
        print(f"  Avg CPU call size (MNK): {avg_cpu:.0f}")
    print()

    # Batching opportunity: find consecutive same-size calls
    print("Batching opportunities (consecutive same-size calls):")
    runs = []
    current_key = None
    current_run = 0
    for c in calls:
        key = (c['func'], c['M'], c['N'], c['K'])
        if key == current_key:
            current_run += 1
        else:
            if current_run > 1:
                runs.append((current_key, current_run))
            current_key = key
            current_run = 1
    if current_run > 1:
        runs.append((current_key, current_run))

    if runs:
        runs.sort(key=lambda x: -x[1])
        total_batchable = sum(r[1] for r in runs)
        print(f"  {len(runs)} batch-able sequences found ({total_batchable} calls total)")
        for (fn, m, n, k), count in runs[:10]:
            print(f"    {fn} {m}×{n}×{k}: {count} consecutive calls")
        if len(runs) > 10:
            print(f"    ... and {len(runs) - 10} more sequences")
    else:
        print("  No consecutive same-size calls found")
    print()

    # Recommendation
    print("Recommendation:")
    if cpu_calls:
        small_sizes = set(f"{c['M']}×{c['N']}×{c['K']}" for c in cpu_calls)
        print(f"  {len(cpu_calls)} calls routed to CPU ({len(small_sizes)} unique sizes)")
        if runs:
            print(f"  Consider ab_dgemm_batched for {total_batchable} batchable GPU calls")
    if not gpu_calls:
        print("  All calls below crossover — lower AB_CROSSOVER_FLOPS to route more to GPU")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <profile_file>")
        print()
        print("Generate a profile by running QE with:")
        print("  AB_PROFILE_FILE=gemm_trace.log pw.x < input.in")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    calls = parse_profile(path)
    if not calls:
        print("No GEMM calls found in profile")
        sys.exit(1)

    analyze(calls)


if __name__ == '__main__':
    main()
