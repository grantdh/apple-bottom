# Documentation guide

## Evaluating apple-bottom

Read in this order:

1. [README](../README.md): what apple-bottom is, when it is faster than Accelerate and when it is not, and its
   limitations.
2. [INTEGRATION.md](INTEGRATION.md): using it from C, Python and Fortran, and in Quantum ESPRESSO.
3. [REPRODUCIBILITY.md](REPRODUCIBILITY.md): each performance number in the README and VAL-1, the command that
   regenerates it, and the dated results folder it came from.
4. [vv/VV_REPORT.md](vv/VV_REPORT.md), the verification and validation report, with
   [vv/PRECISION_ENVELOPE.md](vv/PRECISION_ENVELOPE.md) (error bounds) and [vv/LIMITATIONS.md](vv/LIMITATIONS.md)
   (scope, and what is not claimed).
5. [VAL-1](../tests/validation/VAL001_QE_Si64.md): Quantum ESPRESSO Si64 SCF, DD GEMMs on the GPU against FP64 GEMMs on
   the CPU in one QE build.
6. [COMPARISON.md](COMPARISON.md): the other ways to do FP64 matrix work on Apple Silicon.
7. [ROADMAP.md](ROADMAP.md) for what is planned, [CHANGELOG.md](../CHANGELOG.md) for what has shipped.

## Design notes

- [design/RECTANGULAR_TILING.md](design/RECTANGULAR_TILING.md): the tile configurations, including the tall-skinny
  variant for Quantum ESPRESSO's Davidson shapes.
- [design/FP32_UTILIZATION.md](design/FP32_UTILIZATION.md): how the FP32 utilization figures are derived.
- [vv/DD_PRIMITIVES_CURRENT.md](vv/DD_PRIMITIVES_CURRENT.md): the DD multiplication primitives as investigated for
  1.3.0 (kept for the record; see its status note).

## Contributing and releasing

[CONTRIBUTING.md](../CONTRIBUTING.md), [tests/README.md](../tests/README.md) (the suites and their counts),
[RELEASING.md](RELEASING.md), [SECURITY.md](../SECURITY.md).

## How releases are tested

- **Continuous integration.** GitHub-hosted macOS runners (macos-15 and macos-26) check every push to `main` and every
  pull request into `main` or a release branch:
  - version consistency;
  - a compile of every source file;
  - the static and shared libraries;
  - the exported BLAS symbols;
  - the routing unit tests.

  These runners give jobs no usable Metal GPU, so the GPU suites cannot run there.
- **Before each release,** `make ci-local` runs the full suite on Apple Silicon:
  - precision and correctness;
  - chaos, device API, routing, BLAS ABI and trace;
  - the staged path;
  - the bit-for-bit check of GPU-routed results against the golden file;
  - the V-2 convergence study.

  The build must be from a recorded, clean commit, and the test counts must match
  [tests/README.md](../tests/README.md). [RELEASING.md](RELEASING.md) has the checklist.
- **Measurements.** Performance and validation numbers come from dated folders in
  [benchmarks/results](../benchmarks/results/), which name the machine, OS, toolchain and library build of each run.
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md) maps each published number to its folder and command.
- **Releases so far.** Nine tags, from 1.0.0 (2026-03-22) to 1.5.1 (2026-11-01). Each is recorded in
  [CHANGELOG.md](../CHANGELOG.md).
