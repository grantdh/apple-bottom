# =============================================================================
# apple-bottom Makefile
# =============================================================================
#
# Usage:
#   make           Build library and examples
#   make test      Run correctness tests
#   make bench     Run performance benchmarks
#   make clean     Clean build artifacts
#
# =============================================================================

CC = clang
OBJC = clang

CFLAGS = -Wall -Wextra -O3 -std=c11 -DACCELERATE_NEW_LAPACK
OBJCFLAGS = -std=c++11 -Wall -Wextra -O3 -fobjc-arc -DACCELERATE_NEW_LAPACK
LDFLAGS = -lc++ -framework Metal -framework Foundation -framework Accelerate

# DD multiplication algorithm selector (v1.3 A/C investigation).
#   Default: DWTimesDW2 (JMP 2017 Alg 11, MR 2022 Thm 2.7, error bound < 5u²).
#   Opt-in : DWTimesDW3 (JMP 2017 Alg 12, MR 2022 Thm 2.8, error bound < 4u²).
# Invoke with `make clean && make lib DWTIMESDW3=1`. The flag is also propagated
# into the MSL compile step via MTLCompileOptions.preprocessorMacros.
ifdef DWTIMESDW3
  OBJCFLAGS += -DAPPLEBOTTOM_USE_DWTIMESDW3
endif

# Dylib link flags: re-export Accelerate so host binaries linking only
# -lapplebottom still get the full 150+ BLAS/LAPACK symbols we don't
# shadow ourselves. Our own dgemm_/zgemm_ win at link time via two-level
# namespace ordering (libapplebottom appears before Accelerate).
DYLIB_LDFLAGS = -lc++ -framework Metal -framework Foundation \
                -Wl,-reexport_framework,Accelerate
# Executables link against build/libapplebottom.dylib (install_name @rpath/…).
# Adding @loader_path so in-tree binaries (build/test_*, build/examples/*, build/bench_*)
# resolve the dylib sitting right next to them.
EXE_RPATH = -Wl,-rpath,@loader_path -Wl,-rpath,@loader_path/..

BUILD = build
SRC = src
INCLUDE = include
EXAMPLES = examples

# Library version, from include/apple_bottom.h (scripts/check_version.sh checks
# that every other place stating it agrees: make version-check).
VERSION := $(shell sed -nE 's/^.define APPLE_BOTTOM_VERSION_STRING "([^"]+)"/\1/p' $(INCLUDE)/apple_bottom.h)
VERSION_NUM := $(firstword $(subst -, ,$(VERSION)))

# Build identity reported by ab_build_info() and at the top of every
# AB_PROFILE / AB_REPORT report: the git commit, with -dirty if tracked files
# differ from it, or "unknown" outside a git checkout or when this directory is
# not the top of its checkout, as for a copy vendored inside another repository
# (override with make AB_BUILD_SHA=...). The stamp file changes only when the
# identity does, so a new commit or edit rebuilds just blas_profiler.o.
ifndef AB_BUILD_SHA
AB_BUILD_SHA := $(shell [ -z "$$(git rev-parse --show-prefix 2>/dev/null)" ] \
                  && sha=$$(git rev-parse --short=12 HEAD 2>/dev/null) \
                  && { git diff --quiet HEAD -- 2>/dev/null || sha=$$sha-dirty; echo $$sha; } \
                  || echo unknown)
endif

.PHONY: all lib test bench bench-report bench-rect clean examples install uninstall FORCE

# =============================================================================
# Main targets
# =============================================================================

all: lib examples
	@echo ""
	@echo "✓ Build complete!"
	@echo ""
	@echo "  Library:  $(BUILD)/libapplebottom.a"
	@echo "  Examples: $(BUILD)/examples/"
	@echo ""
	@echo "Quick start:"
	@echo "  ./$(BUILD)/examples/basic_dgemm"
	@echo ""

$(BUILD):
	mkdir -p $(BUILD)
	mkdir -p $(BUILD)/examples

# =============================================================================
# Library
# =============================================================================

$(BUILD)/apple_bottom.o: $(SRC)/apple_bottom.m $(INCLUDE)/apple_bottom.h | $(BUILD)
	$(OBJC) $(OBJCFLAGS) -I$(INCLUDE) -xobjective-c++ -c $(SRC)/apple_bottom.m -o $@

$(BUILD)/blas_wrapper.o: $(SRC)/blas_wrapper.c $(SRC)/blas_trans.h $(SRC)/dispatch.h $(INCLUDE)/apple_bottom.h $(SRC)/profiling/blas_profiler.h | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) -c $(SRC)/blas_wrapper.c -o $@

$(BUILD)/fortran_bridge.o: $(SRC)/fortran_bridge.c $(INCLUDE)/apple_bottom.h | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) -I$(SRC) -c $(SRC)/fortran_bridge.c -o $@

# BLAS-ABI routing; dispatch_table.h is the built-in table `make calibrate` writes.
$(BUILD)/dispatch.o: $(SRC)/dispatch.c $(SRC)/dispatch.h $(SRC)/dispatch_table.h | $(BUILD)
	$(CC) $(CFLAGS) -c $(SRC)/dispatch.c -o $@

$(BUILD)/build_sha.stamp: FORCE | $(BUILD)
	@echo '$(AB_BUILD_SHA)' | cmp -s - $@ || echo '$(AB_BUILD_SHA)' > $@

$(BUILD)/blas_profiler.o: $(SRC)/profiling/blas_profiler.c $(SRC)/profiling/blas_profiler.h $(INCLUDE)/apple_bottom.h $(BUILD)/build_sha.stamp | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) -I$(SRC) '-DAB_BUILD_SHA="$(AB_BUILD_SHA)"' -c $(SRC)/profiling/blas_profiler.c -o $@

$(BUILD)/device_api.o: $(SRC)/device_api.m $(INCLUDE)/apple_bottom.h $(INCLUDE)/apple_bottom_device.h | $(BUILD)
	$(OBJC) $(OBJCFLAGS) -I$(INCLUDE) -xobjective-c++ -c $(SRC)/device_api.m -o $@

LIB_OBJS = $(BUILD)/apple_bottom.o $(BUILD)/blas_wrapper.o \
           $(BUILD)/fortran_bridge.o $(BUILD)/blas_profiler.o \
           $(BUILD)/device_api.o $(BUILD)/dispatch.o

$(BUILD)/libapplebottom.a: $(LIB_OBJS)
	ar rcs $@ $^
	@echo "Built: $@"

# Shared library for Fortran linkage (QE, Yambo, …).
# install_name is set to @rpath so host binaries can locate us via -rpath at link time.
# Accelerate is re-exported so -lapplebottom alone provides full BLAS/LAPACK.
$(BUILD)/libapplebottom.dylib: $(LIB_OBJS)
	$(CC) -dynamiclib -install_name @rpath/libapplebottom.dylib \
	    -current_version $(VERSION_NUM) -compatibility_version 1.0.0 \
	    $^ -o $@ $(DYLIB_LDFLAGS)
	@echo "Built: $@"

dylib: $(BUILD)/libapplebottom.dylib

lib: $(BUILD)/libapplebottom.a $(BUILD)/libapplebottom.dylib

# =============================================================================
# Examples
# =============================================================================

$(BUILD)/examples/basic_dgemm: $(EXAMPLES)/01_basic_dgemm/main.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

examples: $(BUILD)/examples/basic_dgemm

# =============================================================================
# Benchmarks
# =============================================================================

$(BUILD)/bench_dgemm: benchmarks/bench_dgemm.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/bench_zgemm: benchmarks/bench_zgemm.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/bench_sgemm: benchmarks/bench_sgemm.c | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ $(LDFLAGS)
	@echo "Built: $@"

bench: $(BUILD)/bench_dgemm $(BUILD)/bench_sgemm $(BUILD)/bench_pool $(BUILD)/bench_zgemm $(BUILD)/bench_dsyrk $(BUILD)/bench_zherk $(BUILD)/bench_async
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "DGEMM Benchmark"
	@echo "═══════════════════════════════════════════════════════════════════"
	./$(BUILD)/bench_dgemm
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "ZGEMM Benchmark"
	@echo "═══════════════════════════════════════════════════════════════════"
	./$(BUILD)/bench_zgemm
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "SGEMM Reference (AMX FP32)"
	@echo "═══════════════════════════════════════════════════════════════════"
	./$(BUILD)/bench_sgemm

bench-report: bench
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Generating Benchmark Report"
	@echo "═══════════════════════════════════════════════════════════════════"
	@bash scripts/bench_report.sh

# =============================================================================
# Tests
# =============================================================================

$(BUILD)/test_precision: tests/test_precision.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/test_correctness: tests/test_correctness.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/test_device_api: tests/test_device_api.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

# BLAS-ABI conformance harness: column-major ab_*gemm_blas / ab_*gemm_ /
# dgemm_ / zgemm_ vs Accelerate cblas. Known failures are tracked in the
# XFAIL table in tests/test_blas_abi.c (XPASS and stale entries fail the run).
# AB_EXPECT_BUILD_SHA: the exit-report suite checks ab_build_info() names it.
$(BUILD)/test_blas_abi: tests/test_blas_abi.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) '-DAB_EXPECT_BUILD_SHA="$(AB_BUILD_SHA)"' $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/test_convergence: tests/verification/test_convergence.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

# Routing unit tests: dispatch.c alone, no library or GPU. They check the
# built-in table against its calibration data, found under AB_SOURCE_ROOT.
$(BUILD)/test_dispatch: tests/test_dispatch.c $(BUILD)/dispatch.o $(SRC)/dispatch.h $(SRC)/dispatch_table.h | $(BUILD)
	$(CC) $(CFLAGS) -I$(SRC) '-DAB_SOURCE_ROOT="$(CURDIR)"' $< $(BUILD)/dispatch.o -o $@
	@echo "Built: $@"

test: $(BUILD)/test_precision $(BUILD)/test_correctness $(BUILD)/test_device_api $(BUILD)/test_dispatch $(BUILD)/test_blas_abi test-ghost-detection test-symbols test-fortran
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Tests"
	@echo "═══════════════════════════════════════════════════════════════════"
	./$(BUILD)/test_precision
	./$(BUILD)/test_correctness
	./$(BUILD)/test_device_api
	./$(BUILD)/test_dispatch
	./$(BUILD)/test_blas_abi
	@echo ""
	@echo "✓ All tests passed!"

# Ghost-test detector: meta-test for the registry per-test delta guard.
# The inner binary returns the distinguished code 42 on success (ghost
# detected gracefully). The wrapper requires that exact code so a future
# crash (SIGSEGV=139, SIGABRT=134, etc.) cannot masquerade as META-PASS.
$(BUILD)/test_ghost_detection: tests/test_ghost_detection.c | $(BUILD)
	$(CC) $(CFLAGS) -Itests $< -o $@
	@echo "Built: $@"

.PHONY: test-ghost-detection
test-ghost-detection: $(BUILD)/test_ghost_detection
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running ghost-test detector (expecting exit 42)..."
	@echo "═══════════════════════════════════════════════════════════════════"
	@./$(BUILD)/test_ghost_detection; rc=$$?; \
	if [ $$rc -eq 42 ]; then \
		echo "META-PASS: ghost-test detector fired correctly (rc=42)"; \
	elif [ $$rc -eq 0 ]; then \
		echo "META-FAIL: ghost-test detector did NOT fire — registry per-test delta is broken"; \
		exit 1; \
	else \
		echo "META-FAIL: unexpected exit code $$rc (signal? abort?)"; \
		exit 1; \
	fi

# End-to-end Fortran binding probe (H-P0-1): a gfortran CALL DGEMM/ZGEMM,
# linked once against the dylib and once against the static archive, must
# reach the apple-bottom GPU path and give a correct result. Skipped when
# neither gfortran-15 nor gfortran is on PATH; override with GFORTRAN=...
GFORTRAN := $(firstword $(shell command -v gfortran-15 gfortran 2>/dev/null))

$(BUILD)/test_fortran_blas: tests/test_fortran_blas.f90 lib | $(BUILD)
	$(GFORTRAN) -O2 -Wall -Wextra $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/test_fortran_blas_static: tests/test_fortran_blas.f90 lib | $(BUILD)
	$(GFORTRAN) -O2 -Wall -Wextra $< $(BUILD)/libapplebottom.a -o $@ $(LDFLAGS)
	@echo "Built: $@"

.PHONY: test-fortran
ifeq ($(GFORTRAN),)
test-fortran:
	@echo "SKIP test-fortran: neither gfortran-15 nor gfortran is on PATH"
else
test-fortran: $(BUILD)/test_fortran_blas $(BUILD)/test_fortran_blas_static
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Fortran BLAS binding probe (dylib, static archive)"
	@echo "═══════════════════════════════════════════════════════════════════"
	AB_MODE=gpu ./$(BUILD)/test_fortran_blas
	AB_MODE=gpu ./$(BUILD)/test_fortran_blas_static
endif

test-verification: $(BUILD)/test_convergence
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Verification Tests (V&V)"
	@echo "═══════════════════════════════════════════════════════════════════"
	./$(BUILD)/test_convergence $(BUILD)/convergence_data.csv
	@echo ""

# Adversarial & system-level test suite
$(BUILD)/test_chaos: tests/test_chaos.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH) -lpthread
	@echo "Built: $@"

test-chaos: $(BUILD)/test_chaos
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Adversarial & System-Level Tests"
	@echo "═══════════════════════════════════════════════════════════════════"
	./$(BUILD)/test_chaos

# Python bindings smoke test (H-P1-5): stdlib unittest, numpy not required.
# Not part of `make test`. `python3 -I` ignores PYTHONPATH, so the test puts
# python/ on sys.path itself; AB_LIBRARY pins the dylib under test.
PYTHON ?= python3

.PHONY: test-python
test-python: $(BUILD)/libapplebottom.dylib
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Python binding tests"
	@echo "═══════════════════════════════════════════════════════════════════"
	AB_LIBRARY=$(abspath $(BUILD)/libapplebottom.dylib) $(PYTHON) -I tests/python/test_bindings.py

# Rectangular matrix diagnostic test
$(BUILD)/test_rectangular_diag: tests/test_rectangular_diag.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

test-rectangular-diag: $(BUILD)/test_rectangular_diag
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Rectangular Matrix Diagnostic Test"
	@echo "═══════════════════════════════════════════════════════════════════"
	./$(BUILD)/test_rectangular_diag

# Rectangular shapes up to 18277 x 150 x 18277 (correctness, then timings).
# Not part of `make test`: it peaks at about 21 GB of memory (M2 Max: ~10 s).
$(BUILD)/test_rectangular: tests/test_rectangular.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

.PHONY: test-rectangular test-rectangular-diag
test-rectangular: $(BUILD)/test_rectangular
	./$(BUILD)/test_rectangular

# Performance regression: save baseline or compare
bench-baseline: lib
	@echo "Saving performance baseline..."
	./scripts/perf_regression.sh --save

bench-regression: lib
	@echo "Running performance regression check..."
	./scripts/perf_regression.sh --ci

# =============================================================================
# Clean
# =============================================================================

clean:
	rm -rf $(BUILD)
	@echo "Cleaned build directory"

# =============================================================================
# Install (optional)
# =============================================================================

PREFIX ?= /usr/local

install: lib
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include
	install -d $(PREFIX)/lib/pkgconfig
	install -m 644 $(BUILD)/libapplebottom.a $(PREFIX)/lib/
	install -m 755 $(BUILD)/libapplebottom.dylib $(PREFIX)/lib/
	install -m 644 $(INCLUDE)/apple_bottom.h $(PREFIX)/include/
	install -m 644 $(INCLUDE)/apple_bottom_device.h $(PREFIX)/include/
	@sed -e 's|@PREFIX@|$(PREFIX)|g' \
	     -e 's|@VERSION@|$(VERSION)|g' \
	     applebottom.pc.in > $(PREFIX)/lib/pkgconfig/applebottom.pc
	@echo "Installed to $(PREFIX)"

uninstall:
	rm -f $(PREFIX)/lib/libapplebottom.a
	rm -f $(PREFIX)/lib/libapplebottom.dylib
	rm -f $(PREFIX)/include/apple_bottom.h
	rm -f $(PREFIX)/include/apple_bottom_device.h
	rm -f $(PREFIX)/lib/pkgconfig/applebottom.pc
	@echo "Uninstalled from $(PREFIX)"
# =============================================================================
# Add these targets to the end of your Makefile
# =============================================================================

# Sanitizer builds (for local debugging)
.PHONY: test-asan test-ubsan

test-asan: clean
	@echo "Building with AddressSanitizer..."
	$(OBJC) $(OBJCFLAGS) -fsanitize=address -g -I$(INCLUDE) -xobjective-c++ -c $(SRC)/apple_bottom.m -o $(BUILD)/apple_bottom.o
	$(CC) $(CFLAGS) -fsanitize=address -g -I$(INCLUDE) -c $(SRC)/blas_wrapper.c -o $(BUILD)/blas_wrapper.o
	$(CC) $(CFLAGS) -fsanitize=address -g -I$(INCLUDE) -c $(SRC)/fortran_bridge.c -o $(BUILD)/fortran_bridge.o
	ar rcs $(BUILD)/libapplebottom.a $(BUILD)/apple_bottom.o $(BUILD)/blas_wrapper.o $(BUILD)/fortran_bridge.o
	$(CC) $(CFLAGS) -fsanitize=address -g -I$(INCLUDE) tests/test_correctness.c -o $(BUILD)/test_correctness -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Running tests with ASan..."
	ASAN_OPTIONS=detect_leaks=1 ./$(BUILD)/test_correctness

test-ubsan: clean
	@echo "Building with UndefinedBehaviorSanitizer..."
	$(OBJC) $(OBJCFLAGS) -fsanitize=undefined -g -I$(INCLUDE) -xobjective-c++ -c $(SRC)/apple_bottom.m -o $(BUILD)/apple_bottom.o
	$(CC) $(CFLAGS) -fsanitize=undefined -g -I$(INCLUDE) -c $(SRC)/blas_wrapper.c -o $(BUILD)/blas_wrapper.o
	$(CC) $(CFLAGS) -fsanitize=undefined -g -I$(INCLUDE) -c $(SRC)/fortran_bridge.c -o $(BUILD)/fortran_bridge.o
	ar rcs $(BUILD)/libapplebottom.a $(BUILD)/apple_bottom.o $(BUILD)/blas_wrapper.o $(BUILD)/fortran_bridge.o
	$(CC) $(CFLAGS) -fsanitize=undefined -g -I$(INCLUDE) tests/test_correctness.c -o $(BUILD)/test_correctness -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Running tests with UBSan..."
	./$(BUILD)/test_correctness

$(BUILD)/bench_dsyrk: benchmarks/bench_dsyrk.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/bench_zherk: benchmarks/bench_zherk.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/bench_pool: benchmarks/bench_pool.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/bench_async: benchmarks/bench_async.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/bench_dtrsm: benchmarks/bench_dtrsm.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/bench_paper: benchmarks/bench_paper.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/bench_device_residency: benchmarks/bench_device_residency.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/bench_rect_dgemm: benchmarks/bench_rect_dgemm.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

$(BUILD)/bench_rect_zgemm: benchmarks/bench_rect_zgemm.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

bench-rect: $(BUILD)/bench_rect_dgemm $(BUILD)/bench_rect_zgemm
	@echo ""
	@echo "Built bench_rect_{dgemm,zgemm}. Run via benchmarks/scripts/run_rect_sweep.sh"

bench-paper: $(BUILD)/bench_paper
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "HPEC 2026 Paper Benchmark Suite"
	@echo "═══════════════════════════════════════════════════════════════════"
	./$(BUILD)/bench_paper

bench-paper-csv: $(BUILD)/bench_paper
	./$(BUILD)/bench_paper --csv

# AB_MODE=auto calibration (about 15 minutes; run on an idle machine on AC
# power). benchmarks/calibrate.c times every shape of its grid end to end
# through ab_*gemm_blas with the GPU path forced and through Accelerate, in
# one process, and derives the table that sends a BLAS-ABI call to the GPU
# only where the GPU measured >= 1.10x faster. It writes the raw and summary
# CSVs and dispatch_table.txt (use it at once with AB_DISPATCH_TABLE=<path>)
# under CALIBRATE_DIR, and regenerates CALIBRATE_SOURCE, the built-in table:
# rebuild to compile it in, and commit it with CALIBRATE_DIR.
# CALIBRATE_ARGS passes options, e.g. --only qe-paw for one part of the grid
# (which leaves CALIBRATE_SOURCE unchanged).
CALIBRATE_DIR ?= benchmarks/results/$(shell date +%Y-%m-%d)-$(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)-calibrate
CALIBRATE_SOURCE ?= $(SRC)/dispatch_table.h
CALIBRATE_ARGS ?=
GPU_CORES = $(shell ioreg -rc AGXAccelerator -d1 2>/dev/null | sed -nE 's/.*"gpu-core-count" = ([0-9]+).*/\1/p' | head -1)

$(BUILD)/calibrate: benchmarks/calibrate.c $(SRC)/dispatch.h lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) -I$(SRC) '-DAB_CAL_SDK="$(shell xcrun --show-sdk-version 2>/dev/null)"' $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

.PHONY: calibrate
calibrate: $(BUILD)/calibrate
	mkdir -p $(CALIBRATE_DIR)
	caffeinate -is ./$(BUILD)/calibrate --out $(CALIBRATE_DIR) --source $(CALIBRATE_SOURCE) \
	    --gpu-cores "$(GPU_CORES)" $(CALIBRATE_ARGS)
	@echo "Use it now: export AB_DISPATCH_TABLE=$(abspath $(CALIBRATE_DIR))/dispatch_table.txt"

# =============================================================================
# CI Targets
# =============================================================================

.PHONY: version-check
version-check:
	@./scripts/check_version.sh

.PHONY: build-check ci-local verify-symbols test-symbols

# Verify the dylib is a drop-in BLAS for Fortran host codes (QE, Yambo, …):
# it defines the exact Fortran BLAS names (test-symbols) and re-exports
# Accelerate for everything it does not shadow.
verify-symbols: test-symbols
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Verifying libapplebottom.dylib Accelerate re-export"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Accelerate re-export (LC_REEXPORT_DYLIB entries):"
	@otool -l $(BUILD)/libapplebottom.dylib | grep -A 2 LC_REEXPORT_DYLIB \
	    || (echo "  ✗ Accelerate NOT re-exported" && exit 1)
	@echo ""
	@echo "✓ Symbols look correct — this is a drop-in BLAS dylib"
	@echo ""

# Strict Fortran-ABI export check (H-P0-1): a gfortran CALL DGEMM/ZGEMM binds
# to the plain _dgemm_/_zgemm_ symbols, so the dylib and the static archive
# must each define exactly those names (ab_-prefixed exports do not count).
# The $NEWLAPACK variants must not be exported: C code calling dgemm_ through
# the new Accelerate headers keeps getting Accelerate.
test-symbols: $(BUILD)/libapplebottom.dylib $(BUILD)/libapplebottom.a
	@echo "Fortran BLAS exports:"
	@bad=0; \
	for lib in $^; do \
		syms=$$(nm -gU $$lib 2>/dev/null | awk '{print $$NF}'); \
		for s in _dgemm_ _zgemm_; do \
			if echo "$$syms" | grep -qx "$$s"; then \
				echo "  ✓ $$lib exports $$s"; \
			else \
				echo "  ✗ $$lib does not export $$s (H-P0-1)"; bad=1; \
			fi; \
		done; \
		for s in '_dgemm$$NEWLAPACK' '_zgemm$$NEWLAPACK'; do \
			if echo "$$syms" | grep -qxF "$$s"; then \
				echo "  ✗ $$lib exports $$s"; bad=1; \
			fi; \
		done; \
	done; \
	exit $$bad

# Build check: compile all source files to .o without linking
# Used by GitHub Actions to verify syntax and compilation (fast check)
# Objects go to build/check/ so they never replace the library's own objects
# (blas_profiler.o there carries the build SHA).
$(BUILD)/check:
	mkdir -p $@

build-check: | $(BUILD)/check
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Build Check (compile only, no linking)"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Compiling apple_bottom.m..."
	@$(OBJC) $(OBJCFLAGS) -I$(INCLUDE) -xobjective-c++ -c $(SRC)/apple_bottom.m -o $(BUILD)/check/apple_bottom.o
	@echo "Compiling blas_wrapper.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c $(SRC)/blas_wrapper.c -o $(BUILD)/check/blas_wrapper.o
	@echo "Compiling fortran_bridge.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -I$(SRC) -c $(SRC)/fortran_bridge.c -o $(BUILD)/check/fortran_bridge.o
	@echo "Compiling blas_profiler.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -I$(SRC) -c $(SRC)/profiling/blas_profiler.c -o $(BUILD)/check/blas_profiler.o
	@echo "Compiling dispatch.c..."
	@$(CC) $(CFLAGS) -c $(SRC)/dispatch.c -o $(BUILD)/check/dispatch.o
	@echo "Compiling device_api.m..."
	@$(OBJC) $(OBJCFLAGS) -I$(INCLUDE) -xobjective-c++ -c $(SRC)/device_api.m -o $(BUILD)/check/device_api.o
	@echo "Compiling test_correctness.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/test_correctness.c -o $(BUILD)/check/test_correctness.o
	@echo "Compiling test_precision.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/test_precision.c -o $(BUILD)/check/test_precision.o
	@echo "Compiling test_blas_abi.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/test_blas_abi.c -o $(BUILD)/check/test_blas_abi.o
	@echo "Compiling test_dispatch.c..."
	@$(CC) $(CFLAGS) -I$(SRC) -c tests/test_dispatch.c -o $(BUILD)/check/test_dispatch.o
	@echo "Compiling test_device_api.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/test_device_api.c -o $(BUILD)/check/test_device_api.o
	@echo "Compiling calibrate.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -I$(SRC) -c benchmarks/calibrate.c -o $(BUILD)/check/calibrate.o
	@echo "Compiling test_chaos.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/test_chaos.c -o $(BUILD)/check/test_chaos.o
	@echo "Compiling test_convergence.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/verification/test_convergence.c -o $(BUILD)/check/test_convergence.o
	@echo ""
	@echo "✓ All source files compile successfully"
	@echo ""

# Local CI: run full test suite + convergence study, generate CI report
# Only writes CI_REPORT.txt if all tests pass (exits on first failure)
#
# Counting rule (also README.md, tests/README.md): TEST_TOTAL counts tests.
# A test is one test_precision case, one TESTS[] registry entry of
# test_correctness / test_chaos / test_device_api / test_dispatch, or one
# test_blas_abi suite. Device-API CHECKs and BLAS-ABI cases are reported on
# their own lines and are not part of TEST_TOTAL. The report is written only
# after every binary passed, so TEST_PASSED = TEST_TOTAL and TEST_FAILED = 0.
ci-local: version-check $(BUILD)/test_precision $(BUILD)/test_correctness $(BUILD)/test_chaos $(BUILD)/test_device_api $(BUILD)/test_dispatch $(BUILD)/test_blas_abi $(BUILD)/test_convergence test-ghost-detection test-symbols test-fortran
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "CI Local Validation"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Running test suite..."
	@./$(BUILD)/test_precision > $(BUILD)/test_precision.log 2>&1 && echo "✓ test_precision: PASS" || (echo "✗ test_precision: FAIL" && cat $(BUILD)/test_precision.log && exit 1)
	@./$(BUILD)/test_correctness > $(BUILD)/test_correctness.log 2>&1 && echo "✓ test_correctness: PASS" || (echo "✗ test_correctness: FAIL" && cat $(BUILD)/test_correctness.log && exit 1)
	@./$(BUILD)/test_chaos > $(BUILD)/test_chaos.log 2>&1 && echo "✓ test_chaos: PASS" || (echo "✗ test_chaos: FAIL" && cat $(BUILD)/test_chaos.log && exit 1)
	@./$(BUILD)/test_device_api > $(BUILD)/test_device_api.log 2>&1 && echo "✓ test_device_api: PASS" || (echo "✗ test_device_api: FAIL" && cat $(BUILD)/test_device_api.log && exit 1)
	@./$(BUILD)/test_dispatch > $(BUILD)/test_dispatch.log 2>&1 && echo "✓ test_dispatch: PASS" || (echo "✗ test_dispatch: FAIL" && cat $(BUILD)/test_dispatch.log && exit 1)
	@./$(BUILD)/test_blas_abi > $(BUILD)/test_blas_abi.log 2>&1 && echo "✓ test_blas_abi: PASS" || (echo "✗ test_blas_abi: FAIL" && cat $(BUILD)/test_blas_abi.log && exit 1)
	@echo ""
	@echo "Running convergence study (V-2)..."
	@rm -f $(BUILD)/convergence_data.csv
	@./$(BUILD)/test_convergence $(BUILD)/convergence_data.csv > $(BUILD)/test_convergence.log 2>&1 && echo "✓ test_convergence: PASS" || (echo "✗ test_convergence: FAIL" && cat $(BUILD)/test_convergence.log && exit 1)
	@echo ""
	@echo "Generating CI report..."
	@bash -c ' \
		PRECISION=$$(grep -c "✓ PASS" $(BUILD)/test_precision.log); \
		CORRECTNESS=$$(sed -nE "s/^Results: ([0-9]+) passed.*/\1/p" $(BUILD)/test_correctness.log); \
		CHAOS=$$(sed -nE "s/^Results: ([0-9]+) passed.*/\1/p" $(BUILD)/test_chaos.log); \
		DEVICE_API=$$(sed -nE "s/^Summary: ([0-9]+) test.*/\1/p" $(BUILD)/test_device_api.log); \
		DEVICE_API_CHECKS=$$(sed -nE "s/^Summary: .* ([0-9]+) CHECK.*/\1/p" $(BUILD)/test_device_api.log); \
		DISPATCH=$$(sed -nE "s/^Results: ([0-9]+) passed.*/\1/p" $(BUILD)/test_dispatch.log); \
		BLAS_ABI=$$(grep -cE "^\[[0-9]+/[0-9]+\] " $(BUILD)/test_blas_abi.log); \
		for n in "$$PRECISION" "$$CORRECTNESS" "$$CHAOS" "$$DEVICE_API" "$$DEVICE_API_CHECKS" "$$DISPATCH" "$$BLAS_ABI"; do \
			case "$$n" in ""|*[!0-9]*|0) echo "✗ CI report: could not read a test count from the logs"; exit 1;; esac; \
		done; \
		TOTAL=$$((PRECISION + CORRECTNESS + CHAOS + DEVICE_API + DISPATCH + BLAS_ABI)); \
		echo "GIT_SHA: $$(git rev-parse HEAD)" > $(BUILD)/CI_REPORT.txt; \
		echo "GIT_BRANCH: $$(git rev-parse --abbrev-ref HEAD)" >> $(BUILD)/CI_REPORT.txt; \
		echo "AB_BUILD_SHA: $(AB_BUILD_SHA)" >> $(BUILD)/CI_REPORT.txt; \
		echo "TIMESTAMP: $$(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> $(BUILD)/CI_REPORT.txt; \
		echo "HOSTNAME: $$(hostname)" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_PRECISION: PASS ($$PRECISION tests)" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_CORRECTNESS: PASS ($$CORRECTNESS tests)" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_CHAOS: PASS ($$CHAOS tests)" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_DEVICE_API: PASS ($$DEVICE_API tests)" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_DISPATCH: PASS ($$DISPATCH tests)" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_BLAS_ABI: PASS ($$BLAS_ABI tests)" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_CONVERGENCE: PASS" >> $(BUILD)/CI_REPORT.txt; \
		echo "test-ghost-detection: PASS" >> $(BUILD)/CI_REPORT.txt; \
		echo "test-symbols: PASS" >> $(BUILD)/CI_REPORT.txt; \
		echo "test-fortran: $(if $(GFORTRAN),PASS,SKIP (no gfortran))" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_TOTAL: $$TOTAL" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_PASSED: $$TOTAL" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_FAILED: 0" >> $(BUILD)/CI_REPORT.txt; \
		echo "DEVICE_API_CHECKS: $$DEVICE_API_CHECKS" >> $(BUILD)/CI_REPORT.txt; \
		echo "BLAS_ABI_CASES: $$(sed -nE "s/^Results: //p" $(BUILD)/test_blas_abi.log)" >> $(BUILD)/CI_REPORT.txt; \
		if [ -f $(BUILD)/convergence_data.csv ]; then \
			POINTS=$$(tail -n +2 $(BUILD)/convergence_data.csv | wc -l | tr -d " "); \
			SHA=$$(shasum -a 256 $(BUILD)/convergence_data.csv | cut -d" " -f1); \
			echo "CONVERGENCE_POINTS: $$POINTS" >> $(BUILD)/CI_REPORT.txt; \
			echo "CONVERGENCE_DATA_SHA256: $$SHA" >> $(BUILD)/CI_REPORT.txt; \
		else \
			echo "CONVERGENCE_POINTS: 0" >> $(BUILD)/CI_REPORT.txt; \
			echo "CONVERGENCE_DATA_SHA256: missing" >> $(BUILD)/CI_REPORT.txt; \
		fi; \
		echo "STATUS: PASS" >> $(BUILD)/CI_REPORT.txt \
	'
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "✓ CI Validation Complete"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@cat $(BUILD)/CI_REPORT.txt
	@echo ""
	@echo "Report saved to: $(BUILD)/CI_REPORT.txt"
	@echo ""
