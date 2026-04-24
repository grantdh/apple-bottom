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

.PHONY: all lib test bench bench-report bench-rect clean examples install uninstall

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

$(BUILD)/blas_wrapper.o: $(SRC)/blas_wrapper.c $(INCLUDE)/apple_bottom.h | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) -c $(SRC)/blas_wrapper.c -o $@

$(BUILD)/fortran_bridge.o: $(SRC)/fortran_bridge.c $(INCLUDE)/apple_bottom.h $(SRC)/profiling/blas_profiler.h | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) -I$(SRC) -c $(SRC)/fortran_bridge.c -o $@

$(BUILD)/blas_profiler.o: $(SRC)/profiling/blas_profiler.c $(SRC)/profiling/blas_profiler.h | $(BUILD)
	$(CC) $(CFLAGS) -I$(SRC) -c $(SRC)/profiling/blas_profiler.c -o $@

$(BUILD)/device_api.o: $(SRC)/device_api.m $(INCLUDE)/apple_bottom.h $(INCLUDE)/apple_bottom_device.h | $(BUILD)
	$(OBJC) $(OBJCFLAGS) -I$(INCLUDE) -xobjective-c++ -c $(SRC)/device_api.m -o $@

LIB_OBJS = $(BUILD)/apple_bottom.o $(BUILD)/blas_wrapper.o \
           $(BUILD)/fortran_bridge.o $(BUILD)/blas_profiler.o \
           $(BUILD)/device_api.o

$(BUILD)/libapplebottom.a: $(LIB_OBJS)
	ar rcs $@ $^
	@echo "Built: $@"

# Shared library for Fortran linkage (QE, Yambo, …).
# install_name is set to @rpath so host binaries can locate us via -rpath at link time.
# Accelerate is re-exported so -lapplebottom alone provides full BLAS/LAPACK.
$(BUILD)/libapplebottom.dylib: $(LIB_OBJS)
	$(CC) -dynamiclib -install_name @rpath/libapplebottom.dylib \
	    -current_version 1.3.1 -compatibility_version 1.0.0 \
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

$(BUILD)/test_convergence: tests/verification/test_convergence.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS) $(EXE_RPATH)
	@echo "Built: $@"

test: $(BUILD)/test_precision $(BUILD)/test_correctness $(BUILD)/test_device_api
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Tests"
	@echo "═══════════════════════════════════════════════════════════════════"
	./$(BUILD)/test_precision
	./$(BUILD)/test_correctness
	./$(BUILD)/test_device_api
	@echo ""
	@echo "✓ All tests passed!"

test-verification: $(BUILD)/test_convergence
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Verification Tests (V&V)"
	@echo "═══════════════════════════════════════════════════════════════════"
	./$(BUILD)/test_convergence
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
	     -e 's|@VERSION@|1.3.1|g' \
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

# =============================================================================
# CI Targets
# =============================================================================

.PHONY: build-check ci-local verify-symbols

# Verify the dylib exports standard Fortran BLAS names (dgemm_, zgemm_)
# and that Accelerate is properly re-exported. This confirms the dylib
# is a true drop-in for Fortran host codes (QE, Yambo, …).
verify-symbols: $(BUILD)/libapplebottom.dylib
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Verifying libapplebottom.dylib symbol exports"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Shadowed Fortran BLAS symbols (should be defined in our dylib):"
	@nm -gU $(BUILD)/libapplebottom.dylib 2>/dev/null | grep -E " _(dgemm_|zgemm_|ab_dgemm_|ab_zgemm_)$$" \
	    || (echo "  ✗ MISSING standard BLAS exports" && exit 1)
	@echo ""
	@echo "Accelerate re-export (LC_REEXPORT_DYLIB entries):"
	@otool -l $(BUILD)/libapplebottom.dylib | grep -A 2 LC_REEXPORT_DYLIB \
	    || (echo "  ✗ Accelerate NOT re-exported" && exit 1)
	@echo ""
	@echo "✓ Symbols look correct — this is a drop-in BLAS dylib"
	@echo ""

# Build check: compile all source files to .o without linking
# Used by GitHub Actions to verify syntax and compilation (fast check)
build-check: | $(BUILD)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Build Check (compile only, no linking)"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Compiling apple_bottom.m..."
	@$(OBJC) $(OBJCFLAGS) -I$(INCLUDE) -xobjective-c++ -c $(SRC)/apple_bottom.m -o $(BUILD)/apple_bottom.o
	@echo "Compiling blas_wrapper.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c $(SRC)/blas_wrapper.c -o $(BUILD)/blas_wrapper.o
	@echo "Compiling fortran_bridge.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -I$(SRC) -c $(SRC)/fortran_bridge.c -o $(BUILD)/fortran_bridge.o
	@echo "Compiling blas_profiler.c..."
	@$(CC) $(CFLAGS) -I$(SRC) -c $(SRC)/profiling/blas_profiler.c -o $(BUILD)/blas_profiler.o
	@echo "Compiling test_correctness.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/test_correctness.c -o $(BUILD)/test_correctness.o
	@echo "Compiling test_precision.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/test_precision.c -o $(BUILD)/test_precision.o
	@echo "Compiling test_chaos.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/test_chaos.c -o $(BUILD)/test_chaos.o
	@echo "Compiling test_convergence.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/verification/test_convergence.c -o $(BUILD)/test_convergence.o
	@echo ""
	@echo "✓ All source files compile successfully"
	@echo ""

# Local CI: run full test suite + convergence study, generate CI report
# Only writes CI_REPORT.txt if all tests pass (exits on first failure)
ci-local: $(BUILD)/test_precision $(BUILD)/test_correctness $(BUILD)/test_chaos $(BUILD)/test_convergence
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "CI Local Validation"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Running test suite..."
	@./$(BUILD)/test_precision > $(BUILD)/test_precision.log 2>&1 && echo "✓ test_precision: PASS" || (echo "✗ test_precision: FAIL" && cat $(BUILD)/test_precision.log && exit 1)
	@./$(BUILD)/test_correctness > $(BUILD)/test_correctness.log 2>&1 && echo "✓ test_correctness: PASS" || (echo "✗ test_correctness: FAIL" && cat $(BUILD)/test_correctness.log && exit 1)
	@./$(BUILD)/test_chaos > $(BUILD)/test_chaos.log 2>&1 && echo "✓ test_chaos: PASS" || (echo "✗ test_chaos: FAIL" && cat $(BUILD)/test_chaos.log && exit 1)
	@echo ""
	@echo "Running convergence study (V-2)..."
	@./$(BUILD)/test_convergence > $(BUILD)/test_convergence.log 2>&1 && echo "✓ test_convergence: PASS" || (echo "✗ test_convergence: FAIL" && cat $(BUILD)/test_convergence.log && exit 1)
	@echo ""
	@echo "Generating CI report..."
	@bash -c ' \
		PRECISION_PASS=$$(grep -c "✓ PASS" $(BUILD)/test_precision.log || echo 0); \
		CORRECTNESS_LINE=$$(grep "Results:" $(BUILD)/test_correctness.log); \
		CORRECTNESS_PASS=$$(echo "$$CORRECTNESS_LINE" | sed -E "s/.*Results: ([0-9]+) passed.*/\1/"); \
		CORRECTNESS_FAIL=$$(echo "$$CORRECTNESS_LINE" | sed -E "s/.*Results: [0-9]+ passed, ([0-9]+) failed.*/\1/"); \
		TOTAL=$$((PRECISION_PASS + CORRECTNESS_PASS)); \
		FAILED=$$CORRECTNESS_FAIL; \
		echo "GIT_SHA: $$(git rev-parse HEAD)" > $(BUILD)/CI_REPORT.txt; \
		echo "GIT_BRANCH: $$(git rev-parse --abbrev-ref HEAD)" >> $(BUILD)/CI_REPORT.txt; \
		echo "TIMESTAMP: $$(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> $(BUILD)/CI_REPORT.txt; \
		echo "HOSTNAME: $$(hostname)" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_PRECISION: PASS ($$PRECISION_PASS tests)" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_CORRECTNESS: PASS ($$CORRECTNESS_PASS tests)" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_CONVERGENCE: PASS" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_TOTAL: $$TOTAL" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_PASSED: $$TOTAL" >> $(BUILD)/CI_REPORT.txt; \
		echo "TEST_FAILED: $$FAILED" >> $(BUILD)/CI_REPORT.txt; \
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
