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

BUILD = build
SRC = src
INCLUDE = include
EXAMPLES = examples

.PHONY: all lib test bench bench-report clean examples install uninstall

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

$(BUILD)/fortran_bridge.o: $(SRC)/fortran_bridge.c $(INCLUDE)/apple_bottom.h | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) -c $(SRC)/fortran_bridge.c -o $@

$(BUILD)/libapplebottom.a: $(BUILD)/apple_bottom.o $(BUILD)/blas_wrapper.o $(BUILD)/fortran_bridge.o
	ar rcs $@ $^
	@echo "Built: $@"

lib: $(BUILD)/libapplebottom.a

# =============================================================================
# Examples
# =============================================================================

$(BUILD)/examples/basic_dgemm: $(EXAMPLES)/01_basic_dgemm/main.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

examples: $(BUILD)/examples/basic_dgemm

# =============================================================================
# Benchmarks
# =============================================================================

$(BUILD)/bench_dgemm: benchmarks/bench_dgemm.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

$(BUILD)/bench_zgemm: benchmarks/bench_zgemm.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

bench: $(BUILD)/bench_dgemm $(BUILD)/bench_pool $(BUILD)/bench_zgemm $(BUILD)/bench_dsyrk $(BUILD)/bench_zherk $(BUILD)/bench_async
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
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

$(BUILD)/test_correctness: tests/test_correctness.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

$(BUILD)/test_convergence: tests/verification/test_convergence.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

test: $(BUILD)/test_precision $(BUILD)/test_correctness
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Tests"
	@echo "═══════════════════════════════════════════════════════════════════"
	./$(BUILD)/test_precision
	./$(BUILD)/test_correctness
	@echo ""
	@echo "✓ All tests passed!"

test-verification: $(BUILD)/test_convergence
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Verification Tests (V&V)"
	@echo "═══════════════════════════════════════════════════════════════════"
	./$(BUILD)/test_convergence
	@echo ""

# Rectangular matrix diagnostic test
$(BUILD)/test_rectangular_diag: tests/test_rectangular_diag.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
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
	install -m 644 $(INCLUDE)/apple_bottom.h $(PREFIX)/include/
	@sed -e 's|@PREFIX@|$(PREFIX)|g' \
	     -e 's|@VERSION@|1.3.0|g' \
	     applebottom.pc.in > $(PREFIX)/lib/pkgconfig/applebottom.pc
	@echo "Installed to $(PREFIX)"

uninstall:
	rm -f $(PREFIX)/lib/libapplebottom.a
	rm -f $(PREFIX)/include/apple_bottom.h
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
	$(CC) $(CFLAGS) -fsanitize=address -g -I$(INCLUDE) tests/test_correctness.c -o $(BUILD)/test_correctness -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Running tests with ASan..."
	ASAN_OPTIONS=detect_leaks=1 ./$(BUILD)/test_correctness

test-ubsan: clean
	@echo "Building with UndefinedBehaviorSanitizer..."
	$(OBJC) $(OBJCFLAGS) -fsanitize=undefined -g -I$(INCLUDE) -xobjective-c++ -c $(SRC)/apple_bottom.m -o $(BUILD)/apple_bottom.o
	$(CC) $(CFLAGS) -fsanitize=undefined -g -I$(INCLUDE) -c $(SRC)/blas_wrapper.c -o $(BUILD)/blas_wrapper.o
	$(CC) $(CFLAGS) -fsanitize=undefined -g -I$(INCLUDE) -c $(SRC)/fortran_bridge.c -o $(BUILD)/fortran_bridge.o
	ar rcs $(BUILD)/libapplebottom.a $(BUILD)/apple_bottom.o $(BUILD)/blas_wrapper.o $(BUILD)/fortran_bridge.o
	$(CC) $(CFLAGS) -fsanitize=undefined -g -I$(INCLUDE) tests/test_correctness.c -o $(BUILD)/test_correctness -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Running tests with UBSan..."
	./$(BUILD)/test_correctness

$(BUILD)/bench_dsyrk: benchmarks/bench_dsyrk.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

$(BUILD)/bench_zherk: benchmarks/bench_zherk.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

$(BUILD)/bench_pool: benchmarks/bench_pool.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

$(BUILD)/bench_async: benchmarks/bench_async.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

# =============================================================================
# CI Targets
# =============================================================================

.PHONY: build-check ci-local

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
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c $(SRC)/fortran_bridge.c -o $(BUILD)/fortran_bridge.o
	@echo "Compiling test_correctness.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/test_correctness.c -o $(BUILD)/test_correctness.o
	@echo "Compiling test_precision.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/test_precision.c -o $(BUILD)/test_precision.o
	@echo "Compiling test_convergence.c..."
	@$(CC) $(CFLAGS) -I$(INCLUDE) -c tests/verification/test_convergence.c -o $(BUILD)/test_convergence.o
	@echo ""
	@echo "✓ All source files compile successfully"
	@echo ""

# Local CI: run full test suite + convergence study, generate CI report
# Only writes CI_REPORT.txt if all tests pass (exits on first failure)
ci-local: $(BUILD)/test_precision $(BUILD)/test_correctness $(BUILD)/test_convergence
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "CI Local Validation"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Running test suite..."
	@./$(BUILD)/test_precision > $(BUILD)/test_precision.log 2>&1 && echo "✓ test_precision: PASS" || (echo "✗ test_precision: FAIL" && cat $(BUILD)/test_precision.log && exit 1)
	@./$(BUILD)/test_correctness > $(BUILD)/test_correctness.log 2>&1 && echo "✓ test_correctness: PASS" || (echo "✗ test_correctness: FAIL" && cat $(BUILD)/test_correctness.log && exit 1)
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
