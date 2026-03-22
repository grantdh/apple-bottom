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

.PHONY: all lib test bench clean examples

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

$(BUILD)/libapplebottom.a: $(BUILD)/apple_bottom.o
	ar rcs $@ $<
	@echo "Built: $@"

lib: $(BUILD)/libapplebottom.a

# =============================================================================
# Examples
# =============================================================================

$(BUILD)/examples/basic_dgemm: $(EXAMPLES)/01_basic_dgemm/main.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

$(BUILD)/examples/cg_solver: $(EXAMPLES)/02_cg_solver/main.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

$(BUILD)/examples/zgemm: $(EXAMPLES)/03_zgemm/main.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

$(BUILD)/examples/eigenvalue_solver: $(EXAMPLES)/04_eigenvalue_solver/main.c lib | $(BUILD)
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

bench: $(BUILD)/bench_dgemm $(BUILD)/bench_zgemm
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

# =============================================================================
# Tests
# =============================================================================

$(BUILD)/test_precision: tests/test_precision.c lib | $(BUILD)
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@ -L$(BUILD) -lapplebottom $(LDFLAGS)
	@echo "Built: $@"

$(BUILD)/test_correctness: tests/test_correctness.c lib | $(BUILD)
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
	install -m 644 $(BUILD)/libapplebottom.a $(PREFIX)/lib/
	install -m 644 $(INCLUDE)/apple_bottom.h $(PREFIX)/include/
	@echo "Installed to $(PREFIX)"
