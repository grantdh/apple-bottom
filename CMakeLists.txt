cmake_minimum_required(VERSION 3.16)
project(apple_bottom VERSION 1.3.1 LANGUAGES C OBJC)

# Require Apple Silicon
if(NOT APPLE)
    message(FATAL_ERROR "apple-bottom only works on macOS")
endif()

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

# Compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_OBJC_STANDARD 11)
add_compile_options(-Wall -Wextra -O3)
add_compile_definitions(ACCELERATE_NEW_LAPACK)

# Find required frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# =============================================================================
# Library
# =============================================================================

add_library(applebottom STATIC
    src/apple_bottom.m
)

target_include_directories(applebottom PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(applebottom PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
)

# Enable ARC for Objective-C
set_source_files_properties(src/apple_bottom.m PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# =============================================================================
# Examples
# =============================================================================

if(BUILD_EXAMPLES)
    add_executable(basic_dgemm examples/01_basic_dgemm/main.c)
    target_link_libraries(basic_dgemm applebottom ${ACCELERATE_FRAMEWORK})
endif()

# =============================================================================
# Tests
# =============================================================================

if(BUILD_TESTS)
    enable_testing()
    
    add_executable(test_precision tests/test_precision.c)
    target_link_libraries(test_precision applebottom ${ACCELERATE_FRAMEWORK})
    add_test(NAME Precision COMMAND test_precision)
    
    add_executable(test_correctness tests/test_correctness.c)
    target_link_libraries(test_correctness applebottom ${ACCELERATE_FRAMEWORK})
    add_test(NAME Correctness COMMAND test_correctness)
endif()

# =============================================================================
# Benchmarks
# =============================================================================

if(BUILD_BENCHMARKS)
    add_executable(bench_dgemm benchmarks/bench_dgemm.c)
    target_link_libraries(bench_dgemm applebottom ${ACCELERATE_FRAMEWORK})
    
    add_executable(bench_zgemm benchmarks/bench_zgemm.c)
    target_link_libraries(bench_zgemm applebottom ${ACCELERATE_FRAMEWORK})
endif()

# =============================================================================
# Installation
# =============================================================================

include(GNUInstallDirs)

install(TARGETS applebottom
    EXPORT applebottom-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES include/apple_bottom.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT applebottom-targets
    FILE applebottom-config.cmake
    NAMESPACE applebottom::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/applebottom
)

# =============================================================================
# pkg-config
# =============================================================================

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/applebottom.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/applebottom.pc
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/applebottom.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "apple-bottom v${PROJECT_VERSION}")
message(STATUS "  Examples:    ${BUILD_EXAMPLES}")
message(STATUS "  Tests:       ${BUILD_TESTS}")
message(STATUS "  Benchmarks:  ${BUILD_BENCHMARKS}")
message(STATUS "")
