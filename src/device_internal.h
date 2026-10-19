// =============================================================================
// device_internal.h — the device-buffer handle (internal)
// =============================================================================
//
// Owned by device_api.m. blas2.m reads A from a handle in place and
// blas2_abi.c checks arguments against one. The functions are plain C; the
// struct and the wrapper are Objective-C++ only.
//
// A buffer from ab_dev_malloc() starts at byte 0 of its MTLBuffer. One from
// ab_dev_register() is a no-copy wrapper over the page-rounded span of the
// caller's memory, so the caller's first byte is `base` bytes into it.
// =============================================================================
#ifndef AB_DEVICE_INTERNAL_H
#define AB_DEVICE_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "apple_bottom_device.h"

#ifdef __cplusplus
extern "C" {
#endif

// CPU address of the buffer's first byte (the caller's, for a registered one).
void *ab_internal_dev_data(ab_dev_buffer_t buf);

// True if a column-major rows x cols matrix with leading dimension ld >= rows,
// starting `offset` elements of elem_size bytes into buf, lies inside it. An
// empty matrix (rows or cols 0) fits any buffer.
bool ab_internal_dev_matrix_fits(ab_dev_buffer_t buf, size_t offset, int ld, int rows, int cols,
                                 size_t elem_size);

#ifdef __cplusplus
}
#endif

#ifdef __OBJC__
#import <Metal/Metal.h>

struct ab_dev_buffer_s {
    id<MTLBuffer> mtl;
    size_t nbytes;   // bytes the caller sees
    size_t base;     // offset of the caller's first byte in mtl
};

// A no-copy MTLBuffer over the pages holding [ptr, ptr + nbytes), with
// *offset set to ptr's offset in it; nil if part of the span is unmapped or
// unreadable, or Metal refuses it. Every call makes a new wrapper: a wrapper
// kept past free() or munmap() of its memory reads the old pages, so none is
// ever cached by address.
id<MTLBuffer> ab_internal_wrap_host(id<MTLDevice> dev, const void *ptr, size_t nbytes,
                                    size_t *offset);
#endif

#endif // AB_DEVICE_INTERNAL_H
