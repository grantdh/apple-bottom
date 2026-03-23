# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Scope

apple-bottom is a numerical computing library. Security considerations include:

- **Memory safety**: Buffer overflows, use-after-free, double-free
- **Integer overflows**: In matrix dimension calculations
- **Resource exhaustion**: GPU memory allocation failures
- **Data integrity**: Numerical precision guarantees

This library does NOT handle:
- Network communication
- User authentication
- File system access (beyond loading Metal shaders)
- Cryptographic operations

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **DO NOT** open a public issue
2. Email: grantdheileman@gmail.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix timeline**: Depends on severity
  - Critical: 24-48 hours
  - High: 1 week
  - Medium: 2 weeks
  - Low: Next release

## Security Best Practices for Users

### Input Validation

Always validate matrix dimensions before calling apple-bottom:

```c
// Good
if (rows > 0 && cols > 0 && rows <= MAX_DIM && cols <= MAX_DIM) {
    ABMatrix m = ab_matrix_create(rows, cols);
}

// Bad - no validation
ABMatrix m = ab_matrix_create(user_input_rows, user_input_cols);
```

### Memory Limits

Check available GPU memory before large allocations:

```c
// Each NxN matrix uses 8*N*N bytes (DD format)
// 8192x8192 = 512MB per matrix
size_t required = 8 * rows * cols;
// Ensure this fits in available unified memory
```

### Error Handling

Always check return values:

```c
ABStatus status = ab_dgemm(A, B, C);
if (status != AB_OK) {
    // Handle error appropriately
    fprintf(stderr, "DGEMM failed: %s\n", ab_status_string(status));
}
```

## Known Limitations

1. **Single-threaded initialization**: `ab_init()` should only be called from one thread
2. **No concurrent GPU access**: Operations are serialized per command queue
3. **Unified memory**: Large matrices compete with system RAM

## Acknowledgments

We thank security researchers who responsibly disclose vulnerabilities.
