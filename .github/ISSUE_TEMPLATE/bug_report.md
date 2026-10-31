---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Build with '...' (e.g., `make`, or the link line of your program)
2. Run with parameters '...' (including any `AB_*` environment variables)
3. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment:**
 - macOS version: [e.g., macOS 14.3]
 - Chip: [e.g., M2 Max, 38-core GPU]
 - Library build: [the `ab_build_info()` string, e.g. from the first `AB_REPORT` line]
 - Compiler: [run `clang --version`]
 - Xcode version: [if applicable]

**Code/Output**
```
Paste relevant code or error output here (with AB_REPORT=1 for BLAS-ABI calls)
```

**Performance data** (if applicable)
- Matrix dimensions:
- Observed GFLOP/s:
- Expected GFLOP/s:

**Additional context**
Add any other context about the problem here.
