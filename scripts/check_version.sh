#!/bin/bash
# Check that every place that states the library version agrees with
# include/apple_bottom.h. Run via `make version-check`.
set -u
cd "$(dirname "$0")/.."

hdr=include/apple_bottom.h
str=$(sed -nE 's/^#define APPLE_BOTTOM_VERSION_STRING "([^"]+)"/\1/p' "$hdr")
maj=$(sed -nE 's/^#define APPLE_BOTTOM_VERSION_MAJOR ([0-9]+)/\1/p' "$hdr")
min=$(sed -nE 's/^#define APPLE_BOTTOM_VERSION_MINOR ([0-9]+)/\1/p' "$hdr")
pat=$(sed -nE 's/^#define APPLE_BOTTOM_VERSION_PATCH ([0-9]+)/\1/p' "$hdr")
num="$maj.$min.$pat"

fail=0
check() { # <what> <found> <expected>
    if [ "$2" = "$3" ]; then
        printf '  ok    %-34s %s\n' "$1" "$2"
    else
        printf '  FAIL  %-34s %s (expected %s)\n' "$1" "${2:-<missing>}" "$3"
        fail=1
    fi
}

echo "Library version: $str"
check "$hdr MAJOR.MINOR.PATCH" "${str%%-*}" "$num"
check "python/apple_bottom/__init__.py" \
    "$(sed -nE 's/^__version__ = "([^"]+)"/\1/p' python/apple_bottom/__init__.py)" "$str"
check "python/setup.py" \
    "$(sed -nE 's/^ *version="([^"]+)",/\1/p' python/setup.py)" "$str"
check "CITATION.cff" "$(sed -nE 's/^version: *"?([^"]+)"?$/\1/p' CITATION.cff)" "$str"
check "README.md citation" \
    "$(sed -nE 's/^ *version *= *\{([^}]+)\}.*/\1/p' README.md)" "$str"
check "CHANGELOG.md top entry" \
    "$(sed -nE 's/^## \[([^]]+)\].*/\1/p' CHANGELOG.md | head -1)" "$str"

[ $fail -eq 0 ] && echo "version-check: PASS" || echo "version-check: FAIL"
exit $fail
