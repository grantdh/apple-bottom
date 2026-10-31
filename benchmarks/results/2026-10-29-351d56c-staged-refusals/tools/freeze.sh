#!/bin/bash
# freeze.sh <worktree> <label>: copies <worktree>/build/libapplebottom.dylib to
# <ab-frozen>/<sha256[:12]>/ with a copy of the release check's pw.x.ab150
# (QE develop 86c214974) whose first LC_RPATH (the 1.5.0 frozen library's directory)
# is rewritten to the new directory: the same machine code, only the library it loads
# changes. Prints the directory.
set -euo pipefail
WT=$1; LABEL=$2
PW150=<qe>/bin-variants/pw.x.ab150
OLD=<ab-frozen>/755a46613bf2
L=$WT/build/libapplebottom.dylib
H=$(shasum -a 256 "$L" | cut -c1-12); F=<ab-frozen>/$H
if [ ! -f "$F/SHA256SUMS" ]; then
  mkdir -p "$F"; cp "$L" "$F/"; cp "$PW150" "$F/pw.x"; chmod u+w "$F/pw.x"
  install_name_tool -rpath "$OLD" "$F" "$F/pw.x" 2>/dev/null
  codesign -f -s - "$F/pw.x" 2>/dev/null
  (cd "$F" && shasum -a 256 libapplebottom.dylib pw.x > SHA256SUMS)
  { echo "frozen: $(date -u +%FT%TZ) ($LABEL)"
    echo "apple-bottom: $(git -C "$WT" rev-parse HEAD) ($(git -C "$WT" status --porcelain --untracked-files=no | wc -l | tr -d ' ') modified tracked files), make lib in $WT under machlock"
    echo "pw.x: copy of $PW150 (QE develop 86c214974, -O3, see its PROVENANCE.txt); install_name_tool -rpath $OLD -> $F, ad-hoc re-signed; no other change."
    echo "Run from a scratch directory, binary by full path. AB_MODE, AB_REPORT etc. as usual."; } > "$F/PROVENANCE.txt"
fi
otool -l "$F/pw.x" | grep -A2 LC_RPATH | sed -n 3p | grep -q "$F" || { echo "rpath not rewritten in $F/pw.x" >&2; exit 1; }
echo "$F"
