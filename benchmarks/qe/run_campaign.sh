#!/bin/bash
# Run Quantum ESPRESSO pw.x in several AB_MODEs, repeatedly, and collect
# energies, timings and the apple-bottom AB_REPORT of every run.
#
# Usage: PW_X=/path/to/pw.x [PSEUDO_DIR=dir] benchmarks/qe/run_campaign.sh <plan> <out-dir>
#
# pw.x must load libapplebottom (see docs/INTEGRATION.md). Each line of the
# plan file is
#     <label> <input-stem> <np> <omp-threads> <repetitions> <modes,comma-separated>
# e.g. "1.4.0 si64_160b_k222 1 8 2 cpu,auto,gpu"; the label is free text
# (record the library build in it), the input is inputs/<stem>.in. The mode
# order rotates with each repetition. Every run gets a fresh directory,
# <out-dir>/runs/<label>/<stem>/<mode>_r<rep>/, and a line in
# <out-dir>/results.csv. Run on AC power with nothing else on the GPU, e.g.
# under `caffeinate -is`. Summarize with analyze.py and compare.py.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PLAN="$1"; OUT="$(mkdir -p "$2" && cd "$2" && pwd)"
: "${PW_X:?set PW_X to a pw.x that loads libapplebottom}"
PSEUDO_DIR="${PSEUDO_DIR:-$HERE/pseudo}"
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo "label,input,np,omp,mode,rep,start,end,rc,power,load1,energy_Ry,scf_iters,wall_s,cpu_s,cegterg_wall_s,regterg_wall_s,h_psi_wall_s,calbec_wall_s,paw_pot_wall_s,newd_wall_s,addusdens_wall_s,build" > "$CSV"

to_s() { # QE time string (1h02m, 10m15.54s, 95.65s) -> seconds
  python3 -I -c '
import re, sys
s = sys.argv[1].strip()
if not s: print(""); sys.exit()
print(f"{sum(float(v) * {"h": 3600, "m": 60, "s": 1}[u] for v, u in re.findall(r"([0-9.]+)([hms])", s)):.2f}")' "$1"
}
timer() { # <out> <name> -> WALL seconds of that QE timer
  local line; line=$(grep -E "^ +$2 +:" "$1" | head -1)
  [ -z "$line" ] && { echo ""; return; }
  to_s "$(echo "$line" | sed -E 's/.*CPU +([0-9hms.]+) WALL.*/\1/')"
}

while read -r label stem np omp reps modes; do
  [ -z "${label:-}" ] && continue; case "$label" in \#*) continue;; esac
  IFS=, read -r -a M <<< "$modes"
  for ((r = 1; r <= reps; r++)); do
    order=(); for ((i = 0; i < ${#M[@]}; i++)); do order+=("${M[$(( (i + r - 1) % ${#M[@]} ))]}"); done
    for mode in "${order[@]}"; do
      d="$OUT/runs/$label/$stem/${mode}_r$r"; k=1
      while [ -e "$d" ]; do d="$OUT/runs/$label/$stem/${mode}_r${r}_$k"; k=$((k + 1)); done
      mkdir -p "$d/tmp"
      sed -e "s#OUTDIR#$d/tmp#" -e "s#PSEUDO_DIR#$PSEUDO_DIR#" "$HERE/inputs/$stem.in" > "$d/input.in"
      power=$(pmset -g batt | head -1 | sed -E "s/.*'(.*)'.*/\1/")
      load1=$(sysctl -n vm.loadavg | awk '{print $2}')
      start=$(date +%s)
      echo "== $(date +%T) $label $stem np=$np omp=$omp mode=$mode rep=$r (power=$power load=$load1)"
      ( cd "$d" && AB_MODE=$mode AB_REPORT=1 OMP_NUM_THREADS=$omp \
        mpirun -np "$np" -x OMP_NUM_THREADS -x AB_MODE -x AB_REPORT \
        "$PW_X" -in input.in < /dev/null > out.txt 2> err.txt )
      rc=$?; end=$(date +%s); o="$d/out.txt"
      energy=$(grep -E '^!' "$o" | tail -1 | awk '{print $(NF-1)}')
      iters=$(grep -E 'convergence has been achieved in' "$o" | awk '{print $(NF-1)}')
      pw=$(grep -E 'PWSCF +:' "$o" | tail -1)
      wall=$(echo "$pw" | sed -E 's/.*CPU +(.*) WALL.*/\1/'); cpu=$(echo "$pw" | sed -E 's/.*: +(.*) CPU.*/\1/')
      build=$(sed -nE 's/^AB_REPORT\[[0-9]+\] apple-bottom [^ ]+ \(([^)]*)\).*/\1/p' "$d/err.txt" | head -1)
      echo "$label,$stem,$np,$omp,$mode,$r,$start,$end,$rc,$power,$load1,$energy,$iters,$(to_s "$wall"),$(to_s "$cpu"),$(timer "$o" cegterg),$(timer "$o" regterg),$(timer "$o" h_psi),$(timer "$o" calbec),$(timer "$o" PAW_pot),$(timer "$o" newd),$(timer "$o" addusdens),$build" >> "$CSV"
      echo "   rc=$rc E=$energy iters=$iters wall=$(to_s "$wall")s build=$build"
    done
  done
done < "$PLAN"
echo "== done $(date +%T)"
