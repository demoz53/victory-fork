#!/usr/bin/env bash
# shader-sweep.sh: which translated programs does a translation switch (or another build) change, and do the changed
# ones still validate and make pipelines on this card? An offline check: no game start, no game data written
# anywhere but the output folder you name (digest lists: labels and hashes, no shader code).
#
#   tools/shader-sweep.sh <victory exe> "<ENV of side A>" "<ENV of side B>" <folder or container file>...
#   EXE_B=<other build's exe> tools/shader-sweep.sh <exe A> "" "" <paths>...     (two builds, the same switches)
#
#   JOBS=4      worker threads per walk (default 4: a game test usually shares this PC)
#   LOW=0       do not lower the walks' priority (default: idle priority)
#   REUSE=1     keep a side whose list is already complete in OUT (a base build's side A serves many sweeps)
#   OUT=<dir>   where a.txt, b.txt and changed.txt go (default: a new folder under $TMPDIR or /tmp)
#   CHECK=1     also run side B's `translate --validate --pipelines --list` over the changed containers only and
#               print every program that was refused, failed the validator or made no pipeline (uses the GPU briefly)
#
# Give BOTH sides the switches the runtime sets by itself on the card in question. On a card whose waves have 32
# lanes that is VICTORY_HOST_MAX_WAVE=32: without it the walk translates the 64-lane form, which that card never
# runs and refuses as a pipeline ("The parameter is incorrect").
#
# Example (a folder of containers saved with VICTORY_SHADER_SAVE):
#   tools/shader-sweep.sh target-x/diag/victory.exe "VICTORY_HOST_MAX_WAVE=32 VICTORY_LOOP_EXITS=0" "VICTORY_HOST_MAX_WAVE=32 VICTORY_LOOP_EXITS=1" <folder>
# Exit code: 0 = nothing changed, 1 = programs changed, 2 = a walk failed, 3 = CHECK found a changed program that
# no longer validates or makes a pipeline.
#
# The exe is copied to victory-sweep.exe beside itself and run under that name: the test driver ends every game run
# by stopping every process called victory.exe, which would end a sweep in mid-walk.
set -uo pipefail
[ $# -ge 4 ] || { sed -n '2,23p' "$0"; exit 2; }
exe_a=$1; env_a=$2; env_b=$3; shift 3
exe_b=${EXE_B:-$exe_a}
JOBS=${JOBS:-4}
OUT=${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/shader-sweep.XXXXXX")}
mkdir -p "$OUT"
tool() { local t; t="$(dirname "$1")/victory-sweep.exe"; cp -f "$1" "$t" && echo "$t"; }
walk() { # <exe> <env words> <out file> <translate options and paths>...
  local exe=$1 envs=$2 out=$3; shift 3
  # shellcheck disable=SC2086
  env $envs "$exe" translate --jobs "$JOBS" "$@" > "$out" 2> "$out.err" &
  local pid=$!
  # Idle priority on Windows (LOW=0 leaves it alone): a game test may be running beside the walk. By name: the
  # process id bash knows is not the Windows one.
  if [ "${LOW:-1}" != 0 ] && command -v powershell >/dev/null 2>&1; then
    powershell -NoProfile -Command "Get-Process victory-sweep -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'Idle' }" 2>/dev/null
  fi
  wait "$pid"
}
ta=$(tool "$exe_a") || exit 2
tb=$ta; [ "$exe_b" != "$exe_a" ] && { cp -f "$exe_b" "$(dirname "$exe_b")/victory-sweep.exe" && tb="$(dirname "$exe_b")/victory-sweep.exe" || exit 2; }
s=$SECONDS
have() { [ -n "${REUSE:-}" ] && grep -q '^programs ' "$1" 2>/dev/null && echo "reusing $1"; }
have "$OUT/a.txt" || walk "$ta" "$env_a" "$OUT/a.txt" --digest "$@" || { echo "side A's walk failed: $OUT/a.txt.err"; exit 2; }
have "$OUT/b.txt" || walk "$tb" "$env_b" "$OUT/b.txt" --digest "$@" || { echo "side B's walk failed: $OUT/b.txt.err"; exit 2; }
# A digest line: "<label>  emitted <n> bytes <hash>". Everything else (totals, refusals) is compared as text below.
awk -F'  emitted ' 'NF == 2 && FNR == NR { a[$1] = $2; next }
     NF == 2 { n++; if (!($1 in a)) { only_b++; print "only in B: " $1 } else { if (a[$1] != $2) { ch++; print "changed: " $1 } delete a[$1] } }
     END { for (k in a) { only_a++; print "only in A: " k }
           printf "summary: %d programs emitted in B, %d changed, %d only in A, %d only in B\n", n, ch, only_a, only_b }' \
  "$OUT/a.txt" "$OUT/b.txt" | sort > "$OUT/changed.txt"
grep -v '^summary: ' "$OUT/changed.txt"
echo "A: ${env_a:-(no switches)} $exe_a"
echo "B: ${env_b:-(no switches)} $exe_b"
grep '^summary: ' "$OUT/changed.txt"
# Programs one side refuses or fails on and the other does not (the walks' other lines; times left out).
if ! diff <(grep -v '  emitted ' "$OUT/a.txt" | grep -vE '^(containers|programs|structured|DXIL| )' ) \
          <(grep -v '  emitted ' "$OUT/b.txt" | grep -vE '^(containers|programs|structured|DXIL| )' ) > "$OUT/outcomes.diff"; then
  echo "outcomes differ (< A, > B):"; cat "$OUT/outcomes.diff"
fi
echo "two walks in $((SECONDS - s)) s; lists in $OUT"
changed=$(grep -c '^changed: \|^only in ' "$OUT/changed.txt")
[ "$changed" = 0 ] && exit 0
if [ -n "${CHECK:-}" ]; then
  # The changed programs' container files, each once.
  mapfile -t files < <(grep '^changed: \|^only in B: ' "$OUT/changed.txt" | sed -E 's/^(changed|only in B): //; s/(\.cdxil).*/\1/' | sort -u)
  if [ ${#files[@]} -gt 0 ]; then
    s=$SECONDS
    walk "$tb" "$env_b" "$OUT/check.txt" --validate --pipelines --list "${files[@]}" || { echo "the check walk failed: $OUT/check.txt.err"; exit 2; }
    grep -E '^(DXIL| {5,})' "$OUT/check.txt"
    bad=$(grep -cE '  (pipeline: |validator: )' "$OUT/check.txt")
    grep -E '  (pipeline: |validator: )' "$OUT/check.txt"
    echo "check of ${#files[@]} changed containers on side B in $((SECONDS - s)) s: $bad failures"
    [ "$bad" != 0 ] && exit 3
  fi
fi
exit 1
