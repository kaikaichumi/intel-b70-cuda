#!/bin/bash
# M2 / M3: build and run HeCBench benchmarks, CUDA version with b70cc and SYCL
# version with icpx, on the same B70. Run inside the toolchain environment:
#   b70cuda run tests/hecbench/run_hecbench.sh [list-file] [hecbench-dir]
# Writes results.csv next to the logs:
#   name, cuda_build, cuda_run, cuda_verified, cuda_seconds, sycl_build, sycl_run, sycl_verified, sycl_seconds
# "verified" means the program printed PASS (most HeCBench programs self-check).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LIST=${1:-$HERE/list.txt}
HB=${2:-${B70_DATA:-$HOME}/hecbench}
OUT=${B70_HECBENCH_OUT:-$HB/results-$(date +%Y%m%d-%H%M)}
JOBS=${JOBS:-$(nproc)}
TIMEOUT=${TIMEOUT:-600}

[ -d "$HB/src" ] || git clone --depth 1 https://github.com/zjin-lcf/HeCBench "$HB"
mkdir -p "$OUT"
echo "name,cuda_build,cuda_run,cuda_verified,cuda_seconds,sycl_build,sycl_run,sycl_verified,sycl_seconds" > "$OUT/results.csv"

run_one() {  # dir make-args... -> prints "build,run,verified,seconds"
    local dir=$1; shift
    local log=$OUT/$(basename "$dir")
    if ! make -C "$dir" -B -j"$JOBS" "$@" >"$log.build.log" 2>&1; then
        echo "fail,skip,no,"; return
    fi
    local t0 t1 rc
    t0=$(date +%s.%N)
    timeout "$TIMEOUT" make -C "$dir" "$@" run >"$log.run.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    local secs verified=no
    secs=$(awk "BEGIN { printf \"%.2f\", $t1 - $t0 }")
    grep -qE '^ *PASS|PASSED|Result = PASS|: PASS' "$log.run.log" && verified=yes
    grep -qE 'FAIL' "$log.run.log" && verified=FAIL
    if [ $rc -eq 0 ]; then echo "ok,ok,$verified,$secs"
    elif [ $rc -eq 124 ]; then echo "ok,timeout,no,"
    else echo "ok,crash($rc),no,"; fi
}

while read -r name; do
    case "$name" in ''|'#'*) continue ;; esac
    cu=$HB/src/$name-cuda sy=$HB/src/$name-sycl
    [ -d "$cu" ] || { echo "skip $name (no CUDA version)"; continue; }
    c=$(run_one "$cu" CC=b70cc ARCH=sm_80)
    s="skip,skip,no,"
    [ -d "$sy" ] && s=$(run_one "$sy" CC=icpx GCC_TOOLCHAIN= CUDA=no HIP=no GPU=yes)
    echo "$name,$c,$s" | tee -a "$OUT/results.csv"
done < "$LIST"

echo
awk -F, 'NR > 1 { n++; if ($2 == "ok") b++; if ($3 == "ok") r++; if ($4 == "yes") v++ }
         END { printf "CUDA via b70cc: %d benchmarks, %d built, %d ran, %d self-verified PASS\n", n, b, r, v }' "$OUT/results.csv"
echo "results: $OUT/results.csv"
