#!/bin/bash
# M1/M2 acceptance: build every tests/cuda/*.cu with the given CUDA compiler and run it.
#   tests/run_cuda_tests.sh [compiler]      (default: b70cc; works with nvcc too)
# Prints one PASS/FAIL line per check plus compile failures, then a summary.
set -u
CC=${1:-b70cc}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${B70_TEST_OUT:-$(mktemp -d)}
pass=0; fail=0; cfail=0

build_and_run() {  # name source [extra flags...]
    local name=$1 src=$2; shift 2
    if ! "$CC" -O3 -std=c++17 -I"$HERE/cuda" "$@" "$src" -o "$OUT/$name" >"$OUT/$name.log" 2>&1; then
        echo "COMPILE-FAIL $name"
        sed -n '1,12p' "$OUT/$name.log" | sed 's/^/    /'
        cfail=$((cfail + 1))
        return
    fi
    local res
    res=$(timeout 300 "$OUT/$name" 2>&1)
    echo "$res" | grep -E '^(PASS|FAIL|device)' || { echo "FAIL $name (no result, exit $?)"; echo "$res" | tail -5 | sed 's/^/    /'; }
    pass=$((pass + $(echo "$res" | grep -c '^PASS')))
    fail=$((fail + $(echo "$res" | grep -c '^FAIL')))
}

for f in "$HERE"/cuda/*.cu; do
    n=$(basename "$f" .cu)
    build_and_run "$n" "$f"
done
build_and_run reduction_sync "$HERE/cuda/reduction.cu" -DUSE_SYNC
build_and_run fast_intrinsics_use_fast_math "$HERE/cuda/fast_intrinsics.cu" --use_fast_math

echo
echo "summary: $pass passed, $fail failed, $cfail did not compile   (binaries and logs in $OUT)"
[ "$fail" -eq 0 ] && [ "$cfail" -eq 0 ]
