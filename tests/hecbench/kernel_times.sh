#!/bin/bash
# M3, second method: compare the *kernel time each HeCBench program reports itself*
# ("Average kernel execution time: ... (us)" and the like) instead of the wall-clock of
# `make run`.  The wall-clock includes the CPU reference computation, which icpx compiles
# with -fp-model=fast and a vectorizer that handles struct loads, so it says nothing about
# the GPU side (see docs/03-roadmap.md).  Run inside the toolchain environment:
#   b70cuda run tests/hecbench/kernel_times.sh [list-file] [hecbench-dir]
# Env: REBUILD=cuda|sycl|both|none (default cuda: the CUDA side is rebuilt with the current
#      toolchain, the SYCL binaries from run_hecbench.sh are reused), TIMEOUT (s, default 900).
# Each program is run twice and the second log is kept, so a cold JIT compile never counts.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LIST=${1:-$HERE/list-verified.txt}
HB=${2:-${B70_DATA:-$HOME}/hecbench}
OUT=${B70_HECBENCH_OUT:-$HB/kernel-times-$(date +%Y%m%d-%H%M)}
JOBS=${JOBS:-$(nproc)}
TIMEOUT=${TIMEOUT:-900}
REBUILD=${REBUILD:-cuda}
mkdir -p "$OUT"

CUDA_ARGS=(CC=b70cc ARCH=sm_80)
SYCL_ARGS=(CC=icpx GCC_TOOLCHAIN= CUDA=no HIP=no GPU=yes)

build() {  # dir log make-args...
    local dir=$1 log=$2; shift 2
    make -C "$dir" -B -j"$JOBS" "$@" >"$log" 2>&1
}
run_twice() {  # dir log make-args...  (second run's log is kept)
    local dir=$1 log=$2; shift 2
    timeout "$TIMEOUT" make -C "$dir" "$@" run >"$log.warm" 2>&1 || { mv "$log.warm" "$log"; return 1; }
    timeout "$TIMEOUT" make -C "$dir" "$@" run >"$log" 2>&1
}

while read -r name; do
    case "$name" in ''|'#'*) continue ;; esac
    cu=$HB/src/$name-cuda sy=$HB/src/$name-sycl
    [ -d "$cu" ] && [ -d "$sy" ] || { echo "skip $name"; continue; }
    status=""
    case "$REBUILD" in cuda|both) build "$cu" "$OUT/$name-cuda.build.log" "${CUDA_ARGS[@]}" || status="cuda-build-fail" ;; esac
    case "$REBUILD" in sycl|both) build "$sy" "$OUT/$name-sycl.build.log" "${SYCL_ARGS[@]}" || status="$status sycl-build-fail" ;; esac
    [ -z "$status" ] && { run_twice "$cu" "$OUT/$name-cuda.run.log" "${CUDA_ARGS[@]}" || status="cuda-run-fail"; }
    [ -z "$status" ] && { run_twice "$sy" "$OUT/$name-sycl.run.log" "${SYCL_ARGS[@]}" || status="sycl-run-fail"; }
    echo "$name: ${status:-ok}"
done < "$LIST"

echo
python3 "$HERE/summarize_kernel.py" "$OUT" | tee "$OUT/summary.txt"
echo "logs: $OUT"
