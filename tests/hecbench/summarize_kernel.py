"""Summarize kernel_times.sh logs: the kernel time each HeCBench program prints itself.

    python3 summarize_kernel.py <log-dir>

For every <name>-cuda.run.log / <name>-sycl.run.log pair, every line that mentions "time"
and carries a number with a unit (us, ms, s ...) is parsed and the times are summed, so a
program that reports several kernels (or float32 and float64 passes) is compared as a whole.
Lines reporting rates (GB/s, GFLOPS ...) are ignored.  Prints one row per benchmark and the
geometric mean of SYCL / CUDA kernel time (> 1 means the b70cc build is faster).
"""
import math
import os
import re
import sys

NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
UNITS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "usec": 1e-6, "ms": 1e-3, "msec": 1e-3,
         "s": 1.0, "sec": 1.0, "secs": 1.0, "seconds": 1.0, "second": 1.0,
         "milliseconds": 1e-3, "microseconds": 1e-6, "nanoseconds": 1e-9}
PAT = re.compile(NUM + r"\s*\(?\s*(" + "|".join(sorted(UNITS, key=len, reverse=True)) + r")\b\)?",
                 re.IGNORECASE)
RATE = re.compile(r"/s\b|GB|GFLOP|TFLOP|bandwidth|throughput|rate", re.IGNORECASE)


def times_in(path):
    """[(seconds, line)] for every time line in the log."""
    out = []
    for line in open(path, errors="replace"):
        if "time" not in line.lower() or RATE.search(line):
            continue
        m = PAT.search(line)
        if not m:
            continue
        val, unit = float(m.group(1)), m.group(2).lower()
        if val <= 0:
            continue
        out.append((val * UNITS[unit], line.strip()))
    return out


def main(d):
    names = sorted(f[:-len("-cuda.run.log")] for f in os.listdir(d) if f.endswith("-cuda.run.log"))
    rows, skipped = [], []
    for n in names:
        cu, sy = os.path.join(d, n + "-cuda.run.log"), os.path.join(d, n + "-sycl.run.log")
        if not os.path.exists(sy):
            skipped.append((n, "no SYCL log"))
            continue
        tc, ts = times_in(cu), times_in(sy)
        if not tc or not ts:
            skipped.append((n, "no time line (cuda %d, sycl %d)" % (len(tc), len(ts))))
            continue
        note = "" if len(tc) == len(ts) else "  [line count differs: cuda %d, sycl %d]" % (len(tc), len(ts))
        c, s = sum(t for t, _ in tc), sum(t for t, _ in ts)
        rows.append((s / c, n, c, s, note))
    rows.sort()
    print("%-22s %12s %12s %8s" % ("benchmark", "CUDA(b70cc)", "SYCL(icpx)", "SYCL/CUDA"))
    for r, n, c, s, note in rows:
        print("%-22s %12s %12s %8.2f%s" % (n, fmt(c), fmt(s), r, note))
    if rows:
        geo = math.exp(sum(math.log(r) for r, *_ in rows) / len(rows))
        print("\n%d benchmarks; geomean SYCL/CUDA kernel time = %.2f (b70cc build %s)"
              % (len(rows), geo, "faster" if geo > 1 else "slower"))
        print("slowest CUDA relative to SYCL: " + ", ".join("%s %.2f" % (n, r) for r, n, *_ in rows[:5]))
        print("fastest CUDA relative to SYCL: " + ", ".join("%s %.2f" % (n, r) for r, n, *_ in rows[-5:]))
    for n, why in skipped:
        print("skipped %s: %s" % (n, why))


def fmt(sec):
    return "%.3f ms" % (sec * 1e3) if sec < 1 else "%.3f s" % sec


if __name__ == "__main__":
    main(sys.argv[1])
