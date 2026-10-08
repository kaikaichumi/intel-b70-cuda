"""Summarize a run_hecbench.sh results.csv into the M2 / M3 numbers.

    python3 summarize.py results.csv

M2: how many CUDA versions built, ran, and self-verified (PASS).
M3: for benchmarks where both the CUDA (b70cc) and the SYCL (icpx) versions
    self-verified, the geometric mean of sycl_seconds / cuda_seconds
    (> 1 means the CUDA build is faster). Times are wall-clock of `make run`.
"""
import csv
import math
import sys


def main(path):
    rows = list(csv.DictReader(open(path)))
    n = len(rows)
    built = [r for r in rows if r["cuda_build"] == "ok"]
    ran = [r for r in built if r["cuda_run"] == "ok"]
    verified = [r for r in ran if r["cuda_verified"] == "yes"]
    no_check = [r for r in ran if r["cuda_verified"] == "no"]
    print("M2  CUDA via b70cc over %d benchmarks" % n)
    print("    built %d (%.0f%%), ran %d (%.0f%%), self-verified PASS %d (%.0f%%), ran without a PASS line %d"
          % (len(built), 100 * len(built) / n, len(ran), 100 * len(ran) / n,
             len(verified), 100 * len(verified) / n, len(no_check)))
    for label, sel in (("did not build", [r for r in rows if r["cuda_build"] != "ok"]),
                       ("built but did not finish", [r for r in built if r["cuda_run"] != "ok"]),
                       ("reported FAIL", [r for r in ran if r["cuda_verified"] == "FAIL"])):
        if sel:
            print("    %-26s %s" % (label + ":", ", ".join(
                r["name"] + ("(%s)" % r["cuda_run"] if label.startswith("built") else "") for r in sel)))

    both = [r for r in rows if r["cuda_verified"] == "yes" and r["sycl_verified"] == "yes"
            and r["cuda_seconds"] and r["sycl_seconds"]]
    if not both:
        print("M3  no benchmark verified on both sides")
        return
    ratios = sorted(((float(r["sycl_seconds"]) / float(r["cuda_seconds"]), r["name"]) for r in both))
    geo = math.exp(sum(math.log(x) for x, _ in ratios) / len(ratios))
    print("M3  %d benchmarks verified on both sides; geomean SYCL/CUDA time = %.2f (CUDA build %s)"
          % (len(ratios), geo, "faster" if geo > 1 else "slower"))
    print("    slowest CUDA relative to SYCL: " + ", ".join("%s %.2f" % (nm, x) for x, nm in ratios[:6]))
    print("    fastest CUDA relative to SYCL: " + ", ".join("%s %.2f" % (nm, x) for x, nm in ratios[-6:]))


if __name__ == "__main__":
    main(sys.argv[1])
