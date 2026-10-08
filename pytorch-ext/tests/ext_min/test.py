"""M4 acceptance ①: the minimal CUDA extension on XPU tensors, checked against CPU.

Runs under `b70 python` (so "cuda" maps to the B70) after
`python setup.py build_ext --inplace` in this directory.
"""
import os
import sys
import time

os.environ.setdefault("CHIP_BE", "level0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import b70_ext_min as ext  # noqa: E402

dev = "cuda"
fails = 0


def check(name, ok, detail=""):
    global fails
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail))
    fails += 0 if ok else 1


# 1. basic correctness, float and double
for dt in (torch.float32, torch.float64):
    x = torch.randn(1_000_003, device=dev, dtype=dt)
    y = ext.affine(x, 2.0, 1.0)
    ref = x.cpu() * 2.0 + 1.0
    check("affine_%s" % str(dt).split(".")[1], y.device.type == "xpu" and torch.allclose(y.cpu(), ref),
          "max err %.2e" % (y.cpu() - ref).abs().max().item())

# 2. warp shuffles + warpSize
x = torch.randn(4097, 1000, device=dev)
s = ext.row_sum(x)
check("row_sum_warp_shuffle", torch.allclose(s.cpu(), x.cpu().sum(1), atol=1e-3),
      "max err %.2e" % (s.cpu() - x.cpu().sum(1)).abs().max().item())

# 3. ordering: torch op -> extension -> torch op, many times, no explicit syncs
x = torch.randn(2_000_000, device=dev)
z = x
zc = x.cpu()
for i in range(100):
    z = ext.affine(z * 1.001, 1.0, 0.25) - 0.25   # torch kernel before and after each extension call
    zc = (zc * 1.001) * 1.0 + 0.25 - 0.25
check("ordering_torch_ext_torch_x100", torch.allclose(z.cpu(), zc, rtol=1e-4, atol=1e-4),
      "max err %.2e" % (z.cpu() - zc).abs().max().item())

# 4. ordering with torch's own reductions reading the extension's output right away
x = torch.randn(3_000_000, device=dev)
y = ext.affine(x, 3.0, -1.0)
check("ordering_reduction", abs(y.sum().item() - (x.cpu() * 3.0 - 1.0).sum().item()) < 1.0)

# 5. per-call overhead (small tensor)
x = torch.randn(1024, device=dev)
for _ in range(20):
    ext.affine(x, 1.0, 0.0)
torch.xpu.synchronize()
t = time.time()
n = 1000
for _ in range(n):
    y = ext.affine(x, 1.0, 0.0)
torch.xpu.synchronize()
us = (time.time() - t) * 1e6 / n
check("overhead_small_calls", us < 500, "%.0f us per extension call (1024 elements)" % us)

t = time.time()
for _ in range(n):
    y = x * 1.0
torch.xpu.synchronize()
print("      (plain torch op on the same tensor: %.0f us)" % ((time.time() - t) * 1e6 / n))

print("\n%d failed" % fails)
sys.exit(1 if fails else 0)
