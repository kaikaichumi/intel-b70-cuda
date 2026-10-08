"""M4 ③ prerequisite: a Triton kernel written against torch.cuda runs on the B70
through the b70 layer (triton-xpu is bundled with the torch XPU wheel).

    b70 python test_triton.py
"""
import sys
import time

import torch
import triton
import triton.language as tl

print("triton", triton.__version__, "torch", torch.__version__)


@triton.jit
def add_kernel(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)


@triton.jit
def softmax_kernel(out, inp, stride, ncols, B: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, B)
    m = cols < ncols
    x = tl.load(inp + row * stride + cols, mask=m, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(out + row * stride + cols, e / tl.sum(e, axis=0), mask=m)


fails = 0


def check(name, ok, detail=""):
    global fails
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    fails += 0 if ok else 1


n = 1 << 20
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
o = torch.empty_like(x)
add_kernel[(n // 1024,)](x, y, o, n, B=1024)
check("triton_add", torch.allclose(o, x + y), "device %s" % x.device)

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    a = torch.randn(4096, 1000, device="cuda", dtype=dtype)
    out = torch.empty_like(a)
    softmax_kernel[(a.shape[0],)](out, a, a.stride(0), a.shape[1], B=1024)
    ref = torch.softmax(a.float(), dim=1)
    err = (out.float() - ref).abs().max().item()
    check("triton_softmax_%s" % str(dtype).split(".")[1], err < 1e-2, "max err %.2e" % err)

torch.xpu.synchronize()
t = time.time()
for _ in range(200):
    add_kernel[(n // 1024,)](x, y, o, n, B=1024)
torch.xpu.synchronize()
us = (time.time() - t) * 1e6 / 200
t = time.time()
for _ in range(200):
    torch.add(x, y, out=o)
torch.xpu.synchronize()
us_t = (time.time() - t) * 1e6 / 200
print("1M-element add: triton %.0f us, torch %.0f us per call" % (us, us_t))

print("\n%d failed" % fails)
sys.exit(1 if fails else 0)
