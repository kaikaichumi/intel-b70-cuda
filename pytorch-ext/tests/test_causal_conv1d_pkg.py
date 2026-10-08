"""M4 ②: the real causal-conv1d package (Dao-AILab), built from its unmodified
source through the b70torch hook, checked against the package's own reference
implementation. Forward and backward, float32 / float16 / bfloat16.

    cd <dir with .b70venv where causal-conv1d was installed>
    b70 python /path/to/test_causal_conv1d_pkg.py
"""
import itertools
import sys
import time

import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref

dev = "cuda"
fails = 0


def check(name, ok, detail=""):
    global fails
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    fails += 0 if ok else 1


tol = {torch.float32: (1e-4, 1e-4), torch.float16: (1e-2, 1e-2), torch.bfloat16: (3e-2, 3e-2)}
for itype, width, act, channel_last in itertools.product(
        (torch.float32, torch.float16, torch.bfloat16), (2, 4), (None, "silu"), (False, True)):
    batch, dim, seqlen = 2, 96, 1024
    torch.manual_seed(0)
    if channel_last:
        x = torch.randn(batch, seqlen, dim, device=dev, dtype=itype).transpose(1, 2).requires_grad_()
    else:
        x = torch.randn(batch, dim, seqlen, device=dev, dtype=itype).requires_grad_()
    w = torch.randn(dim, width, device=dev, dtype=torch.float32, requires_grad=True)
    b = torch.randn(dim, device=dev, dtype=torch.float32, requires_grad=True)
    x_ref, w_ref, b_ref = [t.detach().clone().requires_grad_() for t in (x, w, b)]

    out = causal_conv1d_fn(x, w, b, activation=act)
    ref = causal_conv1d_ref(x_ref, w_ref, b_ref, activation=act)
    rtol, atol = tol[itype]
    name = "fwd_%s_w%d_%s_%s" % (str(itype).split(".")[1], width, act or "none", "cl" if channel_last else "cf")
    err = (out.float() - ref.float()).abs().max().item()
    check(name, torch.allclose(out.float(), ref.float(), rtol=rtol, atol=atol), "max err %.2e" % err)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    errs = [(x.grad.float() - x_ref.grad.float()).abs().max().item(),
            (w.grad - w_ref.grad).abs().max().item(), (b.grad - b_ref.grad).abs().max().item()]
    ok = all(torch.allclose(a.float(), r.float(), rtol=rtol * 10, atol=atol * 10)
             for a, r in ((x.grad, x_ref.grad), (w.grad, w_ref.grad), (b.grad, b_ref.grad)))
    check(name.replace("fwd", "bwd"), ok, "max err dx %.2e dw %.2e db %.2e" % tuple(errs))

# throughput on a Mamba-sized problem
x = torch.randn(8, 2048, 4096, device=dev, dtype=torch.bfloat16)
w = torch.randn(2048, 4, device=dev)
b = torch.randn(2048, device=dev)
for _ in range(5):
    causal_conv1d_fn(x, w, b, activation="silu")
torch.xpu.synchronize()
t = time.time()
for _ in range(20):
    causal_conv1d_fn(x, w, b, activation="silu")
torch.xpu.synchronize()
ms = (time.time() - t) * 1e3 / 20
t = time.time()
for _ in range(20):
    causal_conv1d_ref(x, w, b, activation="silu")
torch.xpu.synchronize()
ms_ref = (time.time() - t) * 1e3 / 20
print("throughput (8x2048x4096 bf16, silu): kernel %.2f ms, torch reference %.2f ms" % (ms, ms_ref))

print("\n%d failed" % fails)
sys.exit(1 if fails else 0)
