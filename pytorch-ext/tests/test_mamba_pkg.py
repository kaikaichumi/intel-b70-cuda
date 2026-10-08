"""M4 ②: mamba-ssm (state-spaces/mamba), built from its unmodified PyPI source
through the b70torch hook (selective_scan_cuda: CUB BlockLoad/Store/Scan,
BlockRakingLayout, reverse scans, complex dtypes), checked against the
package's own pure-torch references.

    cd <dir with .b70venv where causal-conv1d and mamba-ssm were installed>
    b70 python test_mamba_pkg.py
"""
import sys
import time

import torch

# mamba-ssm 2.2.5 imports two output classes that transformers >= 4.50 renamed
# (fixed on mamba's main branch); nothing to do with the GPU.
import transformers.generation as _tg
for _n in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
    if not hasattr(_tg, _n):
        setattr(_tg, _n, _tg.GenerateDecoderOnlyOutput)

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm.modules.mamba_simple import Mamba

dev = "cuda"
fails = 0


def check(name, ok, detail=""):
    global fails
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    fails += 0 if ok else 1


def maxerr(a, b):
    return (a.float() - b.float()).abs().max().item()


# ---- selective_scan_fn vs selective_scan_ref (forward + backward) -----------
tol = {torch.float32: (1e-4, 1e-4), torch.float16: (1e-2, 1e-2), torch.bfloat16: (3e-2, 3e-2)}
batch, dim, dstate, seqlen = 2, 256, 16, 512
for itype in (torch.float32, torch.float16, torch.bfloat16):
    for has_z, delta_softplus in ((False, False), (True, True)):
        torch.manual_seed(0)
        u = torch.randn(batch, dim, seqlen, device=dev, dtype=itype, requires_grad=True)
        delta = (0.5 * torch.rand(batch, dim, seqlen, device=dev, dtype=itype)).requires_grad_()
        A = (-0.5 * torch.rand(dim, dstate, device=dev, dtype=torch.float32)).requires_grad_()
        B = torch.randn(batch, dstate, seqlen, device=dev, dtype=itype, requires_grad=True)
        C = torch.randn(batch, dstate, seqlen, device=dev, dtype=itype, requires_grad=True)
        D = torch.randn(dim, device=dev, dtype=torch.float32, requires_grad=True)
        z = torch.randn(batch, dim, seqlen, device=dev, dtype=itype, requires_grad=True) if has_z else None
        delta_bias = (0.5 * torch.rand(dim, device=dev, dtype=torch.float32)).requires_grad_()
        refs = [t.detach().clone().requires_grad_() if t is not None else None
                for t in (u, delta, A, B, C, D, z, delta_bias)]

        out = selective_scan_fn(u, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus)
        ref = selective_scan_ref(*refs[:6], z=refs[6], delta_bias=refs[7], delta_softplus=delta_softplus)
        rtol, atol = tol[itype]
        tag = "%s_%s" % (str(itype).split(".")[1], "z_softplus" if has_z else "plain")
        check("selective_scan_fwd_" + tag, torch.allclose(out.float(), ref.float(), rtol=rtol, atol=atol),
              "max err %.2e" % maxerr(out, ref))

        g = torch.randn_like(out)
        out.backward(g)
        ref.backward(g)
        # Precision floor for half types: the same reference computed from the
        # inputs *before* they were rounded to itype. dA sums over batch*seqlen
        # and reaches 1e5 here, so its rounding floor is ~1e2; the kernel is right
        # if it is no further from the itype reference than that floor.
        ins = (u, delta, A, B, C, D, z, delta_bias)
        r32 = [t.detach().float().clone().requires_grad_() if t is not None else None for t in ins]
        ref32 = selective_scan_ref(*r32[:6], z=r32[6], delta_bias=r32[7], delta_softplus=delta_softplus)
        ref32.backward(g.float())
        # Tolerances as in mamba's own tests/ops/test_selective_scan.py (which
        # only enables fp32 inputs); a gradient also passes if it is no further
        # from the itype reference than the input-rounding floor above.
        urtol, uatol = {torch.float32: (6e-4, 2e-3), torch.float16: (3e-3, 5e-3),
                        torch.bfloat16: (3e-2, 5e-2)}[itype]
        rtolw, atolw = (max(1e-3, urtol), max(1e-3, uatol)) if has_z else (1e-3, 1e-3)
        tols = {"du": (urtol * 2, uatol * 2), "ddelta": (urtol * 5, uatol * 10),
                "dA": (rtolw, atolw * 5), "dD": (rtolw, atolw * 5), "ddelta_bias": (rtolw, atolw * 5),
                "dB": (urtol, uatol), "dC": (urtol, uatol), "dz": (urtol, uatol)}
        names = ("du", "ddelta", "dA", "dB", "dC", "dD", "dz", "ddelta_bias")
        errs, ok = [], True
        for n, t, r, r3 in zip(names, ins, refs, r32):
            if t is None:
                continue
            e = maxerr(t.grad, r.grad)
            floor = maxerr(r.grad, r3.grad)
            errs.append("%s %.1e" % (n, e))
            upstream_ok = torch.allclose(t.grad.float(), r.grad.float(), rtol=tols[n][0], atol=tols[n][1])
            ok &= upstream_ok or e <= 2 * floor
        check("selective_scan_bwd_" + tag, ok, " ".join(errs))

# ---- a whole Mamba block: fused fast path vs the layer's own slow path ------
torch.manual_seed(0)
d_model = 768
fast = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, use_fast_path=True, device=dev)
slow = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, use_fast_path=False, device=dev)
slow.load_state_dict(fast.state_dict())
x = torch.randn(2, 256, d_model, device=dev, requires_grad=True)
x2 = x.detach().clone().requires_grad_()
y = fast(x)
y2 = slow(x2)
check("mamba_block_fwd", torch.allclose(y, y2, rtol=1e-3, atol=1e-3), "max err %.2e" % maxerr(y, y2))
g = torch.randn_like(y)
y.backward(g)
y2.backward(g)
err = max(maxerr(x.grad, x2.grad),
          max(maxerr(p.grad, q.grad) for p, q in zip(fast.parameters(), slow.parameters()) if p.grad is not None))
check("mamba_block_bwd", err < 1e-2, "max err %.2e" % err)

# ---- throughput: a 130M-style Mamba layer, training step shapes ------------
layer = Mamba(d_model=768, d_state=16, d_conv=4, expand=2, device=dev, dtype=torch.bfloat16)
xb = torch.randn(8, 2048, 768, device=dev, dtype=torch.bfloat16)
for _ in range(3):
    layer(xb)
torch.xpu.synchronize()
t = time.time()
for _ in range(10):
    layer(xb)
torch.xpu.synchronize()
ms = (time.time() - t) * 1e3 / 10
ref_layer = Mamba(d_model=768, d_state=16, d_conv=4, expand=2, use_fast_path=False, device=dev, dtype=torch.bfloat16)
for _ in range(3):
    ref_layer(xb)
torch.xpu.synchronize()
t = time.time()
for _ in range(10):
    ref_layer(xb)
torch.xpu.synchronize()
ms_ref = (time.time() - t) * 1e3 / 10
print("Mamba layer fwd (8x2048x768 bf16): fused %.2f ms, unfused %.2f ms" % (ms, ms_ref))

print("\n%d failed" % fails)
sys.exit(1 if fails else 0)
