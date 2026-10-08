"""M4 ③: real Triton packages on the B70, installed through `b70 install`
without modification and checked against torch reference implementations.

  * Liger-Kernel (pure Triton; RMSNorm, SwiGLU, fused linear cross-entropy)
  * sageattention 1.0.6 (the all-Triton edition `b70 install` picks instead
    of the CUDA-only 2.x; INT8 QK^T on XMX)

    cd <dir with .b70venv>; b70 install liger-kernel sageattention
    b70 python test_triton_pkgs.py
"""
import sys
import time

import torch
import torch.nn.functional as F

dev = "cuda"
fails = 0


def check(name, ok, detail=""):
    global fails
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    fails += 0 if ok else 1


def bench(fn, n=50):
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.xpu.synchronize()
    return (time.time() - t) * 1e3 / n


# ---------------------------------------------------------------- Liger-Kernel
from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP, LigerFusedLinearCrossEntropyLoss

torch.manual_seed(0)
B, T, H, V = 4, 512, 2048, 32000

for dtype in (torch.float32, torch.bfloat16):
    tag = str(dtype).split(".")[1]
    x = torch.randn(B, T, H, device=dev, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    # RMSNorm
    ln = LigerRMSNorm(H, eps=1e-6).to(dev, dtype)
    ref = torch.nn.RMSNorm(H, eps=1e-6).to(dev, dtype)
    with torch.no_grad():
        ln.weight.copy_(torch.rand(H) + 0.5)
        ref.weight.copy_(ln.weight)
    y = ln(x)
    y_ref = ref(x_ref)
    err = (y.float() - y_ref.float()).abs().max().item()
    check("liger_rmsnorm_fwd_%s" % tag, err < (1e-4 if dtype == torch.float32 else 5e-2), "max err %.2e" % err)
    g = torch.randn_like(y)
    y.backward(g.clone())  # Liger's backward is in-place on the incoming gradient (in_place=True default)
    y_ref.backward(g)
    err = max((x.grad.float() - x_ref.grad.float()).abs().max().item(),
              (ln.weight.grad.float() - ref.weight.grad.float()).abs().max().item() / T / B)
    check("liger_rmsnorm_bwd_%s" % tag, err < (1e-3 if dtype == torch.float32 else 5e-2), "max err %.2e" % err)

    # SwiGLU MLP
    class Cfg:
        hidden_size, intermediate_size, hidden_act = H, 4 * H, "silu"
    mlp = LigerSwiGLUMLP(Cfg()).to(dev, dtype)
    xm = torch.randn(B, T, H, device=dev, dtype=dtype)
    out = mlp(xm)
    out_ref = mlp.down_proj(F.silu(mlp.gate_proj(xm)) * mlp.up_proj(xm))
    err = (out.float() - out_ref.float()).abs().max().item()
    check("liger_swiglu_%s" % tag, err < (1e-3 if dtype == torch.float32 else 1e-1), "max err %.2e" % err)

# fused linear cross-entropy (never materialises the B*T x V logits)
dtype = torch.bfloat16
lin_w = torch.randn(V, H, device=dev, dtype=dtype) * 0.02
hid = torch.randn(B * T, H, device=dev, dtype=dtype, requires_grad=True)
hid_ref = hid.detach().clone().requires_grad_()
tgt = torch.randint(0, V, (B * T,), device=dev)
loss = LigerFusedLinearCrossEntropyLoss()(lin_w, hid, tgt)
loss_ref = F.cross_entropy((hid_ref @ lin_w.t()).float(), tgt)
check("liger_fused_linear_ce_fwd", abs(loss.item() - loss_ref.item()) < 5e-2,
      "liger %.4f torch %.4f" % (loss.item(), loss_ref.item()))
loss.backward()
loss_ref.backward()
err = (hid.grad.float() - hid_ref.grad.float()).abs().max().item()
check("liger_fused_linear_ce_bwd", err < 1e-2, "max err %.2e" % err)
ms = bench(lambda: LigerFusedLinearCrossEntropyLoss()(lin_w, hid.detach(), tgt))
ms_ref = bench(lambda: F.cross_entropy((hid.detach() @ lin_w.t()).float(), tgt))
print("fused linear CE (%dx%d -> %d vocab): liger %.2f ms, torch %.2f ms" % (B * T, H, V, ms, ms_ref))

# --------------------------------------------------------------- sageattention
try:
    import sageattention
    from sageattention import sageattn
    have_sage = True
except ImportError as e:
    print("sageattention not installed (%s); skipping" % e)
    have_sage = False

if have_sage:
    for (b, h, s, d) in ((2, 8, 1024, 64), (1, 16, 2048, 128)):
        q, k, v = (torch.randn(b, h, s, d, device=dev, dtype=torch.float16) for _ in range(3))
        for causal in (False, True):
            out = sageattn(q, k, v, is_causal=causal)
            ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            err = (out.float() - ref.float()).abs().max().item()
            # INT8 QK^T + smoothing: errors around 1e-2 are expected
            check("sageattn_%dx%dx%dx%d_%s" % (b, h, s, d, "causal" if causal else "full"),
                  err < 6e-2, "max err %.2e" % err)
    q, k, v = (torch.randn(2, 16, 4096, 128, device=dev, dtype=torch.float16) for _ in range(3))
    ms = bench(lambda: sageattn(q, k, v, is_causal=True), 20)
    ms_ref = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), 20)
    print("attention (2x16x4096x128 fp16 causal): sageattn %.2f ms, torch SDPA %.2f ms" % (ms, ms_ref))

print("\n%d failed" % fails)
sys.exit(1 if fails else 0)
