"""Standalone reproduction of a torch XPU bug found while running causal-conv1d's
upstream test-suite on the B70 (torch 2.13.0+xpu):

F.conv1d's weight gradient is wrong when the input has length 1 and the length
dimension has a stride other than 1 -- e.g. x = randn(B, 1, C).transpose(1, 2),
strides (C, 1, C). torch regards that tensor as contiguous (size-1 dims are
ignored), but the XPU convolution backward apparently hands the raw strides to
oneDNN and reads the wrong elements. Forward is fine, grad-output layout does
not matter, groups=1 is affected too. CPU is correct, and so is the
causal-conv1d CUDA kernel compiled with b70cc, which is how this surfaced:
204 of the upstream tests (all seqlen=1, channel_last) and most varlen tests
(random length-1 segments) compare against this reference.

    b70 python torch_xpu_conv1d_len1_bug.py
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
b, c, w = 2, 8, 2
weight = torch.randn(c, w)
x0 = torch.randn(b, c, 1)
g0 = torch.randn(b, c, 1)


def weight_grad(dev, x, g):
    xr = x.to(dev).detach().requires_grad_()
    wr = weight.to(dev).detach().requires_grad_()
    out = F.conv1d(xr, wr.unsqueeze(1), padding=w - 1, groups=c)[..., :1]
    return torch.autograd.grad(out, wr, g.to(dev))[0].cpu()


manual = torch.zeros(c, w)
manual[:, 1] = (x0[..., 0] * g0[..., 0]).sum(0)  # only the current tap can be nonzero at length 1

cases = {
    "x strides (c,1,1)  [randn(b,c,1)]": x0,
    "x strides (c,1,c)  [randn(b,1,c).transpose(1,2)]": x0.as_strided((b, c, 1), (c, 1, c)),
}
bad = 0
for name, x in cases.items():
    cpu = weight_grad("cpu", x, g0)
    xpu = weight_grad("xpu", x, g0)
    ok = torch.allclose(xpu, manual, atol=1e-5)
    bad += not ok
    print("%-50s is_contiguous=%s  CPU==manual %s  XPU==manual %s" % (
        name, x.is_contiguous(), torch.allclose(cpu, manual, atol=1e-5), ok))
    if not ok:
        print("   manual:", [round(v, 3) for v in manual.flatten().tolist()[:6]])
        print("   XPU   :", [round(v, 3) for v in xpu.flatten().tolist()[:6]])
print("torch", torch.__version__, "-", "BUG REPRODUCED" if bad else "not reproduced")
