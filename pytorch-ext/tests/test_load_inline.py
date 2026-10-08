"""M4: the JIT path (torch.utils.cpp_extension.load_inline), which research code
uses a lot. Run under `b70 python` (the hook is in the image).
"""
import time

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <ATen/cuda/CUDAContext.h>
__global__ void scale_k(const float* x, float* y, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * s;
}
at::Tensor scale(at::Tensor x, double s) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == at::kFloat);
    auto y = at::empty_like(x);
    int n = x.numel();
    scale_k<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n, (float)s);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""
cpp_src = "at::Tensor scale(at::Tensor x, double s);"

t = time.time()
mod = load_inline(name="b70_inline_test", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["scale"], verbose=False)
print("load_inline build: %.1fs" % (time.time() - t))
x = torch.randn(100_000, device="cuda")
y = mod.scale(x, 3.0)
ok = y.device.type == "xpu" and torch.allclose(y.cpu(), x.cpu() * 3.0)
print("PASS load_inline" if ok else "FAIL load_inline")
raise SystemExit(0 if ok else 1)
