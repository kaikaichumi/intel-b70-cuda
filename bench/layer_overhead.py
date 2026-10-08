"""Same torch workload on device NAME (argv[1]): "cuda" through the b70cuda layer, "xpu" native."""
import sys, time, torch, torch.nn.functional as F
dev = sys.argv[1]
def sync(): (torch.cuda if dev == "cuda" else torch.xpu).synchronize()
def bench(fn, n=20):
    for _ in range(3): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(n): fn()
    sync(); return (time.perf_counter() - t0) / n * 1e3
torch.manual_seed(0)
a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16); b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
ms = bench(lambda: a @ b); print("matmul 4096^2 bf16      %7.2f ms  %5.1f TFLOPS" % (ms, 2 * 4096**3 / ms / 1e9))
q = torch.randn(1, 16, 4096, 64, device=dev, dtype=torch.bfloat16)
ms = bench(lambda: F.scaled_dot_product_attention(q, q, q)); print("sdpa 4k tokens 16 heads %7.2f ms" % ms)
x = torch.randn(8, 64, 256, 256, device=dev, dtype=torch.bfloat16); w = torch.randn(64, 64, 3, 3, device=dev, dtype=torch.bfloat16)
ms = bench(lambda: F.conv2d(x, w, padding=1)); print("conv2d 8x64x256x256     %7.2f ms" % ms)
layer = torch.nn.TransformerEncoderLayer(1024, 16, 4096, batch_first=True).to(dev).to(torch.bfloat16).eval()
inp = torch.randn(4, 512, 1024, device=dev, dtype=torch.bfloat16)
with torch.no_grad(): ms = bench(lambda: layer(inp)); print("transformer layer fwd   %7.2f ms" % ms)
s = torch.zeros(1000, device=dev)
ms = bench(lambda: [s.add_(1) for _ in range(1000)], n=5); print("1000 tiny ops (dispatch)%7.2f ms" % ms)
ms = bench(lambda: torch.randn(1000, 1000, device=dev).to(dev), n=50); print("alloc+to(device) x50    %7.3f ms" % ms)
