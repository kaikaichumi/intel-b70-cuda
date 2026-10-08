"""b70 self-tests: CUDA-style code, written exactly as it would be for an NVIDIA
card, must run on the B70 through the b70cuda layer.

Small enough to run while vLLM holds most of the VRAM (needs < 1 GB).
Run with: b70 test            (or: b70 test -k flash   to pick tests)
"""
import io
import os
import sys
import time
import traceback

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


# --------------------------------------------------------------------------
@test
def cuda_gate():
    import torch
    import b70cuda
    assert b70cuda.is_patched(), "b70cuda did not activate"
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    name = torch.cuda.get_device_name(0)
    assert "B70" in name or "Arc" in name, name
    p = torch.cuda.get_device_properties(0)
    assert p.total_memory > 30 * 2**30 and p.major == 8 and p.multi_processor_count > 0
    free, total = torch.cuda.mem_get_info()
    assert 0 < free <= total
    return "%s, %d Xe cores, %.1f GiB free" % (name, p.multi_processor_count, free / 2**30)


@test
def devices_everywhere():
    import torch
    a = torch.randn(64, 64, device="cuda")
    b = torch.ones(64, 64, device=torch.device("cuda:0"))
    c = torch.arange(64, device="cuda").float()
    d = torch.tensor([1.0, 2.0]).cuda()
    e = torch.zeros(4).to("cuda")
    f = torch.zeros(4).to(torch.device("cuda"), torch.float16)
    g = torch.zeros(4).to(device="cuda", dtype=torch.bfloat16)
    h = a.new_zeros((3,), device="cuda")
    i = torch.zeros(2).to(0)
    for t in (a, b, c, d, e, f, g, h, i):
        assert t.device.type == "xpu", t.device
    assert f.dtype == torch.float16 and g.dtype == torch.bfloat16
    lin = torch.nn.Linear(64, 64).cuda()
    lin2 = torch.nn.Linear(64, 64, device="cuda")
    m = torch.nn.Sequential(torch.nn.Linear(64, 8)).to("cuda")
    y = m(lin(a) + lin2(b) + c)
    torch.cuda.synchronize()
    assert y.device.type == "xpu" and next(m.parameters()).device.type == "xpu"
    return "factories, .to/.cuda, nn.Module all land on xpu"


@test
def device_contexts():
    import torch
    # transformers detects meta-device init exactly like this
    with torch.device("meta"):
        assert torch.tensor([]).device.type == "meta"
        assert torch.zeros(2).device.type == "meta"
        assert torch.nn.Linear(4, 4).weight.device.type == "meta"
    assert torch.tensor([]).device.type == "cpu"
    with torch.device("cuda"):
        assert torch.arange(3).device.type == "xpu"
        assert torch.nn.Linear(4, 4).weight.device.type == "xpu"
    torch.set_default_device("cuda")
    try:
        assert torch.ones(2).device.type == "xpu"
        assert torch.tensor([1, 2]).device.type == "xpu"
    finally:
        torch.set_default_device(None)
    assert torch.ones(2).device.type == "cpu"
    return "with torch.device('meta'/'cuda') and set_default_device('cuda')"


@test
def generator_and_seed():
    import torch
    g = torch.Generator(device="cuda").manual_seed(1234)
    assert isinstance(g, torch.Generator) and g.device.type == "xpu"
    x1 = torch.randn(8, device="cuda", generator=g)
    g2 = torch.Generator("cuda")
    g2.manual_seed(1234)
    x2 = torch.randn(8, device="cuda", generator=g2)
    assert torch.equal(x1, x2)
    torch.cuda.manual_seed_all(0)
    s = torch.cuda.get_rng_state()
    r1 = torch.rand(4, device="cuda")
    torch.cuda.set_rng_state(s)
    r2 = torch.rand(4, device="cuda")
    assert torch.equal(r1, r2)
    return "torch.Generator('cuda'), RNG state round-trip"


@test
def amp_training_step():
    import torch
    model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(),
                                torch.nn.Linear(256, 10)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    x = torch.randn(32, 128, device="cuda")
    yt = torch.randint(0, 10, (32,), device="cuda")
    losses = []
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            out = model(x)
            assert out.dtype == torch.float16, out.dtype
            loss = torch.nn.functional.cross_entropy(out, yt)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert model(x).dtype == torch.bfloat16
    s2 = torch.amp.GradScaler("cuda")
    assert s2 is not None
    assert losses[-1] < losses[0], losses
    return "loss %.3f -> %.3f (cuda.amp autocast + GradScaler)" % (losses[0], losses[-1])


@test
def memory_api():
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.empty(64 * 2**20, dtype=torch.uint8, device="cuda")
    a = torch.cuda.memory_allocated()
    pk = torch.cuda.max_memory_allocated()
    r = torch.cuda.memory_reserved()
    assert a >= 64 * 2**20 and pk >= a and r >= a, (a, pk, r)
    del x
    torch.cuda.empty_cache()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        y = torch.ones(10, device="cuda") * 2
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.device(0):
        pass
    torch.cuda.nvtx.range_push("x")
    torch.cuda.nvtx.range_pop()
    return "allocated/peak/reserved, streams, events, nvtx no-op; y=%s" % y[0].item()


@test
def torch_load_cuda_checkpoint():
    import torch
    import torch.serialization as ser
    sd = {"w": torch.randn(4, 4), "b": torch.randn(4)}
    buf = io.BytesIO()
    real_tag = ser.location_tag
    ser.location_tag = lambda storage: "cuda:0"   # pretend it was saved on an NVIDIA card
    try:
        torch.save(sd, buf)
    finally:
        ser.location_tag = real_tag
    buf.seek(0)
    loaded = torch.load(buf)                      # no map_location, like most repos
    assert loaded["w"].device.type == "xpu"
    buf.seek(0)
    on_cpu = torch.load(buf, map_location="cpu")
    assert on_cpu["w"].device.type == "cpu" and torch.equal(on_cpu["w"], sd["w"])
    buf.seek(0)
    mapped = torch.load(buf, map_location="cuda")
    assert mapped["b"].device.type == "xpu"
    return "a checkpoint tagged cuda:0 loads onto the B70"


@test
def native_libs_see_truth():
    import torch
    import transformers.utils as tu
    # transformers keeps its own XPU path because it sees the real torch.cuda gate
    assert tu.is_torch_xpu_available()
    # ... while user code is told there is a CUDA device
    assert torch.cuda.is_available()
    return "transformers sees xpu, user code sees cuda"


@test
def flash_attn_shim():
    import torch
    import torch.nn.functional as F
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    torch.manual_seed(0)

    def ref(q, k, v, causal=False, window=(-1, -1)):
        # straightforward attention in fp32, flash_attn conventions
        b, sq, h, d = q.shape
        sk, hk = k.shape[1], k.shape[2]
        k = k.repeat_interleave(h // hk, 2)
        v = v.repeat_interleave(h // hk, 2)
        s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / d ** 0.5
        i = torch.arange(sq, device=q.device)[:, None] + sk - sq
        j = torch.arange(sk, device=q.device)[None, :]
        keep = torch.ones(sq, sk, dtype=torch.bool, device=q.device)
        left, right = window
        if causal:
            right = 0
        if right >= 0:
            keep &= j <= i + right
        if left >= 0:
            keep &= j >= i - left
        s = s.masked_fill(~keep, float("-inf"))
        return torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v.float())

    worst = 0.0
    for (sq, sk, h, hk, causal, win) in [(128, 128, 8, 8, False, (-1, -1)),
                                         (128, 128, 8, 2, True, (-1, -1)),
                                         (64, 200, 8, 8, True, (-1, -1)),
                                         (256, 256, 4, 4, False, (32, 32))]:
        q = torch.randn(2, sq, h, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(2, sk, hk, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(2, sk, hk, 64, device="cuda", dtype=torch.float16)
        out = flash_attn_func(q, k, v, causal=causal, window_size=win)
        err = (out.float() - ref(q, k, v, causal, win)).abs().max().item()
        worst = max(worst, err)
        assert out.shape == q.shape and err < 2e-2, (sq, sk, h, hk, causal, win, err)
    # varlen: three sequences of different length packed together
    lens = [37, 100, 64]
    cu = torch.tensor([0, 37, 137, 201], dtype=torch.int32, device="cuda")
    q = torch.randn(201, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(201, 8, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(201, 8, 64, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_varlen_func(q, k, v, cu, cu, 100, 100, causal=True)
    for n, (a, b) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
        r = ref(q[a:b][None], k[a:b][None], v[a:b][None], causal=True)[0]
        err = (out[a:b].float() - r).abs().max().item()
        worst = max(worst, err)
        assert err < 5e-2, (n, err)
    # padding helpers round-trip
    x = torch.randn(3, 100, 16, device="cuda")
    mask = torch.zeros(3, 100, dtype=torch.int32, device="cuda")
    for row, n in enumerate(lens):
        mask[row, :n] = 1
    flat, idx, cus, mx, _ = unpad_input(x, mask)
    assert flat.shape[0] == sum(lens) and mx == 100
    back = pad_input(flat, idx, 3, 100)
    assert torch.equal(back * mask[..., None], x * mask[..., None])
    return "matches reference attention (max err %.4f); varlen, GQA, windows ok" % worst


@test
def sageattention_shim():
    import time
    import torch
    import torch.nn.functional as F
    import sageattention
    from sageattention import sageattn
    assert getattr(sageattention, "B70_SDPA_SHIM", False), "real sageattention installed over the shim?"
    torch.manual_seed(0)

    def ref(q, k, v, causal):  # q, k, v in (B, H, N, D)
        s = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) / q.shape[-1] ** 0.5
        if causal:
            n = q.shape[2]
            s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
        return torch.einsum("bhqk,bhkd->bhqd", s.softmax(-1), v.float())

    worst = 0.0
    for layout, causal, hk in [("HND", False, 8), ("HND", True, 8), ("NHD", False, 8), ("NHD", True, 2)]:
        q = torch.randn(2, 8, 256, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(2, hk, 256, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(2, hk, 256, 64, device="cuda", dtype=torch.float16)
        r = ref(q, k.repeat_interleave(8 // hk, 1), v.repeat_interleave(8 // hk, 1), causal)
        if layout == "NHD":
            q, k, v, r = (t.transpose(1, 2) for t in (q, k, v, r))
        out = sageattn(q, k, v, tensor_layout=layout, is_causal=causal)
        err = (out.float() - r).abs().max().item()
        worst = max(worst, err)
        assert out.shape == q.shape and out.dtype == q.dtype and err < 2e-2, (layout, causal, hk, err)
    # the 2.x kernel entry points exist and agree
    out2 = sageattention.sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="NHD", is_causal=True)
    assert torch.equal(out, out2)
    # and it is SDPA speed, not Triton speed: 4k tokens, 16 heads, bf16
    q = torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        sageattn(q, q, q)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        sageattn(q, q, q)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1e3
    return "matches reference (max err %.4f); HND/NHD, causal, GQA ok; 4k-token attention %.1f ms" % (worst, ms)


@test
def torch_compile():
    import torch

    def f(x, w):
        return torch.nn.functional.gelu(x @ w).sum(-1)

    cf = torch.compile(f)
    x = torch.randn(256, 256, device="cuda")
    w = torch.randn(256, 256, device="cuda")
    t = time.time()
    y = cf(x, w)
    torch.cuda.synchronize()
    assert torch.allclose(y, f(x, w), rtol=1e-3, atol=1e-2)
    return "torch.compile (inductor, triton-xpu) ok, first call %.1fs" % (time.time() - t)


@test
def diffusers_tiny_pipeline():
    import torch
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None)
    pipe = pipe.to("cuda")                                   # model-card style
    g = torch.Generator("cuda").manual_seed(0)
    img = pipe("a photo of a cat", num_inference_steps=4, generator=g,
               output_type="np").images[0]
    assert img.shape[-1] == 3
    assert pipe.unet.device.type == "xpu"
    return "pipe.to('cuda') + Generator('cuda'): image %s" % (img.shape,)


@test
def transformers_tiny_generate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cuda")
    assert next(model.parameters()).device.type == "xpu"
    ids = tok("hello world", return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=8, do_sample=False)
    m2 = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda()
    out2 = m2.generate(**ids, max_new_tokens=8, do_sample=False)
    assert torch.equal(out, out2)
    return "device_map='cuda' and .cuda() both generate on xpu"


@test
def safetensors_cuda_device():
    import torch
    from safetensors.torch import load_file, save_file
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "b70_st_test.safetensors")
    save_file({"a": torch.randn(3, 3)}, path)
    t = load_file(path, device="cuda")
    os.remove(path)
    assert t["a"].device.type == "xpu"
    return "safetensors load_file(device='cuda')"


@test
def bitsandbytes_4bit():
    import torch
    import bitsandbytes as bnb
    lin = torch.nn.Linear(256, 256).half()
    q = bnb.nn.Linear4bit(256, 256, compute_dtype=torch.float16, quant_type="nf4")
    q.load_state_dict(lin.state_dict())
    q = q.to("cuda")
    x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
    ref = lin.cuda()(x)
    err = ((q(x) - ref).norm() / ref.norm()).item()
    assert err < 0.15, err
    return "bitsandbytes %s NF4 Linear on xpu, rel err %.3f" % (bnb.__version__, err)


def main():
    pick = None
    if len(sys.argv) > 2 and sys.argv[1] == "-k":
        pick = sys.argv[2]
    ok = fail = 0
    for fn in RESULTS:
        if pick and pick not in fn.__name__:
            continue
        t = time.time()
        try:
            msg = fn()
            ok += 1
            print("PASS  %-28s %5.1fs  %s" % (fn.__name__, time.time() - t, msg or ""), flush=True)
        except Exception as e:
            fail += 1
            print("FAIL  %-28s %5.1fs  %s: %s" % (fn.__name__, time.time() - t,
                                                  type(e).__name__, e), flush=True)
            if os.environ.get("B70_TEST_TRACE"):
                traceback.print_exc()
    print("\n%d passed, %d failed" % (ok, fail))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
