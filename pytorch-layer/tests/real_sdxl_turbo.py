"""Real-model check: the SDXL-Turbo model-card snippet, unchanged (pipe.to("cuda")).

Needs ~8 GB of VRAM, so free the card first:
    b70 gpu free
    cd ~/b70-kit/tests && b70 python real_sdxl_turbo.py
    b70 gpu llm
Downloads ~7 GB into $B70_DATA/hf on the first run; the image goes to $B70_DATA/home.
"""
import os
import time

import torch
from diffusers import AutoPipelineForText2Image

t0 = time.time()
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")
print("load: %.1fs, device %s" % (time.time() - t0, pipe.unet.device))

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]  # warm-up / JIT

for steps, size in ((1, 512), (4, 512), (4, 1024)):
    torch.cuda.synchronize()
    t = time.time()
    image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0,
                 height=size, width=size).images[0]
    torch.cuda.synchronize()
    print("%d step(s) %dx%d: %.2fs" % (steps, size, size, time.time() - t))

out = os.path.expanduser("~/sdxl_turbo_test.png")  # HOME is $B70_DATA/home in the container
image.save(out)
print("saved", out, "peak VRAM %.1f GiB" % (torch.cuda.max_memory_allocated() / 2**30))
