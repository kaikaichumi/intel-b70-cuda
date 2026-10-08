"""Generate one image through a running ComfyUI (b70 comfyui) over its HTTP API.

Uses the diffusers-format SDXL-Turbo already in the HF cache, linked into
ComfyUI/models/diffusers/sdxl-turbo, so no extra download is needed.
    python3 comfyui_api_test.py [http://127.0.0.1:8188]
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8188"

workflow = {
    "1": {"class_type": "DiffusersLoader", "inputs": {"model_path": "sdxl-turbo"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {
        "clip": ["1", 1],
        "text": "a cinematic photo of a red fox in a snowy forest, golden hour"}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ""}},
    "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "5": {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
        "seed": 42, "steps": 4, "cfg": 1.0, "sampler_name": "euler_ancestral",
        "scheduler": "sgm_uniform", "denoise": 1.0}},
    "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
    "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "b70_test"}},
}


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


stats = call("/system_stats")
for d in stats.get("devices", []):
    print("device: %s  vram_total=%.1f GiB  vram_free=%.1f GiB"
          % (d.get("name"), d.get("vram_total", 0) / 2**30, d.get("vram_free", 0) / 2**30))

t0 = time.time()
pid = call("/prompt", {"prompt": workflow})["prompt_id"]
print("queued", pid)
while True:
    hist = call("/history/" + pid)
    if pid in hist:
        h = hist[pid]
        status = h.get("status", {})
        if status.get("status_str") == "error":
            print("ERROR", json.dumps(status.get("messages", [])[-1:], ensure_ascii=False)[:800])
            sys.exit(1)
        imgs = [i for o in h.get("outputs", {}).values() for i in o.get("images", [])]
        print("done in %.1fs: %s" % (time.time() - t0, [i["filename"] for i in imgs]))
        break
    if time.time() - t0 > 900:
        print("TIMEOUT")
        sys.exit(1)
    time.sleep(2)

# second run: model already loaded, so this is the real generation speed
workflow["5"]["inputs"]["seed"] = 7
t1 = time.time()
pid = call("/prompt", {"prompt": workflow})["prompt_id"]
while pid not in call("/history/" + pid):
    time.sleep(0.5)
print("second image (model loaded): %.1fs" % (time.time() - t1))
