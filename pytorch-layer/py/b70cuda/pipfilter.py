"""Rewrite pip requirements written for NVIDIA so they install cleanly on the B70.

    python -m b70cuda.pipfilter OUTDIR ARG...    (used by `b70 install`)

ARG is anything you would give to `pip install`. Requirement files given with
-r are read (recursively), filtered and written to OUTDIR; the rewritten
argument list is printed one per line on stdout. What was dropped or replaced
is reported on stderr.

Rules
  * torch / torchvision / torchaudio / triton: never reinstalled -- the image's
    XPU builds stay (a CUDA wheel would silently replace them)
  * CUDA-only packages are dropped: flash-attn and sageattention (the image
    ships SDPA-based look-alikes of both), xformers, nvidia-*, cupy, pycuda,
    tensorrt, apex, spconv-cu*, ...
  * packages with a pure-Triton edition can be pinned to it (TRITON_ALT; empty
    at the moment)
  * packages that ship plain CUDA extensions (causal-conv1d, mamba-ssm, ...)
    are kept: the b70torch hook builds them with the b70 toolchain
  * onnxruntime-gpu becomes onnxruntime
  * --index-url / --extra-index-url / --find-links that point at CUDA wheel
    indexes are dropped, as are direct URLs to CUDA wheels
"""
import os
import re
import sys

KEEP_IMAGE = {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton",
              "pytorch-triton-xpu", "triton-xpu"}
# Packages that cannot be built for the B70 even from source (inline PTX /
# cutlass / NVIDIA-only libraries). Plain CUDA extensions such as causal-conv1d
# or mamba-ssm are NOT listed: the b70torch hook builds those with the b70
# toolchain (see pytorch-ext/).
DROP = {"flash-attn", "sageattention", "xformers", "cupy", "pycuda", "apex",
        "nvidia-ml-py", "pynvml", "tensorrt", "flashinfer-python", "flashinfer",
        "deepspeed-kernels", "torch-tensorrt",
        "onnxruntime-training", "bitsandbytes-cuda", "nunchaku"}
DROP_PREFIX = ("nvidia-", "cupy-", "tensorrt-", "tensorrt_", "spconv-cu",
               "flash-attn-", "xformers-")
REPLACE = {"onnxruntime-gpu": "onnxruntime"}
# Packages whose current release is CUDA-only (inline PTX / cutlass) but that
# have a pure-Triton edition; triton-xpu in the image runs those unchanged.
# The version pin replaces whatever the requirement asked for.
TRITON_ALT = {
    # (empty since 2026-10-09: sageattention moved to DROP, the image ships an SDPA-based
    # shim; its 1.0.6 Triton edition ran 17x slower than SDPA on the B70)
}
CUDA_URL = re.compile(r"(/whl/cu\d+|\+cu\d+|cuda\d*|/nightly/cu\d+|nvidia)", re.I)
NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


class Filter:
    def __init__(self, outdir):
        self.outdir = outdir
        self.notes = []
        self.n = 0

    def note(self, what, why):
        self.notes.append("  %-40s %s" % (what, why))

    def requirement(self, req):
        """Return the requirement to keep (maybe rewritten), or None to drop it."""
        s = req.strip()
        if not s:
            return s
        if "://" in s and not s.startswith(("git+", "hg+", "svn+", "bzr+")):
            if CUDA_URL.search(s) and s.endswith(".whl"):
                self.note(s[-60:], "dropped: direct URL to a CUDA wheel")
                return None
            return s
        if s.startswith(("git+", "-e", ".", "/")):
            return s
        mt = NAME.match(s)
        if not mt:
            return s
        name = canon(mt.group(1))
        if name in KEEP_IMAGE:
            self.note(s, "kept the image's XPU build instead")
            return None
        if name in TRITON_ALT:
            new, why = TRITON_ALT[name]
            if canon(s.split(";", 1)[0].strip()) == canon(new):
                return s
            self.note(s, "-> %s (%s)" % (new, why))
            return new
        if name in DROP or name.startswith(DROP_PREFIX):
            why = "dropped: CUDA-only"
            if name == "flash-attn":
                why = "dropped: image ships an SDPA-based flash_attn"
            elif name == "sageattention":
                why = "dropped: image ships an SDPA-based sageattention (the Triton edition is 17x slower here)"
            self.note(s, why)
            return None
        if name in REPLACE:
            new = REPLACE[name] + s[mt.end():]
            self.note(s, "-> " + new)
            return new
        if "+cu" in s:
            new = re.sub(r"\+cu\d+", "", s)
            self.note(s, "-> " + new)
            return new
        return s

    def option_line(self, opt, value):
        """Handle index/find-links options. Returns list of tokens to keep."""
        if CUDA_URL.search(value or ""):
            self.note("%s %s" % (opt, value), "dropped: CUDA wheel index")
            return []
        return [opt, value]

    def file(self, path):
        self.n += 1
        n = self.n
        base = os.path.dirname(os.path.abspath(path))
        out_lines = []
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        joined, buf = [], ""
        for line in lines:  # join backslash continuations
            if line.rstrip().endswith("\\"):
                buf += line.rstrip()[:-1] + " "
                continue
            joined.append(buf + line)
            buf = ""
        if buf:
            joined.append(buf)
        for line in joined:
            body = line.split(" #", 1)[0].strip() if not line.lstrip().startswith("#") else ""
            if not body:
                out_lines.append(line)
                continue
            tok = body.split(None, 1)
            opt = tok[0]
            val = tok[1].strip() if len(tok) > 1 else ""
            if opt in ("-r", "--requirement", "-c", "--constraint"):
                sub = val if os.path.isabs(val) else os.path.join(base, val)
                out_lines.append("%s %s" % (opt, self.file(sub)))
            elif opt.startswith(("-r", "-c")) and len(opt) > 2 and not opt.startswith("--"):
                v = opt[2:]
                sub = v if os.path.isabs(v) else os.path.join(base, v)
                out_lines.append("%s %s" % (opt[:2], self.file(sub)))
            elif opt in ("-i", "--index-url", "--extra-index-url", "-f", "--find-links"):
                out_lines.append(" ".join(self.option_line(opt, val)))
            elif opt.startswith(("--index-url=", "--extra-index-url=", "--find-links=")):
                o, v = opt.split("=", 1)
                out_lines.append(" ".join(self.option_line(o, v)))
            elif opt in ("-e", "--editable"):
                p = val
                if not ("://" in p or os.path.isabs(p)):
                    p = os.path.join(base, p)
                out_lines.append("-e " + p)
            elif opt.startswith("-"):
                out_lines.append(body)
            else:
                if body.startswith(".") or (("/" in body) and "://" not in body
                                            and not body.startswith("git+")):
                    p = body if os.path.isabs(body) else os.path.join(base, body)
                    out_lines.append(p)
                    continue
                r = self.requirement(body)
                if r is not None:
                    out_lines.append(r)
        out = os.path.join(self.outdir, "req%d-%s" % (n, os.path.basename(path)))
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        return out

    def args(self, argv):
        out = []
        i = 0
        while i < len(argv):
            a = argv[i]
            if a in ("-r", "--requirement", "-c", "--constraint") and i + 1 < len(argv):
                out += [a, self.file(argv[i + 1])]
                i += 2
                continue
            if a.startswith(("--requirement=", "--constraint=")):
                o, v = a.split("=", 1)
                out += [o, self.file(v)]
            elif a in ("-i", "--index-url", "--extra-index-url", "-f",
                       "--find-links") and i + 1 < len(argv):
                out += self.option_line(a, argv[i + 1])
                i += 2
                continue
            elif a.startswith(("--index-url=", "--extra-index-url=", "--find-links=")):
                o, v = a.split("=", 1)
                out += self.option_line(o, v)
            elif a.startswith("-"):
                out.append(a)
            else:
                r = self.requirement(a)
                if r is not None:
                    out.append(r)
            i += 1
        return out


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    flt = Filter(argv[1])
    out = flt.args(argv[2:])
    if flt.notes:
        print("[b70 install] rewrote the requirements for the B70:", file=sys.stderr)
        print("\n".join(flt.notes), file=sys.stderr)
    for a in out:
        print(a)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
