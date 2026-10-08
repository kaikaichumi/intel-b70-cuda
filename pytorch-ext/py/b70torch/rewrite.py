"""Source rewrite for CUDA extensions built against PyTorch's XPU build.

torch's inline `is_cuda()`, `kCUDA`, `DeviceType::CUDA`, `DispatchKey::CUDA`
all test for DeviceType::CUDA, which an XPU tensor never is. They cannot be
redefined by macros without breaking torch's own declarations, so the hook
builds extensions from a *shadow copy* of the package tree in which the
extension's sources (not torch's headers) are rewritten with the table below.
Only whole tokens / member calls are touched; the original files are untouched.

    python -m b70torch.rewrite FILE...   prints what would change (dry run)
"""
import os
import re
import sys

# (regex, replacement). Word-bounded; member calls keep their `.`/`->`.
RULES = [
    (r"(?<=[.>])is_cuda\(\)", "is_xpu()"),
    (r"(?<=[.>])cuda\(\)", "xpu()"),                      # x.cuda() -> x.xpu()
    (r"\b(torch|at|c10)::kCUDA\b", r"\1::kXPU"),
    (r"\bkCUDA\b", "kXPU"),
    (r"\bDeviceType::CUDA\b", "DeviceType::XPU"),
    (r"\bDispatchKey::CUDA\b", "DispatchKey::XPU"),
    (r"\bDispatchKey::AutogradCUDA\b", "DispatchKey::AutogradXPU"),
    (r"\bBackend::CUDA\b", "Backend::XPU"),
    (r"\b(TORCH_LIBRARY_IMPL)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*CUDA\s*,", r"\1(\2, XPU,"),
    (r"\b(TORCH_LIBRARY_IMPL)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*AutogradCUDA\s*,", r"\1(\2, AutogradXPU,"),
    (r"\btorch::cuda::is_available\(\)", "torch::xpu::is_available()"),
    (r"\btorch::cuda::device_count\(\)", "torch::xpu::device_count()"),
    (r"\bat::cuda::is_available\(\)", "at::xpu::is_available()"),
    (r"\bDeviceType::CUDA\b", "DeviceType::XPU"),
    (r'"cuda"', '"xpu"'),                                   # torch::Device("cuda") and friends
    (r"\bc10::DeviceType::CUDA\b", "c10::DeviceType::XPU"),
]
COMPILED = [(re.compile(p), r) for p, r in RULES]
SOURCE_EXT = (".cu", ".cuh", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh", ".inl", ".inc")


def rewrite_text(text):
    out = text
    for rx, rep in COMPILED:
        out = rx.sub(rep, out)
    return out


def rewrite_tree(src_root, dst_root, verbose=False):
    """Mirror src_root into dst_root, rewriting source files; returns #changed files."""
    changed = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "build", "__pycache__", ".b70venv", "dist")
                       and not d.endswith(".egg-info")]
        rel = os.path.relpath(dirpath, src_root)
        dst_dir = os.path.join(dst_root, rel) if rel != "." else dst_root
        os.makedirs(dst_dir, exist_ok=True)
        for f in filenames:
            s, d = os.path.join(dirpath, f), os.path.join(dst_dir, f)
            if os.path.exists(d) and os.path.getmtime(d) >= os.path.getmtime(s):
                continue
            if f.endswith(SOURCE_EXT):
                try:
                    text = open(s, encoding="utf-8").read()
                except UnicodeDecodeError:
                    text = None
                if text is not None:
                    new = rewrite_text(text)
                    if new != text:
                        changed += 1
                        if verbose:
                            print("[b70torch] rewrote %s" % os.path.relpath(s, src_root), file=sys.stderr)
                    with open(d, "w", encoding="utf-8") as fh:
                        fh.write(new)
                    continue
            try:
                os.link(s, d)
            except OSError:
                import shutil
                shutil.copy2(s, d)
    return changed


def main(argv):
    for path in argv:
        text = open(path, encoding="utf-8").read()
        new = rewrite_text(text)
        if new == text:
            print("%s: unchanged" % path)
            continue
        import difflib
        sys.stdout.writelines(difflib.unified_diff(text.splitlines(True), new.splitlines(True), path, path + " (b70)"))


if __name__ == "__main__":
    main(sys.argv[1:])
