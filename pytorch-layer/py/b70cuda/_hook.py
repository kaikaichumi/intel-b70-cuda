"""Patch torch the moment it is imported, without importing it ourselves.

b70cuda.pth runs `import b70cuda._hook` at interpreter start-up. Importing torch
there would add ~2 s to every Python process, so instead a meta-path finder
waits for the first `import torch`, lets it finish, then calls b70cuda.patch().
"""
import importlib.abc
import importlib.util
import os
import sys


class _TorchImportHook(importlib.abc.MetaPathFinder):
    """Stays on sys.meta_path until torch has actually been executed.

    Libraries probe for torch with importlib.util.find_spec("torch") before
    importing it (transformers' is_torch_available does). A one-shot finder
    that left after the first find_spec call would fire on the probe and miss
    the real import, so the finder is only removed from exec_and_patch.
    """
    _busy = False

    def find_spec(self, name, path=None, target=None):
        if name != "torch" or self._busy:
            return None
        if "torch" in sys.modules:  # imported behind our back; patch() handles that
            self._retire()
            return None
        self._busy = True  # the find_spec below walks sys.meta_path again
        try:
            spec = importlib.util.find_spec("torch")
        finally:
            self._busy = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            self._retire()
            try:
                from b70cuda import patch
                patch(module)
            except Exception as e:  # never break `import torch`
                print("[b70cuda] not active: %r" % (e,), file=sys.stderr)

        spec.loader.exec_module = exec_and_patch
        return spec

    def _retire(self):
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass


def _install():
    if os.environ.get("B70_CUDA", "1").lower() in ("0", "off", "false", "no"):
        return
    if "torch" in sys.modules:
        from b70cuda import patch
        patch(sys.modules["torch"])
        return
    sys.meta_path.insert(0, _TorchImportHook())


_install()
