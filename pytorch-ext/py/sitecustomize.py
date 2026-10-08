# Python imports this automatically when pytorch-ext/py is on PYTHONPATH, so
# pip's build subprocesses (`pip install --no-build-isolation <cuda package>`)
# get the b70torch hook too. Inside the b70-ai image the same hook is installed
# as b70torch.pth and this file is not needed.
try:
    import b70torch._hook  # noqa: F401
except Exception:
    pass
