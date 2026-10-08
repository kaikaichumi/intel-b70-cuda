# b70-ai: a CUDA-style PyTorch environment for the Intel Arc Pro B70.
#
# Built on the vLLM XPU image already on this box, so torch 2.13+xpu, triton-xpu
# and the oneAPI runtime are the exact builds that are known to work on the B70,
# and the 16 GB of base layers are shared instead of downloaded again.
# `b70 build` tags the current vLLM image as b70-base:<date> first.
ARG BASE=b70-base:latest
FROM ${BASE}

USER root
ENTRYPOINT ["/opt/b70/entry.sh"]
CMD ["bash"]
WORKDIR /

# Freeze the XPU builds: any later pip install that wants a different torch /
# triton fails loudly instead of quietly swapping in a CUDA wheel.
RUN mkdir -p /opt/b70 && \
    pip freeze | grep -E '^(torch|torchvision|torchaudio|triton|triton-xpu|intel-[a-z-]+)==' \
      > /opt/b70/constraints.txt && cat /opt/b70/constraints.txt
ENV PIP_CONSTRAINT=/opt/b70/constraints.txt \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# What image / video / audio model repos usually expect to find already there.
RUN pip install --no-cache-dir \
      diffusers peft gguf ftfy imageio-ffmpeg av torchsde kornia spandrel \
      omegaconf gradio bitsandbytes

# The CUDA -> XPU layer and the SDPA-based flash_attn.
# (build context is the repository root)
COPY pytorch-layer/py/b70cuda /opt/venv/lib/python3.12/site-packages/b70cuda
COPY pytorch-layer/py/b70cuda.pth /opt/venv/lib/python3.12/site-packages/b70cuda.pth
COPY pytorch-layer/py/flash_attn /opt/venv/lib/python3.12/site-packages/flash_attn
COPY pytorch-layer/py/flash_attn-2.8.3.dist-info /opt/venv/lib/python3.12/site-packages/flash_attn-2.8.3.dist-info
COPY pytorch-layer/py/sageattention /opt/venv/lib/python3.12/site-packages/sageattention
COPY pytorch-layer/py/sageattention-2.2.0.dist-info /opt/venv/lib/python3.12/site-packages/sageattention-2.2.0.dist-info
COPY --chmod=755 pytorch-layer/entry.sh /opt/b70/entry.sh
# b70torch: builds CUDA extensions (CUDAExtension / nvcc) with the b70 toolchain
COPY pytorch-ext /opt/b70/pytorch-ext
# CUB (over hipCUB) and thrust/complex.h shims, shared with b70cc
COPY toolchain/include /opt/b70/toolchain/include
RUN printf '%s\n%s\n' /opt/b70/pytorch-ext/py 'import b70torch._hook' \
      > /opt/venv/lib/python3.12/site-packages/b70torch.pth && \
    rm -f /opt/b70/pytorch-ext/py/sitecustomize.py
# No chmod -R on /opt/venv here: touching every file copies the whole venv
# (~10 GB) into the build cache.
RUN python -c "import flash_attn, b70cuda.pipfilter, b70torch; print('shims ok')"

ENV ZE_AFFINITY_MASK=0 \
    NEO_CACHE_PERSISTENT=1 \
    SYCL_CACHE_PERSISTENT=1 \
    B70_IMAGE=1
