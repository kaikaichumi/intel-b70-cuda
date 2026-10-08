"""sageattention look-alike for Intel XPU, built on torch's scaled_dot_product_attention.

SageAttention 2.x is CUDA-only (inline PTX, sm80+ kernels).  Its 1.0.6 Triton edition
does run on the B70 through triton-xpu, but measured 17x slower than torch's SDPA there
(31 ms vs 1.8 ms for a 4k-token attention; docs/03-roadmap.md, M4 ③), because Triton on
Xe2 has no good matmul lowering yet while SDPA is a hand-tuned fused kernel.  So the
`b70 install` filter drops sageattention and this package provides its calling
convention on top of SDPA: `sageattn(q, k, v, tensor_layout="HND"|"NHD", is_causal,
sm_scale, return_lse)` plus the per-kernel entry points of 2.x, which all map to the
same function here.  The INT8/FP8 quantisation is not reproduced: results are the exact
fp16/bf16 attention, i.e. at least as accurate as SageAttention's.

Not supported (raise NotImplementedError): return_lse=True, sageattn_varlen.
"""
import torch
import torch.nn.functional as F

__version__ = "2.2.0"
B70_SDPA_SHIM = True


def _to_hnd(x, layout):
    if layout == "HND":
        return x
    if layout == "NHD":
        return x.transpose(1, 2)
    raise ValueError("tensor_layout must be 'HND' or 'NHD', got %r" % (layout,))


def sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None,
             return_lse=False, attn_mask=None, dropout_p=0.0, **_ignored):
    """SageAttention's main entry point. q/k/v: (B, H, N, D) for "HND" or (B, N, H, D) for
    "NHD"; fp16 or bf16; output has q's layout and dtype.  Extra keyword arguments of the
    various SageAttention versions (smooth_k, qk_quant_gran, pv_accum_dtype, ...) are
    accepted and ignored: they select quantisation details that do not exist here."""
    if return_lse:
        raise NotImplementedError("b70 sageattention shim: return_lse is not supported")
    qh, kh, vh = _to_hnd(q, tensor_layout), _to_hnd(k, tensor_layout), _to_hnd(v, tensor_layout)
    if kh.shape[1] != qh.shape[1]:  # GQA / MQA: SageAttention requires H % Hkv == 0 too
        rep = qh.shape[1] // kh.shape[1]
        kh = kh.repeat_interleave(rep, dim=1)
        vh = vh.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask, dropout_p=dropout_p,
                                         is_causal=is_causal, scale=sm_scale)
    return out if tensor_layout == "HND" else out.transpose(1, 2)


# SageAttention 2.x exposes the individual kernels; they differ only in quantisation.
sageattn_qk_int8_pv_fp16_triton = sageattn
sageattn_qk_int8_pv_fp16_cuda = sageattn
sageattn_qk_int8_pv_fp8_cuda = sageattn
sageattn_qk_int8_pv_fp8_cuda_sm90 = sageattn


def sageattn_varlen(*args, **kwargs):
    raise NotImplementedError("b70 sageattention shim: sageattn_varlen is not supported; "
                              "use flash_attn.flash_attn_varlen_func (also shimmed on SDPA)")


__all__ = ["sageattn", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp16_cuda",
           "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda_sm90", "sageattn_varlen"]
