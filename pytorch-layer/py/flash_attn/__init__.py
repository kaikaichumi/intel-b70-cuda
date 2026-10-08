"""flash_attn look-alike for Intel XPU, built on torch's scaled_dot_product_attention.

Many model repos (Wan, HunyuanVideo, Qwen-VL demos, ...) `import flash_attn`
unconditionally. On the Arc Pro B70 (Xe2) torch's SDPA already runs a fused
flash-attention kernel for fp16/bf16, so this package only translates the
flash_attn calling convention -- (batch, seq, heads, dim) layout, varlen
packing, bottom-right causal alignment, sliding windows, GQA -- onto it.

Not supported (raise NotImplementedError): alibi_slopes, return_attn_probs,
softcap, paged KV (block_table), rotary inside flash_attn_with_kvcache.
"""
from .flash_attn_interface import (  # noqa: F401
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

__version__ = "2.8.3"
B70_SDPA_SHIM = True
