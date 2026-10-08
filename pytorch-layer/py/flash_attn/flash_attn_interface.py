"""flash_attn.flash_attn_interface API on top of torch SDPA (see package docstring)."""
import torch
import torch.nn.functional as F


def _unsupported(what):
    raise NotImplementedError(
        "flash_attn shim (b70): %s is not supported; call "
        "torch.nn.functional.scaled_dot_product_attention directly" % what)


def _check(alibi_slopes, return_attn_probs, softcap=0.0):
    if alibi_slopes is not None:
        _unsupported("alibi_slopes")
    if return_attn_probs:
        _unsupported("return_attn_probs=True")
    if softcap:
        _unsupported("softcap")


def _attend(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
            window_size=(-1, -1)):
    """q (B, Sq, H, D), k/v (B, Sk, Hk, D) -> (B, Sq, H, D)."""
    sq, h = q.shape[1], q.shape[2]
    sk, hk = k.shape[1], k.shape[2]
    if hk != h:  # GQA / MQA: expand so SDPA can stay on its fused kernel
        rep = h // hk
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    left, right = window_size if window_size is not None else (-1, -1)
    if causal:
        right = 0
    mask = None
    is_causal = False
    if left >= 0 or right >= 0:
        if causal and left < 0 and sq == sk:
            is_causal = True
        else:
            # flash_attn aligns the causal/window band to the bottom-right corner
            i = torch.arange(sq, device=q.device).unsqueeze(1) + (sk - sq)
            j = torch.arange(sk, device=q.device).unsqueeze(0)
            mask = torch.ones(sq, sk, dtype=torch.bool, device=q.device)
            if right >= 0:
                mask &= j <= i + right
            if left >= 0:
                mask &= j >= i - left
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal,
        scale=softmax_scale)
    return out.transpose(1, 2)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False, **_):
    _check(alibi_slopes, return_attn_probs, softcap)
    return _attend(q, k, v, dropout_p, softmax_scale, causal, window_size)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None,
                              causal=False, window_size=(-1, -1), softcap=0.0,
                              alibi_slopes=None, deterministic=False,
                              return_attn_probs=False, **_):
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size,
                           softcap, alibi_slopes, deterministic, return_attn_probs)


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None,
                             causal=False, window_size=(-1, -1), softcap=0.0,
                             alibi_slopes=None, deterministic=False,
                             return_attn_probs=False, **_):
    k, v = kv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size,
                           softcap, alibi_slopes, deterministic, return_attn_probs)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                           max_seqlen_k, dropout_p=0.0, softmax_scale=None,
                           causal=False, window_size=(-1, -1), softcap=0.0,
                           alibi_slopes=None, deterministic=False,
                           return_attn_probs=False, block_table=None, **_):
    """q (total_q, H, D), k/v (total_k, Hk, D), cu_seqlens int32 prefix sums."""
    _check(alibi_slopes, return_attn_probs, softcap)
    if block_table is not None:
        _unsupported("block_table (paged KV)")
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    n = len(cq) - 1
    lq = {cq[i + 1] - cq[i] for i in range(n)}
    lk = {ck[i + 1] - ck[i] for i in range(n)}
    if len(lq) == 1 and len(lk) == 1:  # equal lengths: one batched call
        sq, sk = lq.pop(), lk.pop()
        out = _attend(q[cq[0]:cq[-1]].reshape(n, sq, *q.shape[1:]),
                      k[ck[0]:ck[-1]].reshape(n, sk, *k.shape[1:]),
                      v[ck[0]:ck[-1]].reshape(n, sk, *v.shape[1:]),
                      dropout_p, softmax_scale, causal, window_size)
        return out.reshape(n * sq, *out.shape[2:])
    outs = []
    for i in range(n):
        outs.append(_attend(q[cq[i]:cq[i + 1]].unsqueeze(0),
                            k[ck[i]:ck[i + 1]].unsqueeze(0),
                            v[ck[i]:ck[i + 1]].unsqueeze(0),
                            dropout_p, softmax_scale, causal, window_size)[0])
    return torch.cat(outs, dim=0)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0,
                                     softmax_scale=None, causal=False,
                                     window_size=(-1, -1), softcap=0.0,
                                     alibi_slopes=None, deterministic=False,
                                     return_attn_probs=False, **_):
    q, k, v = qkv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen,
                                  max_seqlen, dropout_p, softmax_scale, causal,
                                  window_size, softcap, alibi_slopes,
                                  deterministic, return_attn_probs)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k,
                                    max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                                    softmax_scale=None, causal=False,
                                    window_size=(-1, -1), softcap=0.0,
                                    alibi_slopes=None, deterministic=False,
                                    return_attn_probs=False, **_):
    k, v = kv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                  max_seqlen_q, max_seqlen_k, dropout_p,
                                  softmax_scale, causal, window_size, softcap,
                                  alibi_slopes, deterministic, return_attn_probs)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, rotary_cos=None,
                            rotary_sin=None, cache_seqlens=None,
                            cache_batch_idx=None, cache_leftpad=None,
                            block_table=None, softmax_scale=None, causal=False,
                            window_size=(-1, -1), softcap=0.0,
                            rotary_interleaved=True, alibi_slopes=None,
                            num_splits=0, return_softmax_lse=False, **_):
    """q (B, Sq, H, D); caches (B, S_max, Hk, D). Appends k/v in place, then attends."""
    _check(alibi_slopes, return_softmax_lse, softcap)
    if rotary_cos is not None or rotary_sin is not None:
        _unsupported("rotary inside flash_attn_with_kvcache")
    if block_table is not None or cache_batch_idx is not None or cache_leftpad is not None:
        _unsupported("block_table / cache_batch_idx / cache_leftpad")
    b = q.shape[0]
    if cache_seqlens is None:
        lens = [k_cache.shape[1] if k is None else 0] * b
    elif isinstance(cache_seqlens, int):
        lens = [cache_seqlens] * b
    else:
        lens = cache_seqlens.tolist()
    new = 0 if k is None else k.shape[1]
    outs = []
    for i in range(b):
        start = lens[i]
        if k is not None:
            k_cache[i, start:start + new] = k[i]
            v_cache[i, start:start + new] = v[i]
        end = start + new
        outs.append(_attend(q[i:i + 1], k_cache[i:i + 1, :end], v_cache[i:i + 1, :end],
                            0.0, softmax_scale, causal, window_size))
    return torch.cat(outs, dim=0)
