"""flash_attn.bert_padding in plain torch (the original is plain torch too)."""
import torch
import torch.nn.functional as F


def index_first_axis(x, indices):
    return x[indices]


def index_put_first_axis(values, indices, first_axis_dim):
    out = torch.zeros((first_axis_dim, *values.shape[1:]), device=values.device,
                      dtype=values.dtype)
    out[indices] = values
    return out


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Returns (hidden, indices, cu_seqlens, max_seqlen, used_seqlens) like flash_attn >= 2.7."""
    all_masks = attention_mask + unused_mask if unused_mask is not None else attention_mask
    seqlens = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    return index_first_axis(flat, indices), indices, cu_seqlens, max_seqlen, used_seqlens


def pad_input(hidden_states, indices, batch, seqlen):
    out = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return out.reshape(batch, seqlen, *out.shape[1:])
