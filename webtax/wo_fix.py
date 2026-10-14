"""WebOrganizer's custom gte ('new') code under transformers 5: non-persistent buffers (position_ids, RoPE cos/sin caches) come out
uninitialised (meta-device loading). Rebuild them after from_pretrained."""
import torch


def fix_buffers(m):
    for mod in m.modules():
        if hasattr(mod, "position_ids") and isinstance(getattr(mod, "position_ids"), torch.Tensor):
            n = mod.position_ids.shape[-1]; mod.position_ids = torch.arange(n, device=mod.position_ids.device).expand_as(mod.position_ids).clone()
        if hasattr(mod, "_set_cos_sin_cache"):
            dev = next((b.device for b in mod.buffers()), torch.device("cpu"))
            inv = getattr(mod, "inv_freq", None)
            if inv is not None and hasattr(mod, "base") and hasattr(mod, "dim"):
                mod.inv_freq = 1.0 / (mod.base ** (torch.arange(0, mod.dim, 2, device=dev).float() / mod.dim))
            mod._set_cos_sin_cache(seq_len=getattr(mod, "max_seq_len_cached", getattr(mod, "max_position_embeddings", 8192)), device=dev, dtype=torch.float32)
    return m


def _ext_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
    """transformers 4.x ModuleUtilsMixin.get_extended_attention_mask (removed in 5): [bs, seq] -> additive [bs, 1, 1, seq] mask."""
    dtype = dtype or next(self.parameters()).dtype
    ext = attention_mask[:, None, None, :] if attention_mask.dim() == 2 else attention_mask[:, None, :, :]
    ext = ext.to(dtype=dtype)
    return (1.0 - ext) * torch.finfo(dtype).min


def patch(m):
    for mod in m.modules():
        if type(mod).__name__ == "NewModel" and not hasattr(type(mod), "get_extended_attention_mask"):
            type(mod).get_extended_attention_mask = _ext_mask
    return fix_buffers(m)
