"""Fallback tensor bridge for comfy_aimdo."""

import torch


def hostbuf_to_tensor(hostbuf):
    size = max(0, int(getattr(hostbuf, "size", 0)))
    return torch.empty((size,), dtype=torch.uint8, pin_memory=True)


def aimdo_to_tensor(vbar_alloc, device):
    size = len(vbar_alloc) if vbar_alloc is not None else 0
    return torch.empty((size,), dtype=torch.uint8, device=device)

