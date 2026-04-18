# Stub: real comfy_aimdo provides AIMDO tensor conversion. Forge does not use AIMDO.
import torch


def aimdo_to_tensor(v_handle, device):
    """Stub: return a tensor on device. Not used when vbar_fault returns None."""
    if isinstance(v_handle, torch.Tensor):
        return v_handle.to(device=device)
    size = len(v_handle) if v_handle is not None and hasattr(v_handle, "__len__") else 0
    return torch.empty((size,), dtype=torch.uint8, device=device)


def hostbuf_to_tensor(hostbuf):
    size = max(0, int(getattr(hostbuf, "size", 0)))
    return torch.empty((size,), dtype=torch.uint8, pin_memory=True)
