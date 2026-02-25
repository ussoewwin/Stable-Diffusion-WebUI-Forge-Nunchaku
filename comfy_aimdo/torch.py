# Stub: real comfy_aimdo provides AIMDO tensor conversion. Forge does not use AIMDO.
import torch


def aimdo_to_tensor(v_handle, device):
    """Stub: return a tensor on device. Not used when vbar_fault returns None."""
    if isinstance(v_handle, torch.Tensor):
        return v_handle.to(device=device)
    raise RuntimeError("comfy_aimdo stub: aimdo_to_tensor called without real comfy_aimdo")
