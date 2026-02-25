# Stub: real comfy_aimdo provides VBAR for dynamic weight offloading. Forge does not use it.


def vbar_fault(_v):
    """Return None so ComfyUI uses the normal (non-AIMDO) code path."""
    return None


def vbar_signature_compare(signature, stored):
    """Stub: always False when signature is not None."""
    return False


def vbar_unpin(_v):
    """No-op when AIMDO is not in use."""
    pass


def vbars_reset_watermark_limits():
    """No-op stub for execution.py."""
    pass


class ModelVBAR:
    """Stub: real ModelVBAR manages virtual weight buffers. Forge uses its own loader."""

    def __init__(self, size, device_index):
        self._size = size
        self._device_index = device_index

    def loaded_size(self):
        return 0
