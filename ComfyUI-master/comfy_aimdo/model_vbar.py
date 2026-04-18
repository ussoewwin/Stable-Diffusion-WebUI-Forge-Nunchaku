"""Fallback in-memory VBAR implementation for comfy_aimdo."""


class ModelVBAR:
    def __init__(self, _capacity_bytes, _device_index):
        self._loaded = 0

    def prioritize(self):
        return None

    def alloc(self, size):
        self._loaded += int(size)
        return bytearray(max(0, int(size)))

    def loaded_size(self):
        return int(self._loaded)

    def free_memory(self, target):
        target = max(0, int(target))
        freed = min(self._loaded, target)
        self._loaded -= freed
        return freed


def vbars_analyze():
    return 0


def vbar_fault(_v):
    return None


def vbar_signature_compare(_signature, _stored_signature):
    return False


def vbar_unpin(_v):
    return None


def vbars_reset_watermark_limits():
    return None

