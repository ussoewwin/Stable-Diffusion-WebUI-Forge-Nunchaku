"""Fallback host buffer object for comfy_aimdo."""


class HostBuffer:
    def __init__(self, size):
        self.size = max(0, int(size))
        self.data = bytearray(self.size)

