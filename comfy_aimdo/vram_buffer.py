# Stub vram buffer for comfy_aimdo optional dependency.

class VRAMBuffer:
    def __init__(self, size, device_index):
        self._size = max(0, int(size))
        self._device_index = device_index

    def size(self):
        return self._size

    def get(self, size, offset):
        return bytearray(max(0, int(size)))
