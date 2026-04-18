# Stub host buffer for comfy_aimdo optional dependency.


class HostBuffer:
    def __init__(self, size):
        self.size = max(0, int(size))
        self.data = bytearray(self.size)

