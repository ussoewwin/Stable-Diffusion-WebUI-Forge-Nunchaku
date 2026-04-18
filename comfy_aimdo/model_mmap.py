# Stub memory map wrapper used by comfy.utils.load_safetensors
import mmap


class ModelMMAP:
    def __init__(self, path):
        self._fh = open(path, "rb")
        self._map = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def get(self):
        return self._map

    def __del__(self):
        try:
            self._map.close()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass

