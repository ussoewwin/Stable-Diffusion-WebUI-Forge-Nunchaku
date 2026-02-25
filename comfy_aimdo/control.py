# Stub: real comfy_aimdo.control provides AIMDO device/VRAM control. Forge does not use it.


def init():
    pass


def init_device(device_index):
    return False


def get_total_vram_usage():
    """Return 0 so comfy.windows.get_free_ram() falls back to normal calculation."""
    return 0


def set_log_debug():
    pass


def set_log_critical():
    pass


def set_log_error():
    pass


def set_log_warning():
    pass


def set_log_info():
    pass


def analyze():
    pass
