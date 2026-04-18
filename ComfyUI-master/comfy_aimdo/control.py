"""Fallback no-op control API for comfy_aimdo."""


def init():
    return False


def init_device(_device_index):
    return False


def set_log_debug():
    return None


def set_log_critical():
    return None


def set_log_error():
    return None


def set_log_warning():
    return None


def set_log_info():
    return None


def analyze():
    return None


def get_total_vram_usage():
    return 0

