"""
Compatibility shims for transformers 5+ and diffusers.

Applied lazily via __import__ hook:
  1. HybridCache -> DynamicCache alias (for peft 0.17.x)
  2. modeling_utils.no_init_weights -> local shim (backend/loader.py, etc.)
  3. Re-validate diffusers _flash_attn_available with a real import
     (find_spec alone cannot detect broken C extensions / DLL mismatch)

Important: apply() must not import transformers or diffusers at module load time.
All patches run inside the import hook.
"""
import builtins
from contextlib import contextmanager

_original_import = None
_no_init_weights_patched = False
_diffusers_import_utils_patched = False


@contextmanager
def _no_init_weights():
    """Shim compatible with transformers 4.x modeling_utils.no_init_weights().
    Disables torch.nn.init ops inside the context so weight init is skipped."""
    import torch

    _skip = {
        "uniform_": torch.nn.init.uniform_,
        "normal_": torch.nn.init.normal_,
        "trunc_normal_": torch.nn.init.trunc_normal_,
        "constant_": torch.nn.init.constant_,
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "xavier_normal_": torch.nn.init.xavier_normal_,
        "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
        "kaiming_normal_": torch.nn.init.kaiming_normal_,
        "orthogonal_": torch.nn.init.orthogonal_,
        "sparse_": torch.nn.init.sparse_,
        "zeros_": torch.nn.init.zeros_,
        "ones_": torch.nn.init.ones_,
        "dirac_": torch.nn.init.dirac_,
        "eye_": torch.nn.init.eye_,
    }

    def _noop(*args, **kwargs):
        pass

    try:
        for name in _skip:
            setattr(torch.nn.init, name, _noop)
        yield
    finally:
        for name, fn in _skip.items():
            setattr(torch.nn.init, name, fn)


def _patch_no_init_weights(mod):
    """Inject no_init_weights into modeling_utils if missing."""
    global _no_init_weights_patched
    if _no_init_weights_patched:
        return
    _no_init_weights_patched = True
    if not hasattr(mod, "no_init_weights"):
        mod.no_init_weights = _no_init_weights


def _patch_diffusers_flash_attn_flag(mod):
    """Re-validate diffusers _flash_attn_available with a real import.

    diffusers uses importlib.util.find_spec() for optional packages; that does not
    detect DLL/symbol mismatch in flash_attn_2_cuda. After a torch upgrade an old
    .pyd may remain: find_spec is True but import fails and startup crashes.

    This runs once right after diffusers.utils.import_utils loads. If
    _flash_attn_available is True, try import flash_attn; on failure set the flag
    to False.
    """
    global _diffusers_import_utils_patched
    if _diffusers_import_utils_patched:
        return
    _diffusers_import_utils_patched = True

    if not getattr(mod, "_flash_attn_available", False):
        return

    try:
        _original_import("flash_attn")
    except Exception:
        mod._flash_attn_available = False
        mod._flash_attn_version = "N/A"
        print("[transformers_cache_compat] flash_attn native extension failed to load; "
              "marked unavailable (will be reinstalled on next launch with --flash)")


def _hooked_import(name, *args, **kwargs):
    mod = _original_import(name, *args, **kwargs)

    # HybridCache patch
    if name == "transformers":
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if fromlist and "HybridCache" in fromlist and not hasattr(mod, "HybridCache"):
            from transformers.cache_utils import DynamicCache
            mod.__dict__["HybridCache"] = DynamicCache

    # no_init_weights patch
    if name == "transformers.modeling_utils":
        _patch_no_init_weights(mod)
    elif name == "transformers":
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if fromlist and "modeling_utils" in fromlist:
            import sys
            mu = sys.modules.get("transformers.modeling_utils")
            if mu is not None:
                _patch_no_init_weights(mu)

    # diffusers flash_attn availability patch
    if name == "diffusers.utils.import_utils":
        _patch_diffusers_flash_attn_flag(mod)

    return mod


def apply():
    """Call before importing nunchaku / peft / diffusers; does not import transformers."""
    global _original_import

    if _original_import is not None:
        return  # idempotent

    _original_import = builtins.__import__
    builtins.__import__ = _hooked_import
