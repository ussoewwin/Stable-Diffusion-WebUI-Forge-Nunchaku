"""
transformers 5+ / diffusers 互換パッチ

__import__ フックで遅延適用:
  1. HybridCache  → DynamicCache エイリアス (peft 0.17.x 向け)
  2. modeling_utils.no_init_weights → 自前シム (backend/loader.py 等で使用)
  3. diffusers _flash_attn_available を実 import で検証
     (find_spec だけでは C 拡張の DLL 不整合を検出できない)

重要: apply() は transformers / diffusers を一切インポートしない。
全パッチは __import__ フック内で遅延適用される。
"""
import builtins
from contextlib import contextmanager

_original_import = None
_no_init_weights_patched = False
_diffusers_import_utils_patched = False


@contextmanager
def _no_init_weights():
    """transformers 4.x の modeling_utils.no_init_weights() 互換シム。
    コンテキスト内で torch.nn.init 系の関数を無効にし、重み初期化をスキップする。"""
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
    """modeling_utils モジュールに no_init_weights が無ければ注入する。"""
    global _no_init_weights_patched
    if _no_init_weights_patched:
        return
    _no_init_weights_patched = True
    if not hasattr(mod, "no_init_weights"):
        mod.no_init_weights = _no_init_weights


def _patch_diffusers_flash_attn_flag(mod):
    """diffusers の _flash_attn_available を実 import で再検証する。

    diffusers は importlib.util.find_spec() でパッケージの有無を判定するが、
    これでは C 拡張 (flash_attn_2_cuda) の DLL/シンボル不整合を検出できない。
    torch バージョン更新後に旧ビルドの .pyd が残っていると、
    find_spec=True なのに実 import で ImportError が発生し起動クラッシュする。

    このパッチは diffusers.utils.import_utils 読み込み直後に一度だけ走り、
    _flash_attn_available が True なら実際に import して検証する。
    失敗した場合のみフラグを False に修正する。
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

    # --- HybridCache パッチ ---
    if name == "transformers":
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if fromlist and "HybridCache" in fromlist and not hasattr(mod, "HybridCache"):
            from transformers.cache_utils import DynamicCache
            mod.__dict__["HybridCache"] = DynamicCache

    # --- no_init_weights パッチ ---
    if name == "transformers.modeling_utils":
        _patch_no_init_weights(mod)
    elif name == "transformers":
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if fromlist and "modeling_utils" in fromlist:
            import sys
            mu = sys.modules.get("transformers.modeling_utils")
            if mu is not None:
                _patch_no_init_weights(mu)

    # --- diffusers flash_attn 可用性パッチ ---
    if name == "diffusers.utils.import_utils":
        _patch_diffusers_flash_attn_flag(mod)

    return mod


def apply():
    """nunchaku / peft / diffusers のインポート前に呼ぶこと。transformers は一切インポートしない。"""
    global _original_import

    if _original_import is not None:
        return  # 二重適用防止

    _original_import = builtins.__import__
    builtins.__import__ = _hooked_import
