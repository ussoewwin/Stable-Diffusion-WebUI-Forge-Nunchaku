"""
transformers 5+ 互換パッチ

transformers 5.x で削除・変更された API をモンキーパッチで復元:
  1. HybridCache  → DynamicCache エイリアス (peft 0.17.x 向け)
  2. modeling_utils.no_init_weights → 自前シム (backend/loader.py 等で使用)

重要: apply() は transformers を一切インポートしない。
全パッチは __import__ フック内で遅延適用される。
"""
import builtins
from contextlib import contextmanager

_original_import = None
_no_init_weights_patched = False


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


def _hooked_import(name, *args, **kwargs):
    mod = _original_import(name, *args, **kwargs)

    # --- HybridCache パッチ ---
    if name == "transformers":
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if fromlist and "HybridCache" in fromlist and not hasattr(mod, "HybridCache"):
            from transformers.cache_utils import DynamicCache
            mod.__dict__["HybridCache"] = DynamicCache

    # --- no_init_weights パッチ ---
    # `import transformers.modeling_utils` → name="transformers.modeling_utils"
    if name == "transformers.modeling_utils":
        _patch_no_init_weights(mod)
    # `from transformers import modeling_utils` → name="transformers", fromlist=("modeling_utils",)
    elif name == "transformers":
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if fromlist and "modeling_utils" in fromlist:
            import sys
            mu = sys.modules.get("transformers.modeling_utils")
            if mu is not None:
                _patch_no_init_weights(mu)

    return mod


def apply():
    """nunchaku / peft / diffusers のインポート前に呼ぶこと。transformers は一切インポートしない。"""
    global _original_import

    if _original_import is not None:
        return  # 二重適用防止

    _original_import = builtins.__import__
    builtins.__import__ = _hooked_import
