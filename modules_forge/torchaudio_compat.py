"""
Compatibility shim for torchaudio.

Allows ComfyUI-master, text encoders (e.g. gemma4), and audio nodes (nodes_audio,
nodes_lt, nodes_wandancer, nodes_minimax_h3) to be imported and discovered
without crashing when torchaudio is not installed in the environment.
"""
import importlib.machinery
import sys
import types


class _TorchaudioStubMeta(type):
    def __getattr__(cls, name):
        return _make_stub(f"{cls.__name__}.{name}")

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class _TorchaudioStubObject(metaclass=_TorchaudioStubMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return _make_stub(f"{self.__class__.__name__}.{name}")

    def __iter__(self):
        return iter(())

    def __getitem__(self, item):
        return self


class _TorchaudioStubModule(types.ModuleType):
    def __init__(self, name: str):
        super().__init__(name)
        self.__name__ = name
        self.__file__ = f"<shim {name}>"
        self.__package__ = name.rpartition(".")[0] if "." in name else name
        self.__path__ = []
        self.__version__ = "2.13.0+shim"
        self.__spec__ = importlib.machinery.ModuleSpec(name, None)
        self._cache = {}

    def __getattr__(self, name):
        if name in self._cache:
            return self._cache[name]
        full_name = f"{self.__name__}.{name}"
        obj = _make_stub(full_name)
        self._cache[name] = obj
        setattr(self, name, obj)
        return obj

    def __call__(self, *args, **kwargs):
        return self


def _make_stub(full_name: str):
    parts = full_name.split(".")
    last_part = parts[-1]

    if last_part and last_part[0].isupper():
        cls_stub = type(
            last_part,
            (_TorchaudioStubObject,),
            {
                "__module__": ".".join(parts[:-1]) if len(parts) > 1 else "torchaudio",
                "__name__": last_part,
                "__qualname__": last_part,
            },
        )
        return cls_stub

    mod_stub = _TorchaudioStubModule(full_name)
    sys.modules[full_name] = mod_stub
    return mod_stub


_APPLIED = False


def apply():
    """Install torchaudio stub into sys.modules if real torchaudio is not available."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    try:
        import torchaudio  # noqa: F401
        return
    except Exception:
        pass

    root = _TorchaudioStubModule("torchaudio")
    functional = _TorchaudioStubModule("torchaudio.functional")
    transforms = _TorchaudioStubModule("torchaudio.transforms")
    compliance = _TorchaudioStubModule("torchaudio.compliance")
    sox_effects = _TorchaudioStubModule("torchaudio.sox_effects")
    pipelines = _TorchaudioStubModule("torchaudio.pipelines")

    root.functional = functional
    root.transforms = transforms
    root.compliance = compliance
    root.sox_effects = sox_effects
    root.pipelines = pipelines

    sys.modules["torchaudio"] = root
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio.transforms"] = transforms
    sys.modules["torchaudio.compliance"] = compliance
    sys.modules["torchaudio.sox_effects"] = sox_effects
    sys.modules["torchaudio.pipelines"] = pipelines
