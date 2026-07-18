"""Forge-side fallback when comfy_kitchen layouts are unavailable.

Issue #3 (TimothyDudorov): Krea2 quantized TE load crashes in
``comfy.ops._load_quantized_module``:

    layout_cls = get_layout_class(...)  # None when kitchen missing/stub
    params = layout_cls.Params(...)     # AttributeError

Do not edit ``ComfyUI-master/``. Patch ``comfy.ops`` at runtime so MixedPrecision
Linear (and Embedding) dequantize to float Parameters when layout registration
is missing. TE path uses ``full_precision_mm=True`` already, so float weights
are correct for inference.
"""

from __future__ import annotations

import json
import logging

import torch

_PATCHED = False
_LOG = logging.getLogger("forge.quant_layout_fallback")


def _dequant_to_compute(weight: torch.Tensor, quant_format: str, scales: dict, compute_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize stored weight to ``compute_dtype`` without kitchen layouts."""
    w = weight
    if quant_format in ("float8_e4m3fn", "float8_e5m2"):
        out = w.to(dtype=torch.float32)
        scale = scales.get("scale")
        if scale is not None:
            out = out * scale.to(device=out.device, dtype=torch.float32)
        return out.to(dtype=compute_dtype)

    if quant_format == "int8_tensorwise":
        out = w.to(dtype=torch.float32)
        scale = scales.get("scale")
        if scale is None:
            raise ValueError("Missing INT8 weight_scale for Forge dequant fallback")
        s = scale.to(device=out.device, dtype=torch.float32)
        if s.ndim == 0:
            out = out * s
        elif s.ndim == 1:
            out = out * s.reshape(-1, *([1] * (out.ndim - 1)))
        else:
            out = out * s
        return out.to(dtype=compute_dtype)

    raise RuntimeError(
        f"[Forge] comfy_kitchen layout unavailable for format={quant_format!r}. "
        "Install/update comfy_kitchen, or use a non-quantized checkpoint for this TE/UNet."
    )


def _forge_dequant_load_quantized_module(
    module,
    super_load,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
    load_extra_params=False,
):
    """Same key consumption as ``_load_quantized_module``, but float Parameter output."""
    from comfy.quant_ops import QUANT_ALGOS

    device = module.factory_kwargs["device"]
    compute_dtype = module.factory_kwargs["dtype"]
    layer_name = prefix.rstrip(".")

    weight = state_dict.pop(f"{prefix}weight", None)
    if weight is None:
        logging.warning("Missing weight for layer %s", layer_name)
        module.weight = None
        return
    manually_loaded_keys = [f"{prefix}weight"]

    def pop_scale(name, dtype=None):
        key = f"{prefix}{name}"
        v = state_dict.pop(key, None)
        if v is not None:
            v = v.to(device=device)
            if dtype is not None:
                v = v.view(dtype=dtype)
            manually_loaded_keys.append(key)
        return v

    layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
    if layer_conf is None:
        module.weight = torch.nn.Parameter(weight.to(device=device, dtype=compute_dtype), requires_grad=False)
        super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        for key in manually_loaded_keys:
            if key in missing_keys:
                missing_keys.remove(key)
        return

    layer_conf = json.loads(layer_conf.numpy().tobytes())
    quant_format = layer_conf.get("format", None)
    if quant_format is None:
        raise ValueError(f"Unknown quantization format for layer {layer_name}")
    if quant_format not in QUANT_ALGOS:
        raise ValueError(f"Unsupported quantization format: {quant_format}")

    scales = {}
    if quant_format in ("float8_e4m3fn", "float8_e5m2"):
        scales = {"scale": pop_scale("weight_scale")}
    elif quant_format == "int8_tensorwise":
        scale = pop_scale("weight_scale")
        if scale is None:
            raise ValueError(f"Missing INT8 weight scale for layer {layer_name}")
        scales = {"scale": scale}
        # drop unused scale keys that may exist for other formats
        pop_scale("weight_scale_2")
        pop_scale("input_scale")
    elif quant_format == "mxfp8":
        raise RuntimeError(
            f"[Forge] comfy_kitchen required for mxfp8 layer {layer_name} (no float dequant fallback)."
        )
    elif quant_format == "nvfp4":
        raise RuntimeError(
            f"[Forge] comfy_kitchen required for nvfp4 layer {layer_name} (no float dequant fallback)."
        )
    elif quant_format == "convrot_w4a4":
        raise RuntimeError(
            f"[Forge] comfy_kitchen required for convrot_w4a4 layer {layer_name} (no float dequant fallback)."
        )
    else:
        raise ValueError(f"Unsupported quantization format: {quant_format}")

    if load_extra_params:
        qconfig = QUANT_ALGOS[quant_format]
        for param_name in qconfig["parameters"]:
            if param_name in {"weight_scale", "weight_scale_2"}:
                continue
            param_key = f"{prefix}{param_name}"
            _v = state_dict.pop(param_key, None)
            if _v is None:
                continue
            module.register_parameter(param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False))
            manually_loaded_keys.append(param_key)

    print(
        f"[Forge] comfy_kitchen layout missing for {quant_format} — "
        f"dequantizing {layer_name} to {compute_dtype} (Issue #3 fallback)"
    )
    dequant = _dequant_to_compute(weight, quant_format, scales, compute_dtype)
    module.weight = torch.nn.Parameter(dequant.to(device=device), requires_grad=False)
    module.layout_type = None
    module.quant_format = None
    module._full_precision_mm = True
    module._full_precision_mm_config = True

    super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    for key in manually_loaded_keys:
        if key in missing_keys:
            missing_keys.remove(key)


def _wrap_load_quantized_module(orig):
    def _load_quantized_module(
        module,
        super_load,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        load_extra_params=False,
    ):
        from comfy.quant_ops import QUANT_ALGOS, get_layout_class

        quant_key = f"{prefix}comfy_quant"
        if quant_key in state_dict:
            try:
                conf = json.loads(state_dict[quant_key].numpy().tobytes())
                fmt = conf.get("format")
                layout_name = QUANT_ALGOS.get(fmt, {}).get("comfy_tensor_layout")
                if layout_name is not None and get_layout_class(layout_name) is None:
                    return _forge_dequant_load_quantized_module(
                        module,
                        super_load,
                        state_dict,
                        prefix,
                        local_metadata,
                        strict,
                        missing_keys,
                        unexpected_keys,
                        error_msgs,
                        load_extra_params=load_extra_params,
                    )
            except Exception as e:
                _LOG.debug("layout peek failed for %s: %s", prefix, e)

        return orig(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=load_extra_params,
        )

    return _load_quantized_module


def _wrap_embedding_load(orig_load, compute_dtype: torch.dtype):
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        from comfy.quant_ops import QUANT_ALGOS, get_layout_class

        weight_key = f"{prefix}weight"
        quant_key = f"{prefix}comfy_quant"
        if quant_key in state_dict and weight_key in state_dict:
            try:
                layer_conf = json.loads(state_dict[quant_key].numpy().tobytes())
                quant_format = layer_conf.get("format") if layer_conf is not None else None
                if quant_format in ("float8_e4m3fn", "float8_e5m2"):
                    layout_name = QUANT_ALGOS[quant_format]["comfy_tensor_layout"]
                    if get_layout_class(layout_name) is None:
                        state_dict.pop(quant_key)
                        weight = state_dict.pop(weight_key)
                        scale_key = f"{prefix}weight_scale"
                        scale = state_dict.pop(scale_key, None)
                        device = weight.device
                        scales = {"scale": scale.to(device=device) if scale is not None else None}
                        dequant = _dequant_to_compute(weight, quant_format, scales, compute_dtype)
                        self.weight = torch.nn.Parameter(dequant, requires_grad=False)
                        self.layout_type = None
                        self.quant_format = None
                        print(
                            f"[Forge] comfy_kitchen layout missing for {quant_format} — "
                            f"dequantizing embedding {prefix.rstrip('.')} to {compute_dtype} (Issue #3 fallback)"
                        )
                        torch.nn.Module._load_from_state_dict(
                            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
                        )
                        for k in (weight_key, scale_key, quant_key):
                            if k in missing_keys:
                                missing_keys.remove(k)
                        return
            except Exception as e:
                _LOG.debug("embedding layout peek failed for %s: %s", prefix, e)

        return orig_load(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    return _load_from_state_dict


def _wrap_mixed_precision_ops(orig_mp):
    def mixed_precision_ops(*args, **kwargs):
        MP = orig_mp(*args, **kwargs)
        emb = getattr(MP, "Embedding", None)
        if emb is not None and not getattr(emb, "_forge_layout_fallback", False):
            cd = getattr(MP, "_compute_dtype", torch.bfloat16)
            emb._load_from_state_dict = _wrap_embedding_load(emb._load_from_state_dict, cd)
            emb._forge_layout_fallback = True
        return MP

    return mixed_precision_ops


def ensure_comfy_quant_layout_fallback_patch() -> bool:
    """Idempotent. Returns True when patch is active."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        import comfy.ops as ops
    except ImportError as e:
        print(f"[Forge] quant layout fallback: comfy.ops not importable yet ({e})")
        return False

    if getattr(ops, "_forge_layout_fallback_patched", False):
        _PATCHED = True
        return True

    if not hasattr(ops, "_load_quantized_module"):
        print("[Forge] quant layout fallback: _load_quantized_module missing")
        return False

    ops._load_quantized_module = _wrap_load_quantized_module(ops._load_quantized_module)
    if hasattr(ops, "mixed_precision_ops"):
        ops.mixed_precision_ops = _wrap_mixed_precision_ops(ops.mixed_precision_ops)

    ops._forge_layout_fallback_patched = True
    _PATCHED = True
    print("[Forge] comfy_quant layout fallback patch active (dequant when kitchen layout is None)")
    return True
