# Flash-Attention / SageAttention visibility + runtime switch for Comfy-path models (Krea2 etc.).
# Detection helpers mirror ComfyUI-DistorchMemoryManager nodes/sa.py.
# FA2 UI path follows A1111 md/FA2_direct_load_design.md:
#   flash_attn_func directly (no xformers) → SDPA fallback → once-per-switch log.

from __future__ import annotations

import logging

import torch

_logged_tags: set[str] = set()
_fa2_direct_log_shown = False

# UI choices (GPU Weights row). Default = pytorch SDPA.
ATTENTION_UI_CHOICES = ("Default", "SA2", "SA3", "FA2")
ATTENTION_UI_DEFAULT = "Default"


def attention_fa2_direct(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    """
    A1111-style Flash-Attention 2 direct load.

    Calls ``flash_attn.flash_attn_func`` directly (no xformers wrapper).
    Layout for the kernel: (batch, seqlen, nheads, headdim).
    On failure: fall back to torch SDPA (same chain spirit as A1111 FA → SDP).
    """
    global _fa2_direct_log_shown

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    try:
        if mask is not None:
            raise RuntimeError("Mask must not be set for Flash-Attention direct")

        from flash_attn import flash_attn_func

        # Kernel expects (B, L, H, D); Comfy/Forge tensors here are (B, H, L, D).
        q_f = q.transpose(1, 2).contiguous()
        k_f = k.transpose(1, 2).contiguous()
        v_f = v.transpose(1, 2).contiguous()

        original_dtype = q_f.dtype
        if q_f.dtype not in (torch.float16, torch.bfloat16):
            q_f = q_f.to(torch.float16)
            k_f = k_f.to(torch.float16)
            v_f = v_f.to(torch.float16)

        out = flash_attn_func(q_f, k_f, v_f, dropout_p=0.0, causal=False)

        if not _fa2_direct_log_shown:
            fa_ok, fa_ver, fa_type = get_flash_attention_info()
            label = fa_type or "FA-2"
            ver = fa_ver or "?"
            line = f"[Forge] {label} (Flash-Attention {ver}) called directly"
            print(line)
            logging.info(line)
            _fa2_direct_log_shown = True

        if out.dtype != original_dtype:
            out = out.to(original_dtype)
        out = out.transpose(1, 2)
    except Exception as e:
        line = f"[Forge] Flash-Attention direct failed: {e}"
        print(line)
        logging.warning(line)
        print("[Forge] Fallback: torch.nn.functional.scaled_dot_product_attention")
        sdpa_extra = {}
        if kwargs.get("enable_gqa", False):
            sdpa_extra["enable_gqa"] = True
        if "scale" in kwargs:
            sdpa_extra["scale"] = kwargs["scale"]
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, **sdpa_extra
        )

    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def get_flash_attention_info():
    """Return (is_available, version, type) — Distarch sa.py compatible."""
    flash_is_available = False
    flash_attn_version = None
    flash_attn_type = None
    try:
        import flash_attn

        flash_is_available = True
        try:
            flash_attn_version = flash_attn.__version__
        except AttributeError:
            try:
                import importlib.metadata

                flash_attn_version = importlib.metadata.version("flash-attn")
            except Exception:
                flash_attn_version = "unknown"
        if flash_attn_version and flash_attn_version != "unknown":
            try:
                major_version = int(str(flash_attn_version).split(".")[0])
                flash_attn_type = "FA-3" if major_version >= 3 else "FA-2"
            except Exception:
                flash_attn_type = None
    except Exception:
        flash_is_available = False
    return flash_is_available, flash_attn_version, flash_attn_type


def get_sage_attention_info():
    """Return (version, cuda_version, torch_version) — Distarch sa.py compatible."""
    import torch

    sage_version = None
    cuda_version = "unknown"
    torch_version = "unknown"
    try:
        import sageattention

        try:
            sage_version = sageattention.__version__
        except AttributeError:
            try:
                import importlib.metadata

                sage_version = importlib.metadata.version("sageattention")
            except Exception:
                sage_version = None
        try:
            cuda_version = torch.version.cuda or "unknown"
            torch_version = torch.__version__ or "unknown"
        except Exception:
            pass
    except Exception:
        pass
    return sage_version, cuda_version, torch_version


def get_sage_attention3_info():
    """Return (version, is_available, supports_blackwell) — Distarch sa.py compatible."""
    sage3_version = None
    is_available = False
    supports_blackwell = False
    try:
        from sageattn3.blackwell import __version__ as blackwell_version

        sage3_version = blackwell_version
        is_available = True
        supports_blackwell = True
    except ImportError:
        try:
            import sageattn3  # noqa: F401

            sage3_version = "unknown"
            is_available = True
        except ImportError:
            pass
    return sage3_version, is_available, supports_blackwell


def resolve_active_attention(transformer_options=None):
    """
    Resolve the attention callable Comfy will actually run.
    Returns (fn_name, source) where source is 'override' or 'global'.
    """
    if transformer_options:
        override = transformer_options.get("optimized_attention_override")
        if callable(override):
            return getattr(override, "__name__", type(override).__name__), "override"
    try:
        from comfy.ldm.modules import attention as comfy_attention

        fn = comfy_attention.optimized_attention
        return getattr(fn, "__name__", str(fn)), "global"
    except Exception as e:
        return f"<unresolved:{e}>", "error"


def _describe_active(fn_name: str) -> str:
    fa_ok, fa_ver, fa_type = get_flash_attention_info()
    sa_ver, cuda_ver, torch_ver = get_sage_attention_info()
    sa3_ver, sa3_ok, sa3_bw = get_sage_attention3_info()

    if fn_name in ("attention_sage", "attention3_sage") or "sage" in fn_name.lower():
        if "3" in fn_name or fn_name == "attention3_sage":
            if sa3_ver and sa3_ver != "unknown":
                return f"SageAttention3 {sa3_ver} (Blackwell FP4)" if sa3_bw else f"SageAttention3 {sa3_ver}"
            return "SageAttention3 (Blackwell FP4)" if sa3_bw else "SageAttention3"
        if sa_ver and sa_ver != "unknown":
            if cuda_ver != "unknown" and torch_ver != "unknown":
                return f"SageAttention {sa_ver}+cu{cuda_ver}torch{torch_ver}"
            return f"SageAttention {sa_ver}"
        return "SageAttention (sageattn)"

    if fn_name in ("attention_flash", "attention_fa2_direct") or "flash" in fn_name.lower():
        if fa_ok and fa_type and fa_ver:
            return f"{fa_type} (Flash-Attention {fa_ver}) called directly"
        if fa_ok and fa_ver:
            return f"Flash-Attention {fa_ver} called directly"
        return "Flash-Attention called directly"

    if fn_name == "attention_xformers" or "xformers" in fn_name.lower():
        return "xformers"
    if fn_name == "attention_pytorch" or "pytorch" in fn_name.lower():
        return "pytorch SDPA"
    if fn_name == "attention_split":
        return "split"
    if fn_name == "attention_sub_quad":
        return "sub_quadratic"
    if fn_name == "attention_basic":
        return "basic (eager)"
    return fn_name


def clear_attention_log_once_flags():
    """Allow the next load/first_forward to print again after a UI switch."""
    global _fa2_direct_log_shown
    _logged_tags.clear()
    _fa2_direct_log_shown = False
    try:
        import comfy.ldm.krea2.model as krea2_model

        krea2_model._krea2_attn_logged = False
    except Exception:
        pass


def apply_attention_backend(mode: str, *, log: bool = True) -> str:
    """
    Runtime switch for SA2 / SA3 / FA2 / Default (pytorch SDPA).

    SA3 follows ComfyUI-DistorchMemoryManager nodes/sa.py sageattn3 path
    and Comfy ``attention3_sage`` (sageattn3_blackwell).

    Updates Comfy CLI flags + rebinds optimized_attention, and Forge backend.attention_function.
    Does not reload the UNet; next forward uses the new backend.
    """
    mode = (mode or ATTENTION_UI_DEFAULT).strip()
    if mode not in ATTENTION_UI_CHOICES:
        mode = ATTENTION_UI_DEFAULT

    fa_ok, fa_ver, fa_type = get_flash_attention_info()
    sa_ver, _, _ = get_sage_attention_info()
    sa3_ver, sa3_ok, sa3_bw = get_sage_attention3_info()

    if mode == "SA2" and not sa_ver:
        msg = "[Attention UI] SA2 selected but sageattention is not installed — keeping previous backend"
        print(msg)
        logging.warning(msg)
        return mode
    if mode == "SA3" and not sa3_ok:
        msg = "[Attention UI] SA3 selected but sageattn3 is not installed — keeping previous backend"
        print(msg)
        logging.warning(msg)
        return mode
    if mode == "FA2" and not fa_ok:
        msg = "[Attention UI] FA2 selected but flash_attn is not installed — keeping previous backend"
        print(msg)
        logging.warning(msg)
        return mode

    use_sage = mode == "SA2"
    use_flash = mode == "FA2"
    use_sage3 = mode == "SA3"

    # ComfyUI-master path (Krea2 / optimized_attention_masked)
    try:
        from comfy.cli_args import args as comfy_args

        if hasattr(comfy_args, "use_sage_attention"):
            comfy_args.use_sage_attention = use_sage
        if hasattr(comfy_args, "use_flash_attention"):
            comfy_args.use_flash_attention = use_flash
    except Exception as e:
        logging.warning("[Attention UI] comfy.cli_args update failed: %s", e)

    try:
        import comfy.ldm.modules.attention as comfy_attention

        if use_sage3:
            if not getattr(comfy_attention, "SAGE_ATTENTION3_IS_AVAILABLE", False):
                msg = "[Attention UI] SA3: sageattn3 not available in Comfy attention module — keeping previous backend"
                print(msg)
                logging.warning(msg)
                return mode
            target = comfy_attention.attention3_sage
            if sa3_ver and sa3_ver != "unknown":
                label = f"SageAttention3 {sa3_ver}" + (" (Blackwell FP4)" if sa3_bw else "")
            else:
                label = "SageAttention3 (Blackwell FP4)" if sa3_bw else "SageAttention3"
        elif use_sage:
            target = comfy_attention.attention_sage
            label = "SageAttention (SA2)"
        elif use_flash:
            # A1111 FA2 direct-load (flash_attn_func), not Comfy custom_op / xformers
            target = attention_fa2_direct
            label = f"{fa_type or 'FA-2'} (Flash-Attention {fa_ver or '?'}) called directly"
        else:
            target = comfy_attention.attention_pytorch
            label = "pytorch SDPA (Default)"

        comfy_attention.optimized_attention = target
        comfy_attention.optimized_attention_masked = target
    except Exception as e:
        logging.warning("[Attention UI] comfy.ldm.modules.attention rebind failed: %s", e)
        label = mode

    # Forge classic / non-Comfy UNet path
    try:
        import backend.attention as forge_attention
        import comfy.ldm.modules.attention as comfy_attention

        if use_sage3 and hasattr(comfy_attention, "attention3_sage"):
            forge_attention.attention_function = comfy_attention.attention3_sage
        elif use_sage and hasattr(forge_attention, "attention_sage"):
            forge_attention.attention_function = forge_attention.attention_sage
        elif use_flash:
            forge_attention.attention_function = attention_fa2_direct
        elif hasattr(forge_attention, "attention_pytorch"):
            forge_attention.attention_function = forge_attention.attention_pytorch
    except Exception as e:
        logging.warning("[Attention UI] backend.attention rebind failed: %s", e)

    clear_attention_log_once_flags()

    if log:
        line = f"[Attention UI] switched to {mode} → {label}"
        print(line)
        logging.info(line)
        log_comfy_attention_backend(tag="[Attention UI]", once=False, when="ui_switch")

    return mode


def log_comfy_attention_backend(tag: str = "[Krea2]", transformer_options=None, once: bool = True, when: str = "load"):
    """
    Distarch-style FA/SA visibility. Log only — does not change attention selection.
    """
    key = f"{tag}|{when}"
    if once and key in _logged_tags:
        return
    if once:
        _logged_tags.add(key)

    fn_name, source = resolve_active_attention(transformer_options)
    described = _describe_active(fn_name)
    fa_ok, fa_ver, fa_type = get_flash_attention_info()
    sa_ver, cuda_ver, torch_ver = get_sage_attention_info()
    sa3_ver, sa3_ok, sa3_bw = get_sage_attention3_info()

    use_flash = use_sage = None
    try:
        import comfy.model_management as cmm

        use_flash = cmm.flash_attention_enabled()
        use_sage = cmm.sage_attention_enabled()
    except Exception:
        pass

    lines = [
        f"{tag}[Attention] when={when} active_fn={fn_name} source={source} → {described}",
        (
            f"{tag}[Attention] installed: "
            f"FA={'yes' if fa_ok else 'no'}"
            + (f" {fa_type or ''} {fa_ver or ''}".rstrip() if fa_ok else "")
            + f" | SA={'yes' if sa_ver else 'no'}"
            + (f" {sa_ver}+cu{cuda_ver}torch{torch_ver}" if sa_ver else "")
            + f" | SA3={'yes' if sa3_ok else 'no'}"
            + (f" {sa3_ver}" if sa3_ok and sa3_ver else "")
            + (" Blackwell" if sa3_bw else "")
        ),
        f"{tag}[Attention] flags: sage_attention_enabled={use_sage} flash_attention_enabled={use_flash}",
    ]
    for line in lines:
        print(line)
        logging.info(line)
