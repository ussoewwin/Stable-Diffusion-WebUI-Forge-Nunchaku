"""
Nunchaku SDXL CLIP Normalization Functions

完全独立ファイル: 通常のSDXLには一切影響させない。
Reference: https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.0
"""
import os
import torch


_WARNED_TEXT_PROJ_IDENTITY = set()


def _strip_nunchaku_clip_wrappers(sd: dict[str, torch.Tensor], *, label: str) -> dict[str, torch.Tensor]:
    """
    Peel common outer prefixes used by various SDXL CLIP extractions so that
    downstream normalization can operate on keys like:
      - text_model.embeddings.token_embedding.weight
      - text_projection.weight
      - transformer.resblocks.0.attn.in_proj_weight (OpenAI/OpenCLIP style)
    """
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd

    # Always handle "transformer.text_model.*" / "transformer.text_projection.*" wrapper first.
    if any(
        k.startswith(("transformer.text_model.", "transformer.text_projection.", "transformer.positional_embedding", "transformer.token_embedding"))
        for k in sd.keys()
    ):
        out: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k.startswith("transformer."):
                out[k[len("transformer.") :]] = v
            else:
                out[k] = v
        sd = out

    # Ordered candidates (most specific first).
    candidates: list[str] = [
        # ComfyUI / diffusers-ish wrappers sometimes used in extracted files
        "text_encoders.clip_l.transformer.",
        "text_encoders.clip_g.transformer.",
        "clip_l.transformer.",
        "clip_g.transformer.",
        "clip_l.",
        "clip_g.",
        # SDXL checkpoint-like (A1111 / comfy checkpoint) wrappers
        "conditioner.embedders.0.transformer.",
        "conditioner.embedders.1.model.",
        "conditioner.embedders.1.model.transformer.",
        "conditioner.embedders.0.model.transformer.",
        "cond_stage_model.transformer.",
    ]

    total = len(sd)
    # Require a meaningful number of keys to avoid accidental stripping.
    # CLIP SDs are typically hundreds of keys.
    min_required = 32
    for prefix in candidates:
        matched = [k for k in sd.keys() if isinstance(k, str) and k.startswith(prefix)]
        if len(matched) < min_required:
            continue
        # If a prefix matches a large chunk of the dict, treat it as the wrapper.
        if len(matched) < max(min_required, int(total * 0.25)):
            continue
        out = {k[len(prefix) :]: sd[k] for k in matched}

        # Keep any already-unwrapped CLIP keys that might coexist.
        for k, v in sd.items():
            if k in out:
                continue
            if not isinstance(k, str):
                continue
            if k.startswith(("text_model.", "text_projection.", "token_embedding.", "positional_embedding")) or "transformer.resblocks." in k:
                out[k] = v

        if os.getenv("DEBUG", "").find("NUNCHAKU_SDXL") >= 0:
            try:
                print(f"[Nunchaku SDXL CLIP DEBUG] {label}: stripped wrapper prefix '{prefix}' ({len(matched)}/{total} keys)")
            except Exception:
                pass
        return out

    return sd


def _normalize_comfy_clip_sd(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Normalize common SDXL CLIP extraction variants into ComfyUI expected keys.
    
    Known issues:
    - Some extractors save CLIP-L with a double prefix: "text_model.text_model.*"
    - Some save the projection as "text_model.text_projection.weight" instead of "text_projection.weight"
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        
        # Fix double-prefix variants (most common for CLIP-L extractions)
        if nk.startswith("text_model.text_model."):
            nk = "text_model." + nk[len("text_model.text_model."):]
        
        # Fix projection naming variants
        if nk.startswith("text_model.text_projection."):
            nk = "text_projection." + nk[len("text_model.text_projection."):]
        
        out[nk] = v
    
    return out


def _find_openai_clip_prefix(sd: dict[str, torch.Tensor]) -> str | None:
    """
    Detect OpenAI/OpenCLIP-style text tower keys with an arbitrary prefix.
    Returns the prefix (including trailing ".") up to "transformer.resblocks.".
    
    Examples:
      - "model.transformer.resblocks.0..."           -> "model."
      - "text_model.transformer.resblocks.0..."      -> "text_model."
      - "cond_stage_model.model.transformer..."      -> "cond_stage_model.model."
    """
    needle = "transformer.resblocks."
    for k in sd.keys():
        i = k.find(needle)
        if i >= 0:
            return k[:i]
    return None


def _is_openai_clip_style(sd: dict[str, torch.Tensor]) -> bool:
    """Check if state dict is in OpenAI/OpenCLIP format."""
    prefix = _find_openai_clip_prefix(sd)
    if prefix is None:
        return False
    return (prefix + "token_embedding.weight") in sd or (prefix + "positional_embedding") in sd


def _convert_openai_clip_sd_to_comfy(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert OpenAI/OpenCLIP-style text tower state_dict (keys like "model.transformer.resblocks.*")
    into ComfyUI clip_model.py format (keys like "text_model.encoder.layers.*").
    
    This is needed for some SDXL extractions where CLIP-G is saved in OpenCLIP naming.
    """
    prefix = _find_openai_clip_prefix(sd)
    if prefix is None:
        return sd
    
    print(f"[Nunchaku SDXL CLIP] OpenAI/OpenCLIP style detected. prefix='{prefix}'")
    
    # Infer embed_dim from in_proj_weight
    sample_k = None
    for k in sd.keys():
        if k.startswith(prefix + "transformer.resblocks.") and k.endswith(".attn.in_proj_weight"):
            sample_k = k
            break
    if sample_k is None:
        return sd
    in_proj_w = sd[sample_k]
    embed_dim = in_proj_w.shape[1]
    
    out: dict[str, torch.Tensor] = {}
    
    # Embeddings + final LN + projection
    tok_k = prefix + "token_embedding.weight"
    pos_k = prefix + "positional_embedding"
    ln_w_k = prefix + "ln_final.weight"
    ln_b_k = prefix + "ln_final.bias"
    
    if tok_k in sd:
        out["text_model.embeddings.token_embedding.weight"] = sd[tok_k]
    if pos_k in sd:
        out["text_model.embeddings.position_embedding.weight"] = sd[pos_k]
    if ln_w_k in sd:
        out["text_model.final_layer_norm.weight"] = sd[ln_w_k]
    if ln_b_k in sd:
        out["text_model.final_layer_norm.bias"] = sd[ln_b_k]
    
    # Projection can appear with multiple key spellings
    if (prefix + "text_projection") in sd:
        out["text_projection.weight"] = sd[prefix + "text_projection"]
    elif (prefix + "text_projection.weight") in sd:
        out["text_projection.weight"] = sd[prefix + "text_projection.weight"]
    elif "text_projection.weight" in sd:
        out["text_projection.weight"] = sd["text_projection.weight"]
    
    # Transformer blocks
    block_prefix = prefix + "transformer.resblocks."
    for k, v in sd.items():
        if not k.startswith(block_prefix):
            continue
        rest = k[len(block_prefix):]
        parts = rest.split(".")
        if len(parts) < 3:
            continue
        layer_idx = parts[0]
        tail = ".".join(parts[1:])
        
        layer_prefix = f"text_model.encoder.layers.{layer_idx}."
        
        if tail.startswith("ln_1."):
            out[layer_prefix + "layer_norm1." + tail[len("ln_1."):]] = v
        elif tail.startswith("ln_2."):
            out[layer_prefix + "layer_norm2." + tail[len("ln_2."):]] = v
        elif tail.startswith("mlp.c_fc."):
            out[layer_prefix + "mlp.fc1." + tail[len("mlp.c_fc."):]] = v
        elif tail.startswith("mlp.c_proj."):
            out[layer_prefix + "mlp.fc2." + tail[len("mlp.c_proj."):]] = v
        elif tail.startswith("attn.out_proj."):
            out[layer_prefix + "self_attn.out_proj." + tail[len("attn.out_proj."):]] = v
        elif tail == "attn.in_proj_weight":
            # (3*D, D) => q/k/v: (D, D)
            out[layer_prefix + "self_attn.q_proj.weight"] = v[:embed_dim, :]
            out[layer_prefix + "self_attn.k_proj.weight"] = v[embed_dim:2*embed_dim, :]
            out[layer_prefix + "self_attn.v_proj.weight"] = v[2*embed_dim:, :]
        elif tail == "attn.in_proj_bias":
            out[layer_prefix + "self_attn.q_proj.bias"] = v[:embed_dim]
            out[layer_prefix + "self_attn.k_proj.bias"] = v[embed_dim:2*embed_dim]
            out[layer_prefix + "self_attn.v_proj.bias"] = v[2*embed_dim:]
    
    return out


def _ensure_text_projection_weight(
    sd: dict[str, torch.Tensor],
    *,
    label: str = "clip",
) -> tuple[dict[str, torch.Tensor], bool]:
    """
    ComfyUI's CLIP text models expect 'text_projection.weight' (Linear bias=False).
    Some extracted CLIP-L files omit this key, which causes large 'clip missing' logs.
    
    IMPORTANT:
    Filling identity is a *compatibility fallback* only. It makes the model "run" but it is
    NOT guaranteed to be equivalent to standard SDXL because text_projection is typically a
    learned mapping.
    """
    if "text_projection.weight" in sd:
        return sd, False
    
    mode = str(os.getenv("NUNCHAKU_SDXL_CLIP_TEXT_PROJECTION_MISSING", "identity")).strip().lower()
    # Safety policy: NEVER hard-fail here. Missing projection is common in extracted files;
    # running (with a loud warning) is preferred over stopping the workflow.
    if mode in ("error", "raise", "strict"):
        try:
            key = f"{label}:strict_requested"
            if key not in _WARNED_TEXT_PROJ_IDENTITY:
                _WARNED_TEXT_PROJ_IDENTITY.add(key)
                print(
                    "[NUNCHAKU_SDXL_CLIP_WARNING] "
                    f"{label}: NUNCHAKU_SDXL_CLIP_TEXT_PROJECTION_MISSING='{mode}' was requested, "
                    "but hard-fail is disabled. Falling back to identity to keep running."
                )
        except Exception:
            pass
        mode = "identity"
    if mode in ("skip", "none", "off"):
        try:
            key = f"{label}:skip"
            if key not in _WARNED_TEXT_PROJ_IDENTITY:
                _WARNED_TEXT_PROJ_IDENTITY.add(key)
                print(
                    "[NUNCHAKU_SDXL_CLIP_WARNING] "
                    f"{label}: 'text_projection.weight' is missing; leaving it uninitialized (mode=skip). "
                    "This is NOT equivalent to standard SDXL and may degrade prompt behavior."
                )
        except Exception:
            pass
        return sd, False
    
    tok_key = "text_model.embeddings.token_embedding.weight"
    if tok_key not in sd:
        return sd, False
    
    tok = sd[tok_key]
    if not torch.is_tensor(tok) or tok.ndim != 2:
        return sd, False
    
    hidden = int(tok.shape[1])
    eye = torch.eye(hidden, dtype=tok.dtype, device=tok.device)
    out = dict(sd)
    out["text_projection.weight"] = eye
    try:
        key = f"{label}:identity:{hidden}:{str(tok.dtype)}"
        if key not in _WARNED_TEXT_PROJ_IDENTITY:
            _WARNED_TEXT_PROJ_IDENTITY.add(key)
            print(
                "[NUNCHAKU_SDXL_CLIP_WARNING] "
                f"{label}: 'text_projection.weight' was missing; filled identity ({hidden}x{hidden}). "
                "This makes it run, but is NOT guaranteed equivalent to standard SDXL. "
                "If you have a proper SDXL CLIP file, use that instead."
            )
    except Exception:
        pass
    return out, True


def normalize_nunchaku_clip_state_dict(sd: dict[str, torch.Tensor], label: str = "clip") -> dict[str, torch.Tensor]:
    """
    Normalize Nunchaku SDXL CLIP state dict: handle double prefix, OpenAI/OpenCLIP format, and missing text_projection.
    
    IMPORTANT: この関数はNunchaku SDXL専用。通常のSDXLには絶対に使用しないこと。
    """
    # Step 0: Peel common wrappers (clip_g.*, text_encoders.clip_g.transformer.*, transformer.* etc.)
    sd = _strip_nunchaku_clip_wrappers(sd, label=label)

    # Step 1: Normalize double prefix (text_model.text_model.* -> text_model.*)
    sd = _normalize_comfy_clip_sd(sd)
    
    # Step 2: Convert OpenAI/OpenCLIP format if needed
    if _is_openai_clip_style(sd):
        sd = _convert_openai_clip_sd_to_comfy(sd)
    
    # Step 3: Ensure text_projection.weight exists
    sd, _ = _ensure_text_projection_weight(sd, label=label)
    
    return sd


def convert_nunchaku_clip_to_forge_format(sd: dict[str, torch.Tensor], component_name: str) -> dict[str, torch.Tensor]:
    """
    Convert Nunchaku SDXL CLIP state dict from ComfyUI format to Forge Neo format.
    
    Forge Neo expects: transformer.text_model.*, transformer.text_projection.*
    ComfyUI format: text_model.*, text_projection.*
    
    IMPORTANT: この関数はNunchaku SDXL専用。通常のSDXLには絶対に使用しないこと。
    """
    # First remove clip_l./clip_g. prefix if present
    prefix_to_remove = None
    if component_name == "text_encoder":
        prefix_to_remove = "clip_l."
    elif component_name == "text_encoder_2":
        prefix_to_remove = "clip_g."
    
    if prefix_to_remove:
        state_dict_temp = {}
        prefix_len = len(prefix_to_remove)
        for key, val in sd.items():
            if key.startswith(prefix_to_remove):
                state_dict_temp[key[prefix_len:]] = val
            else:
                state_dict_temp[key] = val
        sd = state_dict_temp
    
    # Normalize (handle double prefix, OpenAI format, etc.)
    sd = normalize_nunchaku_clip_state_dict(sd, label=component_name)
    
    # Convert to Forge Neo format: text_model.* -> transformer.text_model.*
    state_dict_forge = {}
    for key, val in sd.items():
        if key.startswith("text_model.") or key.startswith("text_projection."):
            state_dict_forge["transformer." + key] = val
        else:
            state_dict_forge["transformer." + key] = val
    sd = state_dict_forge
    print(f"[Nunchaku SDXL] Converted {component_name} keys to Forge format ({len(sd)} keys)")

    return sd
