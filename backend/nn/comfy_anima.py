"""Infer native Anima UNet config and remap checkpoint keys (Forge ``backend.nn.anima``)."""

from __future__ import annotations


def _strip_diffusion_prefix(state_dict: dict) -> dict:
    """Forge safetensors often use model.diffusion_model.*; Comfy load uses diffusion_model.*."""
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if any(k.startswith(prefix) for k in state_dict):
            return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _count_blocks(state_dict_keys: list[str], prefix: str) -> int:
    count = 0
    while True:
        if not any(k.startswith(f"{prefix}{count}.") for k in state_dict_keys):
            break
        count += 1
    return count


def _guess_unet_config(state_dict: dict) -> dict:
    """
    Infer Anima kwargs from ComfyUI comfy/model_detection.py (lines 704-753).

    waiANIMA_pw3 uses llm_adapter.blocks.*.cross_attn (not Blueprint llm_adapter.proj.0).
    Must run before generic detect_unet_config (Cosmos predict2 matches blocks.0.mlp first).
    """
    state_dict = _strip_diffusion_prefix(state_dict)
    state_dict_keys = list(state_dict.keys())
    key_prefix = ""

    has_anima_cross = (
        f"{key_prefix}llm_adapter.blocks.0.cross_attn.q_proj.weight" in state_dict_keys
        or any("cross_attn" in k for k in state_dict_keys)
    )
    if not has_anima_cross:
        raise RuntimeError(
            "Not a ComfyUI Anima checkpoint (no cross_attn / llm_adapter.cross_attn). "
            f"Sample keys: {state_dict_keys[:12]}"
        )

    # Production ComfyUI model_detection.py: x_embedder.proj.1 (2-layer) or proj.0 (1-layer)
    x_key = None
    for candidate in (
        f"{key_prefix}x_embedder.proj.1.weight",
        f"{key_prefix}x_embedder.proj.0.weight",
        f"{key_prefix}x_embedder.weight",
    ):
        if candidate in state_dict:
            x_key = candidate
            break
    if x_key is None:
        raise RuntimeError(
            f"Missing x_embedder weight in Anima state_dict. "
            f"Have x_embedder keys: {[k for k in state_dict if 'x_embedder' in k][:8]}"
        )

    w = state_dict[x_key]
    model_channels = int(w.shape[0])
    if "{}blocks.0.norm1.weight".format(key_prefix) in state_dict:
        model_channels = int(state_dict[f"{key_prefix}blocks.0.norm1.weight"].shape[0])

    # Match comfy/model_detection.py 684-737 (Cosmos mlp branch → anima via cross_attn)
    concat_padding_mask = "{}blocks.0.mlp.layer1.weight".format(key_prefix) in state_dict_keys
    in_channels = int(w.shape[1])
    if concat_padding_mask:
        in_channels = in_channels // 4 - int(concat_padding_mask)
    else:
        in_channels = in_channels // 4 if in_channels > 20 else in_channels

    dit_config: dict = {
        "image_model": "anima",
        "max_img_h": 240,
        "max_img_w": 240,
        "max_frames": 128,
        "in_channels": in_channels,
        "out_channels": 16,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "model_channels": model_channels,
        "concat_padding_mask": concat_padding_mask,
        "crossattn_emb_channels": 1024,
        "pos_emb_cls": "rope3d",
        "pos_emb_learnable": True,
        "pos_emb_interpolation": "crop",
        "min_fps": 1,
        "max_fps": 30,
        "use_adaln_lora": True,
        "adaln_lora_dim": 256,
        "extra_h_extrapolation_ratio": 1.0,
        "extra_w_extrapolation_ratio": 1.0,
        "extra_t_extrapolation_ratio": 1.0,
        "rope_enable_fps_modulation": False,
    }

    if model_channels == 2048:
        dit_config["num_blocks"] = 28
        dit_config["num_heads"] = 16
    elif model_channels == 5120:
        dit_config["num_blocks"] = 36
        dit_config["num_heads"] = 40
    else:
        dit_config["num_blocks"] = _count_blocks(state_dict_keys, "blocks.")
        dit_config["num_heads"] = max(1, dit_config["model_channels"] // 128)

    if in_channels == 16:
        dit_config["extra_per_block_abs_pos_emb"] = False
        dit_config["rope_h_extrapolation_ratio"] = 4.0
        dit_config["rope_w_extrapolation_ratio"] = 4.0
        dit_config["rope_t_extrapolation_ratio"] = 1.0
    elif in_channels == 17:
        if model_channels == 2048:
            dit_config["extra_per_block_abs_pos_emb"] = False
            dit_config["rope_h_extrapolation_ratio"] = 3.0
            dit_config["rope_w_extrapolation_ratio"] = 3.0
            dit_config["rope_t_extrapolation_ratio"] = 1.0
        elif model_channels == 5120:
            dit_config["rope_h_extrapolation_ratio"] = 2.0
            dit_config["rope_w_extrapolation_ratio"] = 2.0
            dit_config["rope_t_extrapolation_ratio"] = 0.8333333333333334

    return dit_config


def infer_anima_unet_config_from_state_dict(state_dict: dict, key_prefix: str = "") -> dict:
    """Public entry for ``detect_unet_config`` / loader (Comfy MiniTrainDIT keys)."""
    del key_prefix  # prefix already stripped by ``_strip_diffusion_prefix``
    return _guess_unet_config(_strip_diffusion_prefix(state_dict))


NATIVE_ANIMA_UNET_KEYS = (
    "in_channels",
    "out_channels",
    "patch_spatial",
    "patch_temporal",
    "model_channels",
    "concat_padding_mask",
    "crossattn_emb_channels",
    "adaln_lora_dim",
    "num_blocks",
    "num_heads",
    "rope_h_extrapolation_ratio",
    "rope_w_extrapolation_ratio",
    "rope_t_extrapolation_ratio",
    "mlp_ratio",
)


def native_anima_unet_config(dit_config: dict) -> dict:
    """Forge ``backend.nn.anima.Anima`` kwargs from full dit config (classic CosmosTransformer3DModel)."""
    return {k: dit_config[k] for k in NATIVE_ANIMA_UNET_KEYS if k in dit_config}


def _remap_anima_state_dict(state_dict: dict) -> dict:
    """
    Main DiT blocks (MiniTrainDIT / predict2) use cross_attn.output_proj.
    llm_adapter blocks (ldm.anima Attention) use cross_attn.o_proj.
    waiANIMA checkpoints store o_proj on main blocks and may use either name on llm_adapter.
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith("blocks.") and ".cross_attn.o_proj." in k:
            nk = k.replace(".cross_attn.o_proj.", ".cross_attn.output_proj.")
        elif "llm_adapter.blocks." in k and ".cross_attn.output_proj." in k:
            nk = k.replace(".cross_attn.output_proj.", ".cross_attn.o_proj.")
        else:
            nk = k
        out[nk] = v
    return out