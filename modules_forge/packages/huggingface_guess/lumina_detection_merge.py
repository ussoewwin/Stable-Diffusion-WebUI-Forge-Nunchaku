"""Forge post-merge for Lumina2 / Z-Image after comfy.model_detection delegation."""

import comfy.model_detection as comfy_model_detection


def merge_forge_lumina_extensions(dit_config: dict, state_dict: dict, key_prefix: str) -> dict:
    """Forge-only fields Comfy model_detection does not set (LUMINA_ZIMAGE_COMFY_IMPORT_PLAN §3.3)."""
    state_dict_keys = list(state_dict.keys())

    w_key = "{}cap_embedder.1.weight".format(key_prefix)
    if w_key not in state_dict:
        return dit_config

    w = state_dict[w_key]
    dit_config["dim"] = int(w.shape[0])
    dit_config["cap_feat_dim"] = int(w.shape[1])

    dim = dit_config["dim"]

    if dim == 3840:
        dit_config["nunchaku"] = "{}layers.0.attention.to_out.0.qweight".format(key_prefix) in state_dict_keys
        ff_w1_key = "{}layers.0.feed_forward.w1.weight".format(key_prefix)
        ff_w2_key = "{}layers.0.feed_forward.w2.weight".format(key_prefix)
        if ff_w1_key in state_dict_keys and ff_w2_key in state_dict_keys:
            ff_w1 = state_dict[ff_w1_key]
            ff_w2 = state_dict[ff_w2_key]
            if ff_w1.shape[1] == 1920 and ff_w2.shape[0] == 3840:
                dit_config["ffn_input_dim"] = 1920
                dit_config["ffn_dim_multiplier"] = float(ff_w2.shape[1]) / 1920.0

    elif dim == 1920:
        dit_config["nunchaku"] = "{}layers.0.attention.to_out.0.qweight".format(key_prefix) in state_dict_keys
        dit_config["n_heads"] = 30
        dit_config["n_kv_heads"] = 30
        dit_config["axes_dims"] = [16, 24, 24]
        dit_config["axes_lens"] = [1536, 512, 512]
        dit_config["rope_theta"] = 256.0
        dit_config["z_image_modulation"] = True
        dit_config["time_scale"] = 1000.0
        if "{}cap_pad_token".format(key_prefix) in state_dict_keys:
            dit_config["pad_tokens_multiple"] = 32

    ff_w1_key = "{}layers.0.feed_forward.w1.weight".format(key_prefix)
    if ff_w1_key in state_dict_keys:
        ff_w1 = state_dict[ff_w1_key]
        ff_hidden = int(ff_w1.shape[0])
        dim_actual = int(ff_w1.shape[1])
        if dim_actual == dim and ff_hidden > 0:
            dit_config["ffn_dim_multiplier"] = ff_hidden / float(dim_actual)

    return dit_config


def state_dict_for_comfy_lumina_detection(state_dict: dict, key_prefix: str, state_dict_keys: list) -> dict:
    """Comfy Lumina entry requires k_norm; Nunchaku ZIT uses norm_k until svdq remap at load."""
    k_norm_key = "{}noise_refiner.0.attention.k_norm.weight".format(key_prefix)
    norm_k_key = "{}noise_refiner.0.attention.norm_k.weight".format(key_prefix)
    if k_norm_key in state_dict_keys or norm_k_key not in state_dict_keys:
        return state_dict
    patched = dict(state_dict)
    patched[k_norm_key] = state_dict[norm_k_key]
    return patched


def detect_lumina_comfy_with_forge_merge(state_dict: dict, key_prefix: str, state_dict_keys: list) -> dict:
    """Comfy detect_unet_config + Forge post-merge for Nunchaku / Base / distilled variants."""
    detection_state = state_dict_for_comfy_lumina_detection(state_dict, key_prefix, state_dict_keys)
    dit_config = comfy_model_detection.detect_unet_config(detection_state, key_prefix)
    return merge_forge_lumina_extensions(dit_config, state_dict, key_prefix)
