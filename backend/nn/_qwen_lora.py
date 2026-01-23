# https://github.com/GavChap/ComfyUI-QwenImageLoraLoader/blob/main/nunchaku_code/lora_qwen.py
# Updated for ForgeNeo: Uses backend.utils.load_torch_file instead of safetensors.safe_open

import logging
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from nunchaku.lora.flux.nunchaku_converter import (
    pack_lowrank_weight,
    reorder_adanorm_lora_up,
    unpack_lowrank_weight,
)

from backend.utils import load_torch_file

logger = logging.getLogger(__name__)

# Active mapping override (used for runtime model-structure switching, e.g. NextDiT).
# For Qwen Image, this is always None (no switching needed).
_ACTIVE_KEY_MAPPING = None

# --- Centralized & Optimized Key Mapping for Qwen Image ONLY ---
# This structure is faster to process and easier to maintain than a long if/elif chain.
KEY_MAPPING = [
    # Fused QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]qkv$"), r"\1.\2.attn.to_qkv", "qkv", None),
    # Decomposed QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._](q|k|v)$"), r"\1.\2.attn.to_qkv", "qkv", lambda m: m.group(3).upper()),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._](q|k|v)[._]proj$"), r"\1.\2.attn.to_qkv", "qkv", lambda m: m.group(3).upper()),
    # Fused Add_QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]add[._]qkv[._]proj$"), r"\1.\2.attn.add_qkv_proj", "add_qkv", None),
    # Decomposed Add_QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]add[._](q|k|v)[._]proj$"), r"\1.\2.attn.add_qkv_proj", "add_qkv", lambda m: m.group(3).upper()),
    # Fused QKV (Single Block)
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._]qkv$"), r"\1.\2.attn.to_qkv", "qkv", None),
    # Decomposed QKV (Single Block)
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._](q|k|v)$"), r"\1.\2.attn.to_qkv", "qkv", lambda m: m.group(3).upper()),
    # Output Projections
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]out[._]proj[._]context$"), r"\1.\2.attn.to_add_out", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]add[._]out$"), r"\1.\2.attn.to_add_out", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]out[._]proj$"), r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]out$"), r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]out[._]0$"), r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._]out$"), r"\1.\2.attn.to_out", "regular", None),
    # Feed-Forward / MLP Layers (Standard)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff[._]net[._]0(?:[._]proj)?$"), r"\1.\2.mlp_fc1", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff[._]net[._]2$"), r"\1.\2.mlp_fc2", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff_context[._]net[._]0(?:[._]proj)?$"), r"\1.\2.mlp_context_fc1", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff_context[._]net[._]2$"), r"\1.\2.mlp_context_fc2", "regular", None),
    # --- THIS IS THE CORRECTED SECTION ---
    # Feed-Forward / MLP Layers (img/txt)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](0)[._](proj)$"), r"\1.\2.\3.\4.\5.\6", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](0)$"), r"\1.\2.\3.\4.\5", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](2)$"), r"\1.\2.\3.\4.\5", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](0)[._](proj)$"), r"\1.\2.\3.\4.\5.\6", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](0)$"), r"\1.\2.\3.\4.\5", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](2)$"), r"\1.\2.\3.\4.\5", "regular", None),
    # Mod Layers (img/txt)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mod)[._](1)$"), r"\1.\2.\3.\4", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mod)[._](1)$"), r"\1.\2.\3.\4", "regular", None),
    # ------------------------------------
    # Single Block Projections
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]proj[._]out$"), r"\1.\2.proj_out", "single_proj_out", None),
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]proj[._]mlp$"), r"\1.\2.mlp_fc1", "regular", None),
    # Normalization Layers
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]norm[._]linear$"), r"\1.\2.norm.linear", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]norm1[._]linear$"), r"\1.\2.norm1.linear", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]norm1_context[._]linear$"), r"\1.\2.norm1_context.linear", "regular", None),
    # Mappings for top-level diffusion_model modules
    (re.compile(r"^(img_in)$"), r"\1", "regular", None),
    (re.compile(r"^(txt_in)$"), r"\1", "regular", None),
    (re.compile(r"^(proj_out)$"), r"\1", "regular", None),
    (re.compile(r"^(norm_out)[._](linear)$"), r"\1.\2", "regular", None),
    (re.compile(r"^(time_text_embed)[._](timestep_embedder)[._](linear_1)$"), r"\1.\2.\3", "regular", None),
    (re.compile(r"^(time_text_embed)[._](timestep_embedder)[._](linear_2)$"), r"\1.\2.\3", "regular", None),
]
_RE_LORA_SUFFIX = re.compile(r"\.(?P<tag>lora(?:[._](?:A|B|down|up)))(?:\.[^.]+)*\.weight$")
_RE_LOKR_SUFFIX = re.compile(r"\.(?P<tag>lokr_w[12])(?:\.[^.]+)*$")
_RE_ALPHA_SUFFIX = re.compile(r"\.(?:alpha|lora_alpha)(?:\.[^.]+)*$")


# --- Helper Functions ---
def _rename_layer_underscore_layer_name(old_name: str) -> str:
    """
    Converts specific model layer names by replacing underscore patterns
    with dot notation using an ordered set of regex rules.
    """

    # Rules are ordered from most specific to most general
    # to prevent a general rule from incorrectly matching part
    # of a more specific pattern.
    rules = [
        # Case: transformer_blocks_8_attn_to_out_0 -> transformer_blocks.8.attn.to_out.0
        (r"_(\d+)_attn_to_out_(\d+)", r".\1.attn.to_out.\2"),
        # Case: transformer_blocks_8_img_mlp_net_0_proj -> transformer_blocks.8.img_mlp.net.0.proj
        (r"_(\d+)_img_mlp_net_(\d+)_proj", r".\1.img_mlp.net.\2.proj"),
        # Case: transformer_blocks_8_txt_mlp_net_0_proj -> transformer_blocks.8.txt_mlp.net.0.proj
        (r"_(\d+)_txt_mlp_net_(\d+)_proj", r".\1.txt_mlp.net.\2.proj"),
        # Case: transformer_blocks_8_img_mlp_net_2 -> transformer_blocks.8.img_mlp.net.2
        (r"_(\d+)_img_mlp_net_(\d+)", r".\1.img_mlp.net.\2"),
        # Case: transformer_blocks_8_txt_mlp_net_2 -> transformer_blocks.8.txt_mlp.net.2
        (r"_(\d+)_txt_mlp_net_(\d+)", r".\1.txt_mlp.net.\2"),
        # Case: transformer_blocks_8_img_mod_1 -> transformer_blocks.8.img_mod.1
        (r"_(\d+)_img_mod_(\d+)", r".\1.img_mod.\2"),
        # Case: transformer_blocks_8_txt_mod_1 -> transformer_blocks.8.txt_mod.1
        (r"_(\d+)_txt_mod_(\d+)", r".\1.txt_mod.\2"),
        # General 'attn' case: transformer_blocks_8_attn_... -> transformer_blocks.8.attn....
        # This catches add_k_proj, add_q_proj, to_k, etc.
        (r"_(\d+)_attn_", r".\1.attn."),
    ]

    new_name = old_name
    for pattern, replacement in rules:
        # Apply the substitution. If the pattern doesn't match,
        # re.sub simply returns the original string.
        new_name = re.sub(pattern, replacement, new_name)

    return new_name


def _classify_and_map_key(key: str) -> Optional[Tuple[str, str, Optional[str], str]]:
    """
    Efficiently classifies a LoRA key using the centralized KEY_MAPPING.
    The implementation is new and optimized, but the name and signature are preserved.
    """
    k = key
    if k.startswith("transformer."):
        k = k[len("transformer.") :]
    if k.startswith("diffusion_model."):
        k = k[len("diffusion_model.") :]
    if k.startswith("lora_unet_"):
        k = k[len("lora_unet_") :]
        k = _rename_layer_underscore_layer_name(k)

    base = None
    ab = None

    m = _RE_LORA_SUFFIX.search(k)
    if m:
        tag = m.group("tag")
        base = k[: m.start()]
        if "lora_A" in tag or tag.endswith(".A") or "down" in tag:
            ab = "A"
        elif "lora_B" in tag or tag.endswith(".B") or "up" in tag:
            ab = "B"
    else:
        # Check for LoKR format (lokr_w1, lokr_w2)
        m = _RE_LOKR_SUFFIX.search(k)
        if m:
            tag = m.group("tag")
            base = k[: m.start()]
            if tag == "lokr_w1":
                ab = "lokr_w1"
            elif tag == "lokr_w2":
                ab = "lokr_w2"
        else:
            m = _RE_ALPHA_SUFFIX.search(k)
            if m:
                ab = "alpha"
                base = k[: m.start()]

    if base is None or ab is None:
        return None  # Not a recognized LoRA key format

    mapping_to_use = _ACTIVE_KEY_MAPPING if _ACTIVE_KEY_MAPPING is not None else KEY_MAPPING

    for pattern, template, group, comp_fn in mapping_to_use:
        match = pattern.match(base)
        if match:
            final_key = match.expand(template)
            component = comp_fn(match) if comp_fn else None
            return group, final_key, component, ab

    return None


def _is_indexable_module(m):
    """Checks if a module is a list-like container."""
    return isinstance(m, (nn.ModuleList, nn.Sequential, list, tuple))


def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Traverse a path like 'a.b.3.c' to find and return a module."""
    if not name:
        return model
    module = model
    for part in name.split("."):
        if not part:
            continue

        # Prioritize hasattr check. This works for:
        # 1. Regular attributes ('attn', 'img_mod')
        # 2. Numerically-named children in nn.Sequential/nn.ModuleDict ('0', '1', '2')
        if hasattr(module, part):
            module = getattr(module, part)
        # Fallback to indexing for ModuleList (which fails hasattr for numeric keys)
        elif part.isdigit() and _is_indexable_module(module):
            try:
                module = module[int(part)]
            except (IndexError, TypeError):
                logger.warning(f"Failed to index module {name} with part {part}")
                return None
        # All attempts failed
        else:
            return None
    return module


def _resolve_module_name(model: nn.Module, name: str) -> Tuple[str, Optional[nn.Module]]:
    """Resolve a name string path to a module, attempting fallback paths."""
    m = _get_module_by_name(model, name)
    if m is not None:
        return name, m

    if name.endswith(".attn.to_out.0"):
        alt = name[:-2]
        m = _get_module_by_name(model, alt)
        if m is not None:
            return alt, m
    elif name.endswith(".attn.to_out"):
        alt = name + ".0"
        m = _get_module_by_name(model, alt)
        if m is not None:
            return alt, m

    mapping = {
        ".ff.net.0.proj": ".mlp_fc1",
        ".ff.net.2": ".mlp_fc2",
        ".ff_context.net.0.proj": ".mlp_context_fc1",
        ".ff_context.net.2": ".mlp_context_fc2",
    }
    for src, dst in mapping.items():
        if src in name:
            alt = name.replace(src, dst)
            m = _get_module_by_name(model, alt)
            if m is not None:
                return alt, m

    logger.debug(f"[MISS] Module not found: {name}")
    return name, None


def _load_lora_state_dict(lora_state_dict_or_path: Union[str, Path, Dict[str, torch.Tensor]]) -> Dict[
    str, torch.Tensor]:
    """Load LoRA state dict from path or return existing dict."""
    if isinstance(lora_state_dict_or_path, (str, Path)):
        # Use ForgeNeo's load_torch_file which handles both safetensors and torch files
        return load_torch_file(str(lora_state_dict_or_path), safe_load=False, device=torch.device("cpu"))
    return lora_state_dict_or_path


def _detect_lora_format(lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Detect LoRA formats based on key patterns.
    Returns a dict containing detected format flags and sample keys.

    Formats:
    - Standard LoRA: lora_up/lora_down, lora_A/lora_B, and dot variants.
    - LoKR (Lycoris): lokr_w1/lokr_w2
    - LoHa: hada_w1/hada_w2/hada_t1/hada_t2
    - IA3: ia3_w or '.ia3.' patterns
    """
    keys = list(lora_state_dict.keys())

    standard_patterns = (
        ".lora_up.weight",
        ".lora_down.weight",
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora.up.weight",
        ".lora.down.weight",
        ".lora.A.weight",
        ".lora.B.weight",
    )

    def _sample(match_fn, limit: int = 8) -> List[str]:
        out = []
        for k in keys:
            if match_fn(k):
                out.append(k)
                if len(out) >= limit:
                    break
        return out

    has_standard = any(p in k for k in keys for p in standard_patterns)
    has_lokr = any(".lokr_w1" in k or ".lokr_w2" in k for k in keys)
    has_loha = any(".hada_w1" in k or ".hada_w2" in k or ".hada_t1" in k or ".hada_t2" in k for k in keys)
    has_ia3 = any(".ia3." in k or ".ia3_w" in k or k.endswith(".ia3.weight") for k in keys)

    return {
        "has_standard": has_standard,
        "has_lokr": has_lokr,
        "has_loha": has_loha,
        "has_ia3": has_ia3,
        "sample_standard": _sample(lambda k: any(p in k for p in standard_patterns)),
        "sample_lokr": _sample(lambda k: ".lokr_w1" in k or ".lokr_w2" in k),
        "sample_loha": _sample(lambda k: ".hada_w1" in k or ".hada_w2" in k or ".hada_t1" in k or ".hada_t2" in k),
        "sample_ia3": _sample(lambda k: ".ia3." in k or ".ia3_w" in k or k.endswith(".ia3.weight")),
        "total_keys": len(keys),
    }


def _log_lora_format_detection(lora_name: str, detection: Dict[str, Any]) -> None:
    sep = "=" * 80
    logger.info(sep)
    print(sep)
    logger.info(f"LoRA Format Detection: {lora_name}")
    print(f"LoRA Format Detection: {lora_name}")
    logger.info(sep)
    print(sep)
    logger.info("Detected Formats:")
    print("Detected Formats:")

    has_standard = detection["has_standard"]
    has_lokr = detection["has_lokr"]
    has_loha = detection["has_loha"]
    has_ia3 = detection["has_ia3"]

    if has_standard:
        logger.info("  ✅ Standard LoRA (Rank-Decomposed)")
        print("  ✅ Standard LoRA (Rank-Decomposed)")
    if has_lokr:
        logger.info("  ❌ LoKR (Lycoris) - Not Supported")
        print("  ❌ LoKR (Lycoris) - Not Supported")
    if has_loha:
        logger.info("  ❌ LoHa - Not Supported")
        print("  ❌ LoHa - Not Supported")
    if has_ia3:
        logger.info("  ❌ IA3 - Not Supported")
        print("  ❌ IA3 - Not Supported")

    if not (has_standard or has_lokr or has_loha or has_ia3):
        logger.info("  ❌ Unknown/Unsupported (no known LoRA keys detected)")
        print("  ❌ Unknown/Unsupported (no known LoRA keys detected)")

    if has_standard:
        logger.info("")
        print("")
        logger.info("✅ Standard LoRA Details:")
        print("✅ Standard LoRA Details:")
        logger.info("   Supported weight keys:")
        print("   Supported weight keys:")
        logger.info("   - lora_up.weight / lora_down.weight")
        print("   - lora_up.weight / lora_down.weight")
        logger.info("   - lora.up.weight / lora.down.weight")
        print("   - lora.up.weight / lora.down.weight")
        logger.info("   - lora_A.weight / lora_B.weight")
        print("   - lora_A.weight / lora_B.weight")
        logger.info("   - lora.A.weight / lora.B.weight")
        print("   - lora.A.weight / lora.B.weight")
        logger.info("   These are the standard formats produced by Kohya-ss, Diffusers, and most training scripts.")
        print("   These are the standard formats produced by Kohya-ss, Diffusers, and most training scripts.")

    if has_lokr:
        logger.info("")
        print("")
        logger.info("❌ LoKR (Lycoris) - Not Supported")
        print("❌ LoKR (Lycoris) - Not Supported")
        logger.info("   Issue: LoRAs in LoKR format (created by Lycoris) are not supported.")
        print("   Issue: LoRAs in LoKR format (created by Lycoris) are not supported.")
        logger.info("   Important Note: This limitation applies specifically to Nunchaku quantization models.")
        print("   Important Note: This limitation applies specifically to Nunchaku quantization models.")
        logger.info("   LoKR format LoRAs may work with standard (non-quantized) Qwen Image models, but this node is designed for Nunchaku models only.")
        print("   LoKR format LoRAs may work with standard (non-quantized) Qwen Image models, but this node is designed for Nunchaku models only.")
        logger.info("   LoKR weights are automatically skipped when detected (experimental conversion code is disabled).")
        print("   LoKR weights are automatically skipped when detected (experimental conversion code is disabled).")
        logger.info("   Converting to Standard LoRA using SVD approximation (via external tools or scripts) has also been tested")
        print("   Converting to Standard LoRA using SVD approximation (via external tools or scripts) has also been tested")
        logger.info("   and found to result in noise/artifacts when applied to Nunchaku quantization models.")
        print("   and found to result in noise/artifacts when applied to Nunchaku quantization models.")
        logger.info("   Conclusion: At this time, we have not found a way to successfully apply LoKR weights to Nunchaku models.")
        print("   Conclusion: At this time, we have not found a way to successfully apply LoKR weights to Nunchaku models.")
        logger.info("   Please use Standard LoRA formats.")
        print("   Please use Standard LoRA formats.")
        sample = detection.get("sample_lokr") or []
        if sample:
            logger.info(f"   Sample LoKR keys found: {sample}")
            print(f"   Sample LoKR keys found: {sample}")

    if has_loha:
        logger.info("")
        print("")
        logger.info("❌ LoHa - Not Supported")
        print("❌ LoHa - Not Supported")
        logger.info("   Issue: LoRAs in LoHa format are not supported for Nunchaku quantization models in this loader.")
        print("   Issue: LoRAs in LoHa format are not supported for Nunchaku quantization models in this loader.")
        logger.info("   Please convert LoHa to Standard LoRA format before use.")
        print("   Please convert LoHa to Standard LoRA format before use.")
        sample = detection.get("sample_loha") or []
        if sample:
            logger.info(f"   Sample LoHa keys found: {sample}")
            print(f"   Sample LoHa keys found: {sample}")

    if has_ia3:
        logger.info("")
        print("")
        logger.info("❌ IA3 - Not Supported")
        print("❌ IA3 - Not Supported")
        logger.info("   Issue: IA3 format is not supported for Nunchaku models in this loader.")
        print("   Issue: IA3 format is not supported for Nunchaku models in this loader.")
        logger.info("   Please use Standard LoRA formats.")
        print("   Please use Standard LoRA formats.")
        sample = detection.get("sample_ia3") or []
        if sample:
            logger.info(f"   Sample IA3 keys found: {sample}")
            print(f"   Sample IA3 keys found: {sample}")

    logger.info(sep)
    print(sep)


def _fuse_qkv_lora(qkv_weights: Dict[str, torch.Tensor], model: Optional[nn.Module] = None, base_key: Optional[str] = None) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fuse Q/K/V LoRA weights into a single QKV tensor.
    
    Args:
        qkv_weights: Dictionary containing Q/K/V LoRA weights
        model: Optional model instance to inspect module structure
        base_key: Optional base key to resolve module for Strategy 5
    """
    # Standard LoRA format
    required_keys = ["Q_A", "Q_B", "K_A", "K_B", "V_A", "V_B"]
    if not all(k in qkv_weights for k in required_keys):
        return None, None, None

    A_q, A_k, A_v = qkv_weights["Q_A"], qkv_weights["K_A"], qkv_weights["V_A"]
    B_q, B_k, B_v = qkv_weights["Q_B"], qkv_weights["K_B"], qkv_weights["V_B"]

    if not (A_q.shape == A_k.shape == A_v.shape):
        logger.warning(f"Q/K/V LoRA A dimensions mismatch: {A_q.shape}, {A_k.shape}, {A_v.shape}")
        return None, None, None

    # FACT: Check if B dimensions are consistent
    if not (B_q.shape[1] == B_k.shape[1] == B_v.shape[1]):
        logger.warning(f"Q/K/V LoRA B rank mismatch: {B_q.shape[1]}, {B_k.shape[1]}, {B_v.shape[1]}")
        return None, None, None
    
    # FACT: For QKV modules, get actual module out_features to ensure correct fusion
    qkv_out_features = None
    if model is not None and base_key is not None:
        resolved_name, module = _resolve_module_name(model, base_key)
        if module is not None and hasattr(module, 'out_features'):
            # FACT: QKV module has combined out_features = 3 * individual_out_features
            qkv_out_features = module.out_features
            logger.debug(f"QKV module {resolved_name}: using actual module out_features={qkv_out_features} for fusion")

    alpha_q, alpha_k, alpha_v = qkv_weights.get("Q_alpha"), qkv_weights.get("K_alpha"), qkv_weights.get("V_alpha")
    alpha_fused = None
    if alpha_q is not None and alpha_k is not None and alpha_v is not None and (
            alpha_q.item() == alpha_k.item() == alpha_v.item()):
        alpha_fused = alpha_q

    A_fused = torch.cat([A_q, A_k, A_v], dim=0)

    r = B_q.shape[1]
    
    # FACT: For QKV fusion, use actual module out_features if available
    # Otherwise, use sum of individual out_features (standard case)
    if qkv_out_features is not None:
        # FACT: QKV module has out_features = 3 * individual_out_features
        # Each Q/K/V should have out_features = qkv_out_features / 3
        expected_out_per_component = qkv_out_features // 3
        out_q, out_k, out_v = B_q.shape[0], B_k.shape[0], B_v.shape[0]
        
        # Verify or adjust individual B shapes if needed
        if out_q == expected_out_per_component and out_k == expected_out_per_component and out_v == expected_out_per_component:
            # All correct, use module out_features for fusion
            B_fused = torch.zeros(qkv_out_features, 3 * r, dtype=B_q.dtype, device=B_q.device)
            B_fused[:out_q, :r] = B_q
            B_fused[out_q: out_q + out_k, r: 2 * r] = B_k
            B_fused[out_q + out_k:, 2 * r:] = B_v
        else:
            # B shapes don't match expected - log warning and use standard fusion
            logger.warning(f"Q/K/V B out_features mismatch: Q={out_q}, K={out_k}, V={out_v}, expected={expected_out_per_component} each. Using actual shapes.")
            B_fused = torch.zeros(out_q + out_k + out_v, 3 * r, dtype=B_q.dtype, device=B_q.device)
            B_fused[:out_q, :r] = B_q
            B_fused[out_q: out_q + out_k, r: 2 * r] = B_k
            B_fused[out_q + out_k:, 2 * r:] = B_v
    else:
        # Standard fusion when module info not available
        out_q, out_k, out_v = B_q.shape[0], B_k.shape[0], B_v.shape[0]
        B_fused = torch.zeros(out_q + out_k + out_v, 3 * r, dtype=B_q.dtype, device=B_q.device)
        B_fused[:out_q, :r] = B_q
        B_fused[out_q: out_q + out_k, r: 2 * r] = B_k
        B_fused[out_q + out_k:, 2 * r:] = B_v

    return A_fused, B_fused, alpha_fused


def _fuse_glu_lora(glu_weights: Dict[str, torch.Tensor]) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fuse GLU LoRA weights (gate/w1 and up/w3) into a single tensor for SwiGLU projection.
    
    Args:
        glu_weights: 'w1' (gate) and 'w3' (up) LoRA weights
        
    Returns:
        Fused A, B, alpha
    """
    # w1 is usually gate, w3 is value (up) in LLaMA-like SwiGLU
    # Target module (ff.net[0].proj) has out_features = w1.out + w3.out
    
    if "w1_A" not in glu_weights or "w3_A" not in glu_weights:
        # If one is missing, we could theoretically support partial application,
        # but for now let's require both for simplicity or return None
        return None, None, None

    A_w1, B_w1 = glu_weights["w1_A"], glu_weights["w1_B"]
    A_w3, B_w3 = glu_weights["w3_A"], glu_weights["w3_B"]
    
    alpha_w1 = glu_weights.get("w1_alpha")
    alpha_w3 = glu_weights.get("w3_alpha") # w3 is 'up' or 'value'

    # Check consistency
    if A_w1.shape[0] != A_w3.shape[0]: # in_features should match
         logger.warning(f"GLU LoRA in_features mismatch: {A_w1.shape} vs {A_w3.shape}")
         return None, None, None

    r1 = B_w1.shape[1]
    r3 = B_w3.shape[1]
    
    # Fused A: Concatenate A_w1 and A_w3 (Rank becomes r1 + r3)
    # A shape: (rank, in_features) -> (r1+r3, in)
    A_fused = torch.cat([A_w1, A_w3], dim=0)
    
    # Fused B: Block diagonal
    # B shape: (out_features, rank)
    # Target out_features = out_w1 + out_w3
    out1 = B_w1.shape[0]
    out3 = B_w3.shape[0]
    
    B_fused = torch.zeros(out1 + out3, r1 + r3, dtype=B_w1.dtype, device=B_w1.device)
    B_fused[:out1, :r1] = B_w1
    B_fused[out1:, r1:] = B_w3
    
    # Alpha: If different, we might need to verify logic.
    # Usually they are same. If not, we rely on the fact that scaling is done BEFORE fusion if we were being careful,
    # but here 'compose_loras' applies scale later.
    # If alphas differ, we technically can't use a single scalar alpha for the whole fused layer if rank is used for scaling.
    # But usually alpha is constant.
    alpha_fused = alpha_w1
    if alpha_w1 is not None and alpha_w3 is not None and alpha_w1.item() != alpha_w3.item():
         logger.warning("GLU LoRA alphas differ. Using w1 alpha.")
    
    return A_fused, B_fused, alpha_fused


def _handle_proj_out_split(lora_dict: Dict[str, Dict[str, torch.Tensor]], base_key: str, model: nn.Module) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]], List[str]]:
    """Split single-block proj_out LoRA into two branches."""
    result, consumed = {}, []
    m = re.search(r"single_transformer_blocks\.(\d+)", base_key)
    if not m or base_key not in lora_dict:
        return result, consumed

    block_idx = m.group(1)
    block = _get_module_by_name(model, f"single_transformer_blocks.{block_idx}")
    if block is None:
        return result, consumed

    A_full, B_full, alpha = lora_dict[base_key].get("A"), lora_dict[base_key].get("B"), lora_dict[base_key].get("alpha")
    if A_full is None or B_full is None:
        return result, consumed

    attn_to_out = getattr(getattr(block, "attn", None), "to_out", None)
    mlp_fc2 = getattr(block, "mlp_fc2", None)
    if attn_to_out is None or mlp_fc2 is None or not hasattr(attn_to_out, "in_features") or not hasattr(mlp_fc2, "in_features"):
        return result, consumed

    attn_in, mlp_in = attn_to_out.in_features, mlp_fc2.in_features
    if A_full.shape[1] != attn_in + mlp_in:
        logger.warning(f"{base_key}: A_full shape mismatch {A_full.shape} vs expected in_features {attn_in + mlp_in}")
        return result, consumed

    A_attn, A_mlp = A_full[:, :attn_in], A_full[:, attn_in:]
    result[f"single_transformer_blocks.{block_idx}.attn.to_out"] = (A_attn, B_full.clone(), alpha)
    result[f"single_transformer_blocks.{block_idx}.mlp_fc2"] = (A_mlp, B_full.clone(), alpha)
    consumed.append(base_key)
    return result, consumed


def _is_nunchaku_qwen_image_model(model: nn.Module) -> bool:
    """
    True iff the model is Nunchaku Qwen Image (backend.nn.svdq.NunchakuQwenImageTransformer2DModel).
    AWQ and modulation-layer logic MUST run ONLY for this model.
    Do not affect Flux1, Z-Image, SDXL, or any other model.
    """
    if model is None:
        return False
    cls = type(model)
    return (
        cls.__name__ == "NunchakuQwenImageTransformer2DModel"
        and "backend.nn" in (cls.__module__ or "")
    )


def _apply_lora_to_module(module: nn.Module, A: torch.Tensor, B: torch.Tensor, module_name: str,
                          model: nn.Module) -> None:
    """Helper to append combined LoRA weights to a module."""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"{module_name}: A/B must be 2D, got {A.shape}, {B.shape}")
    # FACT: A must match module.in_features exactly (no padding/estimation)
    if A.shape[1] != module.in_features:
        raise ValueError(f"{module_name}: A shape {A.shape} mismatch with in_features={module.in_features}")
    if B.shape[0] != module.out_features:
        raise ValueError(f"{module_name}: B shape {B.shape} mismatch with out_features={module.out_features}")

    is_awq = (
        module.__class__.__name__ == "AWQW4A16Linear"
        and hasattr(module, "qweight")
        and hasattr(module, "wscales")
        and hasattr(module, "wzeros")
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
    )
    # STRICT BRANCH: AWQ handling ONLY for Nunchaku Qwen Image. Skip entirely for Flux1, SDXL, Z-Image, etc.
    if is_awq and not _is_nunchaku_qwen_image_model(model):
        return

    # Handle AWQ quantized linear layers (e.g. AWQW4A16Linear) by injecting LoRA in forward path.
    # NOTE: We avoid importing the class directly; name/path may differ across environments.
    if is_awq:
        # Save original forward once
        if not hasattr(module, "_lora_original_forward"):
            try:
                module._lora_original_forward = module.forward
            except Exception:
                module._lora_original_forward = None

        # Attach LoRA tensors on the module
        module._lora_A = A
        module._lora_B = B

        def _awq_lora_forward(x, *args, **kwargs):
            orig = getattr(module, "_lora_original_forward", None)
            if orig is None:
                # Fall back, but don't crash (safety)
                out = module.forward(x, *args, **kwargs)
            else:
                out = orig(x, *args, **kwargs)

            A_local = getattr(module, "_lora_A", None)
            B_local = getattr(module, "_lora_B", None)
            if A_local is None or B_local is None:
                return out

            # Compute LoRA residual in forward path:
            # x: [..., in] -> [..., out]
            in_features = int(getattr(module, "in_features"))
            x_in = x
            if not torch.is_tensor(x_in):
                return out
            if x_in.shape[-1] != in_features:
                return out

            x_flat = x_in.reshape(-1, in_features)
            # Ensure compute on same device as A/B
            x_flat = x_flat.to(device=A_local.device, dtype=A_local.dtype)
            lora_mid = x_flat @ A_local.transpose(0, 1)  # [N, rank]
            lora_out = lora_mid @ B_local.transpose(0, 1)  # [N, out]
            lora_out = lora_out.reshape(*x_in.shape[:-1], B_local.shape[0])
            # Cast to out dtype/device and add
            lora_out = lora_out.to(dtype=out.dtype, device=out.device)
            return out + lora_out

        # Patch forward
        module.forward = _awq_lora_forward

        if not hasattr(model, "_lora_slots"):
            model._lora_slots = {}
        # Track for reset
        model._lora_slots[module_name] = {"type": "awq_w4a16"}
        return

    # Handle Nunchaku LoRA-ready modules
    if hasattr(module, "proj_down") and hasattr(module, "proj_up"):
        pd, pu = module.proj_down.data, module.proj_up.data
        pd = unpack_lowrank_weight(pd, down=True)
        pu = unpack_lowrank_weight(pu, down=False)

        base_rank = pd.shape[0] if pd.shape[1] == module.in_features else pd.shape[1]

        if pd.shape[1] == module.in_features:  # [rank, in]
            new_proj_down = torch.cat([pd, A], dim=0)
            axis_down = 0
        else:  # [in, rank]
            new_proj_down = torch.cat([pd, A.T], dim=1)
            axis_down = 1

        new_proj_up = torch.cat([pu, B], dim=1)

        module.proj_down.data = pack_lowrank_weight(new_proj_down, down=True)
        module.proj_up.data = pack_lowrank_weight(new_proj_up, down=False)
        module.rank = base_rank + A.shape[0]

        if not hasattr(model, "_lora_slots"):
            model._lora_slots = {}
        slot = model._lora_slots.setdefault(module_name, {"base_rank": base_rank, "appended": 0, "axis_down": axis_down, "type": "nunchaku"})
        slot["appended"] += A.shape[0]

    # Handle Standard nn.Linear (Fallback)
    elif isinstance(module, nn.Linear):
        if not hasattr(model, "_lora_slots"):
            model._lora_slots = {}
        
        # Initialize slot and backup original weight if not exists
        if module_name not in model._lora_slots:
            # Backup original weight to CPU to save VRAM
            model._lora_slots[module_name] = {
                "type": "linear",
                "original_weight": module.weight.detach().cpu().clone()
            }
        
        # Calculate Delta: B @ A
        # B: [out, rank], A: [rank, in] -> Delta: [out, in]
        delta = B @ A
        if delta.shape != module.weight.shape:
             raise ValueError(f"{module_name}: LoRA delta shape {delta.shape} mismatch with linear weight {module.weight.shape}")
        
        # Apply to weight
        module.weight.data.add_(delta.to(dtype=module.weight.dtype, device=module.weight.device))
    
    else:
        # Should be caught by caller, but safety check
        raise ValueError(f"{module_name}: Unsupported module type {type(module)}")


# --- Main Public API ---


def compose_loras_v2(
    model: torch.nn.Module,
    lora_configs: List[Tuple[Union[str, Path, Dict[str, torch.Tensor]], float]],
) -> bool:
    """
    Resets and composes multiple LoRAs into the model with individual strengths.

    Returns:
        bool: True if the LoRA format is supported and processed, False otherwise.
              This allows wrappers to skip redundant retry logic.
    """
    logger.info(f"Composing {len(lora_configs)} LoRAs...")
    print(f"[Qwen LoRA] Composing {len(lora_configs)} LoRAs...")
    reset_lora_v2(model)
    _first_detection = None  # Initialize for scope safety

    # DEBUG: Inspect all keys in the first LoRA to help debug missing layers (very noisy)
    # NOTE: User requirement: do NOT hide/remove logs.
    # OPTIMIZATION: Cache first LoRA state dict for reuse in processing loop
    _cached_first_lora_state_dict = None
    _first_detection = None
    if lora_configs:
        first_lora_path_or_dict, first_lora_strength = lora_configs[0]
        first_lora_state_dict = _load_lora_state_dict(first_lora_path_or_dict)
        _cached_first_lora_state_dict = first_lora_state_dict  # Cache for reuse
        logger.info(f"--- DEBUG: Inspecting keys for LoRA 1 (Strength: {first_lora_strength}) ---")
        print(f"--- DEBUG: Inspecting keys for LoRA 1 (Strength: {first_lora_strength}) ---")
        
        # OPTIMIZATION: Check format first. If unsupported (e.g. LoKR/LoHa/SD1.5) without ANY standard keys,
        # skipping thousands of UNMATCHED log lines prevents severe lag (Github Issue #44).
        # [USER REQUEST] To restore full logs for unsupported formats, change the condition below to "if True:".
        _first_detection = _detect_lora_format(first_lora_state_dict)
        
        # Log format detection for first LoRA (USER REQUEST: log for ALL LoRAs)
        first_lora_name = first_lora_path_or_dict if isinstance(first_lora_path_or_dict, str) else "dict"
        try:
            _log_lora_format_detection(str(first_lora_name), _first_detection)
        except Exception as e:
            # Safety: never fail compose due to logging, but log the error for debugging
            logger.warning(f"Failed to log LoRA format detection for first LoRA {first_lora_name}: {e}")
            print(f"Failed to log LoRA format detection for first LoRA {first_lora_name}: {e}")
            traceback.print_exc()
        
        if _first_detection["has_standard"]:
            # Standard format (or mixed): Log EVERYTHING as requested.
            for key in first_lora_state_dict.keys():
                parsed_res = _classify_and_map_key(key)
                if parsed_res:
                    group, base_key, comp, ab = parsed_res
                    mapped_name = f"{base_key}.{comp}.{ab}" if comp and ab else (f"{base_key}.{ab}" if ab else base_key)
                    logger.info(f"Key: {key} -> Mapped to: {mapped_name} (Group: {group})")
                    print(f"Key: {key} -> Mapped to: {mapped_name} (Group: {group})")
                else:
                    logger.warning(f"Key: {key} -> UNMATCHED (Ignored)")
                    print(f"Key: {key} -> UNMATCHED (Ignored)")
        else:
            # Unsupported format only: Skip loop to prevent freeze.
            logger.warning("⚠️  Unsupported LoRA format detected (No standard keys).")
            print("⚠️  Unsupported LoRA format detected (No standard keys).")
            logger.warning(f"   Skipping detailed key inspection of {len(first_lora_state_dict)} keys to prevent console freeze.")
            print(f"   Skipping detailed key inspection of {len(first_lora_state_dict)} keys to prevent console freeze.")
            logger.warning("   Note: This LoRA will likely have no effect or will be skipped entirely.")
            print("   Note: This LoRA will likely have no effect or will be skipped entirely.")

        logger.info("--- DEBUG: End key inspection ---")
        print("--- DEBUG: End key inspection ---")

    aggregated_weights: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # 1. Aggregate weights from all LoRAs
    for idx, (lora_path_or_dict, strength) in enumerate(lora_configs):
        lora_name = lora_path_or_dict if isinstance(lora_path_or_dict, str) else "dict"
        # OPTIMIZATION: Reuse cached first LoRA state dict to avoid duplicate file I/O
        if idx == 0 and _cached_first_lora_state_dict is not None:
            lora_state_dict = _cached_first_lora_state_dict
        else:
            lora_state_dict = _load_lora_state_dict(lora_path_or_dict)

        # LoRA format detection + detailed logging (v2.2.3)
        try:
            detection = _detect_lora_format(lora_state_dict)
            _log_lora_format_detection(str(lora_name), detection)
        except Exception as e:
            # Safety: never fail compose due to logging, but log the error for debugging
            logger.warning(f"Failed to log LoRA format detection for {lora_name}: {e}")
            print(f"Failed to log LoRA format detection for {lora_name}: {e}")
            traceback.print_exc()

        lora_grouped: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        lokr_keys_count = 0
        standard_keys_count = 0
        qkv_lokr_keys_count = 0
        unrecognized_keys = []

        for key, value in lora_state_dict.items():
            parsed = _classify_and_map_key(key)
            if parsed is None:
                unrecognized_keys.append(key)
                continue

            group, base_key, comp, ab = parsed
            if ab in ("lokr_w1", "lokr_w2"):
                lokr_keys_count += 1
                # Check if it's QKV format LoKR
                if group in ("qkv", "add_qkv") and comp is not None:
                    qkv_lokr_keys_count += 1
            elif ab in ("A", "B"):
                standard_keys_count += 1

            if group in ("qkv", "add_qkv", "glu") and comp is not None:
                # Handle both standard LoRA (A/B) and LoKR (lokr_w1/lokr_w2) formats
                if ab in ("lokr_w1", "lokr_w2"):
                    lora_grouped[base_key][f"{comp}_{ab}"] = value
                else:
                    lora_grouped[base_key][f"{comp}_{ab}"] = value
            else:
                lora_grouped[base_key][ab] = value

        # Existing lightweight summary is kept at DEBUG to avoid duplicating the v2.2.3 detailed log block.
        has_lokr = lokr_keys_count > 0
        has_standard = standard_keys_count > 0
        if has_lokr and has_standard:
            lora_format = "Mixed (LoKR + Standard LoRA)"
        elif has_lokr:
            lora_format = "LoKR (QKV format)" if qkv_lokr_keys_count > 0 else "LoKR"
        elif has_standard:
            lora_format = "Standard LoRA"
        else:
            lora_format = "Unknown/Unsupported"
        logger.debug(f"LoRA summary: {lora_name} | Format: {lora_format} | Strength: {strength:.3f}")

        # Process grouped weights for this LoRA
        processed_groups = {}
        special_handled = set()
        for base_key, lw in lora_grouped.items():
            if base_key in special_handled:
                continue

            # Check if this is LoKR format (lokr_w1, lokr_w2)
            if "lokr_w1" in lw or "lokr_w2" in lw:
                logger.warning(
                    f"Skipping LoKR weights for {base_key}: LoKR support is currently experimental and disabled due to compatibility issues (produces noise). Please convert LoKR to standard LoRA first."
                )
                continue

            if "qkv" in base_key:
                # Pass model and base_key to _fuse_qkv_lora for actual module inspection
                A, B, alpha = (lw.get("A"), lw.get("B"), lw.get("alpha")) if "A" in lw else _fuse_qkv_lora(lw, model=model, base_key=base_key)
            elif "w1_A" in lw or "w3_A" in lw:  # GLU Fusion detection
                A, B, alpha = _fuse_glu_lora(lw)
            elif ".proj_out" in base_key and "single_transformer_blocks" in base_key:
                split_map, consumed_keys = _handle_proj_out_split(lora_grouped, base_key, model)
                processed_groups.update(split_map)
                special_handled.update(consumed_keys)
                continue
            else:
                A, B, alpha = lw.get("A"), lw.get("B"), lw.get("alpha")

            if A is not None and B is not None:
                processed_groups[base_key] = (A, B, alpha)

        # Warn if no weights were processed for this LoRA
        if not processed_groups:
            if lora_format == "Unknown/Unsupported":
                logger.error(f"❌ {lora_name}: No weights were processed - LoRA format is unsupported and will be skipped!")
            else:
                logger.warning(f"⚠️  {lora_name}: No weights were processed - this LoRA will have no effect!")
                # Debug: show what keys were grouped but not processed
                if lora_grouped:
                    logger.warning(f"   Debug: {len(lora_grouped)} base keys were grouped but none were processed:")
                    for bk, lw in list(lora_grouped.items())[:10]:
                        keys_in_group = list(lw.keys())
                        logger.warning(f"     - {bk}: keys={keys_in_group}")
                    if len(lora_grouped) > 10:
                        logger.warning(f"     ... and {len(lora_grouped) - 10} more grouped keys")
        else:
            logger.debug(f"   {lora_name}: Processed {len(processed_groups)} module groups")

        for module_key, (A, B, alpha) in processed_groups.items():
            aggregated_weights[module_key].append(
                {"A": A, "B": B, "alpha": alpha, "strength": strength, "source": lora_name}
            )

    # 2. Apply aggregated weights to the model
    applied_modules_count = 0

    for module_name, parts in aggregated_weights.items():
        resolved_name, module = _resolve_module_name(model, module_name)
        if module is None:
            logger.debug(f"[MISS] Module not found: {module_name} (resolved: {resolved_name})")
            continue

        is_awq_w4a16 = (
            module.__class__.__name__ == "AWQW4A16Linear"
            and hasattr(module, "qweight")
            and hasattr(module, "wscales")
            and hasattr(module, "wzeros")
            and hasattr(module, "in_features")
            and hasattr(module, "out_features")
        )

        # Check if this is img_mod.1 or txt_mod.1
        is_modulation_layer = (".img_mod.1" in resolved_name or ".txt_mod.1" in resolved_name)

        # STRICT BRANCH: AWQ modulation layers — ONLY for Nunchaku Qwen Image. Skip for all other models.
        if is_awq_w4a16 and is_modulation_layer and not _is_nunchaku_qwen_image_model(model):
            continue

        # Supported module types:
        # - Nunchaku LoRA-ready modules (proj_down/proj_up)
        # - nn.Linear (weight update fallback)
        # - AWQW4A16Linear (forward-path LoRA)
        if (
            not (hasattr(module, "proj_down") and hasattr(module, "proj_up"))
            and not isinstance(module, nn.Linear)
            and not is_awq_w4a16
        ):
            logger.info(
                f"[MISS] Module found but unsupported/missing proj_down/proj_up: {resolved_name} (Type: {type(module)})"
            )
            continue

        all_A = []
        all_B_scaled = []
        for part in parts:
            A, B, alpha, strength = part["A"], part["B"], part["alpha"], part["strength"]
            r_lora = A.shape[0]
            scale_alpha = alpha.item() if alpha is not None else float(r_lora)
            scale = strength * (scale_alpha / max(1.0, float(r_lora)))

            if ".norm1.linear" in resolved_name or ".norm1_context.linear" in resolved_name:
                B = reorder_adanorm_lora_up(B, splits=6)
            elif ".single_transformer_blocks." in resolved_name and ".norm.linear" in resolved_name:
                B = reorder_adanorm_lora_up(B, splits=3)

            # Special reorder for modulation layers: ONLY when Nunchaku Qwen Image (Manual Planar Injection path).
            # Reorder B to match modulation channel layout (shift/scale/gate × 2).
            if is_awq_w4a16 and is_modulation_layer and _is_nunchaku_qwen_image_model(model):
                # Expect out_features divisible by 6
                if B.shape[0] % 6 == 0:
                    try:
                        dim = B.shape[0] // 6
                        B = (
                            B.contiguous()
                            .view(6, dim, B.shape[1])
                            .transpose(0, 1)
                            .reshape(B.shape[0], B.shape[1])
                        )
                    except Exception:
                        # Safety: never fail due to reorder
                        pass
                else:
                    logger.warning(
                        f"{resolved_name}: expected mod up-matrix with out_features divisible by 6, "
                        f"got B({B.shape[0]}, {B.shape[1]}); skipping mod-channel reorder"
                    )

            if hasattr(module, "proj_down"):
                target_dtype = module.proj_down.dtype
                target_device = module.proj_down.device
            elif isinstance(module, nn.Linear):
                target_dtype = module.weight.dtype
                target_device = module.weight.device
            else:
                # AWQ: place LoRA tensors on same device as qweight; compute in fp16 by default.
                qweight = getattr(module, "qweight", None)
                target_device = qweight.device if torch.is_tensor(qweight) else torch.device("cpu")
                target_dtype = torch.float16

            all_A.append(A.to(dtype=target_dtype, device=target_device))
            all_B_scaled.append((B * scale).to(dtype=target_dtype, device=target_device))

        if not all_A:
            continue

        final_A = torch.cat(all_A, dim=0)
        final_B = torch.cat(all_B_scaled, dim=1)

        # STRICT BRANCH: AWQ modulation layers — Manual Planar Injection ONLY for Nunchaku Qwen Image.
        if is_awq_w4a16 and is_modulation_layer and _is_nunchaku_qwen_image_model(model):
            logger.info(f"[AWQ_MOD] {resolved_name}: Storing LoRA weights for manual Planar injection")
            mod = module
            mod._nunchaku_lora_bundle = (final_A, final_B)
            if hasattr(mod, "_lora_original_forward"):
                mod.forward = mod._lora_original_forward
                del mod._lora_original_forward
            mod._is_modulation_layer = True
            if not hasattr(model, "_lora_slots"):
                model._lora_slots = {}
            model._lora_slots[resolved_name] = {"type": "awq_mod_layer"}
            logger.info(f"[APPLY] LoRA stored for manual injection: {resolved_name}")
            print(f"[APPLY] LoRA stored for manual injection: {resolved_name}")
            applied_modules_count += 1
        else:
            _apply_lora_to_module(module, final_A, final_B, resolved_name, model)
            logger.info(f"[APPLY] LoRA applied to: {resolved_name}")
            print(f"[APPLY] LoRA applied to: {resolved_name}")
            applied_modules_count += 1

    total_loras = len(lora_configs)
    # Always output the existing log message
    logger.info(f"Applied LoRA compositions to {applied_modules_count} modules.")
    print(f"[Qwen LoRA] Applied LoRA compositions to {applied_modules_count} modules.")

    # Add additional error message if needed (but keep existing log)
    if total_loras > 0 and applied_modules_count == 0:
        logger.error(f"❌ No LoRA modules were applied! {total_loras} LoRA(s) were loaded but none matched the model structure.")
        logger.error("   This may indicate:")
        logger.error("   - Unsupported LoRA format (check format warnings above)")
        logger.error("   - LoRA for a different model architecture")
        logger.error("   - Corrupted or incompatible LoRA file(s)")

    # Return True if standard keys were found and processed, False otherwise.
    # This allows the wrapper to skip retry logic for unsupported formats.
    is_success = True
    if _first_detection is not None and not _first_detection.get("has_standard", True):
        is_success = False
    
    return is_success


def reset_lora_v2(model: nn.Module) -> None:
    """Removes all appended LoRA weights from the model."""
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        return

    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None:
            continue

        module_type = info.get("type", "nunchaku") # Default to nunchaku for backward compatibility logic

        if module_type == "nunchaku":
             base_rank = info["base_rank"]
             with torch.no_grad():
                 pd = unpack_lowrank_weight(module.proj_down.data, down=True)
                 pu = unpack_lowrank_weight(module.proj_up.data, down=False)
 
                 if info.get("axis_down", 0) == 0:  # [rank, in]
                     pd_reset = pd[:base_rank, :].clone()
                 else:  # [in, rank]
                     pd_reset = pd[:, :base_rank].clone()
                 pu_reset = pu[:, :base_rank].clone()
 
                 module.proj_down.data = pack_lowrank_weight(pd_reset, down=True)
                 module.proj_up.data = pack_lowrank_weight(pu_reset, down=False)
                 module.rank = base_rank

        elif module_type == "linear":
            if "original_weight" in info:
                # Restore original weight
                with torch.no_grad():
                    module.weight.data.copy_(info["original_weight"].to(module.weight.device))

        elif module_type == "awq_w4a16":
            # Restore original forward and remove attached LoRA tensors (Qwen Image AWQ non-mod layers only)
            if hasattr(module, "_lora_original_forward"):
                try:
                    module.forward = module._lora_original_forward
                except Exception:
                    pass
            for attr in ("_lora_A", "_lora_B", "_lora_original_forward"):
                if hasattr(module, attr):
                    try:
                        delattr(module, attr)
                    except Exception:
                        pass

        elif module_type == "awq_mod_layer":
            # Remove LoRA bundle from AWQ modulation layer (Qwen Image only)
            for attr in ("_nunchaku_lora_bundle", "_is_modulation_layer"):
                if hasattr(module, attr):
                    try:
                        delattr(module, attr)
                    except Exception:
                        pass

    model._lora_slots.clear()
    model._lora_strength = 1.0
    logger.info("All LoRA weights have been reset from the model.")
