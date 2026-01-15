# Z-Image LoRA support for Nunchaku Z-Image models only
# Based on ComfyUI-QwenImageLoraLoader for Z-Image (NextDiT/Lumina2)
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
    unpack_lowrank_weight,
)

from backend.utils import load_torch_file

logger = logging.getLogger(__name__)

# --- Centralized & Optimized Key Mapping for Z-Image ONLY ---
# Z-Image uses NextDiT/Lumina2 structure with layers.N.attention.qkv and layers.N.feed_forward.w13/w2
KEY_MAPPING = [
    # NextDiT (ComfyUI Lumina2) mappings for Z-Image-Turbo
    # These are required because NextDiT uses:
    # - layers.N.attention.qkv / layers.N.attention.out
    # - layers.N.feed_forward.w1/w2/w3 (unpatched) OR layers.N.feed_forward.w13 + w2 (nunchaku-patched)
    
    # QKV Fusion (Decomposed Q/K/V -> fused qkv)
    (re.compile(r"^(layers)[._](\d+)[._]attention[._]to[._]([qkv])$"), r"\1.\2.attention.qkv", "qkv", lambda m: m.group(3).upper()),
    # Output projection
    (re.compile(r"^(layers)[._](\d+)[._]attention[._]to[._]out(?:[._]0)?$"), r"\1.\2.attention.out", "regular", None),
    
    # Nunchaku-patched: w1/w3 fused into w13, w2 separate
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._](w1|w3)$"), r"\1.\2.feed_forward.w13", "glu", lambda m: m.group(3)),
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._]w2$"), r"\1.\2.feed_forward.w2", "regular", None),
    
    # Unpatched: w1/w2/w3 separate
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._](w1|w2|w3)$"), r"\1.\2.feed_forward.\3", "regular", None),
    
    # Generic layer mapping (for other layers)
    (re.compile(r"^(layers)[._](\d+)[._](.*)$"), r"\1.\2.\3", "regular", None),
]

_RE_LORA_SUFFIX = re.compile(r"\.(?P<tag>lora(?:[._](?:A|B|down|up)))(?:\.[^.]+)*\.weight$")
_RE_ALPHA_SUFFIX = re.compile(r"\.(?:alpha|lora_alpha)(?:\.[^.]+)*$")


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
    logger.info(f"[Z-Image] LoRA Format Detection: {lora_name}")
    print(f"[Z-Image] LoRA Format Detection: {lora_name}")
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
        logger.info("   LoKR format LoRAs may work with standard (non-quantized) Z-Image models, but this loader is designed for Nunchaku models only.")
        print("   LoKR format LoRAs may work with standard (non-quantized) Z-Image models, but this loader is designed for Nunchaku models only.")
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


def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Traverse a path like 'layers.0.attention.qkv' to find and return a module."""
    if not name:
        return model
    module = model
    for part in name.split("."):
        if not part:
            continue
        if hasattr(module, part):
            module = getattr(module, part)
        elif part.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential, list, tuple)):
            try:
                module = module[int(part)]
            except (IndexError, TypeError):
                logger.warning(f"Failed to index module {name} with part {part}")
                return None
        else:
            return None
    return module


def _classify_and_map_key(key: str) -> Optional[Tuple[str, str, Optional[str], str]]:
    """Classifies a LoRA key using the centralized KEY_MAPPING for Z-Image."""
    k = key
    if k.startswith("transformer."):
        k = k[len("transformer.") :]
    if k.startswith("diffusion_model."):
        k = k[len("diffusion_model.") :]
    
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
        m = _RE_ALPHA_SUFFIX.search(k)
        if m:
            ab = "alpha"
            base = k[: m.start()]

    if base is None or ab is None:
        return None  # Not a recognized LoRA key format

    for pattern, template, group, comp_fn in KEY_MAPPING:
        match = pattern.match(base)
        if match:
            final_key = match.expand(template)
            component = comp_fn(match) if comp_fn else None
            return group, final_key, component, ab

    return None


def _fuse_qkv_lora(qkv_weights: Dict[str, torch.Tensor], model: Optional[nn.Module] = None, base_key: Optional[str] = None) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fuse Q/K/V LoRA weights into a single QKV tensor for Z-Image."""
    required_keys = ["Q_A", "Q_B", "K_A", "K_B", "V_A", "V_B"]
    if not all(k in qkv_weights for k in required_keys):
        return None, None, None

    A_q, A_k, A_v = qkv_weights["Q_A"], qkv_weights["K_A"], qkv_weights["V_A"]
    B_q, B_k, B_v = qkv_weights["Q_B"], qkv_weights["K_B"], qkv_weights["V_B"]

    if not (A_q.shape == A_k.shape == A_v.shape):
        logger.warning(f"Q/K/V LoRA A dimensions mismatch: {A_q.shape}, {A_k.shape}, {A_v.shape}")
        return None, None, None

    if not (B_q.shape[1] == B_k.shape[1] == B_v.shape[1]):
        logger.warning(f"Q/K/V LoRA B rank mismatch: {B_q.shape[1]}, {B_k.shape[1]}, {B_v.shape[1]}")
        return None, None, None

    alpha_q, alpha_k, alpha_v = qkv_weights.get("Q_alpha"), qkv_weights.get("K_alpha"), qkv_weights.get("V_alpha")
    alpha_fused = None
    if alpha_q is not None and alpha_k is not None and alpha_v is not None and (
            alpha_q.item() == alpha_k.item() == alpha_v.item()):
        alpha_fused = alpha_q

    A_fused = torch.cat([A_q, A_k, A_v], dim=0)
    r = B_q.shape[1]
    out_q, out_k, out_v = B_q.shape[0], B_k.shape[0], B_v.shape[0]
    B_fused = torch.zeros(out_q + out_k + out_v, 3 * r, dtype=B_q.dtype, device=B_q.device)
    B_fused[:out_q, :r] = B_q
    B_fused[out_q: out_q + out_k, r: 2 * r] = B_k
    B_fused[out_q + out_k:, 2 * r:] = B_v

    return A_fused, B_fused, alpha_fused


def _fuse_glu_lora(glu_weights: Dict[str, torch.Tensor]) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fuse GLU LoRA weights (gate/w1 and up/w3) into a single tensor for SwiGLU projection (Z-Image w13)."""
    if "w1_A" not in glu_weights or "w3_A" not in glu_weights:
        return None, None, None

    A_w1, B_w1 = glu_weights["w1_A"], glu_weights["w1_B"]
    A_w3, B_w3 = glu_weights["w3_A"], glu_weights["w3_B"]
    
    alpha_w1 = glu_weights.get("w1_alpha")
    alpha_w3 = glu_weights.get("w3_alpha")

    if A_w1.shape[0] != A_w3.shape[0]:
         logger.warning(f"GLU LoRA in_features mismatch: {A_w1.shape} vs {A_w3.shape}")
         return None, None, None

    r1 = B_w1.shape[1]
    r3 = B_w3.shape[1]
    
    A_fused = torch.cat([A_w1, A_w3], dim=0)
    
    out1 = B_w1.shape[0]
    out3 = B_w3.shape[0]
    
    B_fused = torch.zeros(out1 + out3, r1 + r3, dtype=B_w1.dtype, device=B_w1.device)
    B_fused[:out1, :r1] = B_w1
    B_fused[out1:, r1:] = B_w3
    
    alpha_fused = alpha_w1
    if alpha_w1 is not None and alpha_w3 is not None and alpha_w1.item() != alpha_w3.item():
         logger.warning("GLU LoRA alphas differ. Using w1 alpha.")
    
    return A_fused, B_fused, alpha_fused


def _apply_lora_to_module(module: nn.Module, A: torch.Tensor, B: torch.Tensor, module_name: str,
                          model: nn.Module) -> None:
    """Helper to append combined LoRA weights to a module (Z-Image)."""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"{module_name}: A/B must be 2D, got {A.shape}, {B.shape}")
    if A.shape[1] != module.in_features:
        raise ValueError(f"{module_name}: A shape {A.shape} mismatch with in_features={module.in_features}")
    if B.shape[0] != module.out_features:
        raise ValueError(f"{module_name}: B shape {B.shape} mismatch with out_features={module.out_features}")

    # Handle Nunchaku LoRA-ready modules (proj_down/proj_up)
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
    else:
        raise ValueError(f"{module_name}: Unsupported module type {type(module)} (Z-Image requires proj_down/proj_up)")


def compose_loras_v2(
    model: torch.nn.Module,
    lora_configs: List[Tuple[Union[str, Path, Dict[str, torch.Tensor]], float]],
) -> bool:
    """
    Resets and composes multiple LoRAs into the Z-Image model with individual strengths.

    Returns:
        bool: True if the LoRA format is supported and processed, False otherwise.
    """
    logger.info(f"[Z-Image] Composing {len(lora_configs)} LoRAs...")
    print(f"[Z-Image] Composing {len(lora_configs)} LoRAs...")
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
        logger.info(f"--- DEBUG: Inspecting keys for Z-Image LoRA 1 (Strength: {first_lora_strength}) ---")
        print(f"--- DEBUG: Inspecting keys for Z-Image LoRA 1 (Strength: {first_lora_strength}) ---")

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
            logger.warning(f"Failed to log LoRA format detection for first Z-Image LoRA {first_lora_name}: {e}")
            print(f"Failed to log LoRA format detection for first Z-Image LoRA {first_lora_name}: {e}")
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

    # Auto-detect Nunchaku-patched vs unpatched Z-Image structure
    has_w13 = _get_module_by_name(model, "layers.0.feed_forward.w13") is not None

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
            logger.warning(f"Failed to log LoRA format detection for Z-Image LoRA {lora_name}: {e}")
            print(f"Failed to log LoRA format detection for Z-Image LoRA {lora_name}: {e}")
            traceback.print_exc()

        lora_grouped: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        for key, value in lora_state_dict.items():
            parsed = _classify_and_map_key(key)
            if parsed is None:
                continue

            group, base_key, comp, ab = parsed
            # Skip if unpatched w1/w3 when model is patched (has w13)
            if has_w13 and base_key.endswith((".feed_forward.w1", ".feed_forward.w3")):
                logger.debug(f"[Z-Image] Skipping unpatched key {key} (model has w13)")
                continue

            if group in ("qkv", "glu") and comp is not None:
                lora_grouped[base_key][f"{comp}_{ab}"] = value
            else:
                lora_grouped[base_key][ab] = value

        # Process grouped weights for this LoRA
        processed_groups = {}
        for base_key, lw in lora_grouped.items():
            if "qkv" in base_key:
                A, B, alpha = (lw.get("A"), lw.get("B"), lw.get("alpha")) if "A" in lw else _fuse_qkv_lora(lw, model=model, base_key=base_key)
            elif "w1_A" in lw or "w3_A" in lw:  # GLU Fusion detection (w13)
                if not has_w13:
                    logger.warning(f"[Z-Image] Skipping fused w13 LoRA {base_key} (model is unpatched)")
                    continue
                A, B, alpha = _fuse_glu_lora(lw)
            else:
                A, B, alpha = lw.get("A"), lw.get("B"), lw.get("alpha")

            if A is not None and B is not None:
                processed_groups[base_key] = (A, B, alpha)

        for module_key, (A, B, alpha) in processed_groups.items():
            aggregated_weights[module_key].append({"A": A, "B": B, "alpha": alpha, "strength": strength, "source": lora_name})

    # 2. Apply aggregated weights to the model
    applied_modules_count = 0

    for module_name, parts in aggregated_weights.items():
        resolved_name, module = _resolve_module_name(model, module_name)
        if module is None:
            logger.debug(f"[Z-Image] [MISS] Module not found: {module_name} (resolved: {resolved_name})")
            continue

        if not (hasattr(module, "proj_down") and hasattr(module, "proj_up")):
            logger.info(f"[Z-Image] [MISS] Module found but unsupported/missing proj_down/proj_up: {resolved_name} (Type: {type(module)})")
            print(f"[Z-Image] [MISS] Module found but unsupported/missing proj_down/proj_up: {resolved_name} (Type: {type(module)})")
            continue

        all_A = []
        all_B_scaled = []
        for part in parts:
            A, B, alpha, strength = part["A"], part["B"], part["alpha"], part["strength"]
            r_lora = A.shape[0]
            scale_alpha = alpha.item() if alpha is not None else float(r_lora)
            scale = strength * (scale_alpha / max(1.0, float(r_lora)))

            target_dtype = module.proj_down.dtype
            target_device = module.proj_down.device

            all_A.append(A.to(dtype=target_dtype, device=target_device))
            all_B_scaled.append((B * scale).to(dtype=target_dtype, device=target_device))

        if not all_A:
            continue

        final_A = torch.cat(all_A, dim=0)
        final_B = torch.cat(all_B_scaled, dim=1)

        _apply_lora_to_module(module, final_A, final_B, resolved_name, model)
        logger.info(f"[Z-Image] [APPLY] LoRA applied to: {resolved_name}")
        print(f"[Z-Image] [APPLY] LoRA applied to: {resolved_name}")
        applied_modules_count += 1

    logger.info(f"[Z-Image] Applied LoRA compositions to {applied_modules_count} modules.")
    print(f"[Z-Image] Applied LoRA compositions to {applied_modules_count} modules.")
    return applied_modules_count > 0


def _resolve_module_name(model: nn.Module, name: str) -> Tuple[str, Optional[nn.Module]]:
    """Resolve a name string path to a module."""
    m = _get_module_by_name(model, name)
    if m is not None:
        return name, m
    logger.debug(f"[Z-Image] [MISS] Module not found: {name}")
    return name, None


def reset_lora_v2(model: nn.Module) -> None:
    """Removes all appended LoRA weights from the Z-Image model."""
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        return

    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None:
            continue

        module_type = info.get("type", "nunchaku")

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
            
    model._lora_slots.clear()
    model._lora_strength = 1.0
    logger.info("[Z-Image] All LoRA weights have been reset from the model.")
    print("[Z-Image] All LoRA weights have been reset from the model.")