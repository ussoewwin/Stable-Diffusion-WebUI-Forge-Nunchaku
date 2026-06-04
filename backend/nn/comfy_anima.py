"""
Forge-only checkpoint key remap for Comfy ``comfy.ldm.anima.model.Anima``.

Do not edit ComfyUI-master; upstream updates frequently. All Anima glue lives here
and under backend/ / modules_forge/ (import comfy.* only).
"""

from __future__ import annotations


def remap_anima_state_dict(state_dict: dict) -> dict:
    """
    Main DiT blocks (MiniTrainDIT / predict2) use cross_attn.output_proj.
    llm_adapter blocks (ldm.anima Attention) use cross_attn.o_proj.
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
