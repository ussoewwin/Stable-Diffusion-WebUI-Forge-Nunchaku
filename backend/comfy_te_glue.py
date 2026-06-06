"""
Forge glue for Comfy ``sd.CLIP`` text encoders (Lumina2 / Z-Image / Anima Step 5).

VRAM contract (match v1.7.3 Forge ``Gemma2_2B`` / ``Qwen3_4B`` path):
- TE weights init on CPU (``using_forge_operations(device=cpu)`` equivalent)
- GPU only inside ``encode_from_tokens`` → ``CLIP.load_model``
- TE off GPU before UNet sampling when callers invoke ``offload_comfy_clip``
"""

from backend import memory_management


def comfy_te_model_options() -> dict:
    """
    Comfy ``CLIP.__init__`` (``sd.py`` L278-279) force_full_load when
    ``initial_device == load_device``.  v1.7.3 always built Forge TE on CPU
    regardless of ``--always-gpu`` / HIGH_VRAM — mirror that here.
    """
    load_device = memory_management.text_encoder_device()
    offload_device = memory_management.text_encoder_offload_device()
    return {
        "load_device": load_device,
        "offload_device": offload_device,
        "initial_device": memory_management.cpu,
    }


def is_comfy_sd_clip(obj) -> bool:
    from comfy.sd import CLIP as ComfyCLIP

    return isinstance(obj, ComfyCLIP)


def offload_comfy_clip(clip) -> None:
    """
    Drop Comfy TE from GPU after conditioning.

    Must use Comfy ``unload_model_and_clones`` (``free_memory`` path).  Never call
    ``LoadedModel.model_unload()`` directly — that leaves ``real_model=None`` in
    ``current_loaded_models`` and breaks the next ``encode_from_tokens``.
    """
    if clip is None or not hasattr(clip, "patcher") or clip.patcher is None:
        return
    try:
        import comfy.model_management as mm

        mm.unload_model_and_clones(clip.patcher)
    except Exception:
        pass
    memory_management.soft_empty_cache()
