"""
Forge glue for Lumina2 / Z-Image (NextDiT) in ``KModel``.

Upstream reference: ``comfy.model_base.Lumina2.extra_conds`` and ``BaseModel._apply_model``.
Do not patch ComfyUI-master.
"""

from __future__ import annotations

import copy
from typing import Any

import torch


def is_nextdit_model(diffusion_model) -> bool:
    """True when ``diffusion_model`` is Comfy ``NextDiT`` (Lumina2 / Z-Image)."""
    try:
        from comfy.ldm.lumina.model import NextDiT

        if isinstance(diffusion_model, NextDiT):
            return True
        return type(diffusion_model).__name__ == "NextDiT"
    except (ImportError, AttributeError, TypeError):
        return False


def _zit_patches_in(patches: dict) -> bool:
    return "noise_refiner" in patches or "double_block" in patches


def merge_kmodel_transformer_options(
    model_options: dict,
    transformer_options: dict | None,
    diffusion_model,
) -> dict:
    """
    Merge ``transformer_options`` before ``WrapperExecutor`` (all KModel types).

    NextDiT-only: skip ``model_options`` merge when ZIT patches already provided.
    Any model: clear stale ZIT patches left after switching away from NextDiT.
    """
    if transformer_options is None:
        transformer_options = {}

    skip_model_options_merge = False
    provided_patches = transformer_options.get("patches", {})
    zit_patches_in_provided = _zit_patches_in(provided_patches)

    self_patches = model_options.get("transformer_options", {}).get("patches", {})
    zit_patches_in_self = _zit_patches_in(self_patches)

    if zit_patches_in_provided and is_nextdit_model(diffusion_model):
        skip_model_options_merge = True
    elif zit_patches_in_self and not zit_patches_in_provided:
        model_options["transformer_options"] = {}
        skip_model_options_merge = True

    final_transformer_options: dict[str, Any] = {}
    if not skip_model_options_merge and "transformer_options" in model_options:
        final_transformer_options = copy.deepcopy(model_options["transformer_options"])

    if transformer_options:
        if "patches" in transformer_options:
            if "patches" not in final_transformer_options:
                final_transformer_options["patches"] = {}
            cur_patches = final_transformer_options["patches"].copy()
            for patch_name, patches in transformer_options["patches"].items():
                if patch_name in cur_patches:
                    cur_patches[patch_name] = cur_patches[patch_name] + patches
                else:
                    cur_patches[patch_name] = patches
            final_transformer_options["patches"] = cur_patches
        for key, value in transformer_options.items():
            if key != "patches":
                final_transformer_options[key] = value

    return final_transformer_options


def resolve_lumina2_forward_conds(
    extra_conds: dict,
    context: torch.Tensor | None,
    transformer_options: dict,
) -> tuple[dict, torch.Tensor | None, int | None]:
    """
    Apply Comfy ``Lumina2.extra_conds`` rules at Forge forward time.

    - ``attention_mask``: kept only when not all-ones (padding present)
    - ``num_tokens``: from mask sum, else context length, else explicit kwarg
    - ``transformer_options``: passed only via ``**kwargs`` to ``NextDiT.forward``
    """
    extra_conds = dict(extra_conds)
    attention_mask = extra_conds.pop("attention_mask", None)
    num_tokens = None

    if attention_mask is not None:
        if torch.numel(attention_mask) != attention_mask.sum():
            extra_conds["attention_mask"] = attention_mask
        num_tokens = max(1, int(torch.sum(attention_mask).item()))

    if num_tokens is None and context is not None and hasattr(context, "shape") and len(context.shape) >= 2:
        num_tokens = int(context.shape[1])
    elif "num_tokens" in extra_conds:
        num_tokens = extra_conds.pop("num_tokens")

    extra_conds["transformer_options"] = transformer_options
    return extra_conds, attention_mask, num_tokens


def forward_nextdit(
    diffusion_model,
    xc: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor | None,
    control,
    extra_conds: dict,
    transformer_options: dict,
) -> torch.Tensor:
    """Run ``NextDiT`` forward with Lumina2 extra_conds semantics."""
    forward_conds, attention_mask, num_tokens = resolve_lumina2_forward_conds(
        extra_conds, context, transformer_options
    )
    if num_tokens is not None:
        return diffusion_model(
            xc,
            t,
            context=context,
            num_tokens=num_tokens,
            attention_mask=attention_mask,
            control=control,
            **forward_conds,
        )
    return diffusion_model(xc, t, context=context, control=control, **forward_conds)

