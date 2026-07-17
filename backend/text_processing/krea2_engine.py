"""
Forge glue for ComfyUI Krea 2 (K2) TE (do not patch ComfyUI-master).

Load: ``comfy.sd.load_text_encoder_state_dicts`` (``CLIPType.KREA2``)
Encode: ``comfy.sd.CLIP.encode_from_tokens`` (same as Comfy nodes)

The Krea2 TE (Qwen3-VL-4B, 12-layer tap) flattens its ``(B, 12, seq, 2560)``
hidden-state stack to ``(B, seq, 12*2560)`` inside ``Krea2TEModel``. The
``SingleStreamDiT`` UNet unpacks that fused feature dim in ``_unpack_context``.
This engine only forwards the flattened conditioning; no ``preprocess_text_embeds``
step exists (unlike Anima) because fusion happens inside the DiT's ``txtfusion``.
"""

import torch


class Krea2TextProcessingEngine:
    def __init__(self, clip):
        self.clip = clip

    def tokenize(self, texts):
        batches = []
        for text in texts:
            pairs = self.clip.tokenize(text)
            batches.append([t[0] for t in pairs["qwen3vl_4b"][0]])
        return batches

    def _encode_line(self, line: str) -> torch.Tensor:
        tokens = self.clip.tokenize(line)
        out = self.clip.encode_from_tokens(tokens, return_dict=True)
        z = out["cond"]
        if z.ndim == 3:
            z = z[0]
        return z

    def __call__(self, texts):
        cross_attn_list = []
        cache: dict[str, torch.Tensor] = {}

        for line in texts:
            if line in cache:
                z = cache[line]
            else:
                z = self._encode_line(line)
                cache[line] = z
            cross_attn_list.append(z)

        del cache

        return dict(
            crossattn=torch.cat([t.unsqueeze(0) if t.ndim == 2 else t for t in cross_attn_list], dim=0),
        )
