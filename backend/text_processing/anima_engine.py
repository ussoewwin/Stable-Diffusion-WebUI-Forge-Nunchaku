"""
Forge glue for ComfyUI Anima TE (do not patch ComfyUI-master).

Load: ``comfy.sd.load_text_encoder_state_dicts``
Encode: ``comfy.sd.CLIP.encode_from_tokens`` (same as Comfy nodes)
UNet ``preprocess_text_embeds``: ``comfy.ldm.anima.model.Anima`` only.
"""

import torch


class AnimaTextProcessingEngine:
    def __init__(self, clip, qwen_tokenizer=None, t5_tokenizer=None):
        del qwen_tokenizer, t5_tokenizer
        self.clip = clip

    def tokenize(self, texts):
        qwen_batches = []
        t5_batches = []
        for text in texts:
            pairs = self.clip.tokenize(text)
            qwen_batches.append([t[0] for t in pairs["qwen3_06b"][0]])
            t5_batches.append([(t[0], t[1]) for t in pairs["t5xxl"][0]])
        return qwen_batches, t5_batches

    def _encode_line(self, line: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.clip.tokenize(line)
        out = self.clip.encode_from_tokens(tokens, return_dict=True)
        z = out["cond"]
        if z.ndim == 3:
            z = z[0]
        return z, out["t5xxl_ids"], out["t5xxl_weights"]

    def __call__(self, texts):
        cross_attn_list = []
        t5_ids_list = []
        t5_weights_list = []
        cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for line in texts:
            if line in cache:
                z, tok, mul = cache[line]
            else:
                z, tok, mul = self._encode_line(line)
                cache[line] = (z, tok, mul)

            cross_attn_list.append(z)
            t5_ids_list.append(tok)
            t5_weights_list.append(mul)

        del cache

        return dict(
            crossattn=torch.cat([t.unsqueeze(0) if t.ndim == 2 else t for t in cross_attn_list], dim=0),
            t5xxl_ids=torch.stack(t5_ids_list),
            t5xxl_weights=torch.stack(t5_weights_list),
        )
