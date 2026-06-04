"""
Forge glue for ComfyUI Anima TE path (do not patch ComfyUI-master).

Upstream: ``comfy.text_encoders.anima.AnimaTokenizer`` + Forge-loaded Qwen3 weights.
``llm_adapter`` / ``preprocess_text_embeds`` live on ``comfy.ldm.anima.model.Anima`` only.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.nn.llm.llama import Qwen3_06B

import torch

from backend import memory_management
from comfy.text_encoders.anima import AnimaTokenizer


class AnimaTextProcessingEngine:
    def __init__(self, text_encoder, qwen_tokenizer=None, t5_tokenizer=None):
        del qwen_tokenizer, t5_tokenizer
        self.text_encoder: "Qwen3_06B" = text_encoder
        self.comfy_tokenizer = AnimaTokenizer()
        self.id_pad = 151643

    def tokenize(self, texts):
        qwen_batches = []
        t5_batches = []
        for text in texts:
            pairs = self.comfy_tokenizer.tokenize_with_weights(text, return_word_ids=False)
            qwen_batches.append([t[0] for t in pairs["qwen3_06b"][0]])
            t5_batches.append([(t[0], t[1]) for t in pairs["t5xxl"][0]])
        return qwen_batches, t5_batches

    def __call__(self, texts):
        cross_attn_list = []
        t5_ids_list = []
        t5_weights_list = []
        cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for line in texts:
            if line in cache:
                z, tok, mul = cache[line]
            else:
                pairs = self.comfy_tokenizer.tokenize_with_weights(line, return_word_ids=False)
                qwen_tokens = [t[0] for t in pairs["qwen3_06b"][0]]
                t5_tokens = [t[0] for t in pairs["t5xxl"][0]]
                t5_multipliers = [t[1] for t in pairs["t5xxl"][0]]

                z: torch.Tensor = self._encode_qwen(qwen_tokens)
                tok = torch.tensor(t5_tokens, dtype=torch.long)
                mul = torch.tensor(t5_multipliers, dtype=torch.float32)
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

    def _encode_qwen(self, tokens: list[int]) -> torch.Tensor:
        device = memory_management.text_encoder_device()
        attention_mask = []
        tokens_temp = []
        eos = False
        for token in tokens:
            attention_mask.append(0 if eos else 1)
            tokens_temp.append(int(token))
            if not eos and token == self.id_pad:
                eos = True

        embeds = self.text_encoder.get_input_embeddings()(torch.tensor([tokens_temp], device=device, dtype=torch.long))
        mask = torch.tensor([attention_mask], device=device, dtype=torch.long)
        z, _ = self.text_encoder(input_ids=None, embeds=embeds, attention_mask=mask, num_tokens=[sum(attention_mask)], embeds_info=[])
        return z[0]
