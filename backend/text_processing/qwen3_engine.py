# Forge glue: comfy.text_encoders.z_image.ZImageTokenizer + Forge Qwen3_4B weights.
# https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/text_encoders/z_image.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch

from backend import memory_management
from backend.text_processing import emphasis, parsing
from comfy.text_encoders.z_image import ZImageTokenizer
from modules.shared import opts

_COMFY_CLIP_KEY = "qwen3_4b"


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


def _flat_comfy_token_batch(pairs: dict, clip_key: str, segment_weight: float) -> tuple[list[int], list[float]]:
    tokens: list[int] = []
    multipliers: list[float] = []
    for batch in pairs[clip_key]:
        for item in batch:
            token_id = item[0]
            if isinstance(token_id, (int, float)):
                tokens.append(int(token_id))
                multipliers.append(float(segment_weight))
    return tokens, multipliers


class Qwen3TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer=None):
        super().__init__()

        self.text_encoder = text_encoder
        self.comfy_tokenizer = ZImageTokenizer()

        self.id_pad = 151643
        self.intermediate_output = -2
        self.layer_norm_hidden_state = False

    def _comfy_tokenize_text(self, text: str, weight: float) -> tuple[list[int], list[float]]:
        # ZImageTokenizer already passes disable_weights=True to super(); do not pass it here.
        pairs = self.comfy_tokenizer.tokenize_with_weights(text, return_word_ids=False)
        return _flat_comfy_token_batch(pairs, _COMFY_CLIP_KEY, weight)

    def tokenize(self, texts):
        tokenized = []
        for text in texts:
            tokens, _ = self._comfy_tokenize_text(text, 1.0)
            tokenized.append(tokens)
        return tokenized

    def tokenize_line(self, line: str):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)

        chunks = []
        chunk = PromptChunk()

        def next_chunk():
            nonlocal chunk

            chunks.append(chunk)
            chunk = PromptChunk()

        for text, weight in parsed:
            tokens, multipliers = self._comfy_tokenize_text(text, weight)
            chunk.tokens.extend(tokens)
            chunk.multipliers.extend(multipliers)

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks

    def __call__(self, texts: "SdConditioning"):
        zs = []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        for line in texts:
            if line in cache:
                line_z_values = cache[line]
            else:
                chunks = self.tokenize_line(line)
                line_z_values = []

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers

                    z = self.process_tokens([tokens], [multipliers])[0]
                    line_z_values.append(z)
                cache[line] = line_z_values

            zs.extend(line_z_values)

        return zs

    def process_embeds(self, batch_tokens):
        device = memory_management.text_encoder_device()
        self.text_encoder.to(device)

        embeds_out = []
        attention_masks = []
        num_tokens = []

        for tokens in batch_tokens:
            attention_mask = []
            tokens_temp = []
            eos = False

            for t in tokens:
                token = int(t)
                attention_mask.append(0 if eos else 1)
                tokens_temp += [token]
                if not eos and token == self.id_pad:
                    eos = True

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.text_encoder.get_input_embeddings()(tokens_embed)

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens

    def process_tokens(self, batch_tokens, batch_multipliers):
        embeds, mask, count = self.process_embeds(batch_tokens)
        _, z = self.text_encoder(
            None,
            attention_mask=mask,
            embeds=embeds,
            num_tokens=count,
            intermediate_output=self.intermediate_output,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
        )
        return z
