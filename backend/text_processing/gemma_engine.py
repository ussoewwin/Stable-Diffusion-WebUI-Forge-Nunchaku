# Forge glue: comfy.text_encoders.lumina2.LuminaTokenizer + Forge Gemma2_2B weights.
# https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/text_encoders/lumina2.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch

from backend import memory_management
from backend.text_processing import emphasis, parsing
from comfy.text_encoders.lumina2 import LuminaTokenizer
from modules.shared import opts

_COMFY_CLIP_KEY = "gemma2_2b"


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


class GemmaTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer=None):
        super().__init__()

        self.text_encoder = text_encoder
        spiece_model = getattr(text_encoder, "forge_spiece_model", None)
        tokenizer_data = {}
        if spiece_model is not None:
            tokenizer_data["spiece_model"] = spiece_model
        self.comfy_tokenizer = LuminaTokenizer(tokenizer_data=tokenizer_data)

        self.id_start = 2
        self.id_pad = 0

        self.intermediate_output = -2
        self.layer_norm_hidden_state = False

    def _comfy_tokenize_text(self, text: str, weight: float) -> tuple[list[int], list[float]]:
        pairs = self.comfy_tokenizer.tokenize_with_weights(text, return_word_ids=False, disable_weights=True)
        return _flat_comfy_token_batch(pairs, _COMFY_CLIP_KEY, weight)

    def tokenize(self, texts):
        tokenized = []
        for text in texts:
            tokens, _ = self._comfy_tokenize_text(text, 1.0)
            tokenized.append(tokens)
        return tokenized

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        first_in_chunk = True

        def next_chunk():
            nonlocal chunk
            nonlocal first_in_chunk

            if chunk.tokens and chunk.tokens[0] != self.id_start:
                chunk.tokens = [self.id_start] + chunk.tokens
                chunk.multipliers = [1.0] + chunk.multipliers

            chunks.append(chunk)
            chunk = PromptChunk()
            first_in_chunk = True

        for text, weight in parsed:
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            tokens, multipliers = self._comfy_tokenize_text(text, weight)
            if not first_in_chunk and tokens and tokens[0] == self.id_start:
                tokens = tokens[1:]
                multipliers = multipliers[1:]

            chunk.tokens.extend(tokens)
            chunk.multipliers.extend(multipliers)
            token_count += len(tokens)
            first_in_chunk = False

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks, token_count

    @staticmethod
    def process_template(text: str, negative: bool) -> str:
        if "<Prompt Start>" in text:
            return text

        from modules.shared import opts

        if negative:
            return "\n".join([opts.neta_template_negative, text])
        else:
            return "\n".join([opts.neta_template_positive, text])

    def __call__(self, texts: "SdConditioning"):
        zs = []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        for line in texts:
            line = self.process_template(line, texts.is_negative_prompt)

            if line in cache:
                line_z_values = cache[line]
            else:
                chunks, token_count = self.tokenize_line(line)
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
