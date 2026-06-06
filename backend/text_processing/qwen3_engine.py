"""
Forge glue for ComfyUI Z-Image TE (do not patch ComfyUI-master).

Load: ``comfy.sd.load_text_encoder_state_dicts``
Encode: ``comfy.sd.CLIP.encode_from_tokens`` (same as Comfy nodes)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

from backend.text_processing import emphasis, parsing
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
    def __init__(self, clip, tokenizer=None):
        del tokenizer
        self.clip = clip

    def _comfy_tokenize_text(self, text: str, weight: float) -> tuple[list[int], list[float]]:
        pairs = self.clip.tokenize(text)
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
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            tokens, multipliers = self._comfy_tokenize_text(text, weight)
            chunk.tokens.extend(tokens)
            chunk.multipliers.extend(multipliers)

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks

    def _encode_chunk(self, tokens: list[int], multipliers: list[float]):
        token_weight_pairs = {
            _COMFY_CLIP_KEY: [[(int(t), float(m)) for t, m in zip(tokens, multipliers)]]
        }
        z = self.clip.encode_from_tokens(token_weight_pairs)
        if z.ndim == 3:
            return z[0]
        return z

    def _encode_line(self, line: str):
        tokens = self.clip.tokenize(line)
        z = self.clip.encode_from_tokens(tokens)
        if z.ndim == 3:
            return z[0]
        return z

    def __call__(self, texts: "SdConditioning"):
        zs = []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        for line in texts:
            if line in cache:
                line_z_values = cache[line]
            else:
                parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
                has_break = any(text == "BREAK" and weight == -1 for text, weight in parsed)

                if not has_break and all(w == 1.0 for _, w in parsed):
                    line_z_values = [self._encode_line(line)]
                else:
                    chunks = self.tokenize_line(line)
                    line_z_values = []
                    for chunk in chunks:
                        z = self._encode_chunk(chunk.tokens, chunk.multipliers)
                        line_z_values.append(z)
                cache[line] = line_z_values

            zs.extend(line_z_values)

        return zs
