"""
Forge glue for ComfyUI Lumina2 TE (do not patch ComfyUI-master).

Load: ``comfy.sd.load_text_encoder_state_dicts``
Encode: ``comfy.sd.CLIP.encode_from_tokens`` (same as Comfy nodes)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

from backend.text_processing import emphasis, parsing
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
    def __init__(self, clip, tokenizer=None):
        del tokenizer
        self.clip = clip

        self.id_start = 2
        self.id_pad = 0

    def _comfy_tokenize_text(self, text: str, weight: float) -> tuple[list[int], list[float]]:
        pairs = self.clip.tokenize(text)
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

    def _encode_chunk(self, tokens: list[int], multipliers: list[float]):
        token_weight_pairs = {
            _COMFY_CLIP_KEY: [[(int(t), float(m)) for t, m in zip(tokens, multipliers)]]
        }
        z = self.clip.encode_from_tokens(token_weight_pairs)
        if z.ndim == 3:
            return z[0]
        return z

    def _encode_line(self, line: str):
        """Encode one template-processed line (Comfy ``clip.tokenize`` + ``encode_from_tokens``)."""
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
            line = self.process_template(line, texts.is_negative_prompt)

            if line in cache:
                line_z_values = cache[line]
            else:
                parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
                has_break = any(text == "BREAK" and weight == -1 for text, weight in parsed)

                if not has_break and all(w == 1.0 for _, w in parsed):
                    line_z_values = [self._encode_line(line)]
                else:
                    chunks, token_count = self.tokenize_line(line)
                    line_z_values = []
                    for chunk in chunks:
                        z = self._encode_chunk(chunk.tokens, chunk.multipliers)
                        line_z_values.append(z)
                cache[line] = line_z_values

            zs.extend(line_z_values)

        return zs
