#!/usr/bin/env python3
"""Build bilingual v1.7.1 release notes (EN in RELEASE_NOTES/, ZH in zhmd/)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "ANIMA_COMFY_IMPORT_GUIDE_7061e66.md"
PROSE_ZH = ROOT / "zhmd" / "_v1.7.1_prose_zh.md"
OUT_EN = ROOT / "RELEASE_NOTES" / "v1.7.1.md"
OUT_ZH = ROOT / "zhmd" / "v1.7.1.md"

EN_SWITCHER = """\
<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/zhmd/v1.7.1.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

"""

ZH_SWITCHER = """\
<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.1"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

"""

# Section 13+ prose lines (outside code fences) — file-path headers stay; descriptions translated.
SECTION13_LINE_ZH: dict[str, str] = {
    "## 13. Full source (7061e66, block comments, no omissions)": (
        "## 13. 完整源码（7061e66，块注释，无省略）"
    ),
    "Files **added, changed, or deleted** in commit `7061e66` are included **with no line omitted**. Each block is preceded by **`#` comment notes** (meaning of ComfyUI import). No fragmented end-of-line comments.": (
        "commit `7061e66` 中**新增、修改或删除**的文件**一行不漏**全部收录。每个代码块前附有 **`#` 注释说明**（ComfyUI import 的含义）。不使用行尾碎片化注释。"
    ),
    "Forge WebUI engine. Canonical DiT, `llm_adapter`, and `preprocess_text_embeds` live on Comfy `Anima`; this file only wires loaded parts (§2, §7.1).": (
        "Forge WebUI 引擎。规范 DiT、`llm_adapter` 和 `preprocess_text_embeds` 位于 Comfy `Anima`；本文件仅连接已加载的组件（§2、§7.1）。"
    ),
    "Splits T5/Qwen via Comfy `AnimaTokenizer`; Qwen embeddings only on Forge. `llm_adapter` fusion is unified on the UNet (§2.2, §7.2).": (
        "通过 Comfy `AnimaTokenizer` 拆分 T5/Qwen；Qwen embeddings 仅在 Forge 侧。**`llm_adapter` 融合统一在 UNet 上**（§2.2、§7.2）。"
    ),
    "For Anima, only `Qwen3_06B` (hidden 1024) is loaded as TE. No `llm_adapter`; unified on Comfy `Anima` on the UNet (§7.3). Full file also covers other models (e.g. Qwen3_4B).": (
        "Anima 仅加载 `Qwen3_06B`（hidden 1024）作为 TE。无 `llm_adapter`；统一使用 UNet 上的 Comfy `Anima`（§7.3）。完整文件还涵盖其他模型（如 Qwen3_4B）。"
    ),
    "Fixes ckpt vs Comfy attention projection name differences (`o_proj` / `output_proj`) only. No DiT reimplementation (§6.1).": (
        "仅修正 checkpoint 与 Comfy attention projection 键名差异（`o_proj` / `output_proj`）。不重新实现 DiT（§6.1）。"
    ),
    "Load hub. In 7061e66, **removed moving llm_adapter to TE**. UNet builds `comfy.ldm.anima.model.Anima` (§7.4). Anima-related blocks in the full source have comment notes.": (
        "加载中心。7061e66 中**移除了将 llm_adapter 移至 TE**。UNet 构建 `comfy.ldm.anima.model.Anima`（§7.4）。完整源码中 Anima 相关块附有注释说明。"
    ),
    "UNet call during sampling. For Comfy `Anima` only: expand **4D latent → 5D `(B,C,T,H,W)`** and squeeze back to 4D on output (§7.6).": (
        "采样时的 UNet 调用。仅对 Comfy `Anima`：将 **4D latent 扩展为 5D `(B,C,T,H,W)`**，输出时再 squeeze 回 4D（§7.6）。"
    ),
    "Converts Forge conditioning dict to Comfy form. Anima has **no pooled (vector)**, so do not set `model_conds[\"y\"]` (§9).": (
        "将 Forge conditioning dict 转换为 Comfy 形式。Anima **无 pooled（vector）**，因此不要设置 `model_conds[\"y\"]`（§9）。"
    ),
    "UNet config inference. For Anima (`x_embedder` + `llm_adapter`), **delegate to Comfy `detect_unet_config` at the top** to prevent Lumina2 misdetection (§7.5).": (
        "UNet 配置推断。对 Anima（`x_embedder` + `llm_adapter`），**优先委托 Comfy `detect_unet_config`**，防止 Lumina2 误检（§7.5）。"
    ),
    "Forge guess class definitions. `AnimaBase` / `AnimaWai68` / `Anima` are branch keys for `split_state_dict` and `loader` (§7.2).": (
        "Forge guess class 定义。`AnimaBase` / `AnimaWai68` / `Anima` 为 `split_state_dict` 与 `loader` 的分支键（§7.2）。"
    ),
    "### `backend/nn/anima.py` (deleted in `7061e66` — full source from prior commit)": (
        "### `backend/nn/anima.py`（7061e66 中已删除 — 先前 commit 的完整源码）"
    ),
    "Native DiT implementation (655 lines). Deleted in `7061e66` and replaced by `comfy.ldm.anima.model.Anima`.": (
        "Native DiT 实现（655 行）。7061e66 中删除，由 `comfy.ldm.anima.model.Anima` 替代。"
    ),
    "Key remap helper only. DiT lives in `comfy.ldm.anima.model.Anima` (§6.1).": (
        "仅键名 remap 辅助。DiT 位于 `comfy.ldm.anima.model.Anima`（§6.1）。"
    ),
    "Anima branch loads Comfy `Anima`; **does not** move `llm_adapter` to TE (§7.4).": (
        "Anima 分支加载 Comfy `Anima`；**不会**将 `llm_adapter` 移至 TE（§7.4）。"
    ),
    "Early Anima detection delegates to Comfy `detect_unet_config` (§6.4, §7.5).": (
        "Anima 早期检测委托给 Comfy `detect_unet_config`（§6.4、§7.5）。"
    ),
    "Metadata guesses only; no DiT class (§7.6).": (
        "仅 metadata guess；无 DiT 类（§7.6）。"
    ),
    "When `pooled_output is None`, omit `y` from `model_conds` (§7.7).": (
        "当 `pooled_output is None` 时，从 `model_conds` 省略 `y`（§7.7）。"
    ),
    "4D↔5D wrap for Comfy `Anima` still-image latents (§6.5).": (
        "Comfy `Anima` 静止图像 latent 的 4D↔5D 包装（§6.5）。"
    ),
    "**Deleted** in 7061e66 — native DiT removed; use Comfy import (§7.8).": (
        "**7061e66 中已删除** — native DiT 已移除；请使用 Comfy import（§7.8）。"
    ),
    "## Appendix — ComfyUI-master (do not edit)": "## 附录 — ComfyUI-master（请勿编辑）",
    "| Path | Why import |": "| 路径 | import 原因 |",
    "| Canonical DiT + `llm_adapter` on UNet |": "| UNet 上的规范 DiT + `llm_adapter` |",
    "| `AnimaTokenizer` (`t5xxl_ids` vocab) |": "| `AnimaTokenizer`（`t5xxl_ids` 词汇表）|",
    "| Reference for `preprocess_text_embeds` timing |": "| `preprocess_text_embeds` 时机的参考 |",
    "| UNet config inference from ckpt |": "| 从 ckpt 推断 UNet 配置 |",
}


def release_body_from_source(text: str) -> str:
    """Match GitHub release: start at Table of contents (skip title block)."""
    marker = "## Table of contents"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[idx:].lstrip("\n")


def translate_section13_line(line: str) -> str:
    stripped = line.rstrip("\n")
    if stripped in SECTION13_LINE_ZH:
        return SECTION13_LINE_ZH[stripped] + "\n"
    # Table row partial matches
    for en, zh in SECTION13_LINE_ZH.items():
        if en.startswith("|") and en in stripped:
            return stripped.replace(en.split("|")[2].strip(), zh.split("|")[2].strip()) + "\n"
    return line


def build_section13_zh(en_lines: list[str]) -> str:
    out: list[str] = []
    in_fence = False
    for line in en_lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
        else:
            out.append(translate_section13_line(line))
    return "".join(out)


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"Source not found: {SOURCE}")
    if not PROSE_ZH.is_file():
        raise SystemExit(f"Chinese prose not found: {PROSE_ZH}")

    source = SOURCE.read_text(encoding="utf-8")
    body = release_body_from_source(source)

    OUT_EN.parent.mkdir(parents=True, exist_ok=True)
    OUT_EN.write_text(EN_SWITCHER + body, encoding="utf-8", newline="\n")

    prose_zh = PROSE_ZH.read_text(encoding="utf-8")
    # prose_zh should start at ## Table of contents
    idx13_en = body.find("## 13. Full source")
    if idx13_en == -1:
        raise SystemExit("Section 13 marker not found in English body")

    section13_en = body[idx13_en:]
    section13_zh = build_section13_zh(section13_en.splitlines(keepends=True))

    zh_body = prose_zh.rstrip() + "\n\n" + section13_zh.lstrip("\n")
    OUT_ZH.write_text(ZH_SWITCHER + zh_body, encoding="utf-8", newline="\n")

    print(f"Wrote {OUT_EN} ({OUT_EN.stat().st_size} bytes)")
    print(f"Wrote {OUT_ZH} ({OUT_ZH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
