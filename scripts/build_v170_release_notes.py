#!/usr/bin/env python3
"""Maintainer script: regenerate v1.7.0 release files from local (untracked) docs/ANIMA_IMPLEMENTATION_GUIDE.md."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "ANIMA_IMPLEMENTATION_GUIDE.md"
PROSE_ZH = ROOT / "zhmd" / "_v1.7.0_prose_zh.md"
OUT_EN = ROOT / "RELEASE_NOTES" / "v1.7.0.md"
OUT_ZH = ROOT / "zhmd" / "v1.7.0.md"
OUT_GH = ROOT / "RELEASE_NOTES" / "v1.7.0_github_body.md"

EN_SWITCHER = """<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/zhmd/v1.7.0.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

"""

ZH_SWITCHER = """<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.0"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

"""

GH_BODY = """<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/zhmd/v1.7.0.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## Overview

Documentation release: **Anima Integration Guide (Forge Nunchaku — native pipeline)** for v1.7.0, with bilingual language switchers on the English and Chinese release-note pages.

The **full guide** (sections 1–9, including complete source with block comments) is in the repository files linked below. This GitHub Release page carries the overview only.

---

## Full release notes (complete)

| Language | File |
|----------|------|
| **EN** | [`RELEASE_NOTES/v1.7.0.md`](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/RELEASE_NOTES/v1.7.0.md) |
| **中文** | [`zhmd/v1.7.0.md`](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/zhmd/v1.7.0.md) |

---

## What's new (v1.7.0)

### Anima model support

- Native Forge support for [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) and compatible single-file checkpoints (e.g. `anima-base-v1.0.safetensors`, community merges such as `waiANIMA_pw3.safetensors`).
- New **UI Preset: Anima** in the checkpoint manager; load **Additional modules** `qwen_3_06b_base.safetensors` (Qwen3 text encoder) and `qwen_image_vae.safetensors` (VAE). T5 tokenizer vocabulary is used for `llm_adapter` cross-attention without a separate T5/UMT5 weight file.
- Native `backend.nn.anima` UNet with flow-matching sampling; **Shift** in the UI applies to the noise schedule.

### Bilingual language switchers

- **`RELEASE_NOTES/v1.7.0.md`**: EN ↔ 中文 switcher linking to `zhmd/v1.7.0.md`.
- **`zhmd/v1.7.0.md`**: 中文 ↔ EN switcher linking back to this release page.
- **`Changelog/CHANGELOG.md`** and **`zhmd/CHANGELOG.md`**: v1.7.0 entry links to both EN release and 中文 notes.

---

## Repository layout (docs)

| English | Chinese |
|---------|---------|
| `README.md` | `zhmd/README.md` |
| `Changelog/CHANGELOG.md` | `zhmd/CHANGELOG.md` |
| `RELEASE_NOTES/v1.7.0.md` | `zhmd/v1.7.0.md` |
| GitHub Release body (EN overview) | Linked from switcher → `zhmd/v1.7.0.md` on `main` |

Changelog bullets in both READMEs are kept in sync under **Changelog** / **更新日志**.

---

## Upgrade notes

- Pull or refresh docs to read the full implementation guide in EN or 中文.
- Open [中文 release notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/zhmd/v1.7.0.md) from the switcher above.

"""

# Prose lines outside code fences in §5–§7 (exact EN line → ZH line).
SECTION567_LINE_ZH: dict[str, str] = {
    "## 5. Full source of new files": "## 5. 新增文件的完整源码",
    "Below is the **full source (A)** from commit `a8f6373` (no omissions).": (
        "以下为 commit `a8f6373` 的**完整源码（A）**（无省略）。"
    ),
    "**Notes appear immediately before each heading**; the next fenced block is the complete file.": (
        "**说明紧接在每个标题之前**；下一个代码块为该文件的完整源码。"
    ),
    "### 5.1 Python (core)": "### 5.1 Python（核心）",
    "### 5.2 JSON / HF metadata": "### 5.2 JSON / HF 元数据",
    "### 5.3 Large tokenizer assets": "### 5.3 大型 tokenizer 资源",
    "Large HF tokenizer files are **not inlined** (151k+ lines). They ship under `backend/huggingface/circlestone-labs/Anima/tokenizer*` on disk.": (
        "大型 HF tokenizer 文件**未内联**（15 万行以上）。它们位于磁盘上的 `backend/huggingface/circlestone-labs/Anima/tokenizer*`。"
    ),
    "## 6. Full source of modified files": "## 6. 修改文件的完整源码",
    "Entire files at commit **`a8f6373`** (no omissions). Notes before each file; fenced block is the complete file.": (
        "commit **`a8f6373`** 时点的完整文件（无省略）。每个文件前有说明；代码块为完整源码。"
    ),
    "## 7. Full source of removed files": "## 7. 删除文件的完整源码",
    "Removed at **`a8f6373`**; full source from prior commit **`95832e0`**.": (
        "在 **`a8f6373`** 删除；完整源码来自先前 commit **`95832e0`**。"
    ),
    "#### Notes": "#### 说明",
    "| Path | Role |": "| 路径 | 职责 |",
    "| Path | Why import |": "| 路径 | import 原因 |",
}


def body_from_guide(text: str) -> str:
    marker = "## Table of contents"
    idx = text.find(marker)
    if idx == -1:
        raise SystemExit("Guide missing Table of contents")
    return text[idx:]


def translate_line(line: str) -> str:
    stripped = line.rstrip("\n")
    if stripped in SECTION567_LINE_ZH:
        return SECTION567_LINE_ZH[stripped] + "\n"
    for en, zh in SECTION567_LINE_ZH.items():
        if en.startswith("|") and en in stripped:
            return stripped.replace(en.split("|")[2].strip(), zh.split("|")[2].strip()) + "\n"
    return line


def build_section567_zh(en_lines: list[str]) -> str:
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
            out.append(translate_line(line))
    return "".join(out)


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"Source not found: {SOURCE}")
    if not PROSE_ZH.is_file():
        raise SystemExit(f"Chinese prose not found: {PROSE_ZH}")

    guide = SOURCE.read_text(encoding="utf-8")
    body = body_from_guide(guide)

    OUT_EN.parent.mkdir(parents=True, exist_ok=True)
    OUT_EN.write_text(EN_SWITCHER + body, encoding="utf-8", newline="\n")
    OUT_GH.write_text(GH_BODY, encoding="utf-8", newline="\n")

    prose_zh = PROSE_ZH.read_text(encoding="utf-8")
    idx5 = body.find("## 5. Full source")
    idx8 = body.find("## 8. Recommended")
    if idx5 == -1 or idx8 == -1:
        raise SystemExit("Section 5 or 8 marker not found in English body")

    section567_en = body[idx5:idx8]
    section567_zh = build_section567_zh(section567_en.splitlines(keepends=True))

    idx5z = prose_zh.find("<!-- BUILD_INSERT_S567 -->")
    idx8z = prose_zh.find("## 8.")
    if idx5z == -1 or idx8z == -1:
        raise SystemExit("Chinese prose missing BUILD_INSERT_S567 or section 8")
    prose_1_4 = prose_zh[:idx5z].rstrip()
    prose_8_9 = prose_zh[idx8z:].strip()

    zh_body = prose_1_4 + "\n\n" + section567_zh.strip() + "\n\n" + prose_8_9 + "\n"
    OUT_ZH.write_text(ZH_SWITCHER + zh_body, encoding="utf-8", newline="\n")

    print(f"Wrote {OUT_EN} ({OUT_EN.stat().st_size} bytes)")
    print(f"Wrote {OUT_ZH} ({OUT_ZH.stat().st_size} bytes)")
    print(f"Wrote {OUT_GH} ({OUT_GH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
