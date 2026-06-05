#!/usr/bin/env python3
"""Sync RELEASE_NOTES/v1.7.1.md and zhmd/v1.7.1.md from ANIMA_COMFY_IMPORT_GUIDE line 12+."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "ANIMA_COMFY_IMPORT_GUIDE_7061e66.md"
OUT_EN = ROOT / "RELEASE_NOTES" / "v1.7.1.md"
OUT_ZH = ROOT / "zhmd" / "v1.7.1.md"
BUILD = ROOT / "scripts" / "build_v171_release_notes.py"

EN_SWITCHER = """<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/zhmd/v1.7.1.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

"""

ZH_SWITCHER = """<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.1"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

"""


def load_build_helpers():
    spec = importlib.util.spec_from_file_location("build_v171", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def body_from_guide_line12(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) < 12:
        raise SystemExit("Guide has fewer than 12 lines")
    return "".join(lines[11:])


def main() -> None:
    guide = SOURCE.read_text(encoding="utf-8")
    body = body_from_guide_line12(guide)

    OUT_EN.write_text(EN_SWITCHER + body, encoding="utf-8", newline="\n")
    print(f"Wrote {OUT_EN} ({len(EN_SWITCHER) + len(body)} chars)")

    build = load_build_helpers()
    zh_old = OUT_ZH.read_text(encoding="utf-8")
    i_toc = zh_old.find("## 目录")
    i13z = zh_old.find("## 13.")
    if i_toc == -1 or i13z == -1:
        raise SystemExit("zhmd/v1.7.1.md missing ## 目录 or ## 13.")
    prose_zh = zh_old[i_toc:i13z].rstrip() + "\n\n"

    idx13 = body.find("## 13. Full source")
    if idx13 == -1:
        raise SystemExit("Section 13 not found in guide body")
    s13_en = body[idx13:]
    s13_zh = build.build_section13_zh(s13_en.splitlines(keepends=True))
    zh_body = prose_zh + s13_zh.lstrip("\n")
    OUT_ZH.write_text(ZH_SWITCHER + zh_body, encoding="utf-8", newline="\n")
    print(f"Wrote {OUT_ZH} ({len(ZH_SWITCHER) + len(zh_body)} chars)")


if __name__ == "__main__":
    main()
