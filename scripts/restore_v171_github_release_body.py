#!/usr/bin/env python3
"""Restore GitHub Release v1.7.1 body: bilingual switcher + original body (unchanged)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "RELEASE_NOTES" / "_v1.7.1_original_body.md"
OUT = ROOT / "RELEASE_NOTES" / "v1.7.1_github_body.md"

SWITCHER = """<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/zhmd/v1.7.1.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

"""


def main() -> None:
    original = ORIGINAL.read_text(encoding="utf-8")
    body = SWITCHER + original
    if len(body) > 125_000:
        raise SystemExit(f"Body too long for GitHub: {len(body)} chars (max 125000)")
    OUT.write_text(body, encoding="utf-8")
    print(f"Wrote {OUT} ({len(body)} chars)")


if __name__ == "__main__":
    main()
