# v1.3.9 Release Notes

## Config-Presets as Built-in Extension

This release adds **Config-Presets** as a built-in extension for sd-webui-forge-classic-neo.

### What is Config-Presets?

Config-Presets is an extension (originally [Zyin055/Config-Presets](https://github.com/Zyin055/Config-Presets), MIT licensed) that adds a configurable dropdown to the txt2img and img2img tabs, allowing you to quickly switch between different generation settings.

### Key Features

- **Preset Selection**: Choose from existing presets to apply settings in bulk
- **Create/Delete Presets**: Save current UI values as new presets, or delete existing ones
- **Custom Fields**: Track components from other extensions via `*custom-tracked-components.txt`
- **Manual Editing**: Edit JSON config files directly and reload with the refresh button

### Included Presets (txt2img)

- SD1.5 / SD2.1 / SDXL / Flux.1 Dev & Schnell resolutions
- SD1.5 Low / Medium / High quality
- High res with Hires fix
- SD1.5 1080p / 1440p / 4k

### Included Presets (img2img)

- Low / Medium / High denoising

### Configuration

- Use `--configpresets-dir` to specify a custom directory for config files (e.g., to share settings across multiple installations)
- Config files: `config-txt2img.json`, `config-img2img.json`
- See [Config-Presets Complete Guide](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/extensions-builtin/Config-Presets/COMPLETE-GUIDE.md) for full documentation

---

For full changelog, see [CHANGELOG.md](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/Changelog/CHANGELOG.md).
