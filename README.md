<h1 align="center">Stable Diffusion WebUI Forge - Nunchaku</h1>

⚠️ **This project is in beta testing and is not ready for general users.**

⚠️ **This project requires Python 3.13 exclusively.**

## About

This program is a fork that integrates Nunchaku support into Stable Diffusion WebUI Forge. It is built upon the following repositories:

### Base Repositories

- **[stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)**  
  Original Forge implementation by [@lllyasviel](https://github.com/lllyasviel)

- **[sd-webui-forge-classic (neo branch)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**  
  Classic Forge implementation by [@Haoming02](https://github.com/Haoming02)

- **[ComfyUI-nunchaku-unofficial-loader](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader)**  
  Nunchaku integration reference implementation by [@ussoewwin](https://github.com/ussoewwin)

- **[ADetailer_without_mediapipe](https://github.com/ussoewwin/ADetailer_without_mediapipe)**  
  ADetailer with Python 3.13 support using InsightFace instead of MediaPipe by [@ussoewwin](https://github.com/ussoewwin)

## Changelog

### Version 1.0.6

- **Nunchaku SDXL Loader**
  - Implemented `SVDQUNet2DConditionModel` for Nunchaku SDXL UNet loading
  - Automatic model detection via `qweight` format
  - CLIP text encoder conversion for Nunchaku SDXL format
  - SVDQ quantization support with bfloat16 precision

- **Nunchaku LoRA Loader**
  - Runtime LoRA application using forward-add method for quantized UNet
  - A1111 LoRA format support with automatic key conversion
  - Multiple LoRAs with individual strength control
  - Reference: ComfyUI-nunchaku-unofficial-loader v2.1

- **ControlNet Support**
  - ControlNet signal conversion from Forge Neo format to diffusers format
  - Full compatibility with Nunchaku SDXL UNet
  - Reference: ComfyUI-nunchaku-unofficial-loader v2.6

### Version 1.0.7

- **Added ADetailer as built-in extension**
  - Integrated [ADetailer_without_mediapipe](https://github.com/ussoewwin/ADetailer_without_mediapipe) as a standard built-in feature
  - Python 3.13 compatible with InsightFace instead of MediaPipe
  - Includes YOLOv8, YOLOv11, and InsightFace hybrid detection system
  - Fixed ControlNet preprocessor initialization issue
  - Added `extensions-builtin/adetailer/models/` to `.gitignore`