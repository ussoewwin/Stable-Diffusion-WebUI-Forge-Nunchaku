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

## Features

### 🎯 Key Features

- **Nunchaku SDXL Support**
  - Full support for Nunchaku SDXL models with LoRA and ControlNet
  - Optimized 4-bit quantized SDXL models with SVDQ W4A4 engine (Rank 128)
  - Automatic model detection and loading
  - LoRA loader support for Nunchaku SDXL models
  - ControlNet support for Nunchaku SDXL models

- **Union ControlNet for Flux1 and Nunchaku Flux1**
  - ✅ **Union ControlNet support for both Flux1 and Nunchaku Flux1 models**
  - Multiple ControlNet models can be used simultaneously (Union ControlNet)
  - Supports Flux Union ControlNet models (e.g., `flux_shakker_labs_union_pro-2-fp8.safetensors`)
  - Automatic model detection and loading via `controlnet_x_embedder.weight` key detection
  - VAE wrapper for seamless Forge VAE integration with ComfyUI ControlNet interface

  <img src="png/f1cn.png" alt="Flux1 ControlNet Union" width="400">

  *Flux1 Union ControlNet workflow example*

- **Built-in ADetailer**
  - Python 3.13 compatible face detection and enhancement
  - YOLOv8, YOLOv11, and InsightFace hybrid detection system
  - Enhanced face detection accuracy with complementary detection
  - Automatic model download and management
  - SDXL/Pony optimized detection thresholds

- **Python 3.13 Exclusive**
  - Latest Python features and performance improvements
  - Optimized for modern hardware and workflows
  - Future-proof architecture

## Models

### Nunchaku SDXL Models

Nunchaku SDXL models are available from the following repository:

- **[Nunchaku-R128-SDXL-Series](https://huggingface.co/ussoewwin/Nunchaku-R128-SDXL-Series)**  
  High-fidelity 4-bit quantized SDXL models optimized using Nunchaku (SVDQ W4A4) engine with Rank 128 (r128) for maximum quality preservation.

**Installation:**
1. Download the `.safetensors` files from the repository
2. Place them in `models/Stable-diffusion/` directory
3. The models will be automatically detected and loaded

## Changelog

### Version 1.2.0

- **Added LoRA support for Nunchaku Qwen Image and Z-Image models**
  - Full LoRA support for Nunchaku Qwen Image (QI) models
  - Full LoRA support for Nunchaku Z-Image (ZIT) models
  - Completely separated implementations for Qwen Image and Z-Image
  - Comprehensive logging with format detection for all LoRAs
  - Robust change detection to handle model reloads correctly
  - Support for standard LoRA formats (lora_A/lora_B, lora_up/lora_down)
  - AWQ quantization layer handling with safety switch
  - See [Release Notes](docs/RELEASE_NOTES_v1.2.0.md) for details

### Version 1.1.0

- **Added Union ControlNet support for Flux1 and Nunchaku Flux1**
  - Full Union ControlNet support for both Flux1 and Nunchaku Flux1 models
  - Multiple ControlNet models can be used simultaneously
  - Automatic model detection via `controlnet_x_embedder.weight` key
  - VAE wrapper for seamless Forge VAE integration
  - See [Release Notes](docs/RELEASE_NOTES_v1.1.0.md) for details

### Version 1.0.7

- **Added ADetailer as built-in extension**
  - Integrated [ADetailer_without_mediapipe](https://github.com/ussoewwin/ADetailer_without_mediapipe) as a standard built-in feature
  - Python 3.13 compatible with InsightFace instead of MediaPipe
  - Includes YOLOv8, YOLOv11, and InsightFace hybrid detection system
  - Fixed ControlNet preprocessor initialization issue
  - Added `extensions-builtin/adetailer/models/` to `.gitignore`

### Version 1.0.6

- Nunchaku SDXL loader, LoRA loader, and ControlNet support completed