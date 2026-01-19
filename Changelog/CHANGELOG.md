# Changelog

## Version 1.3.1

- **Added Diffsynth Union ControlNet support for ZIT (standard and Nunchaku)**
  - Full Diffsynth Union ControlNet support for both standard Z-Image Turbo (ZIT) and Nunchaku ZIT models
  - Multiple ControlNet models can be used simultaneously (Union ControlNet)
  - Supports ZIT ControlNet models (e.g., `z-image-turbo-controlnet.safetensors`)
  - Automatic model detection for ZIT models via NextDiT model type
  - Strict model type checking to ensure compatibility only with ZIT models
  - VAE wrapper for seamless Forge VAE integration with ComfyUI ControlNet interface
  - Complete implementation based on ComfyUI's nodes_model_patch.py
  - Fixed double patching and stale patches issues causing RecursionError
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.3.1) for details

## Version 1.3.0

- **Added RES4LYF Sampler Support**
  - Full support for RES4LYF (RES4) samplers for all model types
  - Support for Nunchaku Qwen Image, Nunchaku Flux1, Nunchaku SDXL, standard SDXL, and standard Flux1 models
  - Comprehensive sampler collection including multistep (res_2m, res_3m, etc.) and exponential (res_2s, res_3s, etc.) variants
  - ODE version support for non-implicit samplers
  - Robust model detection and handling for both Forge and ComfyUI model structures
  - Automatic CONST and EPS model type detection for proper sampling behavior
  - Fixed model_sampling access for Forge models via KModel wrapper
  - Improved compatibility with ComfyUI-master directory structure
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.3.0) for details

## Version 1.2.1

- **Added Union ControlNet support for Nunchaku Qwen Image**
  - Full Union ControlNet support for Nunchaku Qwen Image (QI) models
  - Multiple ControlNet models can be used simultaneously (Union ControlNet)
  - Supports Qwen Image Union ControlNet models (e.g., `Qwen-Image-InstantX-ControlNet-Union.safetensors`)
  - Automatic model detection via `transformer_blocks.0.img_mlp.net.0.proj.weight` key
  - Strict model type checking to ensure compatibility only with Nunchaku Qwen Image models
  - VAE wrapper for seamless Forge VAE integration with ComfyUI ControlNet interface
  - Complete and independent implementation separate from Flux ControlNet
  - Fixed device placement issues for ControlNet model loading
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.2.1) for details

## Version 1.2.0

- **Added LoRA support for Nunchaku Qwen Image and Z-Image models**
  - Full LoRA support for Nunchaku Qwen Image (QI) models
  - Full LoRA support for Nunchaku Z-Image (ZIT) models
  - Completely separated implementations for Qwen Image and Z-Image
  - Comprehensive logging with format detection for all LoRAs
  - Robust change detection to handle model reloads correctly
  - Support for standard LoRA formats (lora_A/lora_B, lora_up/lora_down)
  - AWQ quantization layer handling with safety switch
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.2.0) for details

## Version 1.1.0

- **Added Union ControlNet support for Flux1 and Nunchaku Flux1**
  - Full Union ControlNet support for both Flux1 and Nunchaku Flux1 models
  - Multiple ControlNet models can be used simultaneously
  - Automatic model detection via `controlnet_x_embedder.weight` key
  - VAE wrapper for seamless Forge VAE integration
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.1.0) for details

## Version 1.0.7

- **Added ADetailer as built-in extension**
  - Integrated [ADetailer_without_mediapipe](https://github.com/ussoewwin/ADetailer_without_mediapipe) as a standard built-in feature
  - Python 3.13 compatible with InsightFace instead of MediaPipe
  - Includes YOLOv8, YOLOv11, and InsightFace hybrid detection system
  - Fixed ControlNet preprocessor initialization issue
  - Added `extensions-builtin/adetailer/models/` to `.gitignore`

## Version 1.0.6

- Nunchaku SDXL loader, LoRA loader, and ControlNet support completed
