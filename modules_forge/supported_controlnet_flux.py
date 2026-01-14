"""
Flux ControlNet support for Forge (Nunchaku SVDQ).
This module handles the integration of ComfyUI's ControlNet system with Forge's
SVDQFluxTransformer2DModel.
"""

import os
import sys
import torch

# ComfyUI標準のControlNetシステムを使用するため、ComfyUIへのパスを追加
comfy_path = r"D:\USERFILES\ComfyUI\ComfyUI"
if comfy_path not in sys.path:
    sys.path.insert(0, comfy_path)

import comfy.controlnet
import comfy.model_detection
import comfy.utils
import comfy.model_management
import comfy.model_base
from modules_forge.supported_controlnet import ControlModelPatcher


class FluxVAEWrapper:
    """
    Wrapper for Forge's VAE to provide ComfyUI-compatible interface.
    
    ComfyUI's ControlNet.get_control calls vae.encode(self.cond_hint.movedim(1, -1)),
    so input is always [B, H, W, C] format.
    Forge VAE.encode also expects [B, H, W, C] with values in [0, 1].
    """
    def __init__(self, vae):
        self.vae = vae
        print(f"[FluxVAEWrapper] Initialized with VAE: {type(vae)}")

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def spacial_compression_encode(self):
        """Return the spatial compression ratio for the VAE encoder."""
        if hasattr(self.vae, "first_stage_model") and hasattr(self.vae.first_stage_model, "downscale_ratio"):
            return self.vae.first_stage_model.downscale_ratio
        if hasattr(self.vae, "downscale_ratio"):
            return self.vae.downscale_ratio
        return 8  # Default for Flux VAE

    def encode(self, x):
        """
        Encode input image to latent space.
        
        Input x is [B, H, W, C], values in [0, 1].
        Forge VAE.encode expects the same format.
        """
        print(f"[FluxVAEWrapper] encode called. Input shape: {x.shape}, dtype: {x.dtype}")
        print(f"[FluxVAEWrapper] Input range: min={x.min().item():.4f}, max={x.max().item():.4f}")
        
        # Direct pass-through to Forge VAE
        result = self.vae.encode(x)
        print(f"[FluxVAEWrapper] Output shape: {result.shape}, range: min={result.min().item():.4f}, max={result.max().item():.4f}")
        return result

# Note: ControlNet.get_control and T2IAdapter.get_control patches are now in
# modules_forge/supported_controlnet.py to ensure they're applied for all ControlNet types



class FluxControlNetPatcher(ControlModelPatcher):
    """
    ControlNet patcher specifically designed for Nunchaku Flux1 models.
    Uses ComfyUI's standard ControlNet system for compatibility.
    """
    
    @staticmethod
    def try_build_from_state_dict(controlnet_data, ckpt_path):
        """
        Attempt to load a Flux ControlNet from state dict.
        Uses ComfyUI's standard load_controlnet_state_dict function.
        """
        try:
            # Use ComfyUI's standard loader which handles Diffusers format conversion
            control = comfy.controlnet.load_controlnet_state_dict(controlnet_data)
            
            if control is None:
                print(f"[FluxControlNet] Failed to load Flux ControlNet: {ckpt_path}")
                return None
            
            print(f"[FluxControlNet] Successfully loaded Flux ControlNet: {ckpt_path}")
            print(f"[FluxControlNet] Model type: {type(control.control_model)}")
            if hasattr(control.control_model, 'latent_input'):
                print(f"[FluxControlNet] latent_input: {control.control_model.latent_input}")
            if hasattr(control.control_model, 'num_union_modes'):
                print(f"[FluxControlNet] num_union_modes: {control.control_model.num_union_modes}")
            if hasattr(control, 'latent_format'):
                print(f"[FluxControlNet] latent_format: {control.latent_format}")
            if hasattr(control, 'compression_ratio'):
                print(f"[FluxControlNet] compression_ratio: {control.compression_ratio}")
            
            return FluxControlNetPatcher(control)
        except Exception as e:
            print(f"[FluxControlNet] Error loading Flux ControlNet: {e}")
            import traceback
            traceback.print_exc()
            return None

    def __init__(self, control_model):
        super().__init__(control_model)
        self.control_model = control_model
        
        # Override get_models() to prevent model loading conflicts
        def get_models_wrapper():
            return []
        control_model.get_models = get_models_wrapper

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        """
        Set up ControlNet for Nunchaku Flux1 sampling.
        
        This method:
        1. Verifies the model is a Nunchaku Flux1 model
        2. Wraps the VAE for ComfyUI compatibility
        3. Sets up the ControlNet linked list
        """
        unet = process.sd_model.forge_objects.unet
        
        # Verify this is a Nunchaku Flux1 model
        from backend.nn.svdq import SVDQFluxTransformer2DModel
        is_nunchaku_flux1 = False
        if hasattr(unet, 'model') and hasattr(unet.model, 'diffusion_model'):
            is_nunchaku_flux1 = isinstance(unet.model.diffusion_model, SVDQFluxTransformer2DModel)
        
        if not is_nunchaku_flux1:
            print("[FluxControlNet] WARNING: This ControlNet is designed for Nunchaku Flux1 only!")
            return
        
        print(f"[FluxControlNet] Processing ControlNet for Nunchaku Flux1")
        print(f"[FluxControlNet] Strength: {self.strength}, Range: {self.start_percent}-{self.end_percent}")
        
        # Get and wrap VAE
        vae = process.sd_model.forge_objects.vae
        if not hasattr(vae, "spacial_compression_encode"):
            vae = FluxVAEWrapper(vae)
        
        # Create ControlNet copy with condition hint
        cnet = self.control_model.copy()
        cnet = cnet.set_cond_hint(cond, self.strength, (self.start_percent, self.end_percent), vae=vae)
        
        # Debug: Print cond hint info
        if hasattr(cnet, 'cond_hint_original') and cnet.cond_hint_original is not None:
            print(f"[FluxControlNet] cond_hint_original shape: {cnet.cond_hint_original.shape}")
            print(f"[FluxControlNet] cond_hint_original range: {cnet.cond_hint_original.min().item():.4f} to {cnet.cond_hint_original.max().item():.4f}")
        
        # Override get_models() on the copy too
        def get_models_wrapper():
            return []
        cnet.get_models = get_models_wrapper
        
        # Add to ControlNet linked list
        unet.add_patched_controlnet(cnet)
        
        print(f"[FluxControlNet] ControlNet set up complete")
        return
