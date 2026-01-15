"""
Flux ControlNet support for Forge.
This module handles the integration of Flux ControlNet with Forge's
SVDQFluxTransformer2DModel and IntegratedFluxTransformer2DModel.

Uses Forge's bundled ComfyUI package (no external ComfyUI installation required).
"""

import torch

# Use Forge's bundled comfy package (located at repository root)
import comfy.controlnet
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
        Uses Forge's bundled comfy.controlnet loader.
        """
        try:
            # Use Forge's bundled loader which handles Diffusers format conversion
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
        Set up ControlNet for Flux1 sampling.
        
        This method:
        1. Verifies the model is a Flux1 model (Nunchaku or standard)
        2. Wraps the VAE for ComfyUI compatibility
        3. Sets up the ControlNet linked list
        """
        unet = process.sd_model.forge_objects.unet
        
        # Check for Flux1 models (both Nunchaku and standard)
        from backend.nn.svdq import SVDQFluxTransformer2DModel
        from backend.nn.flux import IntegratedFluxTransformer2DModel
        
        is_flux1 = False
        flux_type = None
        if hasattr(unet, 'model') and hasattr(unet.model, 'diffusion_model'):
            diffusion_model = unet.model.diffusion_model
            if isinstance(diffusion_model, SVDQFluxTransformer2DModel):
                is_flux1 = True
                flux_type = "Nunchaku"
            elif isinstance(diffusion_model, IntegratedFluxTransformer2DModel):
                is_flux1 = True
                flux_type = "Standard"
        
        if not is_flux1:
            print("[FluxControlNet] WARNING: This ControlNet is designed for Flux1 models only!")
            return
        
        print(f"[FluxControlNet] Processing ControlNet for {flux_type} Flux1")
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
