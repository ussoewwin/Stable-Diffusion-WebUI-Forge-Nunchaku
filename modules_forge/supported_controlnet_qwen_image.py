"""
Nunchaku Qwen Image ControlNet support for Forge.
This module handles the integration of Qwen Image ControlNet with Forge's
NunchakuQwenImageTransformer2DModel ONLY.

IMPORTANT: This is for Nunchaku Qwen Image models ONLY.
- Works with: NunchakuQwenImageTransformer2DModel (from backend.nn.svdq)
- Does NOT work with: Standard QwenImageTransformer2DModel (from backend.nn.qwen)
- Does NOT work with: Flux models (handled by supported_controlnet_flux.py)

Uses Forge's bundled ComfyUI package (no external ComfyUI installation required).
"""

import torch

# Use Forge's bundled comfy package (located at repository root)
import comfy.controlnet
from modules_forge.supported_controlnet import ControlModelPatcher


class QwenImageVAEWrapper:
    """
    Wrapper for Forge's VAE to provide ComfyUI-compatible interface.
    
    ComfyUI's ControlNet.get_control calls vae.encode(self.cond_hint.movedim(1, -1)),
    so input is always [B, H, W, C] format.
    Forge VAE.encode also expects [B, H, W, C] with values in [0, 1].
    """
    def __init__(self, vae):
        self.vae = vae
        print(f"[QwenImageVAEWrapper] Initialized with VAE: {type(vae)}")

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def spacial_compression_encode(self):
        """Return the spatial compression ratio for the VAE encoder."""
        # Use the same logic as ComfyUI's VAE.spacial_compression_encode():
        # Try to access [-1] for tuple (3D VAE), fallback to direct value for int (2D VAE)
        compression = None
        if hasattr(self.vae, "first_stage_model") and hasattr(self.vae.first_stage_model, "downscale_ratio"):
            compression = self.vae.first_stage_model.downscale_ratio
        elif hasattr(self.vae, "downscale_ratio"):
            compression = self.vae.downscale_ratio
        else:
            return 8  # Default for Qwen Image VAE (same as Flux)
        
        # Same logic as ComfyUI: try [-1] for tuple, fallback to direct value for int
        try:
            result = compression[-1]
        except (TypeError, IndexError):
            result = compression
        
        # Ensure we return an integer
        if isinstance(result, (int, float)):
            return int(result)
        else:
            # Fallback if result is somehow not a number
            return 8

    def encode(self, x):
        """
        Encode input image to latent space.
        
        Input x is [B, H, W, C], values in [0, 1].
        Forge VAE.encode expects the same format.
        """
        print(f"[QwenImageVAEWrapper] encode called. Input shape: {x.shape}, dtype: {x.dtype}")
        print(f"[QwenImageVAEWrapper] Input range: min={x.min().item():.4f}, max={x.max().item():.4f}")
        
        # Direct pass-through to Forge VAE
        result = self.vae.encode(x)
        print(f"[QwenImageVAEWrapper] Output shape: {result.shape}, range: min={result.min().item():.4f}, max={result.max().item():.4f}")
        return result

# Note: ControlNet.get_control and T2IAdapter.get_control patches are now in
# modules_forge/supported_controlnet.py to ensure they're applied for all ControlNet types



class QwenImageControlNetPatcher(ControlModelPatcher):
    """
    ControlNet patcher EXCLUSIVELY for Nunchaku Qwen Image models.
    
    This patcher:
    - ONLY works with NunchakuQwenImageTransformer2DModel (from backend.nn.svdq)
    - Will REJECT standard QwenImageTransformer2DModel (from backend.nn.qwen)
    - Will REJECT Flux models (handled by FluxControlNetPatcher)
    - Will REJECT all other model types
    
    Uses ComfyUI's standard ControlNet system for compatibility.
    """
    
    @staticmethod
    def try_build_from_state_dict(controlnet_data, ckpt_path):
        """
        Attempt to load a Qwen Image ControlNet from state dict.
        
        NOTE: This loads the ControlNet model, but it will ONLY be applied
        to NunchakuQwenImageTransformer2DModel (checked in process_before_every_sampling).
        Uses Forge's bundled comfy.controlnet loader.
        """
        try:
            # Use Forge's bundled comfy.controlnet.load_controlnet_state_dict() which:
            # 1. Detects Qwen Image ControlNet by "transformer_blocks.0.img_mlp.net.0.proj.weight"
            # 2. Calls load_controlnet_qwen_instantx() which uses comfy.ldm.qwen_image.controlnet.QwenImageControlNetModel
            # This ensures we use the correct Qwen Image ControlNet logic from comfy/ldm/qwen_image/controlnet.py
            control = comfy.controlnet.load_controlnet_state_dict(controlnet_data)
            
            if control is None:
                print(f"[QwenImageControlNet] Failed to load Qwen Image ControlNet: {ckpt_path}")
                return None
            
            # Verify that we're using comfy.ldm.qwen_image.controlnet.QwenImageControlNetModel
            control_model_type = type(control.control_model).__name__
            control_model_module = type(control.control_model).__module__
            print(f"[QwenImageControlNet] Successfully loaded Qwen Image ControlNet: {ckpt_path}")
            print(f"[QwenImageControlNet] Using {control_model_type} from {control_model_module}")
            print(f"[QwenImageControlNet] Expected: QwenImageControlNetModel from comfy.ldm.qwen_image.controlnet")
            if hasattr(control.control_model, 'latent_input'):
                print(f"[QwenImageControlNet] latent_input: {control.control_model.latent_input}")
            if hasattr(control.control_model, 'num_union_modes'):
                print(f"[QwenImageControlNet] num_union_modes: {control.control_model.num_union_modes}")
            if hasattr(control, 'latent_format'):
                print(f"[QwenImageControlNet] latent_format: {control.latent_format}")
            if hasattr(control, 'compression_ratio'):
                print(f"[QwenImageControlNet] compression_ratio: {control.compression_ratio}")
            
            return QwenImageControlNetPatcher(control)
        except Exception as e:
            print(f"[QwenImageControlNet] Error loading Qwen Image ControlNet: {e}")
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
        Set up ControlNet for Nunchaku Qwen Image sampling ONLY.
        
        This method:
        1. STRICTLY verifies the model is NunchakuQwenImageTransformer2DModel (NOT standard QwenImageTransformer2DModel)
        2. Returns immediately if model is not Nunchaku Qwen Image (to avoid affecting other models)
        3. Wraps the VAE for ComfyUI compatibility
        4. Sets up the ControlNet linked list
        """
        unet = process.sd_model.forge_objects.unet
        
        # STRICT check: Only allow NunchakuQwenImageTransformer2DModel (from backend.nn.svdq)
        # Reject standard QwenImageTransformer2DModel (from backend.nn.qwen) and all other models
        from backend.nn.svdq import NunchakuQwenImageTransformer2DModel
        from backend.nn.flux import IntegratedFluxTransformer2DModel
        from backend.nn.svdq import SVDQFluxTransformer2DModel
        
        is_nunchaku_qwen_image = False
        diffusion_model = None
        
        if hasattr(unet, 'model') and hasattr(unet.model, 'diffusion_model'):
            diffusion_model = unet.model.diffusion_model
            
            # Only accept NunchakuQwenImageTransformer2DModel from svdq.py
            if isinstance(diffusion_model, NunchakuQwenImageTransformer2DModel):
                # Additional check: make sure it's the correct one (has NunchakuModelMixin)
                if hasattr(diffusion_model, 'offload') and hasattr(diffusion_model, 'set_offload'):
                    is_nunchaku_qwen_image = True
        
        # REJECT all other models - return immediately without any processing
        if not is_nunchaku_qwen_image:
            if diffusion_model is not None:
                model_type = type(diffusion_model).__name__
                module_name = type(diffusion_model).__module__
                print(f"[QwenImageControlNet] REJECTED: This ControlNet is designed for Nunchaku Qwen Image models ONLY!")
                print(f"[QwenImageControlNet] Current model: {model_type} (from {module_name})")
                print(f"[QwenImageControlNet] Expected: NunchakuQwenImageTransformer2DModel (from backend.nn.svdq)")
                
                # Check what type it actually is
                if isinstance(diffusion_model, IntegratedFluxTransformer2DModel):
                    print(f"[QwenImageControlNet] Detected: Standard Flux1 model - ControlNet will be handled by FluxControlNetPatcher")
                elif isinstance(diffusion_model, SVDQFluxTransformer2DModel):
                    print(f"[QwenImageControlNet] Detected: Nunchaku Flux1 model - ControlNet will be handled by FluxControlNetPatcher")
                elif hasattr(diffusion_model, '__class__'):
                    from backend.nn.qwen import QwenImageTransformer2DModel as StdQwenImageTransformer2DModel
                    if isinstance(diffusion_model, StdQwenImageTransformer2DModel):
                        print(f"[QwenImageControlNet] Detected: Standard Qwen Image model - NOT SUPPORTED by this ControlNet")
                    else:
                        print(f"[QwenImageControlNet] Detected: Unknown model type")
            else:
                print(f"[QwenImageControlNet] REJECTED: Cannot access diffusion_model - ControlNet will not be applied")
            
            # CRITICAL: Return immediately - do NOT process for other models
            return
        
        # Only proceed if it's confirmed to be NunchakuQwenImageTransformer2DModel
        print(f"[QwenImageControlNet] Processing ControlNet for Nunchaku Qwen Image")
        print(f"[QwenImageControlNet] Strength: {self.strength}, Range: {self.start_percent}-{self.end_percent}")
        
        # Get target device from UNet model (ensure ControlNet model is on the same device)
        target_device = None
        if hasattr(unet, 'model') and hasattr(unet.model, 'diffusion_model'):
            # Get device from diffusion_model's first parameter
            for param in diffusion_model.parameters():
                if param.device.type == 'cuda':
                    target_device = param.device
                    break
        
        # If no CUDA device found, try to get from UNet model directly
        if target_device is None:
            try:
                target_device = next(diffusion_model.parameters()).device
            except StopIteration:
                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move ControlNet model to target device
        if target_device is not None:
            if hasattr(self.control_model, 'control_model'):
                # Get device of control_model (the actual PyTorch model)
                control_model_device = next(self.control_model.control_model.parameters()).device
                if control_model_device != target_device:
                    print(f"[QwenImageControlNet] Moving ControlNet model from {control_model_device} to {target_device}")
                    self.control_model.control_model = self.control_model.control_model.to(target_device)
                    # Also ensure control_model_wrapped is updated if it exists
                    if hasattr(self.control_model, 'control_model_wrapped'):
                        # ModelPatcher will handle device loading, but we can ensure it's set up correctly
                        pass
        
        # Get and wrap VAE
        vae = process.sd_model.forge_objects.vae
        if not hasattr(vae, "spacial_compression_encode"):
            vae = QwenImageVAEWrapper(vae)
        
        # Create ControlNet copy with condition hint
        cnet = self.control_model.copy()
        # Ensure the copy's control_model is also on the correct device
        if target_device is not None and hasattr(cnet, 'control_model'):
            cnet.control_model = cnet.control_model.to(target_device)
        cnet = cnet.set_cond_hint(cond, self.strength, (self.start_percent, self.end_percent), vae=vae)
        
        # Debug: Print cond hint info
        if hasattr(cnet, 'cond_hint_original') and cnet.cond_hint_original is not None:
            print(f"[QwenImageControlNet] cond_hint_original shape: {cnet.cond_hint_original.shape}")
            print(f"[QwenImageControlNet] cond_hint_original range: {cnet.cond_hint_original.min().item():.4f} to {cnet.cond_hint_original.max().item():.4f}")
        
        # Override get_models() on the copy too
        def get_models_wrapper():
            return []
        cnet.get_models = get_models_wrapper
        
        # Add to ControlNet linked list
        unet.add_patched_controlnet(cnet)
        
        print(f"[QwenImageControlNet] ControlNet set up complete for Nunchaku Qwen Image")
        return
