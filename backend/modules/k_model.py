import math

import torch

from backend import memory_management
from backend.modules.k_prediction import k_prediction_from_diffusers_scheduler

# Import comfy modules for WrapperExecutor
import comfy.patcher_extension
import comfy.ldm.lumina.model  # For NextDiT check in apply_model


def reshape_sigma(sigma, noise_dim):
    if sigma.nelement() == 1:
        return sigma.view(())
    else:
        return sigma.view(sigma.shape[:1] + (1,) * (noise_dim - 1))


class CONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = reshape_sigma(sigma, model_output.ndim)
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = reshape_sigma(sigma, noise.ndim)
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        sigma = reshape_sigma(sigma, latent.ndim)
        return latent / (1.0 - sigma)


def flux_time_shift(mu: float, sigma: float, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class ModelSamplingFlux(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        self.set_parameters(shift=sampling_settings.get("shift", 1.15))

    def set_parameters(self, shift=1.15, timesteps=10000):
        self.shift = shift
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps))
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return flux_time_shift(self.shift, 1.0, timestep)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return flux_time_shift(self.shift, 1.0, 1.0 - percent)


class ModelSamplingFluxWithConst(ModelSamplingFlux, CONST):
    pass


class KModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, diffusers_scheduler, k_predictor=None, config=None):
        super().__init__()

        self.config = config

        self.storage_dtype = model.storage_dtype
        self.computation_dtype = model.computation_dtype

        print(f"K-Model Created: {dict(storage_dtype=self.storage_dtype, computation_dtype=self.computation_dtype)}")

        self.diffusion_model = model
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)

        if k_predictor is None:
            self.predictor = k_prediction_from_diffusers_scheduler(diffusers_scheduler)
        else:
            self.predictor = k_predictor

        # ComfyUI標準のmodel_samplingを追加（ControlNet.pre_run()がmodel.model_samplingを期待するため）
        if config is not None and hasattr(config, "model_type"):
            model_type = config.model_type
            if hasattr(model_type, "name"):
                import comfy.model_base
                from comfy.model_base import ModelType
                if model_type.name == "FLUX":
                    self.model_sampling = ModelSamplingFluxWithConst(config)
                elif model_type.name == "FLOW":
                    # Qwen Image uses ModelType.FLOW
                    # Use ComfyUI's model_sampling function to create the correct ModelSampling class
                    model_type_enum = ModelType.FLOW
                    self.model_sampling = comfy.model_base.model_sampling(config, model_type_enum)
                elif model_type.name == "EPS":
                    # SDXL and other EPS models use ModelType.EPS
                    # Use ComfyUI's model_sampling function to create the correct ModelSampling class
                    model_type_enum = ModelType.EPS
                    self.model_sampling = comfy.model_base.model_sampling(config, model_type_enum)
                else:
                    # For other model types, try to use ComfyUI's model_sampling function
                    # This handles V_PREDICTION, V_PREDICTION_EDM, STABLE_CASCADE, EDM, etc.
                    try:
                        model_type_enum = getattr(ModelType, model_type.name, None)
                        if model_type_enum is not None:
                            self.model_sampling = comfy.model_base.model_sampling(config, model_type_enum)
                        else:
                            self.model_sampling = None
                    except (AttributeError, TypeError):
                        self.model_sampling = None
            else:
                self.model_sampling = None
        else:
            self.model_sampling = None
        
        # Initialize model_options for transformer_options (used by ControlNet patches, etc.)
        self.model_options = {"transformer_options": {}}

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        # Use WrapperExecutor same as ComfyUI BaseModel.apply_model
        # This ensures APPLY_MODEL level wrappers are executed
        if transformer_options is None:
            transformer_options = {}
        
        # ZIT-ONLY FIX: Handle ZIT patches to prevent double application and stale patches
        # 
        # Problem 1: Double application - ZIT patches exist in both provided_patches AND self_patches
        # Solution: Skip self.model_options merge when ZIT patches are already in provided_patches
        #
        # Problem 2: Stale patches - After model switch, ZIT patches remain in self_patches
        #            but no new ZIT patches are in provided_patches, causing RecursionError
        # Solution: Clear self.model_options ZIT patches when detected as stale
        #
        # This fix ONLY affects NextDiT models - other models use the original merge logic.
        skip_model_options_merge = False
        provided_patches = transformer_options.get("patches", {})
        zit_patches_in_provided = "noise_refiner" in provided_patches or "double_block" in provided_patches
        
        # Get current model type and self patches
        model_type_name = type(self.diffusion_model).__name__
        self_patches = self.model_options.get("transformer_options", {}).get("patches", {})
        zit_patches_in_self = "noise_refiner" in self_patches or "double_block" in self_patches
        
        # Case 1: ZIT patches in provided - skip merge to prevent duplication
        if zit_patches_in_provided and model_type_name == "NextDiT":
            skip_model_options_merge = True
        
        # Case 2: Stale ZIT patches detected - self has ZIT patches but provided doesn't
        # This happens after model switch: old patches remain in self.model_options
        elif zit_patches_in_self and not zit_patches_in_provided:
            # Clear the stale transformer_options to prevent applying old patches
            self.model_options["transformer_options"] = {}
            skip_model_options_merge = True  # Nothing to merge now
        
        # Merge transformer_options from model_options before passing to WrapperExecutor
        # Same as ComfyUI: model_options["transformer_options"] is merged with provided transformer_options
        final_transformer_options = {}
        
        # ZIT-ONLY: Skip self.model_options merge if ZIT patches already in transformer_options
        if not skip_model_options_merge and "transformer_options" in self.model_options:
            import copy
            final_transformer_options = copy.deepcopy(self.model_options["transformer_options"])
        
        # Merge with provided transformer_options (from sampling_function_inner)
        if transformer_options:
            # Merge patches (same logic as sampling_function.py lines 250-259)
            if "patches" in transformer_options:
                if "patches" not in final_transformer_options:
                    final_transformer_options["patches"] = {}
                cur_patches = final_transformer_options["patches"].copy()
                for patch_name, patches in transformer_options["patches"].items():
                    if patch_name in cur_patches:
                        cur_patches[patch_name] = cur_patches[patch_name] + patches
                    else:
                        cur_patches[patch_name] = patches
                final_transformer_options["patches"] = cur_patches
            # Merge other transformer_options (override with provided values)
            for key, value in transformer_options.items():
                if key != "patches":
                    final_transformer_options[key] = value
        
        transformer_options = final_transformer_options
        
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._apply_model,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.predictor.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.computation_dtype

        xc = xc.to(dtype)
        t = self.predictor.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            # Skip transformer_options as it's passed explicitly
            if o == "transformer_options":
                continue
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        # Remove transformer_options from extra_conds to avoid duplicate argument
        extra_conds_clean = {k: v for k, v in extra_conds.items() if k != "transformer_options"}
        
        # CRITICAL: Check if this is a ZIT model (NextDiT) BEFORE applying ZIT-specific logic
        # ZIT models use NextDiT as diffusion_model, other models (SD1.5, SDXL, Flux1, Qwen Image) do NOT
        is_zit_model = False
        try:
            NextDiT = comfy.ldm.lumina.model.NextDiT
            if isinstance(self.diffusion_model, NextDiT):
                is_zit_model = True
            else:
                # Fallback: check by type name
                model_type_name = type(self.diffusion_model).__name__
                if model_type_name == "NextDiT":
                    is_zit_model = True
        except (ImportError, AttributeError, TypeError):
            # If we can't import NextDiT or check, assume it's not a ZIT model
            is_zit_model = False
        
        # ZIT models (NextDiT) ONLY: Apply ComfyUI Lumina2.extra_conds logic for attention_mask and num_tokens
        # Same as ComfyUI model_base.py Lumina2.extra_conds (lines 1154-1172)
        # Other models (SD1.5, SDXL, Flux1, Qwen Image): Skip this logic completely
        if is_zit_model:
            attention_mask = extra_conds_clean.pop("attention_mask", None)
            num_tokens = None
            
            if attention_mask is not None:
                # ComfyUI: if torch.numel(attention_mask) != attention_mask.sum():
                #   out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
                # out['num_tokens'] = comfy.conds.CONDConstant(max(1, torch.sum(attention_mask).item()))
                if torch.numel(attention_mask) != attention_mask.sum():
                    # attention_mask has zeros, keep it
                    extra_conds_clean["attention_mask"] = attention_mask
                num_tokens = max(1, torch.sum(attention_mask).item())
            
            # ComfyUI: cross_attn = kwargs.get("cross_attn", None)
            # if cross_attn is not None:
            #     out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
            #     if 'num_tokens' not in out:
            #         out['num_tokens'] = comfy.conds.CONDConstant(cross_attn.shape[1])
            # Note: Forge uses c_crossattn parameter instead of cross_attn in kwargs
            if num_tokens is None and context is not None and hasattr(context, 'shape') and len(context.shape) >= 2:
                # Fallback: calculate num_tokens from context shape
                num_tokens = context.shape[1]
            elif "num_tokens" in extra_conds_clean:
                # Use num_tokens from kwargs if explicitly provided
                num_tokens = extra_conds_clean.pop("num_tokens")
            
            # CRITICAL: transformer_options must be in kwargs for NextDiT.forward() to get wrappers
            # NextDiT.forward() uses kwargs.get("transformer_options", {}) to get wrappers
            # Even though _forward() accepts transformer_options as explicit argument,
            # the WrapperExecutor in forward() needs it in kwargs
            # BUT: We must NOT pass it as explicit argument AND in kwargs to avoid "multiple values" error
            # Pass it ONLY in kwargs, not as explicit argument
            extra_conds_clean["transformer_options"] = transformer_options
            
            # Pass transformer_options ONLY in kwargs (not as explicit argument)
            # This allows NextDiT.forward() to get wrappers via kwargs.get("transformer_options", {})
            # and _forward() will receive it from kwargs as well
            # Same as ComfyUI BaseModel._apply_model: model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
            if num_tokens is not None:
                model_output = self.diffusion_model(xc, t, context=context, num_tokens=num_tokens, attention_mask=attention_mask, control=control, **extra_conds_clean).float()
            else:
                # Fallback: try without num_tokens (for models that don't require it)
                model_output = self.diffusion_model(xc, t, context=context, control=control, **extra_conds_clean).float()
        else:
            # Other models (SD1.5, SDXL, Flux1, Qwen Image): Normal forward without ZIT-specific parameters
            # DO NOT add transformer_options to kwargs, DO NOT pass num_tokens or attention_mask
            model_output = self.diffusion_model(xc, t, context=context, control=control, **extra_conds_clean).float()
        
        return self.predictor.calculate_denoised(sigma, model_output, x)

    def memory_required(self, input_shape: list[int]) -> float:
        """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/model_base.py#L354"""
        input_shapes = [input_shape]
        area = sum(map(lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes))

        if memory_management.xformers_enabled():
            return (area * memory_management.dtype_size(self.computation_dtype) * 0.01 * self.config.memory_usage_factor) * (1024 * 1024)
        else:
            return (area * 0.15 * self.config.memory_usage_factor) * (1024 * 1024)

    def cleanup(self):
        del self.config
        del self.predictor
        del self.diffusion_model
