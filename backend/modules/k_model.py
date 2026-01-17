import math

import torch

from backend import memory_management
from backend.modules.k_prediction import k_prediction_from_diffusers_scheduler


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
                if model_type.name == "FLUX":
                    self.model_sampling = ModelSamplingFluxWithConst(config)
                elif model_type.name == "FLOW":
                    # Qwen Image uses ModelType.FLOW
                    # Use ComfyUI's model_sampling function to create the correct ModelSampling class
                    import comfy.model_base
                    # Convert model_type.name to ModelType Enum
                    from comfy.model_base import ModelType
                    model_type_enum = ModelType.FLOW
                    self.model_sampling = comfy.model_base.model_sampling(config, model_type_enum)
                else:
                    self.model_sampling = None
            else:
                self.model_sampling = None
        else:
            self.model_sampling = None

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
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
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
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
