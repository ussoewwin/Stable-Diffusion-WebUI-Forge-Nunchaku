import inspect
import logging
import sys

import torch

import modules.shared as shared
from backend.sampling.sampling_function import sampling_cleanup, sampling_prepare
from modules import devices, sd_samplers_common, sd_samplers_timesteps_impl
from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback
from modules.sd_samplers_cfg_denoiser import CFGDenoiser
from modules.shared import opts

samplers_timesteps = [
    ("DDIM", sd_samplers_timesteps_impl.ddim, ["ddim"], {}),
    ("PLMS", sd_samplers_timesteps_impl.plms, ["plms"], {}),
]


samplers_data_timesteps = [sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: CompVisSampler(funcname, model), aliases, options) for label, funcname, aliases, options in samplers_timesteps]


class CompVisTimestepsDenoiser(torch.nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_model = model
        self.inner_model.alphas_cumprod = 1.0 / (self.inner_model.forge_objects.unet.model.predictor.sigmas**2.0 + 1.0)

    def forward(self, input, timesteps, **kwargs):
        return self.inner_model.apply_model(input, timesteps, **kwargs)


class CFGDenoiserTimesteps(CFGDenoiser):
    def __init__(self, sampler):
        super().__init__(sampler)
        self.classic_ddim_eps_estimation = True

    @property
    def inner_model(self):
        if self.model_wrap is None:
            self.model_wrap = CompVisTimestepsDenoiser(shared.sd_model)

        return self.model_wrap


class CompVisSampler(sd_samplers_common.Sampler):
    def __init__(self, funcname, sd_model):
        super().__init__(funcname)

        self.eta_option_field = "eta_ddim"
        self.eta_infotext_field = "Eta DDIM"
        self.eta_default = 0.0

        self.model_wrap_cfg = CFGDenoiserTimesteps(self)
        self.model_wrap = self.model_wrap_cfg.inner_model

    def get_timesteps(self, p, steps):
        discard_next_to_last_sigma = self.config is not None and self.config.options.get("discard_next_to_last_sigma", False)
        if opts.always_discard_next_to_last_sigma and not discard_next_to_last_sigma:
            discard_next_to_last_sigma = True
            p.extra_generation_params["Discard penultimate sigma"] = True

        steps += 1 if discard_next_to_last_sigma else 0

        timesteps = torch.clip(torch.asarray(list(range(0, 1000, 1000 // steps)), device=devices.device) + 1, 0, 999)

        return timesteps

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        def _tensor_stats(name, t):
            if t is None:
                return f"{name}=None"
            with torch.no_grad():
                td = t.detach()
                finite = torch.isfinite(td)
                finite_count = int(finite.sum().item())
                total_count = td.numel()
                nan_count = int(torch.isnan(td).sum().item())
                inf_count = int(torch.isinf(td).sum().item())
                if finite_count > 0:
                    fv = td[finite]
                    min_v = float(fv.min().item())
                    max_v = float(fv.max().item())
                    mean_v = float(fv.mean().item())
                    std_v = float(fv.std(unbiased=False).item())
                else:
                    min_v = max_v = mean_v = std_v = None
                return (
                    f"{name}: shape={tuple(td.shape)} dtype={td.dtype} "
                    f"finite={finite_count}/{total_count} nan={nan_count} inf={inf_count} "
                    f"min={min_v} max={max_v} mean={mean_v} std={std_v}"
                )

        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(self.model_wrap.inner_model.forge_objects.unet, x=x)

        self.model_wrap.inner_model.alphas_cumprod = self.model_wrap.inner_model.alphas_cumprod.to(x.device)

        logging.warning(
            "[HRDBG] timesteps sample_img2img enter "
            f"is_hr_pass={getattr(p, 'is_hr_pass', False)} "
            f"enable_hr={getattr(p, 'enable_hr', False)} "
            f"denoising_strength={getattr(p, 'denoising_strength', None)} "
            f"steps_arg={steps} "
            f"p.steps={getattr(p, 'steps', None)} "
            f"x_shape={tuple(x.shape)} noise_shape={tuple(noise.shape)} "
            f"img_cond_shape={None if image_conditioning is None else tuple(image_conditioning.shape)}"
        )
        print(f"[HRDBG] {_tensor_stats('x_in', x)}")
        print(f"[HRDBG] {_tensor_stats('noise_in', noise)}")

        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)

        timesteps = self.get_timesteps(p, steps).to(x.device)
        timesteps_sched = timesteps[:t_enc]

        alphas_cumprod = shared.sd_model.alphas_cumprod
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[timesteps[t_enc]])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[timesteps[t_enc]])

        xi = x.to(noise) * sqrt_alpha_cumprod + noise * sqrt_one_minus_alpha_cumprod
        logging.warning(
            "[HRDBG] timesteps sample_img2img noise-mix "
            f"steps={steps} t_enc={t_enc} "
            f"timesteps_len={len(timesteps)} timesteps_sched_len={len(timesteps_sched)} "
            f"alpha_idx={int(timesteps[t_enc].item()) if t_enc < len(timesteps) else 'out_of_range'}"
        )
        print(f"[HRDBG] {_tensor_stats('xi', xi)}")

        if opts.img2img_extra_noise > 0:
            p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
            extra_noise_params = ExtraNoiseParams(noise, x, xi)
            extra_noise_callback(extra_noise_params)
            noise = extra_noise_params.noise
            xi += noise * opts.img2img_extra_noise * sqrt_alpha_cumprod

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        if "timesteps" in parameters:
            extra_params_kwargs["timesteps"] = timesteps_sched
        if "is_img2img" in parameters:
            extra_params_kwargs["is_img2img"] = True

        self.model_wrap_cfg.init_latent = x
        self.last_latent = x
        self.sampler_extra_args = {
            "cond": conditioning,
            "image_cond": image_conditioning,
            "uncond": unconditional_conditioning,
            "cond_scale": p.cfg_scale,
            "s_min_uncond": self.s_min_uncond,
        }

        samples = self.launch_sampling(
            t_enc + 1,
            lambda: self.func(self.model_wrap_cfg, xi, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs),
        )
        print(f"[HRDBG] {_tensor_stats('samples_out', samples)}")

        self.add_infotext(p)

        sampling_cleanup(unet_patcher)

        return samples

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(self.model_wrap.inner_model.forge_objects.unet, x=x)

        self.model_wrap.inner_model.alphas_cumprod = self.model_wrap.inner_model.alphas_cumprod.to(x.device)

        steps = steps or p.steps
        timesteps = self.get_timesteps(p, steps).to(x.device)

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        if "timesteps" in parameters:
            extra_params_kwargs["timesteps"] = timesteps

        self.last_latent = x
        self.sampler_extra_args = {
            "cond": conditioning,
            "image_cond": image_conditioning,
            "uncond": unconditional_conditioning,
            "cond_scale": p.cfg_scale,
            "s_min_uncond": self.s_min_uncond,
        }
        samples = self.launch_sampling(
            steps,
            lambda: self.func(self.model_wrap_cfg, x, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs),
        )

        self.add_infotext(p)

        sampling_cleanup(unet_patcher)

        return samples


sys.modules["modules.sd_samplers_compvis"] = sys.modules[__name__]
VanillaStableDiffusionSampler = CompVisSampler  # backward compatibility
