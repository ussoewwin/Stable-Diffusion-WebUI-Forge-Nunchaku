import torch
from modules_forge.packages.huggingface_guess import model_list

from backend.comfy_te_glue import offload_comfy_clip
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionFlux
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.krea2_engine import Krea2TextProcessingEngine


class Krea2(ForgeDiffusionEngine):
    """Forge glue only: Comfy ``SingleStreamDiT`` UNet + Comfy ``sd.CLIP`` (Krea2 Qwen3-VL-4B TE) + Wan VAE.

    Comfy Krea2 is ``ModelType.FLUX``; its sigma schedule is ``flux_time_shift(shift=1.15, 1.0, t)``.
    Forge-Nunchaku ``PredictionFlux`` takes that shift through its ``mu`` kwarg verbatim
    (``mu == 1.15``, NOT ``math.log(1.15)`` — same slot as Comfy ``ModelSamplingFlux.shift``).
    See ``docs/KREA2_INTEGRATION_PLAN.md`` §4.
    """

    matched_guesses = [model_list.Krea2]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = huggingface_components["text_encoder"]

        vae = VAE(model=huggingface_components["vae"], is_wan=True)
        vae.first_stage_model.latent_format = self.model_config.latent_format

        shift = float(self.model_config.sampling_settings.get("shift", 1.15))
        k_predictor = PredictionFlux(mu=shift)

        unet = UnetPatcher.from_model(
            model=huggingface_components["transformer"],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config,
        )

        self.text_processing_engine_krea2 = Krea2TextProcessingEngine(clip)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        cond = self.text_processing_engine_krea2(prompt)
        offload_comfy_clip(self.forge_objects.clip)
        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        tokens = self.text_processing_engine_krea2.tokenize([prompt])
        return len(tokens[0]), max(256, len(tokens[0]))

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        samples: list[torch.Tensor] = []
        for b in range(x.size(0)):
            y = x[b].unsqueeze(0)
            sample = self.forge_objects.vae.encode(y.movedim(1, -1) * 0.5 + 0.5)
            sample = self.forge_objects.vae.first_stage_model.process_in(sample)
            samples.append(sample)
        return torch.cat(samples).to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor):
        samples: list[torch.Tensor] = []
        for b in range(x.size(0)):
            y = x[b].unsqueeze(0)
            sample = self.forge_objects.vae.first_stage_model.process_out(y)
            sample = self.forge_objects.vae.decode(sample).movedim(-1, 2) * 2.0 - 1.0
            samples.append(sample)
        return torch.cat(samples).to(x)
