import torch
from huggingface_guess import model_list

from backend.comfy_te_glue import offload_comfy_clip
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.gemma_engine import GemmaTextProcessingEngine


class Lumina2(ForgeDiffusionEngine):
    """Forge glue only: Comfy ``NextDiT`` UNet + Comfy ``sd.CLIP`` (Lumina2 TE)."""

    matched_guesses = [model_list.Lumina2]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = huggingface_components["text_encoder"]

        vae = VAE(model=huggingface_components["vae"])

        k_predictor = PredictionDiscreteFlow(estimated_config)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_gemma = GemmaTextProcessingEngine(clip)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.use_shift = True
        self.is_flux = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        shift = getattr(prompt, "distilled_cfg_scale", 6.0)
        self.forge_objects.unet.model.predictor.set_parameters(shift=shift)
        cond = self.text_processing_engine_gemma(prompt)
        offload_comfy_clip(self.forge_objects.clip)
        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_gemma.tokenize([prompt])[0])
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)
