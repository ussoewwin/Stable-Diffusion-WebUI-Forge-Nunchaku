import torch
from modules_forge.packages.huggingface_guess import model_list

from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.anima_engine import AnimaTextProcessingEngine


class Anima(ForgeDiffusionEngine):
    """Forge glue only: ``comfy.ldm.anima`` + ``comfy.text_encoders.anima`` (import, no duplicated modules)."""

    matched_guesses = [model_list.Anima, model_list.AnimaBase, model_list.AnimaWai68]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(
            model_dict={"qwen3_06b": huggingface_components["text_encoder"]},
            tokenizer_dict={
                "qwen3_06b": huggingface_components["tokenizer"],
                "t5xxl": huggingface_components["tokenizer_2"],
            },
        )

        vae = VAE(model=huggingface_components["vae"], is_wan=True)
        vae.first_stage_model.latent_format = self.model_config.latent_format

        k_predictor = PredictionDiscreteFlow(estimated_config)

        unet = UnetPatcher.from_model(
            model=huggingface_components["transformer"],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config,
        )

        self.text_processing_engine_anima = AnimaTextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen3_06b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True
        self.use_shift = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        shift = getattr(prompt, "distilled_cfg_scale", None)
        if shift is None:
            shift = self.model_config.sampling_settings.get("shift", 3.0)
        self.forge_objects.unet.model.predictor.set_parameters(shift=float(shift))

        cond = self.text_processing_engine_anima(prompt)

        memory_management.load_model_gpu(self.forge_objects.unet)
        diffusion_model = self.forge_objects.unet.model.diffusion_model
        device = memory_management.get_torch_device()
        dtype = self.forge_objects.unet.model.computation_dtype

        crossattn = cond["crossattn"].to(device=device, dtype=dtype)
        t5xxl_ids = cond["t5xxl_ids"].to(device=device)
        t5xxl_weights = cond.get("t5xxl_weights")
        if t5xxl_weights is not None:
            t5xxl_weights = t5xxl_weights.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)

        return diffusion_model.preprocess_text_embeds(crossattn, t5xxl_ids, t5xxl_weights=t5xxl_weights)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        qwen_tokens, _ = self.text_processing_engine_anima.tokenize([prompt])
        return len(qwen_tokens[0]), max(512, len(qwen_tokens[0]))

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
