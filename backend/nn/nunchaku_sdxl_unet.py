from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transformers import T5EncoderModel

import gc
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Add DLL directories for nunchaku
if os.name == 'nt':
    try:
        # Add torch directory and torch/lib directory
        torch_root = os.path.dirname(torch.__file__)
        torch_lib = os.path.join(torch_root, "lib")
        
        if os.path.exists(torch_root):
            os.add_dll_directory(torch_root)
        if os.path.exists(torch_lib):
            os.add_dll_directory(torch_lib)
            
        # Add system CUDA paths as fallback
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        ]
        for path in cuda_paths:
            if os.path.exists(path):
                try:
                    os.add_dll_directory(path)
                except Exception:
                    pass
    except Exception:
        pass

# Safe import for nunchaku
NUNCHAKU_AVAILABLE = False
try:
    from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
    from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_transformer
    from nunchaku.caching.fbcache import cache_context, create_cache_context
    from nunchaku.lora.flux.compose import compose_lora
    from nunchaku.models.linear import AWQW4A16Linear, SVDQW4A4Linear
    from nunchaku.models.utils import CPUOffloadManager
    from nunchaku.ops.fused import fused_gelu_mlp
    from nunchaku.utils import load_state_dict_in_safetensors
    NUNCHAKU_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"[WARNING] Nunchaku not available: {e}")
    # Define dummy classes to prevent NameError if imported
    class NunchakuFluxTransformer2dModel: pass
    class NunchakuT5EncoderModel: pass
    class AWQW4A16Linear: pass
    class SVDQW4A4Linear: pass
    def apply_cache_on_transformer(*args, **kwargs): pass
    def cache_context(*args, **kwargs): pass
    def create_cache_context(*args, **kwargs): pass
    def compose_lora(*args, **kwargs): pass
    def fused_gelu_mlp(*args, **kwargs): pass
    def load_state_dict_in_safetensors(*args, **kwargs): pass
    class CPUOffloadManager: pass

from backend.args import dynamic_args
from backend.nn._qwen_lora import compose_loras_v2, reset_lora_v2
from backend.utils import process_img
from modules import shared


class NunchakuModelMixin(nn.Module):
    offload: bool = False

    def set_offload(self, offload: bool, use_pin_memory: bool, num_blocks_on_gpu: int):
        raise NotImplementedError

    def to(self, *args, **kwargs):
        args = (arg for arg in args if not isinstance(arg, torch.dtype))
        kwargs.pop("dtype", None)

        dev: bool = any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs

        if self.offload and dev:
            return self
        else:
            return super().to(*args, **kwargs)


# ========== Flux ========== #


class SVDQFluxTransformer2DModel(nn.Module):
    """https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/v1.0.0/wrappers/flux.py"""

    def __init__(self, config: dict):
        super().__init__()
        model = NunchakuFluxTransformer2dModel.from_pretrained(config.pop("filename"), offload=shared.opts.svdq_cpu_offload)
        model = apply_cache_on_transformer(transformer=model, residual_diff_threshold=shared.opts.svdq_cache_threshold)
        model.set_attention_impl(shared.opts.svdq_attention)

        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        self.loras = []

        # for first-block cache
        self._prev_timestep = None
        self._cache_context = None

    def forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            assert isinstance(timestep, float)
            timestep_float = timestep

        model = self.model
        assert isinstance(model, NunchakuFluxTransformer2dModel)

        bs, c, h_orig, w_orig = x.shape
        patch_size = self.config.get("patch_size", 2)
        h_len = (h_orig + (patch_size // 2)) // patch_size
        w_len = (w_orig + (patch_size // 2)) // patch_size

        img, img_ids = process_img(x)
        img_tokens = img.shape[1]

        ref_latents = dynamic_args.get("ref_latents", None)

        if ref_latents is not None:
            h = 0
            w = 0
            for ref in ref_latents:
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h

                kontext, kontext_ids = process_img(ref.to(x), index=1, h_offset=h_offset, w_offset=w_offset)
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)

        # load and compose LoRA
        if self.loras != model.comfy_lora_meta_list:
            lora_to_be_composed = []
            for _ in range(max(0, len(model.comfy_lora_meta_list) - len(self.loras))):
                model.comfy_lora_meta_list.pop()
                model.comfy_lora_sd_list.pop()
            for i in range(len(self.loras)):
                meta = self.loras[i]
                if i >= len(model.comfy_lora_meta_list):
                    sd = load_state_dict_in_safetensors(meta[0])
                    model.comfy_lora_meta_list.append(meta)
                    model.comfy_lora_sd_list.append(sd)
                elif model.comfy_lora_meta_list[i] != meta:
                    if meta[0] != model.comfy_lora_meta_list[i][0]:
                        sd = load_state_dict_in_safetensors(meta[0])
                        model.comfy_lora_sd_list[i] = sd
                    model.comfy_lora_meta_list[i] = meta
                lora_to_be_composed.append(({k: v for k, v in model.comfy_lora_sd_list[i].items()}, meta[1]))

            composed_lora = compose_lora(lora_to_be_composed)

            if len(composed_lora) == 0:
                model.reset_lora()
            else:
                if "x_embedder.lora_A.weight" in composed_lora:
                    new_in_channels = composed_lora["x_embedder.lora_A.weight"].shape[1]
                    current_in_channels = model.x_embedder.in_features
                    if new_in_channels < current_in_channels:
                        model.reset_x_embedder()
                try:
                    model.update_lora_params(composed_lora)
                except RuntimeError as e:
                    if "size mismatch" in str(e):
                        print(f"[SVDQ] Warning: LoRA update failed due to shape mismatch in nunchaku. Skipping LoRA application: {e}")
                        print("[SVDQ] This may be due to incompatibility between the LoRA format and the current nunchaku version.")
                        model.reset_lora()
                    else:
                        raise

        controlnet_block_samples = None if control is None else [y.to(x.dtype) for y in control["input"]]
        controlnet_single_block_samples = None if control is None else [y.to(x.dtype) for y in control["output"]]

        if getattr(model, "_is_cached", False) or getattr(model, "residual_diff_threshold_multi", 0) != 0:
            # A more robust caching strategy
            cache_invalid = False

            # Check if timestamps have changed or are out of valid range
            if self._prev_timestep is None:
                cache_invalid = True
            elif self._prev_timestep < timestep_float + 1e-5:  # allow a small tolerance to reuse the cache
                cache_invalid = True

            if cache_invalid:
                self._cache_context = create_cache_context()

            # Update the previous timestamp
            self._prev_timestep = timestep_float
            with cache_context(self._cache_context):
                out = model(
                    hidden_states=img,
                    encoder_hidden_states=context,
                    pooled_projections=y,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                ).sample
        else:
            out = model(
                hidden_states=img,
                encoder_hidden_states=context,
                pooled_projections=y,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance if self.config["guidance_embed"] else None,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
            ).sample

        out = out[:, :img_tokens]
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=patch_size, pw=patch_size)
        out = out[:, :, :h_orig, :w_orig]

        self._prev_timestep = timestep_float
        return out

    def load_state_dict(self, *args, **kwargs):
        return [], []


# ========== T5 ========== #


def _forward(self: "T5EncoderModel", input_ids: torch.LongTensor, *args, **kwargs):
    outputs = self.encoder(input_ids=input_ids, *args, **kwargs)
    return outputs.last_hidden_state


class WrappedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input: torch.Tensor, *args, **kwargs):
        return self.embedding(input)

    @property
    def weight(self):
        return self.embedding.weight


class SVDQT5(torch.nn.Module):
    """https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/v1.0.0/nodes/models/text_encoder.py"""

    def __init__(self, path: str):
        super().__init__()

        transformer = NunchakuT5EncoderModel.from_pretrained(path)
        transformer.forward = types.MethodType(_forward, transformer)
        transformer.shared = WrappedEmbedding(transformer.shared)

        self.transformer = transformer
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))


# ========== Qwen ========== #

from backend.memory_management import xformers_enabled

if xformers_enabled():
    from backend.attention import attention_xformers as attention_function
else:
    from backend.attention import attention_pytorch as attention_function

from backend.nn.flux import EmbedND
from backend.nn.qwen import (
    GELU,
    FeedForward,
    LastLayer,
    QwenImageTransformer2DModel,
    QwenTimestepProjEmbeddings,
    apply_rotary_emb,
)


class NunchakuGELU(GELU):

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
        **kwargs,
    ):
        super(GELU, self).__init__()
        self.proj = SVDQW4A4Linear(dim_in, dim_out, bias=bias, **kwargs)
        self.approximate = approximate


class NunchakuFeedForward(FeedForward):

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        inner_dim=None,
        bias: bool = True,
        **kwargs,
    ):
        super(FeedForward, self).__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([])
        self.net.append(NunchakuGELU(dim, inner_dim, approximate="tanh", bias=bias, **kwargs))
        self.net.append(nn.Dropout(dropout))
        self.net.append(
            SVDQW4A4Linear(
                inner_dim,
                dim_out,
                bias=bias,
                act_unsigned=kwargs.get("precision", "int4") == "int4",
                **kwargs,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.net[0], NunchakuGELU):
            return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])
        else:
            for module in self.net:
                hidden_states = module(hidden_states)
            return hidden_states


class Attention(nn.Module):

    def __init__(
        self,
        query_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-5,
        out_bias: bool = True,
        out_dim: int = None,
        out_context_dim: int = None,
        **kwargs,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.heads = heads
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.dropout = dropout

        # Q/K normalization for both streams
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)
        self.norm_added_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_added_k = nn.RMSNorm(dim_head, eps=eps)

        # Image stream projections: fused QKV for speed
        self.to_qkv = SVDQW4A4Linear(query_dim, self.inner_dim + self.inner_kv_dim * 2, bias=bias, **kwargs)

        # Text stream projections: fused QKV for speed
        self.add_qkv_proj = SVDQW4A4Linear(query_dim, self.inner_dim + self.inner_kv_dim * 2, bias=bias, **kwargs)

        # Output projections
        self.to_out = nn.ModuleList(
            [
                SVDQW4A4Linear(self.inner_dim, self.out_dim, bias=out_bias, **kwargs),
                nn.Dropout(dropout),
            ]
        )
        self.to_add_out = SVDQW4A4Linear(self.inner_dim, self.out_context_dim, bias=out_bias, **kwargs)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        seq_txt = encoder_hidden_states.shape[1]

        img_qkv = self.to_qkv(hidden_states)
        img_query, img_key, img_value = img_qkv.chunk(3, dim=-1)

        # Compute QKV for text stream (context projections)
        txt_qkv = self.add_qkv_proj(encoder_hidden_states)
        txt_query, txt_key, txt_value = txt_qkv.chunk(3, dim=-1)

        img_query = img_query.unflatten(-1, (self.heads, -1))
        img_key = img_key.unflatten(-1, (self.heads, -1))
        img_value = img_value.unflatten(-1, (self.heads, -1))

        txt_query = txt_query.unflatten(-1, (self.heads, -1))
        txt_key = txt_key.unflatten(-1, (self.heads, -1))
        txt_value = txt_value.unflatten(-1, (self.heads, -1))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        # Concatenate image and text streams for joint attention
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Apply rotary embeddings
        joint_query = apply_rotary_emb(joint_query, image_rotary_emb)
        joint_key = apply_rotary_emb(joint_key, image_rotary_emb)

        joint_query = joint_query.flatten(start_dim=2)
        joint_key = joint_key.flatten(start_dim=2)
        joint_value = joint_value.flatten(start_dim=2)

        # Compute joint attention
        joint_hidden_states = attention_function(joint_query, joint_key, joint_value, self.heads, attention_mask)

        # Split results back to separate streams
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class NunchakuQwenImageTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        scale_shift: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.scale_shift = scale_shift
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Modulation and normalization for image stream
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            AWQW4A16Linear(dim, 6 * dim, bias=True),
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = NunchakuFeedForward(dim=dim, dim_out=dim, **kwargs)

        # Modulation and normalization for text stream
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            AWQW4A16Linear(dim, 6 * dim, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = NunchakuFeedForward(dim=dim, dim_out=dim, **kwargs)

        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
            **kwargs,
        )

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if self.scale_shift != 0:
            scale.add_(self.scale_shift)
        return x * scale.unsqueeze(1) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Nunchaku's mod_params is [B, 6*dim] instead of [B, dim*6]
        img_mod_params = img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(img_mod_params.shape[0], -1)
        txt_mod_params = txt_mod_params.view(txt_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(txt_mod_params.shape[0], -1)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Joint attention computation (DoubleStreamLayerMegatron logic)
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream ("sample")
            encoder_hidden_states=txt_modulated,  # Text stream ("context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        return encoder_hidden_states, hidden_states


class NunchakuQwenImageTransformer2DModel(NunchakuModelMixin, QwenImageTransformer2DModel):
    """https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/v1.0.1/models/qwenimage.py"""

    def __init__(
        self,
        filename: str = None,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        scale_shift: float = 1.0,
        **kwargs,
    ):
        super(QwenImageTransformer2DModel, self).__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
        )

        self.txt_norm = nn.RMSNorm(joint_attention_dim, eps=1e-6)
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                NunchakuQwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    scale_shift=scale_shift,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = LastLayer(
            self.inner_dim,
            self.inner_dim,
        )
        self.proj_out = nn.Linear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True,
        )

        self.set_offload(
            offload=shared.opts.svdq_cpu_offload,
            use_pin_memory=shared.opts.svdq_use_pin_memory,
            num_blocks_on_gpu=shared.opts.svdq_num_blocks_on_gpu,
        )

        self.loras = []
        self._applied_loras = []

    def forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        guidance: torch.Tensor = None,
        ref_latents=None,
        transformer_options={},
        control=None,
        **kwargs,
    ):

        device = x.device
        if self.offload:
            self.offload_manager.set_device(device)

        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        ref_latents = dynamic_args.get("ref_latents", ref_latents)

        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            index_ref_method = kwargs.get("ref_latents_method", "index") == "index"
            for ref in ref_latents:
                if index_ref_method:
                    index += 1
                    h_offset = 0
                    w_offset = 0
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids, _ = self.process_img(ref.to(x), index=index, h_offset=h_offset, w_offset=w_offset)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        txt_start = round(
            max(
                ((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2,
                ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2,
            )
        )
        txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).squeeze(1).unsqueeze(2).to(x.dtype)
        del ids, txt_ids, img_ids

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance * 1000

        if self.loras != self._applied_loras:
            self._applied_loras = self.loras.copy()

            reset_lora_v2(self)
            self.set_offload(False, None, None)

            print("[Qwen] Composing LoRAs...")
            compose_loras_v2(self, self.loras)
            print("[Qwen] LoRAs Composed~")

            self.set_offload(
                offload=shared.opts.svdq_cpu_offload,
                use_pin_memory=shared.opts.svdq_use_pin_memory,
                num_blocks_on_gpu=shared.opts.svdq_num_blocks_on_gpu,
            )

        temb = self.time_text_embed(timestep, hidden_states) if guidance is None else self.time_text_embed(timestep, guidance, hidden_states)

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        # Setup compute stream for offloading
        compute_stream = torch.cuda.current_stream()
        if self.offload:
            self.offload_manager.initialize(compute_stream)

        for i, block in enumerate(self.transformer_blocks):
            with torch.cuda.stream(compute_stream):
                if self.offload:
                    block = self.offload_manager.get_block(i)
                if ("double_block", i) in blocks_replace:

                    def block_wrap(args):
                        out = {}
                        out["txt"], out["img"] = block(
                            hidden_states=args["img"],
                            encoder_hidden_states=args["txt"],
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            temb=args["vec"],
                            image_rotary_emb=args["pe"],
                        )
                        return out

                    out = blocks_replace[("double_block", i)](
                        {"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb},
                        {"original_block": block_wrap},
                    )
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                    )
                # ControlNet helpers(device/dtype-safe residual adds)
                _control = control if control is not None else (transformer_options.get("control", None) if isinstance(transformer_options, dict) else None)
                if isinstance(_control, dict):
                    control_i = _control.get("input")
                    try:
                        _scale = float(_control.get("weight", _control.get("scale", 1.0)))
                    except Exception:
                        _scale = 1.0
                else:
                    control_i = None
                    _scale = 1.0
                if control_i is not None and i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        if getattr(add, "device", None) != hidden_states.device or getattr(add, "dtype", None) != hidden_states.dtype:
                            add = add.to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
                        t = min(hidden_states.shape[1], add.shape[1])
                        if t > 0:
                            hidden_states[:, :t].add_(add[:, :t], alpha=_scale)

            if self.offload:
                self.offload_manager.step(compute_stream)

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(orig_shape)[:, :, :, : x.shape[-2], : x.shape[-1]]

    def set_offload(self, offload: bool, use_pin_memory: bool, num_blocks_on_gpu: int):
        if offload == self.offload:
            return

        self.offload = offload
        if offload:
            self.offload_manager = CPUOffloadManager(
                self.transformer_blocks,
                use_pin_memory=use_pin_memory,
                on_gpu_modules=[
                    self.img_in,
                    self.txt_in,
                    self.txt_norm,
                    self.time_text_embed,
                    self.norm_out,
                    self.proj_out,
                ],
                num_blocks_on_gpu=num_blocks_on_gpu,
            )
        else:
            self.offload_manager = None
            gc.collect()
            torch.cuda.empty_cache()

    def load_state_dict(self, sd, *args, **kwargs):
        state_dict = self.state_dict()
        for k in state_dict.keys():
            if k not in sd:
                if "dummy" in k:
                    continue
                if ".wcscales" not in k:
                    raise ValueError(f"Key {k} not found in state_dict")
                sd[k] = torch.ones_like(state_dict[k])
        for n, m in self.named_modules():
            if isinstance(m, SVDQW4A4Linear):
                if m.wtscale is not None:
                    m.wtscale = sd.pop(f"{n}.wtscale", 1.0)

        return super().load_state_dict(sd, *args, **kwargs)


# ========== SDXL ========== #


class SVDQUNet2DConditionModel(NunchakuModelMixin):
    """
    Wrapper for Nunchaku SDXL UNet with SVDQ quantization.
    Reference: https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/releases/tag/2.0
    """

    def __init__(self, config: dict):
        super().__init__()
        try:
            from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel, convert_sdxl_state_dict
            from nunchaku.utils import get_precision
        except ImportError as e:
            error_msg = str(e)
            if "transformer_z_image" in error_msg:
                raise ImportError(
                    f"Nunchaku package requires a newer version of diffusers that includes transformer_z_image. "
                    f"Current error: {error_msg}\n"
                    f"Please update diffusers: pip install --upgrade diffusers"
                ) from e
            raise ImportError(
                f"Failed to import nunchaku modules: {error_msg}\n"
                f"Please ensure nunchaku is properly installed and compatible with your diffusers version."
            ) from e

        self.filename = config.pop("filename", None)
        self.config = config
        self.dtype = torch.bfloat16

        # Device management
        self.load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offload_device = torch.device("cpu")
        self.initial_device = self.load_device
        self.storage_dtype = torch.bfloat16
        self.computation_dtype = torch.bfloat16

        # Will be loaded in load_state_dict
        self._nunchaku_unet = None
        self._convert_state_dict = convert_sdxl_state_dict
        self._get_precision = get_precision
        self._unet_cls = NunchakuSDXLUNet2DConditionModel

        # LoRA support: list of (filename, strength) tuples
        self.loras = []
        self._lora_cache = {}  # Cache for loaded LoRA state dicts

    def set_offload(self, offload: bool, use_pin_memory: bool, num_blocks_on_gpu: int):
        # SDXL UNet doesn't use the same offload mechanism as Flux
        pass

    def _build_and_load(self, sd: dict, metadata: dict = None):
        """Build and load the Nunchaku SDXL UNet from state dict."""
        import json
        from nunchaku.models.linear import SVDQW4A4Linear

        metadata = metadata or {}
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))

        # Determine precision
        precision = quantization_config.get("precision", None)
        if precision is None:
            precision = self._get_precision()
        if precision == "fp4":
            precision = "nvfp4"

        # Determine rank
        # IMPORTANT: The raw `sd` we receive may not expose proj_down/proj_up keys yet.
        # The actual rank used by the patched UNet must match the rank in the converted state dict.
        inferred_rank = None
        try:
            preview_sd = self._convert_state_dict(sd)
        except Exception:
            preview_sd = None

        if isinstance(preview_sd, dict):
            for k, v in preview_sd.items():
                if not torch.is_tensor(v) or v.ndim != 2:
                    continue
                if ("proj_down" in k) or ("proj_up" in k):
                    inferred_rank = int(v.shape[-1])
                    break
        else:
            for k, v in sd.items():
                if not torch.is_tensor(v) or v.ndim != 2:
                    continue
                if ("proj_down" in k) or ("proj_up" in k):
                    inferred_rank = int(v.shape[-1])
                    break

        rank = quantization_config.get("rank", None)
        try:
            rank = int(rank) if rank is not None else None
        except Exception:
            rank = None

        if inferred_rank is not None and (rank is None or rank != inferred_rank):
            # Keep this as a single-line notice; avoids crashing on rank mismatches (e.g., r128 vs metadata rank=32).
            print(f"[SVDQ SDXL] Rank override: metadata={rank} -> inferred={inferred_rank}")
            rank = inferred_rank

        if rank is None:
            rank = 32

        # Build config from metadata or default SDXL config
        unet_config = json.loads(metadata.get("config", "{}"))
        if not unet_config:
            # Default SDXL config
            unet_config = {
                "sample_size": 128,
                "in_channels": 4,
                "out_channels": 4,
                "center_input_sample": False,
                "flip_sin_to_cos": True,
                "freq_shift": 0,
                "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
                "mid_block_type": "UNetMidBlock2DCrossAttn",
                "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
                "only_cross_attention": False,
                "block_out_channels": [320, 640, 1280],
                "layers_per_block": 2,
                "downsample_padding": 1,
                "mid_block_scale_factor": 1,
                "dropout": 0.0,
                "act_fn": "silu",
                "norm_num_groups": 32,
                "norm_eps": 1e-05,
                "cross_attention_dim": 2048,
                "transformer_layers_per_block": [1, 2, 10],
                "reverse_transformer_layers_per_block": None,
                "encoder_hid_dim": None,
                "encoder_hid_dim_type": None,
                "attention_head_dim": [5, 10, 20],
                "num_attention_heads": None,
                "dual_cross_attention": False,
                "use_linear_projection": True,
                "class_embed_type": None,
                "addition_embed_type": "text_time",
                "addition_time_embed_dim": 256,
                "num_class_embeds": None,
                "upcast_attention": None,
                "resnet_time_scale_shift": "default",
                "resnet_skip_time_act": False,
                "resnet_out_scale_factor": 1.0,
                "time_embedding_type": "positional",
                "time_embedding_dim": None,
                "time_embedding_act_fn": None,
                "timestep_post_act": None,
                "time_cond_proj_dim": None,
                "conv_in_kernel": 3,
                "conv_out_kernel": 3,
                "projection_class_embeddings_input_dim": 2816,
                "attention_type": "default",
                "class_embeddings_concat": False,
                "mid_block_only_cross_attention": None,
                "cross_attention_norm": None,
                "addition_embed_type_num_heads": 64
            }

        # Build model on meta device
        with torch.device("meta"):
            unet = self._unet_cls.from_config(unet_config).to(self.dtype)

        # Patch with quantization
        unet._patch_model(precision=precision, rank=rank)

        # Move to real device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet = unet.to_empty(device=device)

        # Convert and load state dict
        converted_sd = preview_sd if isinstance(preview_sd, dict) else self._convert_state_dict(sd)

        # Handle wtscale (stored as float, not tensor)
        for n, m in unet.named_modules():
            if isinstance(m, SVDQW4A4Linear):
                if getattr(m, "wtscale", None) is not None:
                    key = f"{n}.wtscale"
                    if key in converted_sd:
                        val = converted_sd.pop(key)
                        m.wtscale = float(val.item()) if torch.is_tensor(val) else float(val)
                    else:
                        m.wtscale = 1.0
                # Fill missing wcscales
                wc_key = f"{n}.wcscales"
                if getattr(m, "wcscales", None) is not None and wc_key not in converted_sd:
                    converted_sd[wc_key] = torch.ones_like(m.wcscales)

        # Fill missing proj_down/proj_up with zeros
        for n, m in unet.named_modules():
            if isinstance(m, SVDQW4A4Linear):
                down_k = f"{n}.proj_down"
                up_k = f"{n}.proj_up"
                if down_k not in converted_sd:
                    converted_sd[down_k] = torch.zeros_like(m.proj_down)
                if up_k not in converted_sd:
                    converted_sd[up_k] = torch.zeros_like(m.proj_up)

        missing, unexpected = unet.load_state_dict(converted_sd, strict=False)
        if missing:
            print(f"[SVDQ SDXL] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[SVDQ SDXL] Unexpected keys: {len(unexpected)}")

        self._nunchaku_unet = unet
        return unet

    def _detect_lora_format(self, lora_sd: dict, lora_name: str) -> dict:
        """
        Detect LoRA formats based on key patterns.
        Reference: ComfyUI-QwenImageLoraLoader v2.2.3
        """
        keys = list(lora_sd.keys())
        
        standard_patterns = (
            ".lora_up.weight", ".lora_down.weight",
            ".lora_A.weight", ".lora_B.weight",
            ".lora.up.weight", ".lora.down.weight",
            ".lora.A.weight", ".lora.B.weight",
        )
        
        def _sample(match_fn, limit: int = 5) -> list:
            return [k for k in keys if match_fn(k)][:limit]
        
        has_standard = any(p in k for k in keys for p in standard_patterns)
        has_lokr = any(".lokr_w1" in k or ".lokr_w2" in k for k in keys)
        has_loha = any(".hada_w1" in k or ".hada_w2" in k for k in keys)
        has_ia3 = any(".ia3." in k or ".ia3_w" in k for k in keys)
        
        # Count UNet keys
        unet_keys = [k for k in keys if k.startswith("lora_unet_") or "down_blocks" in k or "up_blocks" in k or "mid_block" in k]
        
        return {
            "has_standard": has_standard,
            "has_lokr": has_lokr,
            "has_loha": has_loha,
            "has_ia3": has_ia3,
            "sample_standard": _sample(lambda k: any(p in k for p in standard_patterns)),
            "sample_lokr": _sample(lambda k: ".lokr_w1" in k or ".lokr_w2" in k),
            "total_keys": len(keys),
            "unet_keys": len(unet_keys),
        }

    def _log_lora_detection(self, lora_name: str, detection: dict, strength: float) -> None:
        """Log LoRA format detection results."""
        sep = "=" * 80
        print(sep)
        print(f"[SVDQ SDXL LoRA] Format Detection: {os.path.basename(lora_name)}")
        print(sep)
        print(f"  Strength: {strength:.2f}")
        print(f"  Total keys: {detection['total_keys']}")
        print(f"  UNet keys: {detection['unet_keys']}")
        print("  Detected Formats:")
        
        if detection["has_standard"]:
            print("    ✅ Standard LoRA (Rank-Decomposed) - Supported")
        if detection["has_lokr"]:
            print("    ❌ LoKR (Lycoris) - Not Supported for SVDQ")
        if detection["has_loha"]:
            print("    ❌ LoHa - Not Supported for SVDQ")
        if detection["has_ia3"]:
            print("    ❌ IA3 - Not Supported for SVDQ")
        
        if not (detection["has_standard"] or detection["has_lokr"] or detection["has_loha"] or detection["has_ia3"]):
            print("    ⚠️ Unknown format (no recognized LoRA keys)")
        
        if detection["sample_standard"]:
            print(f"  Sample keys: {detection['sample_standard'][:3]}")
        
        print(sep)

    def _lora_base_to_module_path(self, base: str) -> str | None:
        """
        Convert LoRA base key to module dot-path.
        
        Examples:
        - lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_qkv
          → down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_qkv
        
        Reference: ComfyUI-nunchaku-unofficial-loader v2.1
        """
        import re
        
        if not isinstance(base, str) or not base.startswith("lora_unet_"):
            return None
        
        rest = base[len("lora_unet_"):]
        
        # Direct attention module projections (NOT inside transformer_blocks)
        # These showed up in your unmatched list: *_attentions_*_proj_in / proj_out
        m = re.match(r"^(down_blocks|up_blocks)_(\d+)_attentions_(\d+)_(proj_in|proj_out)$", rest)
        if m is not None:
            block_type, bi, ai, which = m.group(1), m.group(2), m.group(3), m.group(4)
            return f"{block_type}.{bi}.attentions.{ai}.{which}"

        m = re.match(r"^mid_block_attentions_(\d+)_(proj_in|proj_out)$", rest)
        if m is not None:
            ai, which = m.group(1), m.group(2)
            return f"mid_block.attentions.{ai}.{which}"

        # Resnet / sampler blocks (these were the remaining unmapped bases in coverage logs)
        m = re.match(r"^(down_blocks|up_blocks)_(\d+)_resnets_(\d+)_(.+)$", rest)
        if m is not None:
            bt, bi, ri, tail = m.group(1), m.group(2), m.group(3), m.group(4)
            return f"{bt}.{bi}.resnets.{ri}.{tail}"

        m = re.match(r"^mid_block_resnets_(\d+)_(.+)$", rest)
        if m is not None:
            ri, tail = m.group(1), m.group(2)
            return f"mid_block.resnets.{ri}.{tail}"

        m = re.match(r"^(down_blocks)_(\d+)_downsamplers_(\d+)_(.+)$", rest)
        if m is not None:
            bt, bi, di, tail = m.group(1), m.group(2), m.group(3), m.group(4)
            return f"{bt}.{bi}.downsamplers.{di}.{tail}"

        m = re.match(r"^(up_blocks)_(\d+)_upsamplers_(\d+)_(.+)$", rest)
        if m is not None:
            bt, bi, ui, tail = m.group(1), m.group(2), m.group(3), m.group(4)
            return f"{bt}.{bi}.upsamplers.{ui}.{tail}"

        # Top-level blocks commonly present in SDXL LoRAs
        if rest == "conv_in":
            return "conv_in"
        if rest == "conv_out":
            return "conv_out"
        m = re.match(r"^time_embedding_linear_(\d+)$", rest)
        if m is not None:
            return f"time_embedding.linear_{m.group(1)}"
        m = re.match(r"^add_embedding_linear_(\d+)$", rest)
        if m is not None:
            return f"add_embedding.linear_{m.group(1)}"

        # Fast path: only handle transformer-block internals we know are quantized in Nunchaku SDXL
        tail_map = {
            "attn1_to_qkv": "attn1.to_qkv",
            "attn1_to_out_0": "attn1.to_out.0",
            "attn1_to_out": "attn1.to_out.0",
            "attn2_to_q": "attn2.to_q",
            "attn2_to_k": "attn2.to_k",
            "attn2_to_v": "attn2.to_v",
            "attn2_to_out_0": "attn2.to_out.0",
            "attn2_to_out": "attn2.to_out.0",
            "ff_net_0_proj": "ff.net.0.proj",
            "ff_net_2": "ff.net.2",
        }
        
        for tail_key, tail_path in tail_map.items():
            suffix = f"_{tail_key}"
            if not rest.endswith(suffix):
                continue
            
            prefix = rest[:-len(suffix)]
            
            # Match block patterns
            m = re.match(
                r"^(down_blocks|up_blocks)_(\d+)_attentions_(\d+)_transformer_blocks_(\d+)$",
                prefix,
            )
            if m is not None:
                block_type, bi, ai, ti = m.group(1), m.group(2), m.group(3), m.group(4)
                return f"{block_type}.{bi}.attentions.{ai}.transformer_blocks.{ti}.{tail_path}"
            
            m = re.match(r"^mid_block_attentions_(\d+)_transformer_blocks_(\d+)$", prefix)
            if m is not None:
                ai, ti = m.group(1), m.group(2)
                return f"mid_block.attentions.{ai}.transformer_blocks.{ti}.{tail_path}"
        
        return None

    def _apply_loras(self):
        """
        Apply LoRAs to the quantized UNet using runtime forward-add method.
        
        Nunchaku SDXL UNet uses SVDQW4A4Linear modules which don't have built-in LoRA support.
        We patch the forward methods to add LoRA deltas at runtime.
        
        Reference: ComfyUI-nunchaku-unofficial-loader v2.1
        """
        # Skip if no LoRAs
        if not self.loras:
            # Clear any existing runtime LoRA patches
            if getattr(self, "_lora_runtime_active", False):
                self._clear_runtime_lora()
                print("[SVDQ SDXL LoRA] Cleared all runtime LoRA patches")
            return

        # Check if LoRA config changed
        lora_key = tuple((fn, s) for fn, s in self.loras)
        if hasattr(self, "_applied_lora_key") and self._applied_lora_key == lora_key:
            return  # Already applied

        unet = self._nunchaku_unet
        if unet is None:
            # Reset applied_lora_key so LoRA will be reapplied when unet becomes available
            if hasattr(self, "_applied_lora_key"):
                delattr(self, "_applied_lora_key")
            return

        # Clear previous LoRA patches
        self._clear_runtime_lora()

        # Load and convert LoRAs
        from backend.utils import load_torch_file

        print("\n" + "=" * 80)
        print("[SVDQ SDXL LoRA] Loading LoRAs...")
        print(f"  Total LoRAs to apply: {len(self.loras)}")
        print("=" * 80)

        lora_weights = []
        for idx, (filename, strength) in enumerate(self.loras, 1):
            print(f"\n[LoRA {idx}/{len(self.loras)}] Loading: {os.path.basename(filename)}")
            
            if filename not in self._lora_cache:
                try:
                    lora_sd = load_torch_file(filename, safe_load=True)
                    
                    # Detect and log format
                    detection = self._detect_lora_format(lora_sd, filename)
                    self._log_lora_detection(filename, detection, strength)
                    
                    # Skip unsupported formats
                    if not detection["has_standard"]:
                        print(f"  ⚠️ SKIPPED: No Standard LoRA keys found")
                        continue
                    
                    converted = self._convert_lora_keys(lora_sd)
                    self._lora_cache[filename] = converted
                    
                    # Debug: show sample converted keys
                    sample_keys = [k for k in list(converted.keys())[:5]]
                    print(f"  ✅ Loaded and converted: {len(converted)} keys")
                    if sample_keys:
                        print(f"  Sample converted keys: {sample_keys}")
                except Exception as e:
                    import traceback
                    print(f"  ❌ Failed to load: {e}")
                    traceback.print_exc()
                    continue
            else:
                converted = self._lora_cache[filename]
                print(f"  ✅ Using cached: {len(converted)} keys")
            
            lora_weights.append((converted, strength, os.path.basename(filename)))

        if not lora_weights:
            print("\n[SVDQ SDXL LoRA] No valid LoRAs to apply")
            self._applied_lora_key = lora_key
            return

        # Apply runtime LoRA patches to SVDQW4A4Linear modules
        try:
            from nunchaku.models.linear import SVDQW4A4Linear
        except ImportError:
            print("[SVDQ SDXL LoRA] Failed to import SVDQW4A4Linear")
            self._applied_lora_key = lora_key
            return

        # ComfyUI-QwenImageLoraLoader-style "coverage / mapping" logging.
        # User explicitly requested "no compromise" until complete mapping.
        try:
            coverage_debug = str(os.getenv("NUNCHAKU_SDXL_LORA_COVERAGE", "1")).strip().lower() not in ("0", "false", "off", "no")
        except Exception:
            coverage_debug = True

        print("\n" + "=" * 80)
        print("[SVDQ SDXL LoRA] Patching SVDQW4A4Linear modules...")
        print("=" * 80)

        # Build base -> (down, up, alpha, strength) mapping
        # Group by LoRA base key
        bases_all = {}  # base_key -> [(down, up, alpha, strength, lora_name), ...]
        for lora_sd, strength, lora_name in lora_weights:
            for k in lora_sd.keys():
                if not isinstance(k, str):
                    continue
                if k.endswith(".lora_down.weight"):
                    base = k[:-len(".lora_down.weight")]
                    up_key = base + ".lora_up.weight"
                    alpha_key = base + ".alpha"
                    if up_key in lora_sd:
                        down = lora_sd[k]
                        up = lora_sd[up_key]
                        alpha = lora_sd.get(alpha_key)
                        if base not in bases_all:
                            bases_all[base] = []
                        bases_all[base].append((down, up, alpha, strength, lora_name))

        patched_modules = []
        patched_count = 0

        # Coverage buckets (base-level)
        base_total = 0
        base_has_pair = 0
        base_resolved_path = 0
        base_resolve_failed = []  # bases where base->path mapping failed
        base_path_lookup_failed = []  # bases where path exists but getattr/indexing failed
        base_type_not_supported = []  # bases where module exists but type not supported
        base_patched = 0

        # Track duplicates and partials
        base_missing_up = 0
        base_missing_down = 0
        
        for base, lora_entries in bases_all.items():
            base_total += 1

            # Convert base to module path
            mod_path = self._lora_base_to_module_path(base)
            if mod_path is None:
                base_resolve_failed.append(base)
                continue
            base_resolved_path += 1
            
            # Find the module
            try:
                mod = unet
                for seg in mod_path.split("."):
                    if seg.isdigit():
                        mod = mod[int(seg)]
                    else:
                        mod = getattr(mod, seg)
            except (AttributeError, KeyError, IndexError) as e:
                base_path_lookup_failed.append(f"{base} (path: {mod_path}, error: {e})")
                continue
            
            # We only count as "has_pair" if at least one entry has both down/up present.
            # (We already constructed bases_all from down->up pairs, so this should be true,
            # but keep counters explicit for coverage logs.)
            base_has_pair += 1
            
            # Prepare LoRA data for this module
            mod_loras = []
            matched_info = []
            for down, up, alpha, strength, lora_name in lora_entries:
                rank = int(down.shape[0]) if hasattr(down, "shape") else 1
                alpha_val = float(alpha.item()) if alpha is not None and hasattr(alpha, "item") else (float(alpha) if alpha is not None else None)
                alpha_scale = (alpha_val / rank) if (alpha_val is not None and rank > 0) else 1.0
                scale = float(strength) * float(alpha_scale)
                mod_loras.append((down.detach().cpu(), up.detach().cpu(), scale))
                matched_info.append(f"{lora_name}(r={rank},s={strength:.2f},α={alpha_val})")
            
            if mod_loras:
                # Patch depending on module type.
                # 1) SVDQW4A4Linear: runtime forward-add (quantized)
                # 2) torch.nn.Linear: runtime forward-add (non-quantized) so we can still "fully map"
                # 3) torch.nn.Conv2d: runtime forward-add (needed for full SDXL LoRA coverage)
                if isinstance(mod, SVDQW4A4Linear):
                    self._patch_module_forward(mod, mod_loras)
                    patched_count += 1
                    base_patched += 1
                    patched_modules.append((mod_path, matched_info))
                elif isinstance(mod, nn.Linear):
                    self._patch_linear_forward(mod, mod_loras)
                    patched_count += 1
                    base_patched += 1
                    patched_modules.append((mod_path, matched_info))
                elif isinstance(mod, nn.Conv2d):
                    self._patch_conv2d_forward(mod, mod_loras)
                    patched_count += 1
                    base_patched += 1
                    patched_modules.append((mod_path, matched_info))
                else:
                    base_type_not_supported.append(f"{base} -> {mod_path} type={type(mod)}")

        # Log patched modules summary
        print(f"\n[SVDQ SDXL LoRA] Patched {patched_count} modules:")
        
        # Group by block type for cleaner output
        block_counts = {"down_blocks": 0, "mid_block": 0, "up_blocks": 0, "other": 0}
        for name, loras in patched_modules:
            if "down_blocks" in name:
                block_counts["down_blocks"] += 1
            elif "mid_block" in name:
                block_counts["mid_block"] += 1
            elif "up_blocks" in name:
                block_counts["up_blocks"] += 1
            else:
                block_counts["other"] += 1
        
        print(f"  - Down blocks: {block_counts['down_blocks']} modules")
        print(f"  - Mid block: {block_counts['mid_block']} modules")
        print(f"  - Up blocks: {block_counts['up_blocks']} modules")
        if block_counts["other"] > 0:
            print(f"  - Other: {block_counts['other']} modules")
        
        # Show first few patched modules as examples
        print("\n  Sample patched modules:")
        for name, loras in patched_modules[:5]:
            print(f"    - {name}")
            for lora_info in loras:
                print(f"      └─ {lora_info}")
        if len(patched_modules) > 5:
            print(f"    ... and {len(patched_modules) - 5} more")
        
        if coverage_debug:
            print("\n" + "=" * 80)
            print("[SVDQ SDXL LoRA COVERAGE] base-level mapping report (no compromise mode)")
            print("=" * 80)
            print(f"  bases_total:             {base_total}")
            print(f"  bases_has_pair(down+up): {base_has_pair}")
            print(f"  bases_resolved_path:     {base_resolved_path}")
            print(f"  bases_patched:           {base_patched}")
            unresolved = base_total - base_patched
            print(f"  bases_unpatched_total:   {unresolved}")

            if base_resolve_failed:
                print(f"\n  [UNMAPPED] base->module_path failed: {len(base_resolve_failed)}")
                for ub in base_resolve_failed:
                    print(f"    - {ub}")

            if base_path_lookup_failed:
                print(f"\n  [UNRESOLVED] module_path lookup failed: {len(base_path_lookup_failed)}")
                for ub in base_path_lookup_failed:
                    print(f"    - {ub}")

            if base_type_not_supported:
                print(f"\n  [UNSUPPORTED] module type not supported: {len(base_type_not_supported)}")
                for ub in base_type_not_supported:
                    print(f"    - {ub}")

            print("=" * 80 + "\n")

        if patched_count > 0:
            print(f"\n✅ [SVDQ SDXL LoRA] Successfully applied runtime LoRA to {patched_count} modules")
            self._lora_runtime_active = True
        else:
            print("\n⚠️ [SVDQ SDXL LoRA] No modules were patched (LoRA keys may not match model)")

        print("=" * 80 + "\n")
        self._applied_lora_key = lora_key

    def _patch_linear_forward(self, mod: nn.Linear, loras):
        """
        Runtime forward-add for non-quantized nn.Linear (to reach full mapping coverage).
        Uses the same (x @ down.T @ up.T) formulation as SVDQW4A4Linear patch.
        """
        if hasattr(mod, "_orig_forward_before_lora"):
            mod._runtime_lora_data = loras
            # Invalidate prepared cache (new LoRA set / scales)
            try:
                mod._runtime_lora_version = int(getattr(mod, "_runtime_lora_version", 0)) + 1
            except Exception:
                mod._runtime_lora_version = 1
            return

        orig_forward = mod.forward
        mod._orig_forward_before_lora = orig_forward
        mod._runtime_lora_data = loras
        mod._runtime_lora_prepared = {}  # (device, dtype) -> {"ver": int, "down": Tensor, "up": Tensor}
        mod._runtime_lora_version = 1

        def _forward_with_lora(x):
            base_out = orig_forward(x)
            lora_data = getattr(mod, "_runtime_lora_data", None)
            if not lora_data:
                return base_out

            x_shape = x.shape
            x2 = x.reshape(-1, x_shape[-1])
            device = base_out.device
            # Compute dtype for runtime LoRA matmuls:
            # Default: match base_out.dtype (fast). Allow override for debugging / stability.
            try:
                mode = os.getenv("NUNCHAKU_SDXL_SVDQ_RUNTIME_DTYPE", "out").strip().lower()
            except Exception:
                mode = "out"
            if mode in ("fp32", "float32"):
                dtype = torch.float32
            elif mode in ("x", "input"):
                dtype = x2.dtype
            else:
                dtype = base_out.dtype
            if x2.device != device:
                x2 = x2.to(device=device)

            # Prepare/fuse LoRAs ONCE per (device, dtype, version)
            ver = int(getattr(mod, "_runtime_lora_version", 0))
            cache_key = (str(device), str(dtype))
            prepared = getattr(mod, "_runtime_lora_prepared", None)
            if not isinstance(prepared, dict):
                prepared = {}
                mod._runtime_lora_prepared = prepared

            entry = prepared.get(cache_key, None)
            if not (isinstance(entry, dict) and entry.get("ver", None) == ver):
                downs = []
                ups = []
                for down, up, scale in lora_data:
                    d = down.to(device=device, dtype=dtype, non_blocking=True).contiguous()
                    u = up.to(device=device, dtype=dtype, non_blocking=True).contiguous()
                    if scale != 1.0:
                        u = (u * float(scale)).contiguous()
                    downs.append(d)
                    ups.append(u)
                # Fuse ranks: down_cat (sum_r, in), up_cat (out, sum_r)
                down_cat = torch.cat(downs, dim=0).contiguous() if len(downs) > 1 else downs[0]
                up_cat = torch.cat(ups, dim=1).contiguous() if len(ups) > 1 else ups[0]
                entry = {"ver": ver, "down": down_cat, "up": up_cat}
                prepared[cache_key] = entry

            down_cat = entry["down"]
            up_cat = entry["up"]
            # delta = (x @ down_cat.T) @ up_cat.T
            delta = (x2.to(dtype=dtype) @ down_cat.T) @ up_cat.T

            if delta is None:
                return base_out
            delta = delta.reshape(*x_shape[:-1], base_out.shape[-1])
            # Safety guard (Nunchaku-only): prevent NaN/Inf from turning the whole image black.
            # Enable with NUNCHAKU_SDXL_LORA_GUARD=1 (default: ON).
            try:
                _guard = str(os.getenv("NUNCHAKU_SDXL_LORA_GUARD", "1")).strip().lower() not in ("0", "false", "off", "no")
            except Exception:
                _guard = True
            if _guard:
                try:
                    if not torch.isfinite(delta).all():
                        if not getattr(mod, "_nunchaku_lora_nonfinite_warned", False):
                            mod._nunchaku_lora_nonfinite_warned = True
                            try:
                                print("[NUNCHAKU_SDXL_LORA] WARNING: non-finite delta detected (nn.Linear). Replacing NaN/Inf with 0.")
                            except Exception:
                                pass
                        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    pass
            # Keep output dtype identical to original module output
            if delta.dtype != base_out.dtype:
                delta = delta.to(dtype=base_out.dtype)
            return base_out + delta

        mod.forward = _forward_with_lora

    def _patch_conv2d_forward(self, mod: nn.Conv2d, loras):
        """
        Runtime forward-add for nn.Conv2d (needed for full SDXL LoRA coverage).
        Supports common LoRA conv shapes:
          - down: (r, in, k, k) and up: (out, r, 1, 1)
          - down: (r, in, 1, 1) and up: (out, r, k, k)
          - down/up both (., ., k, k) -> einsum to build deltaW
        """
        if hasattr(mod, "_orig_forward_before_lora"):
            mod._runtime_lora_data = loras
            # Invalidate prepared cache (new LoRA set / scales)
            try:
                mod._runtime_lora_version = int(getattr(mod, "_runtime_lora_version", 0)) + 1
            except Exception:
                mod._runtime_lora_version = 1
            return

        orig_forward = mod.forward
        mod._orig_forward_before_lora = orig_forward
        mod._runtime_lora_data = loras
        mod._runtime_lora_prepared = {}  # (device, dtype) -> {"ver": int, "pairs": [(d,u,scale)], "delta_w": Tensor|None}
        mod._runtime_lora_version = 1

        k_hw = tuple(mod.kernel_size) if isinstance(mod.kernel_size, tuple) else (int(mod.kernel_size), int(mod.kernel_size))

        def _as_4d(w):
            if w is None:
                return None
            if not torch.is_tensor(w):
                return w
            if w.ndim == 2:
                return w[:, :, None, None]
            return w

        def _forward_with_lora(x):
            base_out = orig_forward(x)
            lora_data = getattr(mod, "_runtime_lora_data", None)
            if not lora_data:
                return base_out

            device = base_out.device
            # Compute dtype for runtime LoRA conv ops:
            try:
                mode = os.getenv("NUNCHAKU_SDXL_SVDQ_RUNTIME_DTYPE", "out").strip().lower()
            except Exception:
                mode = "out"
            if mode in ("fp32", "float32"):
                dtype = torch.float32
            elif mode in ("x", "input"):
                dtype = x.dtype
            else:
                dtype = base_out.dtype

            # Prepare ONCE per (device, dtype, version) to avoid per-forward .to() overhead.
            ver = int(getattr(mod, "_runtime_lora_version", 0))
            cache_key = (str(device), str(dtype))
            prepared = getattr(mod, "_runtime_lora_prepared", None)
            if not isinstance(prepared, dict):
                prepared = {}
                mod._runtime_lora_prepared = prepared
            entry = prepared.get(cache_key, None)
            if not (isinstance(entry, dict) and entry.get("ver", None) == ver):
                pairs = []
                # For the (k,k)+(k,k) case, we can precompute and sum deltaW once.
                delta_w_sum = None
                for down, up, scale in lora_data:
                    d = _as_4d(down)
                    u = _as_4d(up)
                    if not (torch.is_tensor(d) and torch.is_tensor(u)):
                        continue
                    d = d.to(device=device, dtype=dtype, non_blocking=True).contiguous()
                    u = u.to(device=device, dtype=dtype, non_blocking=True).contiguous()
                    s = float(scale)

                    dhw = tuple(d.shape[2:])
                    uhw = tuple(u.shape[2:])
                    if dhw == k_hw and uhw == k_hw:
                        # Precompute deltaW: (out, r, k, k) x (r, in, k, k) -> (out, in, k, k)
                        dw = torch.einsum("orhw,rihw->oihw", u, d)
                        if s != 1.0:
                            dw = dw * s
                        delta_w_sum = dw if delta_w_sum is None else (delta_w_sum + dw)
                    else:
                        pairs.append((d, u, s))

                entry = {"ver": ver, "pairs": pairs, "delta_w": delta_w_sum}
                prepared[cache_key] = entry

            delta = None
            # Precomputed deltaW path (fast, single conv)
            dw = entry.get("delta_w", None) if isinstance(entry, dict) else None
            if torch.is_tensor(dw):
                y = F.conv2d(x.to(dtype=dtype), dw, bias=None, stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups)
                delta = y if delta is None else (delta + y)

            # Remaining mixed-kernel pairs (still cached tensors; no per-forward .to()).
            for d, u, s in (entry.get("pairs", []) if isinstance(entry, dict) else []):
                dhw = tuple(d.shape[2:])
                uhw = tuple(u.shape[2:])

                if dhw == k_hw and uhw == (1, 1):
                    y = F.conv2d(x.to(dtype=dtype), d, bias=None, stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups)
                    y = F.conv2d(y, u, bias=None, stride=1, padding=0, dilation=1, groups=1)
                elif uhw == k_hw and dhw == (1, 1):
                    y = F.conv2d(x.to(dtype=dtype), d, bias=None, stride=1, padding=0, dilation=1, groups=1)
                    y = F.conv2d(y, u, bias=None, stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups)
                else:
                    continue

                if s != 1.0:
                    y = y * float(s)
                delta = y if delta is None else (delta + y)

            if delta is None:
                return base_out
            # Safety guard (Nunchaku-only): prevent NaN/Inf from turning the whole image black.
            try:
                _guard = str(os.getenv("NUNCHAKU_SDXL_LORA_GUARD", "1")).strip().lower() not in ("0", "false", "off", "no")
            except Exception:
                _guard = True
            if _guard:
                try:
                    if not torch.isfinite(delta).all():
                        if not getattr(mod, "_nunchaku_lora_nonfinite_warned", False):
                            mod._nunchaku_lora_nonfinite_warned = True
                            try:
                                print("[NUNCHAKU_SDXL_LORA] WARNING: non-finite delta detected (nn.Conv2d). Replacing NaN/Inf with 0.")
                            except Exception:
                                pass
                        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    pass
            if delta.dtype != base_out.dtype:
                delta = delta.to(dtype=base_out.dtype)
            return base_out + delta

        mod.forward = _forward_with_lora

    def _clear_runtime_lora(self):
        """Clear runtime LoRA patches from all modules."""
        unet = self._nunchaku_unet
        if unet is None:
            # Even if unet is None, reset the applied_lora_key to force re-application
            # when unet becomes available again
            if hasattr(self, "_applied_lora_key"):
                delattr(self, "_applied_lora_key")
            self._lora_runtime_active = False
            return

        for name, mod in unet.named_modules():
            if hasattr(mod, "_orig_forward_before_lora"):
                mod.forward = mod._orig_forward_before_lora
                delattr(mod, "_orig_forward_before_lora")
            if hasattr(mod, "_runtime_lora_data"):
                delattr(mod, "_runtime_lora_data")
            # Clear runtime LoRA prepared cache (GPU memory)
            if hasattr(mod, "_runtime_lora_prepared"):
                try:
                    mod._runtime_lora_prepared.clear()
                except Exception:
                    pass
                try:
                    delattr(mod, "_runtime_lora_prepared")
                except Exception:
                    pass
            # Clear runtime LoRA version
            if hasattr(mod, "_runtime_lora_version"):
                try:
                    delattr(mod, "_runtime_lora_version")
                except Exception:
                    pass
            # Clear any non-finite warning flag
            if hasattr(mod, "_nunchaku_lora_nonfinite_warned"):
                try:
                    delattr(mod, "_nunchaku_lora_nonfinite_warned")
                except Exception:
                    pass

        # Reset applied_lora_key to force re-application
        if hasattr(self, "_applied_lora_key"):
            delattr(self, "_applied_lora_key")
        self._lora_runtime_active = False

    def _patch_module_forward(self, mod, loras):
        """
        Patch a SVDQW4A4Linear module's forward to add LoRA delta.
        
        loras: List of (down, up, scale) tuples
        """
        if hasattr(mod, "_orig_forward_before_lora"):
            # Already patched, update loras
            mod._runtime_lora_data = loras
            # Invalidate prepared cache (new LoRA set / scales)
            try:
                mod._runtime_lora_version = int(getattr(mod, "_runtime_lora_version", 0)) + 1
            except Exception:
                mod._runtime_lora_version = 1
            return

        orig_forward = mod.forward
        mod._orig_forward_before_lora = orig_forward
        mod._runtime_lora_data = loras
        mod._runtime_lora_prepared = {}  # (device, dtype) -> {"ver": int, "down": Tensor, "up": Tensor}
        mod._runtime_lora_version = 1

        def _forward_with_lora(x, output=None):
            base_out = orig_forward(x, output)

            lora_data = getattr(mod, "_runtime_lora_data", None)
            if not lora_data:
                return base_out

            # Compute LoRA delta: sum of (x @ down.T @ up.T) * scale for each LoRA
            x_shape = x.shape
            x2 = x.reshape(-1, x_shape[-1])
            device = base_out.device
            # Compute dtype for runtime LoRA matmuls:
            # Default: match base_out.dtype (fast). Allow override for stability.
            try:
                mode = os.getenv("NUNCHAKU_SDXL_SVDQ_RUNTIME_DTYPE", "out").strip().lower()
            except Exception:
                mode = "out"
            if mode in ("fp32", "float32"):
                dtype = torch.float32
            elif mode in ("x", "input"):
                dtype = x2.dtype
            else:
                dtype = base_out.dtype

            if x2.device != device:
                x2 = x2.to(device=device)

            # Prepare/fuse LoRAs ONCE per (device, dtype, version)
            ver = int(getattr(mod, "_runtime_lora_version", 0))
            cache_key = (str(device), str(dtype))
            prepared = getattr(mod, "_runtime_lora_prepared", None)
            if not isinstance(prepared, dict):
                prepared = {}
                mod._runtime_lora_prepared = prepared
            entry = prepared.get(cache_key, None)
            if not (isinstance(entry, dict) and entry.get("ver", None) == ver):
                downs = []
                ups = []
                for down, up, scale in lora_data:
                    d = down.to(device=device, dtype=dtype, non_blocking=True).contiguous()
                    u = up.to(device=device, dtype=dtype, non_blocking=True).contiguous()
                    if scale != 1.0:
                        u = (u * float(scale)).contiguous()
                    downs.append(d)
                    ups.append(u)
                down_cat = torch.cat(downs, dim=0).contiguous() if len(downs) > 1 else downs[0]
                up_cat = torch.cat(ups, dim=1).contiguous() if len(ups) > 1 else ups[0]
                entry = {"ver": ver, "down": down_cat, "up": up_cat}
                prepared[cache_key] = entry

            down_cat = entry["down"]
            up_cat = entry["up"]
            delta = (x2.to(dtype=dtype) @ down_cat.T) @ up_cat.T

            if delta is not None:
                delta = delta.reshape(*x_shape[:-1], base_out.shape[-1])
                # Safety guard (Nunchaku-only): prevent NaN/Inf from turning the whole image black.
                try:
                    _guard = str(os.getenv("NUNCHAKU_SDXL_LORA_GUARD", "1")).strip().lower() not in ("0", "false", "off", "no")
                except Exception:
                    _guard = True
                if _guard:
                    try:
                        if not torch.isfinite(delta).all():
                            if not getattr(mod, "_nunchaku_lora_nonfinite_warned", False):
                                mod._nunchaku_lora_nonfinite_warned = True
                                try:
                                    print("[NUNCHAKU_SDXL_LORA] WARNING: non-finite delta detected (SVDQW4A4Linear). Replacing NaN/Inf with 0.")
                                except Exception:
                                    pass
                            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:
                        pass
                if delta.dtype != base_out.dtype:
                    delta = delta.to(dtype=base_out.dtype)
                base_out = base_out + delta

            return base_out

        mod.forward = _forward_with_lora

    def _build_lora_mapping(self, lora_weights, unet):
        """Build a mapping of module paths to LoRA weights."""
        # This is a placeholder - actual implementation depends on key format
        return {}

    def _convert_lora_keys(self, lora_sd: dict) -> dict:
        """
        Convert A1111 LoRA keys to Nunchaku SDXL diffusers format.
        
        This handles:
        1. A1111 format (input_blocks/output_blocks/middle_block) → diffusers format (down_blocks/up_blocks/mid_block)
        2. Q/K/V → QKV fusion for fused attention layers
        3. Key normalization (.lora_down/.lora_up → .lora_A/.lora_B)
        
        Reference: ComfyUI-nunchaku-unofficial-loader v2.1
        """
        import re
        
        # First, build the A1111 → diffusers mapping
        a1111_to_diffusers = self._build_a1111_to_diffusers_map()
        
        # Ultra-verbose (ComfyUI-level) debugging toggle:
        # Default ON, but allow temporary muting to keep ControlNet logs visible.
        try:
            _mute = str(os.getenv("NUNCHAKU_SDXL_LORA_DEBUG_MUTE", "0")).strip().lower() not in ("0", "false", "off", "no")
        except Exception:
            _mute = False
        try:
            verbose = (not _mute) and (str(os.getenv("NUNCHAKU_SDXL_LORA_DEBUG", "1")).strip().lower() not in ("0", "false", "off", "no"))
        except Exception:
            verbose = (not _mute)

        # Normalize PEFT-style keys (.lora_A/.lora_B → .lora_down/.lora_up)
        normalized = {}
        for k, v in lora_sd.items():
            nk = k
            if ".lora_A.weight" in nk:
                nk = nk.replace(".lora_A.weight", ".lora_down.weight")
            elif ".lora_B.weight" in nk:
                nk = nk.replace(".lora_B.weight", ".lora_up.weight")
            elif ".lora_A.default.weight" in nk:
                nk = nk.replace(".lora_A.default.weight", ".lora_down.weight")
            elif ".lora_B.default.weight" in nk:
                nk = nk.replace(".lora_B.default.weight", ".lora_up.weight")
            normalized[nk] = v
        lora_sd = normalized

        if verbose:
            try:
                total_keys = len(lora_sd)
                unet_keys = sum(1 for k in lora_sd.keys() if isinstance(k, str) and k.startswith("lora_unet_"))
                print("\n" + "=" * 80)
                print("[SVDQ SDXL LoRA DEBUG] Begin key conversion (A1111/PEFT -> diffusers-like)")
                print(f"  total keys: {total_keys}")
                print(f"  unet keys:  {unet_keys}")
                print("  NOTE: printing per-key mapping (very long) because NUNCHAKU_SDXL_LORA_DEBUG=1")
                print("=" * 80)
            except Exception:
                pass
        
        # Convert A1111 keys to diffusers format
        converted = {}
        qkv_pending = {}  # For QKV fusion: base_key -> {"q": (down, up, alpha), "k": ..., "v": ...}

        stats = {
            "kept_nonunet": 0,
            "kept_other_suffix": 0,
            "unmapped_base": 0,
            "regular_mapped": 0,
            "qkv_pending": 0,
        }
        
        for k, v in lora_sd.items():
            if not k.startswith("lora_unet_"):
                if verbose:
                    try:
                        print(f"[SVDQ SDXL LoRA DEBUG] SKIP non-unet: {k}")
                    except Exception:
                        pass
                stats["kept_nonunet"] += 1
                continue
            
            # Extract base and suffix
            if k.endswith(".lora_down.weight"):
                base = k[:-len(".lora_down.weight")]
                suffix = ".lora_down.weight"
            elif k.endswith(".lora_up.weight"):
                base = k[:-len(".lora_up.weight")]
                suffix = ".lora_up.weight"
            elif k.endswith(".alpha"):
                base = k[:-len(".alpha")]
                suffix = ".alpha"
            else:
                # Keep other keys as-is
                converted[k] = v
                stats["kept_other_suffix"] += 1
                if verbose:
                    try:
                        print(f"[SVDQ SDXL LoRA DEBUG] KEEP other suffix: {k}")
                    except Exception:
                        pass
                continue
            
            # Convert A1111 base to diffusers base
            diffusers_base = self._convert_a1111_base_to_diffusers(base, a1111_to_diffusers)
            if diffusers_base is None:
                # No mapping found, keep original
                converted[k] = v
                stats["unmapped_base"] += 1
                if verbose:
                    try:
                        print(f"[SVDQ SDXL LoRA DEBUG] UNMAPPED base: {k}  (base={base})")
                    except Exception:
                        pass
                continue
            
            # Check if this is Q/K/V that needs fusion to QKV
            # Nunchaku SDXL uses attn1.to_qkv (fused) instead of separate to_q/to_k/to_v
            qkv_match = re.search(r"_attn1_to_(q|k|v)$", diffusers_base)
            if qkv_match:
                qkv_type = qkv_match.group(1)  # "q", "k", or "v"
                qkv_base = diffusers_base[:-len(f"_to_{qkv_type}")] + "_to_qkv"
                
                if qkv_base not in qkv_pending:
                    qkv_pending[qkv_base] = {"q": {}, "k": {}, "v": {}}
                qkv_pending[qkv_base][qkv_type][suffix] = v
                stats["qkv_pending"] += 1
                if verbose:
                    try:
                        print(f"[SVDQ SDXL LoRA DEBUG] QKV pending: {k} -> {qkv_base}{suffix} (part={qkv_type})")
                    except Exception:
                        pass
            else:
                # Regular key, just convert
                new_key = diffusers_base + suffix
                converted[new_key] = v
                stats["regular_mapped"] += 1
                if verbose:
                    try:
                        print(f"[SVDQ SDXL LoRA DEBUG] MAP: {k} -> {new_key}")
                    except Exception:
                        pass
        
        # Fuse Q/K/V into QKV
        fused_stats = {"fused_ok": 0, "fused_failed": 0, "fused_incomplete": 0}
        for qkv_base, qkv_data in qkv_pending.items():
            q_data = qkv_data.get("q", {})
            k_data = qkv_data.get("k", {})
            v_data = qkv_data.get("v", {})
            
            q_down = q_data.get(".lora_down.weight")
            q_up = q_data.get(".lora_up.weight")
            k_down = k_data.get(".lora_down.weight")
            k_up = k_data.get(".lora_up.weight")
            v_down = v_data.get(".lora_down.weight")
            v_up = v_data.get(".lora_up.weight")
            
            # Need all three for proper fusion
            if all(x is not None for x in [q_down, q_up, k_down, k_up, v_down, v_up]):
                # Nunchaku SDXL uses fused QKV projection:
                #   to_qkv: weight shape = (3 * inner_dim, in_features)
                # Standard LoRA for fused projection should be:
                #   down: (rank_total, in_features)  where rank_total = rq + rk + rv
                #   up:   (out_features, rank_total) where out_features = 3 * inner_dim
                #
                # Q/K/V -> QKV fusion note (important):
                # - Target module is fused `to_qkv` with out_features = 3 * inner_dim.
                # - Source LoRAs are typically trained for separate to_q/to_k/to_v,
                #   where each `lora_up` has out_features = inner_dim (NOT 3*inner_dim).
                #
                # Therefore:
                #   - down_fused must keep in_features unchanged -> CONCAT rows:
                #       down_fused = cat([q_down, k_down, v_down], dim=0)         # (rq+rk+rv, in)
                #   - up_fused must expand out_features to 3 blocks -> BLOCK-DIAG:
                #       up_fused   = block_diag(q_up, k_up, v_up)                 # (3*inner_dim, rq+rk+rv)
                try:
                    import torch

                    # Validate shapes early to avoid hard-to-debug matmul crashes later.
                    # Expected:
                    #   q_down/k_down/v_down: (r, in)
                    #   q_up/k_up/v_up:       (out, r)
                    if (
                        q_down.ndim != 2
                        or k_down.ndim != 2
                        or v_down.ndim != 2
                        or q_up.ndim != 2
                        or k_up.ndim != 2
                        or v_up.ndim != 2
                    ):
                        raise ValueError("QKV LoRA tensors must be 2D (rank-decomposed).")

                    in_q = int(q_down.shape[1])
                    in_k = int(k_down.shape[1])
                    in_v = int(v_down.shape[1])
                    if not (in_q == in_k == in_v):
                        raise ValueError(f"QKV down in_features mismatch: q={in_q} k={in_k} v={in_v}")

                    out_q = int(q_up.shape[0])
                    out_k = int(k_up.shape[0])
                    out_v = int(v_up.shape[0])
                    if not (out_q == out_k == out_v):
                        raise ValueError(f"QKV up out_features mismatch: q={out_q} k={out_k} v={out_v}")

                    fused_down = torch.cat([q_down, k_down, v_down], dim=0).contiguous()
                    # IMPORTANT: `block_diag` is required here to produce (3*out_q, rq+rk+rv)
                    fused_up = torch.block_diag(q_up, k_up, v_up).contiguous()
                    
                    converted[qkv_base + ".lora_down.weight"] = fused_down
                    converted[qkv_base + ".lora_up.weight"] = fused_up

                    fused_stats["fused_ok"] += 1
                    if verbose:
                        try:
                            print(
                                "[SVDQ SDXL LoRA DEBUG] QKV FUSED OK: "
                                f"{qkv_base} down={tuple(fused_down.shape)} up={tuple(fused_up.shape)}"
                            )
                        except Exception:
                            pass
                    
                    # Alpha handling:
                    # Our runtime scaling later does: scale *= alpha / rank
                    # After fusion rank_total = rq+rk+rv, so setting alpha=rank_total preserves ~1.0 scaling
                    # for the common case alpha==rank per component.
                    alphas = [q_data.get(".alpha"), k_data.get(".alpha"), v_data.get(".alpha")]
                    vals = []
                    for a in alphas:
                        if a is None:
                            continue
                        try:
                            vals.append(float(a.item()) if hasattr(a, "item") else float(a))
                        except Exception:
                            pass
                    if len(vals) == 3:
                        # If all present, summing maintains per-component alpha semantics for the fused rank_total.
                        converted[qkv_base + ".alpha"] = torch.tensor(float(vals[0] + vals[1] + vals[2]))
                    elif len(vals) > 0:
                        # At least one alpha exists; keep scaling sane by defaulting to rank_total.
                        converted[qkv_base + ".alpha"] = torch.tensor(float(fused_down.shape[0]))
                except Exception as e:
                    print(f"[SVDQ SDXL LoRA] QKV fusion failed for {qkv_base}: {e}")
                    fused_stats["fused_failed"] += 1
                    # Fall back to keeping separate keys
                    for qkv_type, data in [("q", q_data), ("k", k_data), ("v", v_data)]:
                        orig_base = qkv_base.replace("_to_qkv", f"_to_{qkv_type}")
                        for suf, val in data.items():
                            converted[orig_base + suf] = val
            else:
                # Incomplete QKV, keep separate keys
                fused_stats["fused_incomplete"] += 1
                for qkv_type, data in [("q", q_data), ("k", k_data), ("v", v_data)]:
                    orig_base = qkv_base.replace("_to_qkv", f"_to_{qkv_type}")
                    for suf, val in data.items():
                        converted[orig_base + suf] = val
        
        if verbose:
            try:
                print("\n" + "=" * 80)
                print("[SVDQ SDXL LoRA DEBUG] Conversion summary")
                for kk in sorted(stats.keys()):
                    print(f"  {kk}: {stats[kk]}")
                for kk in sorted(fused_stats.keys()):
                    print(f"  {kk}: {fused_stats[kk]}")
                print(f"  converted_total_keys: {len(converted)}")
                print("=" * 80 + "\n")
            except Exception:
                pass

        return converted
    
    def _build_a1111_to_diffusers_map(self) -> dict:
        """
        Build A1111 → diffusers key mapping for SDXL.
        Returns a dict mapping A1111 key prefixes to diffusers key prefixes.
        """
        # SDXL config
        num_res_blocks = [2, 2, 2]
        channel_mult = [1, 2, 4]
        transformer_depth = [0, 0, 2, 2, 10, 10]
        transformer_depth_output = [0, 0, 0, 2, 2, 2, 10, 10, 10]
        transformers_mid = 10
        num_blocks = len(channel_mult)
        
        # Build diffusers → A1111 first, then invert
        d2a = {}
        
        # Down blocks
        td = transformer_depth[:]
        for x in range(num_blocks):
            n = 1 + (num_res_blocks[x] + 1) * x
            for i in range(num_res_blocks[x]):
                num_transformers = td.pop(0)
                if num_transformers > 0:
                    for t in range(num_transformers):
                        d_prefix = f"down_blocks_{x}_attentions_{i}_transformer_blocks_{t}"
                        a_prefix = f"input_blocks_{n}_1_transformer_blocks_{t}"
                        d2a[d_prefix] = a_prefix
                    # Also map proj_in/proj_out
                    d2a[f"down_blocks_{x}_attentions_{i}_proj_in"] = f"input_blocks_{n}_1_proj_in"
                    d2a[f"down_blocks_{x}_attentions_{i}_proj_out"] = f"input_blocks_{n}_1_proj_out"
                n += 1
            # Downsampler
            d2a[f"down_blocks_{x}_downsamplers_0_conv"] = f"input_blocks_{n}_0_op"
        
        # Mid block
        for t in range(transformers_mid):
            d2a[f"mid_block_attentions_0_transformer_blocks_{t}"] = f"middle_block_1_transformer_blocks_{t}"
        d2a["mid_block_attentions_0_proj_in"] = "middle_block_1_proj_in"
        d2a["mid_block_attentions_0_proj_out"] = "middle_block_1_proj_out"
        
        # Up blocks
        nrb = list(reversed(num_res_blocks))
        tdo = transformer_depth_output[:]
        for x in range(num_blocks):
            n = (nrb[x] + 1) * x
            l = nrb[x] + 1
            for i in range(l):
                num_transformers = tdo.pop()
                if num_transformers > 0:
                    for t in range(num_transformers):
                        d_prefix = f"up_blocks_{x}_attentions_{i}_transformer_blocks_{t}"
                        a_prefix = f"output_blocks_{n}_1_transformer_blocks_{t}"
                        d2a[d_prefix] = a_prefix
                    d2a[f"up_blocks_{x}_attentions_{i}_proj_in"] = f"output_blocks_{n}_1_proj_in"
                    d2a[f"up_blocks_{x}_attentions_{i}_proj_out"] = f"output_blocks_{n}_1_proj_out"
                n += 1
        
        # Invert: A1111 → diffusers
        a2d = {v: k for k, v in d2a.items()}
        return a2d
    
    def _convert_a1111_base_to_diffusers(self, base: str, a2d_map: dict) -> str | None:
        """
        Convert an A1111-format LoRA base key to diffusers format.
        
        Example:
        - lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q
        → lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q
        """
        if not base.startswith("lora_unet_"):
            return None
        
        rest = base[len("lora_unet_"):]
        
        # Check if already in diffusers format
        if rest.startswith(("down_blocks_", "up_blocks_", "mid_block_")):
            return base  # Already diffusers format
        
        # Try to match A1111 prefixes
        for a1111_prefix, diffusers_prefix in a2d_map.items():
            if rest.startswith(a1111_prefix):
                suffix = rest[len(a1111_prefix):]
                return "lora_unet_" + diffusers_prefix + suffix
        
        # Handle special cases
        # input_blocks_0_0 → conv_in
        if rest.startswith("input_blocks_0_0"):
            return "lora_unet_conv_in" + rest[len("input_blocks_0_0"):]
        
        # No mapping found
        return None

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor = None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        """
        Forward pass converting Forge format to diffusers format.
        Reference: ComfyUI-nunchaku-unofficial-loader model_base/sdxl.py
        """
        if self._nunchaku_unet is None:
            raise RuntimeError("Model not loaded. Call load_state_dict first.")

        # Apply LoRAs if any
        self._apply_loras()

        unet = self._nunchaku_unet
        device = x.device
        input_dtype = x.dtype
        
        # Nunchaku SDXL UNet uses bfloat16 internally
        # Convert inputs to match UNet dtype and device to avoid dtype/device mismatch errors
        unet_dtype = self.dtype  # bfloat16 by default
        x = x.to(device=device, dtype=unet_dtype)
        context = context.to(device=device, dtype=unet_dtype) if context is not None else None
        timesteps = timesteps.to(device=device, dtype=unet_dtype) if timesteps.dtype.is_floating_point else timesteps.to(device=device)

        # Build added_cond_kwargs for SDXL
        # Forge Neo format for y (vector):
        #   - First part: clip_pooled (can be 1280 for CLIP-G only, or 2048 for CLIP-L+G)
        #   - Second part: flat time embedding (6 * 256 = 1536 dims from Timestep embedder)
        # Diffusers SDXL format:
        #   - text_embeds: 1280 dims (CLIP-G pooled output only)
        #   - time_ids: 6 dims (raw values, internally embedded by UNet)
        added_cond_kwargs = {}
        if y is not None:
            bs = x.shape[0]
            y_dim = y.shape[-1]

            # Determine clip_pooled size from y dimension
            # y = clip_pooled + flat(1536)
            # If y_dim = 2816: clip_pooled = 1280 (CLIP-G only) - standard SDXL
            # If y_dim = 3584: clip_pooled = 2048 (CLIP-L 768 + CLIP-G 1280)
            flat_dim = 1536  # 6 * 256 from Timestep(256) embedder
            clip_pooled_dim = y_dim - flat_dim

            if clip_pooled_dim == 1280:
                # Standard: CLIP-G pooled only
                text_embeds = y[:, :1280]
            elif clip_pooled_dim == 2048:
                # Combined: CLIP-L (768) + CLIP-G (1280)
                # For diffusers SDXL, we only need CLIP-G (last 1280 dims of pooled)
                text_embeds = y[:, 768:2048]  # Skip CLIP-L, take CLIP-G
            else:
                # Fallback: take first 1280 dims
                text_embeds = y[:, :1280]

            added_cond_kwargs["text_embeds"] = text_embeds.to(device=device, dtype=unet_dtype)

            # Generate time_ids (diffusers expects raw 6-dim values)
            # Default: 1024x1024 image, no crop, target 1024x1024
            time_ids = torch.tensor(
                [1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0],
                device=device,
                dtype=unet_dtype,
            ).unsqueeze(0).repeat(bs, 1)
            added_cond_kwargs["time_ids"] = time_ids

        # ControlNet support
        # Forge Neo format (from control_merge in controlnet.py):
        #   - control["output"]: down block residuals (all but last from ControlNet output)
        #   - control["middle"]: mid block residual (last from ControlNet output)
        # Diffusers format:
        #   - down_block_additional_residuals: tuple of tensors for down blocks
        #   - mid_block_additional_residual: single tensor for mid block
        down_block_additional_residuals = None
        mid_block_additional_residual = None

        # Debug (opt-in): Log ControlNet structure once (even if control is None)
        try:
            _dbg_cn = str(os.getenv("NUNCHAKU_SDXL_DEBUG", "0")).strip().lower() not in ("0", "false", "off", "no")
        except Exception:
            _dbg_cn = False
        if _dbg_cn and not getattr(self, "_controlnet_debug_logged", False):
            self._controlnet_debug_logged = True
            print("\n" + "=" * 80)
            print("[NUNCHAKU_SDXL_DEBUG] SVDQ SDXL ControlNet: first forward call")
            print(f"  control is None: {control is None}")
            print(f"  control type: {type(control)}")
            if control is not None:
                if isinstance(control, dict):
                    print(f"  Keys: {list(control.keys())}")
                    for k, v in control.items():
                        if v is None:
                            print(f"    {k}: None")
                        elif isinstance(v, (list, tuple)):
                            print(f"    {k}: list/tuple with {len(v)} items")
                            for i, item in enumerate(v[:3]):  # Show first 3
                                if item is None:
                                    print(f"      [{i}]: None")
                                elif hasattr(item, "shape"):
                                    print(f"      [{i}]: Tensor {tuple(item.shape)} {item.dtype}")
                                else:
                                    print(f"      [{i}]: {type(item)}")
                            if len(v) > 3:
                                print(f"      ... and {len(v) - 3} more")
                        elif hasattr(v, "shape"):
                            print(f"    {k}: Tensor {tuple(v.shape)} {v.dtype}")
                        else:
                            print(f"    {k}: {type(v)}")
                else:
                    print(f"  control value: {control}")
            print("=" * 80 + "\n")

        if control is not None and isinstance(control, dict):
            # ControlNet signal conversion (fixed based on ComfyUI-nunchaku-unofficial-loader v2.6)
            # Forge Neo format (from control_merge in controlnet.py):
            #   - control["output"]: down block residuals (all but last from ControlNet output)
            #   - control["middle"]: mid block residual (last from ControlNet output)
            #   - control["input"]: sometimes populated by adapters / edge cases
            # Diffusers format:
            #   - down_block_additional_residuals: tuple of tensors for down blocks
            #   - mid_block_additional_residual: single tensor for mid block
            
            # Extract control signals from dict (ComfyUI format)
            inp = control.get("output", None)
            if inp is None:
                # Fallback to "input" if "output" is not available
                inp = control.get("input", None)
            mid = control.get("middle", None)
            
            # Convert down block residuals
            if isinstance(inp, (list, tuple)):
                # Filter out None values (ComfyUI approach)
                down_list = [v for v in inp if v is not None]
                if len(down_list) > 0:
                    # Convert device and dtype to match UNet requirements
                    down_block_additional_residuals = tuple(
                        v.to(device=device, dtype=unet_dtype) for v in down_list
                    )
                    if _dbg_cn and not getattr(self, "_controlnet_down_logged", False):
                        self._controlnet_down_logged = True
                        print(f"[NUNCHAKU_SDXL_DEBUG] SVDQ SDXL ControlNet: down blocks = {len(down_block_additional_residuals)}")
                        for i, t in enumerate(down_block_additional_residuals[:3]):
                                print(f"  [{i}]: {tuple(t.shape)}")

            # Convert mid block residual
            if isinstance(mid, (list, tuple)):
                for v in mid:
                    if v is not None:
                        # Convert device and dtype to match UNet requirements
                        mid_block_additional_residual = v.to(device=device, dtype=unet_dtype)
                    if _dbg_cn and not getattr(self, "_controlnet_mid_logged", False):
                        self._controlnet_mid_logged = True
                        print(f"[NUNCHAKU_SDXL_DEBUG] SVDQ SDXL ControlNet: mid block = {tuple(mid_block_additional_residual.shape)}")
                    break

        # Always-on (Nunchaku-only) one-shot: show residual magnitudes actually passed into diffusers UNet.
        # This helps distinguish "CN computed zero tensors" vs "CN tensors exist but effect is lost elsewhere".
        if not getattr(self, "_nunchaku_cn_pass_stats_once", False):
            try:
                if control is not None and isinstance(control, dict):
                    self._nunchaku_cn_pass_stats_once = True
                    def _mean_abs(t):
                        try:
                            return float(t.detach().abs().mean().item())
                        except Exception:
                            return None
                    down_stats = None
                    if isinstance(down_block_additional_residuals, tuple):
                        non_none = [t for t in down_block_additional_residuals if t is not None][:2]
                        down_stats = [_mean_abs(t) for t in non_none]
                    mid_stat = _mean_abs(mid_block_additional_residual) if mid_block_additional_residual is not None else None
                    print("\n" + "=" * 80)
                    print("[NUNCHAKU_SDXL] ControlNet residuals passed to diffusers UNet (one-shot)")
                    print(f"  down_block_additional_residuals: {0 if down_block_additional_residuals is None else len(down_block_additional_residuals)}")
                    print(f"  down mean_abs(first2 non-None): {down_stats}")
                    print(f"  mid_block_additional_residual is None: {mid_block_additional_residual is None}")
                    print(f"  mid mean_abs: {mid_stat}")
                    print("=" * 80 + "\n")
            except Exception:
                pass

        # Log ControlNet status once per generation
        if _dbg_cn and not getattr(self, "_controlnet_status_logged", False):
            self._controlnet_status_logged = True
            if down_block_additional_residuals is not None or mid_block_additional_residual is not None:
                print(f"[NUNCHAKU_SDXL_DEBUG] SVDQ SDXL ControlNet: passing to UNet")
                print(f"  - down_block_additional_residuals: {len(down_block_additional_residuals) if down_block_additional_residuals else 0} tensors")
                print(f"  - mid_block_additional_residual: {'Yes' if mid_block_additional_residual is not None else 'No'}")
            elif control is not None:
                print(f"[NUNCHAKU_SDXL_DEBUG] SVDQ SDXL ControlNet: control dict exists but no valid tensors extracted")
            # Reset for next generation
            self._controlnet_debug_logged = False
            self._controlnet_down_logged = False
            self._controlnet_mid_logged = False

        # Call diffusers UNet
        output = unet(
            sample=x,
            timestep=timesteps,
            encoder_hidden_states=context,
            added_cond_kwargs=added_cond_kwargs if added_cond_kwargs else None,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            return_dict=False,
        )

        # Extract output and convert back to input dtype
        if isinstance(output, tuple):
            result = output[0]
        else:
            result = output

        # Convert back to original input dtype (float16) and device for compatibility with rest of pipeline
        return result.to(device=device, dtype=input_dtype)

    def load_state_dict(self, sd, *args, **kwargs):
        """Build and load the model from state dict."""
        metadata = kwargs.pop("metadata", {})
        self._build_and_load(sd, metadata)
        return [], []  # Return empty missing/unexpected lists
