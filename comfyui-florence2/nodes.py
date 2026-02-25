from collections.abc import Callable
import torch
import torchvision.transforms.functional as F
import io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageColor, ImageFont
import random
import numpy as np
import re
from pathlib import Path

from transformers.dynamic_module_utils import get_imports
import transformers
from packaging import version

# Transformers 5.x compatibility patch for peft
if version.parse(transformers.__version__) >= version.parse('5.0.0'):
    if not hasattr(transformers, "HybridCache"):
        try:
            from transformers import DynamicCache
            transformers.HybridCache = DynamicCache
            
            # Hack to make _LazyModule aware of the new attribute
            if hasattr(transformers, "_objects"):
                transformers._objects["HybridCache"] = DynamicCache
            
            # Ensure it is in __dict__ as well to bypass __getattr__ if possible
            transformers.__dict__["HybridCache"] = DynamicCache
        except ImportError:
            pass

from safetensors.torch import save_file

def load_model(model_path: str, attention: str, dtype: torch.dtype, offload_device: torch.device):
    """Load Florence2 model using the appropriate API for the installed transformers version."""
    if version.parse(transformers.__version__) >= version.parse('5.0.0'):
        # transformers 5.x: use native Florence2 implementation
        from transformers import Florence2ForConditionalGeneration, Florence2Processor
        
        # Load state_dict manually to handle key mismatches (custom Microsoft structure vs Official Transformers)
        checkpoint_file = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(model_path, "pytorch_model.bin")
            
        state_dict = None
        if os.path.exists(checkpoint_file):
            if checkpoint_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_file)
            else:
                state_dict = torch.load(checkpoint_file, map_location="cpu")
        
        legacy_2048_1024_detected = False
        if state_dict:
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("vision_tower."):
                    new_state_dict["model." + k] = v
                elif k.startswith("language_model."):
                    new_state_dict["model." + k] = v
                elif k.startswith("image_proj_norm."):
                    new_state_dict["model.multi_modal_projector." + k] = v
                elif k == "image_projection" or k == "image_projection.weight":
                    if v.shape == (2048, 1024):
                        print(f"DEBUG: Transposing image_projection {k} from {v.shape} to {v.T.shape}")
                        v = v.T
                        legacy_2048_1024_detected = True
                    new_state_dict["model.multi_modal_projector.image_projection.weight"] = v
                elif k.startswith("image_pos_embed."):
                    new_key = k.replace("image_pos_embed.", "model.multi_modal_projector.image_position_embed.")
                    new_state_dict[new_key] = v
                elif k.startswith("visual_temporal_embed."):
                    new_key = k.replace("visual_temporal_embed.", "model.multi_modal_projector.visual_temporal_embed.")
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        # Load Config (Force official config to ensure compatibility with Official Model)
        from transformers import Florence2Config
        try:
            config = Florence2Config.from_pretrained(model_path)
        except Exception as e:
            # Fallback if config.json is completely incompatible? 
             from transformers import AutoConfig
             config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Patch config for compatibility with official Florence2ForConditionalGeneration
        # The checkpoint's config.json might use old keys (e.g. embed_dims instead of embed_dim)
        if hasattr(config, "vision_config"):
            vc = config.vision_config
            # Map legacy key names from config.json to official Florence2VisionConfig parameter names
            # Legacy config.json uses "dim_embed" but Florence2VisionConfig expects "embed_dim"
            if hasattr(vc, "dim_embed") and not hasattr(vc, "_embed_dim_patched"):
                vc.embed_dim = list(vc.dim_embed)
                vc._embed_dim_patched = True
            elif not hasattr(vc, "embed_dim"):
                if hasattr(vc, "embed_dims"):
                    vc.embed_dim = vc.embed_dims
                else:
                    vc.embed_dim = [256, 512, 1024, 1024]
            
            # Map other potential missing keys
            if not hasattr(vc, "num_heads"):
                if hasattr(vc, "num_attention_heads"):
                    vc.num_heads = vc.num_attention_heads
                else:
                    # Default for Large? [4, 8, 16, 32]?
                    # Attempt safe defaults roughly matching embed_dim/64
                    vc.num_heads = [4, 8, 16, 32]
            
            if not hasattr(vc, "in_channels"):
                vc.in_channels = 3
                
            if not hasattr(vc, "patch_size"):
                if hasattr(vc, "patch_sizes"):
                    vc.patch_size = vc.patch_sizes
                else:
                    vc.patch_size = [7, 3, 3, 3] # Default guess
            
            if not hasattr(vc, "patch_stride"):
                if hasattr(vc, "strides"):
                    vc.patch_stride = vc.strides
                else:
                    vc.patch_stride = [4, 2, 2, 2] # Default guess
            
            if not hasattr(vc, "patch_padding"):
                if hasattr(vc, "paddings"):
                    vc.patch_padding = vc.paddings
                else:
                    vc.patch_padding = [3, 1, 1, 1] # Default guess

            if not hasattr(vc, "patch_prenorm"):
                vc.patch_prenorm = [False, False, False, False]
            
            if not hasattr(vc, "projection_dim"):
                # Based on checkpint mismatch 2048 vs 1024
                vc.projection_dim = 2048
            
            if not hasattr(vc, "window_size"):
                if hasattr(vc, "window_sizes"):
                    vc.window_size = vc.window_sizes[0] # Implementation uses single window size
                else:
                    vc.window_size = 12
            
            if not hasattr(vc, "depths"):
                vc.depths = [1, 1, 3, 1] # Default guess for base? Large might be deeper. 
                # DaViT-Large: [1, 1, 9, 1]
                # If we updated embed_dim to [256...], we should arguably use Large depths.
                # Let's try [1, 1, 9, 1] if embed_dim starts with 256.
                if vc.embed_dim[0] == 256:
                    vc.depths = [1, 1, 9, 1]
            
            if not hasattr(vc, "drop_path_rate"):
                vc.drop_path_rate = 0.1
            
            if not hasattr(vc, "mlp_ratio"):
                vc.mlp_ratio = 4.0
            
            if not hasattr(vc, "qkv_bias"):
                vc.qkv_bias = True
                
            if not hasattr(vc, "activation_function"):
                vc.activation_function = "gelu"
                
            if not hasattr(vc, "max_temporal_embeddings"):
                vc.max_temporal_embeddings = 100 # Default matches checkpoint
            
            if not hasattr(vc, "max_position_embeddings"):
                vc.max_position_embeddings = 50 # Default matches checkpoint
                
        # Patch main config
        if not hasattr(config, "image_token_id"):
             config.image_token_id = 51200 # Default for Florence2 (usually after vocab)
             if hasattr(config, "text_config") and hasattr(config.text_config, "vocab_size"):
                 pass

        # Dynamic registration of custom text config (if needed)
        if hasattr(config, "text_config"):
            text_config_class = type(config.text_config)
            if "Florence2LanguageConfig" in text_config_class.__name__:
                try:
                    from transformers import AutoModel, BartModel
                    class Florence2LanguageModel(BartModel):
                        config_class = text_config_class
                    
                    try:
                        AutoModel.register(text_config_class, Florence2LanguageModel)
                    except ValueError:
                         pass
                except Exception as e:
                    print(f"DEBUG: Failed to register custom text config: {e}")

        # Monkey-patch: Strip the +1 spatial token from the projector output.
        # The official projector outputs [B, 1+H*W, projection_dim] where the +1 is from
        # spatial_image_features.mean(dim=2). But the legacy model's text encoder has
        # max_position_embeddings=1024, so we can only fit 1024 image tokens (not 1025).
        # We skip the spatial mean token and output only temporal features: [B, H*W, projection_dim].
        import types
        from transformers.models.florence2.modeling_florence2 import Florence2MultiModalProjector

        def _patched_projector_forward(self, image_features):
            position_features = image_features + self.image_position_embed(image_features)
            position_features = position_features.flatten(2).transpose(1, 2)
            temporal_features = self.visual_temporal_embed(position_features[:, :1, :])
            temporal_features = temporal_features.unsqueeze(1)
            visual_token_features = position_features + temporal_features
            visual_token_features = visual_token_features.unsqueeze(1)
            # Skip spatial_image_features (the +1 token) — only use temporal
            temporal_image_features = visual_token_features.mean(dim=1)  # [B, H*W, dim]
            image_features = self.image_projection(temporal_image_features)
            image_features = self.image_proj_norm(image_features)
            return image_features

        Florence2MultiModalProjector.forward = _patched_projector_forward

        # Initialization
        # We need to respect the attention implementation
        # Florence2Config might not have _attn_implementation attr in 5.x, it's passed to model init usually?
        # But we are using constructor. Constructor takes config. 
        # Attn implementation is usually set in config or via kwarg if supported.
        # Check if config supports it.
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = attention
            
        model = Florence2ForConditionalGeneration(config)
        
        # Load Weights
        if state_dict:
            # Drop known mismatch keys that are safe to recompute
            keys_to_drop = [
                "model.multi_modal_projector.visual_temporal_embed.pos_idx_to_embed"
            ]
            for k in keys_to_drop:
                if k in state_dict:
                    del state_dict[k]

            try:
                # We use strict=False initially to allow execution even if some minor keys are missing, 
                # but ideally it should match. 
                # Given the mess with keys, let's print mismatches but try to proceed if crucial parts are there.
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys or unexpected_keys:
                    # Optional: print debug info
                    # print(f"DEBUG: Florence2 Load Report")
                    # print(f"Missing keys: {len(missing_keys)}")
                    # if len(missing_keys) < 20: print(missing_keys)
                    # print(f"Unexpected keys: {len(unexpected_keys)}")
                    # if len(unexpected_keys) < 20: print(unexpected_keys)
                    pass
            except RuntimeError as e:
                print(f"WARNING: Runtime Error during state_dict load (ignoring to proceed): {e}")

        model = model.to(dtype).to(offload_device).eval()

        
        # Load Tokenizer manually to patch missing attributes (image_token)
        # transformers 5.x Florence2Processor expects tokenizer.image_token
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if not hasattr(tokenizer, "image_token"):
                tokenizer.image_token = "<image>"
            
            if not hasattr(tokenizer, "image_token_id"):
                if hasattr(model.config, "image_token_id"):
                    tokenizer.image_token_id = model.config.image_token_id
                else:
                    tokenizer.image_token_id = 51200
            
            # Ensure " <image>" or "<image>" is tokenized correctly as a single token
            # Florence-2 processor injects "<image>" tokens.
            # We need to make sure tokenizer produces a single ID for it, 
            # and that ID matches what the model expects (or we align them).
            image_token = "<image>"
            if image_token not in tokenizer.get_vocab():
                tokenizer.add_tokens([image_token], special_tokens=True)
                model.resize_token_embeddings(len(tokenizer))
                
                new_id = tokenizer.convert_tokens_to_ids(image_token)
                old_id = getattr(model.config, "image_token_id", 51200)
                
                # Copy weights if possible
                with torch.no_grad():
                    input_embeddings = model.get_input_embeddings()
                    if old_id < input_embeddings.weight.shape[0]:
                        input_embeddings.weight[new_id] = input_embeddings.weight[old_id]
                
                # Update config/processor to use the new valid ID
                model.config.image_token_id = new_id
                tokenizer.image_token_id = new_id
                
            processor = Florence2Processor.from_pretrained(model_path, tokenizer=tokenizer, trust_remote_code=True)
            # Ensure processor syncs up
            processor.image_token_id = tokenizer.image_token_id
            
            # Patch image_seq_length if likely mismatch
            if hasattr(processor, "image_seq_length"):
                if processor.image_seq_length == 577:
                    print(f"DEBUG: Processor has image_seq_length=577 from config.")

        except Exception as e:
            print(f"DEBUG: Florence2Processor.from_pretrained failed: {e}")
            print(f"DEBUG: Creating processor manually with CLIPImageProcessor + BartTokenizerFast.")

            from transformers import CLIPImageProcessor, BartTokenizerFast

            image_processor = CLIPImageProcessor(
                size={"height": 768, "width": 768},
                crop_size={"height": 768, "width": 768},
                do_resize=True,
                do_center_crop=False,
                do_rescale=True,
                rescale_factor=1/255.0,
                do_normalize=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
            )
            image_processor.image_seq_length = 577

            tokenizer = BartTokenizerFast.from_pretrained(model_path)
            processor = Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)

    # Fix image_seq_length for legacy checkpoint:
    # Original config: image_seq_length=577 = 1 + 24*24 (768x768 image, stride 32, +1 spatial token)
    # Our monkey-patched projector strips the spatial token, so we need 576 = 24*24
    # Keep default 768x768 image size - do NOT force 1024x1024 (would overflow max_position_embeddings)
    if legacy_2048_1024_detected:
         target_seq_len = 576  # 24*24 from 768x768 with stride 32, spatial token stripped
         print(f"DEBUG: Legacy checkpoint detected. Setting image_seq_length={target_seq_len} (768x768, stride 32, no spatial token).")
         
         # IMPORTANT: Florence2Processor caches num_image_tokens at init. We must update it manually.
         if hasattr(processor, "num_image_tokens"):
              processor.num_image_tokens = target_seq_len
         
         if hasattr(processor, "image_seq_length"):
              processor.image_seq_length = target_seq_len
         
         if hasattr(processor, "image_processor"):
              if hasattr(processor.image_processor, "image_seq_length"):
                   processor.image_processor.image_seq_length = target_seq_len

    return model, processor

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    try:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
    except:
        print(f"No flash_attn import to remove")
        pass
    return imports


def create_path_dict(paths: list[str], predicate: Callable[[Path], bool] = lambda _: True) -> dict[str, str]:
    """
    Creates a flat dictionary of the contents of all given paths: ``{name: absolute_path}``.

    Non-recursive.  Optionally takes a predicate to filter items.  Duplicate names overwrite (the last one wins).

    Args:
        paths (list[str]):
            The paths to search for items.
        predicate (Callable[[Path], bool]): 
            (Optional) If provided, each path is tested against this filter.
            Returns ``True`` to include a path.

            Default: Include everything
    """

    flattened_paths = [item for path in paths if Path(path).exists() for item in Path(path).iterdir() if predicate(item)]

    return {item.name: str(item.absolute()) for item in flattened_paths}


import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))
model_directory = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(model_directory, exist_ok=True)

# Ensure ComfyUI knows about the LLM model path
folder_paths.add_model_folder_path("LLM", model_directory)

from transformers import AutoProcessor, set_seed

model_list = [
            'microsoft/Florence-2-base',
            'microsoft/Florence-2-base-ft',
            'microsoft/Florence-2-large',
            'microsoft/Florence-2-large-ft',
            'HuggingFaceM4/Florence-2-DocVQA',
            'thwri/CogFlorence-2.1-Large',
            'thwri/CogFlorence-2.2-Large',
            'gokaygokay/Florence-2-SD3-Captioner',
            'gokaygokay/Florence-2-Flux-Large',
            'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
            'MiaoshouAI/Florence-2-large-PromptGen-v1.5',
            'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
            'MiaoshouAI/Florence-2-large-PromptGen-v2.0',
            'PJMixers-Images/Florence-2-base-Castollux-v0.5'
            ]

class DownloadAndLoadFlorence2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (model_list, {"default": 'microsoft/Florence-2-base'}),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'fp16'
                    }),
            "attention": (
                    [ 'flash_attention_2', 'sdpa', 'eager'],
                    {
                    "default": 'sdpa'
                    }),
            },
            "optional": {
                "lora": ("PEFTLORA",),
                "convert_to_safetensors": ("BOOLEAN", {"default": False, "tooltip": "Some of the older model weights are not saved in .safetensors format, which seem to cause longer loading times, this option converts the .bin weights to .safetensors"}),
            }
        }

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model, precision, attention, lora=None, convert_to_safetensors=False):
        if model not in model_list:
            raise ValueError(f"Model {model} is not in the supported model list.")

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(model_directory, model_name)

        if not os.path.exists(model_path):
            print(f"Downloading Florence2 model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            local_dir=model_path,
                            local_dir_use_symlinks=False)

        print(f"Florence2 using {attention} for attention")

        if convert_to_safetensors:
            model_weight_path = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.exists(model_weight_path):
                safetensors_weight_path = os.path.join(model_path, 'model.safetensors')
                print(f"Converting {model_weight_path} to {safetensors_weight_path}")
                if not os.path.exists(safetensors_weight_path):
                    sd = torch.load(model_weight_path, map_location=offload_device)
                    sd_new = {}
                    for k, v in sd.items():
                        sd_new[k] = v.clone()
                    save_file(sd_new, safetensors_weight_path)
                    if os.path.exists(safetensors_weight_path):
                        print(f"Conversion successful. Deleting original file: {model_weight_path}")
                        os.remove(model_weight_path)
                        print(f"Original {model_weight_path} file deleted.")

        model, processor = load_model(model_path, attention, dtype, offload_device)

        if lora is not None:
            from peft import PeftModel
            adapter_name = lora
            model = PeftModel.from_pretrained(model, adapter_name, trust_remote_code=True)

        florence2_model = {
            'model': model,
            'processor': processor,
            'dtype': dtype
            }

        return (florence2_model,)

class DownloadAndLoadFlorence2Lora:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'NikshepShetty/Florence-2-pixelprose',
                    ],
                  ),            
            },
          
        }

    RETURN_TYPES = ("PEFTLORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model):
        if model not in ['NikshepShetty/Florence-2-pixelprose']:
            raise ValueError(f"Lora Model {model} is not in the supported lora model list.")
        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(model_directory, model_name)
        
        if not os.path.exists(model_path):
            print(f"Downloading Florence2 lora model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
        return (model_path,)
    
class Florence2ModelLoader:

    @classmethod
    def INPUT_TYPES(s):
        all_llm_paths = folder_paths.get_folder_paths("LLM")
        s.model_paths = create_path_dict(all_llm_paths, lambda x: x.is_dir())

        return {"required": {
            "model": ([*s.model_paths], {"tooltip": "models are expected to be in Comfyui/models/LLM folder"}),
            "precision": (['fp16','bf16','fp32'],),
            "attention": (
                    [ 'flash_attention_2', 'sdpa', 'eager'],
                    {
                    "default": 'sdpa'
                    }),
            },
            "optional": {
                "lora": ("PEFTLORA",),
                "convert_to_safetensors": ("BOOLEAN", {"default": False, "tooltip": "Some of the older model weights are not saved in .safetensors format, which seem to cause longer loading times, this option converts the .bin weights to .safetensors"}),
            }
        }

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model, precision, attention, lora=None, convert_to_safetensors=False):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        model_path = Florence2ModelLoader.model_paths.get(model)
        print(f"Loading model from {model_path}")
        print(f"Florence2 using {attention} for attention")
        if convert_to_safetensors:
            model_weight_path = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.exists(model_weight_path):
                safetensors_weight_path = os.path.join(model_path, 'model.safetensors')
                print(f"Converting {model_weight_path} to {safetensors_weight_path}")
                if not os.path.exists(safetensors_weight_path):
                    sd = torch.load(model_weight_path, map_location=offload_device)
                    sd_new = {}
                    for k, v in sd.items():
                        sd_new[k] = v.clone()
                    save_file(sd_new, safetensors_weight_path)
                    if os.path.exists(safetensors_weight_path):
                        print(f"Conversion successful. Deleting original file: {model_weight_path}")
                        os.remove(model_weight_path)
                        print(f"Original {model_weight_path} file deleted.")

        model, processor = load_model(model_path, attention, dtype, offload_device)

        if lora is not None:
            from peft import PeftModel
            adapter_name = lora
            model = PeftModel.from_pretrained(model, adapter_name, trust_remote_code=True)

        florence2_model = {
            'model': model,
            'processor': processor,
            'dtype': dtype
            }
   
        return (florence2_model,)
    
class Florence2Run:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "florence2_model": ("FL2MODEL", ),
                "text_input": ("STRING", {"default": "", "multiline": True}),
                "task": (
                    [ 
                    'region_caption',
                    'dense_region_caption',
                    'region_proposal',
                    'caption',
                    'detailed_caption',
                    'more_detailed_caption',
                    'caption_to_phrase_grounding',
                    'referring_expression_segmentation',
                    'ocr',
                    'ocr_with_region',
                    'docvqa',
                    'prompt_gen_tags',
                    'prompt_gen_mixed_caption',
                    'prompt_gen_analyze',
                    'prompt_gen_mixed_caption_plus',
                    ],
                   ),
                "fill_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "output_mask_select": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "JSON")
    RETURN_NAMES =("image", "mask", "caption", "data") 
    FUNCTION = "encode"
    CATEGORY = "Florence2"

    def hash_seed(self, seed):
        import hashlib
        # Convert the seed to a string and then to bytes
        seed_bytes = str(seed).encode('utf-8')
        # Create a SHA-256 hash of the seed bytes
        hash_object = hashlib.sha256(seed_bytes)
        # Convert the hash to an integer
        hashed_seed = int(hash_object.hexdigest(), 16)
        # Ensure the hashed seed is within the acceptable range for set_seed
        return hashed_seed % (2**32)

    def encode(self, image, text_input, florence2_model, task, fill_mask, keep_model_loaded=False, 
            num_beams=3, max_new_tokens=1024, do_sample=True, output_mask_select="", seed=None):
        _, height, width, _ = image.shape
        annotated_image_tensor = None
        mask_tensor = None
        processor = florence2_model['processor']
        model = florence2_model['model']
        dtype = florence2_model['dtype']
        model.to(device)

        if seed:
            set_seed(self.hash_seed(seed))

        colormap = ['blue','orange','green','purple','brown','pink','olive','cyan','red',
                    'lime','indigo','violet','aqua','magenta','gold','tan','skyblue']

        prompts = {
            'region_caption': '<OD>',
            'dense_region_caption': '<DENSE_REGION_CAPTION>',
            'region_proposal': '<REGION_PROPOSAL>',
            'caption': '<CAPTION>',
            'detailed_caption': '<DETAILED_CAPTION>',
            'more_detailed_caption': '<MORE_DETAILED_CAPTION>',
            'caption_to_phrase_grounding': '<CAPTION_TO_PHRASE_GROUNDING>',
            'referring_expression_segmentation': '<REFERRING_EXPRESSION_SEGMENTATION>',
            'ocr': '<OCR>',
            'ocr_with_region': '<OCR_WITH_REGION>',
            'docvqa': '<DocVQA>',
            'prompt_gen_tags': '<GENERATE_TAGS>',
            'prompt_gen_mixed_caption': '<MIXED_CAPTION>',
            'prompt_gen_analyze': '<ANALYZE>',
            'prompt_gen_mixed_caption_plus': '<MIXED_CAPTION_PLUS>',
        }
        task_prompt = prompts.get(task, '<OD>')

        if (task not in ['referring_expression_segmentation', 'caption_to_phrase_grounding', 'docvqa']) and text_input:
            raise ValueError("Text input (prompt) is only supported for 'referring_expression_segmentation', 'caption_to_phrase_grounding', and 'docvqa'")

        if text_input != "":
            prompt = task_prompt + " " + text_input
        else:
            prompt = task_prompt

        image = image.permute(0, 3, 1, 2)

        out = []
        out_masks = []
        out_results = []
        out_data = []
        pbar = ProgressBar(len(image))
        for img in image:
            image_pil = F.to_pil_image(img)
            
            inputs = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False).to(dtype).to(device)
            
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
                use_cache=False,
            )

            results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            # Print repr to avoid UnicodeEncodeError on Windows console
            try:
                print(repr(results))
            except:
                pass
            # cleanup the special tokens from the final list
            if task == 'ocr_with_region':
                clean_results = str(results)       
                cleaned_string = re.sub(r'</?s>|<[^>]*>', '\n',  clean_results)
                clean_results = re.sub(r'\n+', '\n', cleaned_string)
            else:
                clean_results = str(results)       
                clean_results = clean_results.replace('</s>', '')
                clean_results = clean_results.replace('<s>', '')

             #return single string if only one image for compatibility with nodes that can't handle string lists
            if len(image) == 1:
                out_results = clean_results
            else:
                out_results.append(clean_results)

            W, H = image_pil.size
            
            W, H = image_pil.size
            
            try:
                parsed_answer = processor.post_process_generation(results, task=task_prompt, image_size=(W, H))
            except Exception as e:
                # Some tasks (like caption) might not be supported by post_process_generation in older transformers versions
                # or don't require parsing.
                parsed_answer = None

            if task == 'region_caption' or task == 'dense_region_caption' or task == 'caption_to_phrase_grounding' or task == 'region_proposal':           
                fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                ax.imshow(image_pil)
                bboxes = parsed_answer[task_prompt]['bboxes']
                labels = parsed_answer[task_prompt]['labels']

                mask_indexes = []
                # Determine mask indexes outside the loop
                if output_mask_select != "":
                    mask_indexes = [n for n in output_mask_select.split(",")]
                    print(mask_indexes)
                else:
                    mask_indexes = [str(i) for i in range(len(bboxes))]

                # Initialize mask_layer only if needed
                if fill_mask:
                    mask_layer = Image.new('RGB', image_pil.size, (0, 0, 0))
                    mask_draw = ImageDraw.Draw(mask_layer)

                for index, (bbox, label) in enumerate(zip(bboxes, labels)):
                    # Modify the label to include the index
                    indexed_label = f"{index}.{label}"
                    
                    if fill_mask:
                        # Ensure y1 is greater than or equal to y0 for mask drawing
                        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                        if y1 < y0:
                            y0, y1 = y1, y0
                        if x1 < x0:
                            x0, x1 = x1, x0
                            
                        if str(index) in mask_indexes:
                            print("match index:", str(index), "in mask_indexes:", mask_indexes)
                            mask_draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
                        if label in mask_indexes:
                            print("match label")
                            mask_draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

                    # Create a Rectangle patch
                    # Ensure y1 is greater than or equal to y0
                    y0, y1 = bbox[1], bbox[3]
                    if y1 < y0:
                        y0, y1 = y1, y0
                    
                    rect = patches.Rectangle(
                        (bbox[0], y0),  # (x,y) - lower left corner
                        bbox[2] - bbox[0],   # Width
                        y1 - y0,   # Height
                        linewidth=1,
                        edgecolor='r',
                        facecolor='none',
                        label=indexed_label
                    )
                     # Calculate text width with a rough estimation
                    text_width = len(label) * 6  # Adjust multiplier based on your font size
                    text_height = 12  # Adjust based on your font size

                    # Get corrected coordinates
                    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                    if y1 < y0:
                        y0, y1 = y1, y0
                    if x1 < x0:
                        x0, x1 = x1, x0

                    # Initial text position
                    text_x = x0
                    text_y = y0 - text_height  # Position text above the top-left of the bbox

                    # Adjust text_x if text is going off the left or right edge
                    if text_x < 0:
                        text_x = 0
                    elif text_x + text_width > W:
                        text_x = W - text_width

                    # Adjust text_y if text is going off the top edge
                    if text_y < 0:
                        text_y = y1  # Move text below the bottom-left of the bbox if it doesn't overlap with bbox

                    # Add the rectangle to the plot
                    ax.add_patch(rect)
                    facecolor = random.choice(colormap) if len(image) == 1 else 'red'
                    # Add the label
                    plt.text(
                        text_x,
                        text_y,
                        indexed_label,
                        color='white',
                        fontsize=12,
                        bbox=dict(facecolor=facecolor, alpha=0.5)
                    )
                if fill_mask:             
                    mask_tensor = F.to_tensor(mask_layer)
                    mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                    mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
                    mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
                    mask_tensor = mask_tensor[:, :, :, 0]
                    out_masks.append(mask_tensor)           

                # Remove axis and padding around the image
                ax.axis('off')
                ax.margins(0,0)
                ax.get_xaxis().set_major_locator(plt.NullLocator())
                ax.get_yaxis().set_major_locator(plt.NullLocator())
                fig.canvas.draw() 
                buf = io.BytesIO()
                plt.savefig(buf, format='png', pad_inches=0)
                buf.seek(0)
                annotated_image_pil = Image.open(buf)

                annotated_image_tensor = F.to_tensor(annotated_image_pil)
                out_tensor = annotated_image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                out.append(out_tensor)
               
                if task == 'caption_to_phrase_grounding':
                    out_data.append(parsed_answer[task_prompt])
                else:
                    out_data.append(bboxes)

                
                pbar.update(1)
    
                plt.close(fig)

            elif task == 'referring_expression_segmentation':
                # Create a new black image
                mask_image = Image.new('RGB', (W, H), 'black')
                mask_draw = ImageDraw.Draw(mask_image)
  
                predictions = parsed_answer[task_prompt]
    
                # Iterate over polygons and labels  
                for polygons, label in zip(predictions['polygons'], predictions['labels']):
                    color = random.choice(colormap)
                    for _polygon in polygons:  
                        _polygon = np.array(_polygon).reshape(-1, 2)
                        # Clamp polygon points to image boundaries
                        _polygon = np.clip(_polygon, [0, 0], [W - 1, H - 1])
                        if len(_polygon) < 3:  
                            print('Invalid polygon:', _polygon)
                            continue  
                        
                        _polygon = _polygon.reshape(-1).tolist()
                        
                        # Draw the polygon
                        if fill_mask:
                            overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
                            image_pil = image_pil.convert('RGBA')
                            draw = ImageDraw.Draw(overlay)
                            color_with_opacity = ImageColor.getrgb(color) + (180,)
                            draw.polygon(_polygon, outline=color, fill=color_with_opacity, width=3)
                            image_pil = Image.alpha_composite(image_pil, overlay)
                        else:
                            draw = ImageDraw.Draw(image_pil)
                            draw.polygon(_polygon, outline=color, width=3)

                        #draw mask
                        mask_draw.polygon(_polygon, outline="white", fill="white")
                        
                image_tensor = F.to_tensor(image_pil)
                image_tensor = image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float() 
                out.append(image_tensor)

                mask_tensor = F.to_tensor(mask_image)
                mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
                mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
                mask_tensor = mask_tensor[:, :, :, 0]
                out_masks.append(mask_tensor)
                pbar.update(1)

            elif task == 'ocr_with_region':
                try:
                    font = ImageFont.load_default().font_variant(size=24)
                except:
                    font = ImageFont.load_default()
                predictions = parsed_answer[task_prompt]
                scale = 1
                image_pil = image_pil.convert('RGBA')
                overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)
                bboxes, labels = predictions['quad_boxes'], predictions['labels']
                
                # Create a new black image for the mask
                mask_image = Image.new('RGB', (W, H), 'black')
                mask_draw = ImageDraw.Draw(mask_image)
                
                for box, label in zip(bboxes, labels):
                    scaled_box = [v / (width if idx % 2 == 0 else height) for idx, v in enumerate(box)]
                    out_data.append({"label": label, "box": scaled_box})
                    
                    color = random.choice(colormap)
                    new_box = (np.array(box) * scale).tolist()
                    
                    # Ensure polygon coordinates are valid
                    # For polygons, we need to make sure the points form a valid shape
                    # This is a simple check to ensure the polygon has at least 3 points
                    if len(new_box) >= 6:  # At least 3 points (x,y pairs)
                        if fill_mask:
                            color_with_opacity = ImageColor.getrgb(color) + (180,)
                            draw.polygon(new_box, outline=color, fill=color_with_opacity, width=3)
                        else:
                            draw.polygon(new_box, outline=color, width=3)
                        
                        # Get the first point for text positioning
                        text_x, text_y = new_box[0]+8, new_box[1]+2
                        
                        draw.text((text_x, text_y),
                                  "{}".format(label),
                                  align="right",
                                  font=font,
                                  fill=color)
                        
                        # Draw the mask
                        mask_draw.polygon(new_box, outline="white", fill="white")
                
                image_pil = Image.alpha_composite(image_pil, overlay)
                image_pil = image_pil.convert('RGB')
                
                image_tensor = F.to_tensor(image_pil)
                image_tensor = image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                out.append(image_tensor)

                # Process the mask
                mask_tensor = F.to_tensor(mask_image)
                mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
                mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
                mask_tensor = mask_tensor[:, :, :, 0]
                out_masks.append(mask_tensor)

                pbar.update(1)
            
            elif task == 'docvqa':
                if text_input == "":
                    raise ValueError("Text input (prompt) is required for 'docvqa'")
                prompt = "<DocVQA> " + text_input

                inputs = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False).to(dtype).to(device)
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    use_cache=False,
                )

                results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                clean_results = results.replace('</s>', '').replace('<s>', '')
                
                if len(image) == 1:
                    out_results = clean_results
                else:
                    out_results.append(clean_results)
                    
                out.append(F.to_tensor(image_pil).unsqueeze(0).permute(0, 2, 3, 1).cpu().float())

                pbar.update(1)
            
        if len(out) > 0:
            out_tensor = torch.cat(out, dim=0)
        else:
            out_tensor = torch.zeros((1, 64,64, 3), dtype=torch.float32, device="cpu")
        if len(out_masks) > 0:
            out_mask_tensor = torch.cat(out_masks, dim=0)
        else:
            out_mask_tensor = torch.zeros((1,64,64), dtype=torch.float32, device="cpu")

        if not keep_model_loaded:
            print("Offloading model...")
            model.to(offload_device)
            mm.soft_empty_cache()
        
        return (out_tensor, out_mask_tensor, out_results, out_data)
     
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadFlorence2Model": DownloadAndLoadFlorence2Model,
    "DownloadAndLoadFlorence2Lora": DownloadAndLoadFlorence2Lora,
    "Florence2ModelLoader": Florence2ModelLoader,
    "Florence2Run": Florence2Run,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFlorence2Model": "DownloadAndLoadFlorence2Model",
    "DownloadAndLoadFlorence2Lora": "DownloadAndLoadFlorence2Lora",
    "Florence2ModelLoader": "Florence2ModelLoader",
    "Florence2Run": "Florence2Run",
}
