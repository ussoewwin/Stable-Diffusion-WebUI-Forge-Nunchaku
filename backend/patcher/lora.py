# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.77/comfy/lora.py

import weakref

import torch

from backend import memory_management, utils
from modules_forge.packages.comfy.lora import (  # noqa
    load_lora,
    model_lora_keys_clip,
    model_lora_keys_unet,
    weight_adapter,
)

extra_weight_calculators = {}


@torch.inference_mode()
def weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype, function):
    # https://github.com/comfyanonymous/ComfyUI/blob/v0.3.77/comfy/weight_adapter/base.py#L62

    dora_scale = memory_management.cast_to_device(dora_scale, weight.device, computation_dtype)
    lora_diff *= alpha
    weight_calc = weight + function(lora_diff).type(weight.dtype)

    wd_on_output_axis = dora_scale.shape[0] == weight_calc.shape[0]
    if wd_on_output_axis:
        weight_norm = weight.reshape(weight.shape[0], -1).norm(dim=1, keepdim=True).reshape(weight.shape[0], *[1] * (weight.dim() - 1))
    else:
        weight_norm = weight_calc.transpose(0, 1).reshape(weight_calc.shape[1], -1).norm(dim=1, keepdim=True).reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1)).transpose(0, 1)
    weight_norm = weight_norm + torch.finfo(weight.dtype).eps

    weight_calc *= (dora_scale / weight_norm).type(weight.dtype)
    if strength != 1.0:
        weight_calc -= weight
        weight += strength * (weight_calc)
    else:
        weight[:] = weight_calc
    return weight


@torch.inference_mode()
def merge_lora_to_weight(patches, weight, key="online_lora", computation_dtype=torch.float32):
    # https://github.com/comfyanonymous/ComfyUI/blob/v0.3.77/comfy/lora.py#L361

    weight_dtype_backup = None

    if computation_dtype == weight.dtype:
        weight = weight.clone()
    else:
        weight_dtype_backup = weight.dtype
        weight = weight.to(dtype=computation_dtype)

    for p in patches:
        strength = p[0]
        v = p[1]
        strength_model = p[2]
        offset = p[3]
        function = p[4]
        if function is None:
            function = lambda a: a

        old_weight = None
        if offset is not None:
            old_weight = weight
            weight = weight.narrow(offset[0], offset[1], offset[2])

        if strength_model != 1.0:
            weight *= strength_model

        if isinstance(v, list):
            v = (merge_lora_to_weight(v[1:], v[0][1](memory_management.cast_to_device(v[0][0], weight.device, computation_dtype, copy=True), inplace=True), key, computation_dtype=computation_dtype),)

        if isinstance(v, weight_adapter.WeightAdapterBase):
            output = v.calculate_weight(weight, key, strength, strength_model, offset, function, computation_dtype)
            if output is None:
                print("Calculate Weight Failed: {} {}".format(v.name, key))
            else:
                weight = output
                if old_weight is not None:
                    weight = old_weight
            continue

        if len(v) == 1:
            patch_type = "diff"
        elif len(v) == 2:
            patch_type = v[0]
            v = v[1]

        if patch_type == "diff":
            diff: torch.Tensor = v[0]
            # An extra flag to pad the weight if the diff's shape is larger than the weight
            do_pad_weight = len(v) > 1 and v[1]["pad_weight"]
            if do_pad_weight and diff.shape != weight.shape:
                print("Pad weight {} from {} to shape: {}".format(key, weight.shape, diff.shape))
                weight = weight_adapter.base.pad_tensor_to_shape(weight, diff.shape)

            if strength != 0.0:
                if diff.shape != weight.shape:
                    print("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, diff.shape, weight.shape))
                else:
                    weight += function(strength * memory_management.cast_to_device(diff, weight.device, weight.dtype))
        elif patch_type == "set":
            weight.copy_(v[0])
        elif patch_type == "model_as_lora":
            raise NotImplementedError('"patch_type" is not supported...')
        else:
            print("patch type not recognized {} {}".format(patch_type, key))

        if old_weight is not None:
            weight = old_weight

    if weight_dtype_backup is not None:
        weight = weight.to(dtype=weight_dtype_backup)

    return weight


def get_parameter_devices(model):
    parameter_devices = {}
    for key, p in model.named_parameters():
        parameter_devices[key] = p.device
    return parameter_devices


def set_parameter_devices(model, parameter_devices):
    for key, device in parameter_devices.items():
        p = utils.get_attr(model, key)
        if p.device != device:
            p = utils.tensor2parameter(p.to(device=device))
            utils.set_attr_raw(model, key, p)
    return model


from backend import operations


class LoraLoader:
    def __init__(self, model):
        self._model = weakref.ref(model)
        self.backup = {}
        self.online_backup = []
        self.loaded_hash = str([])

    @property
    def model(self):
        return self._model()

    @torch.inference_mode()
    def refresh(self, lora_patches, offload_device=torch.device("cpu"), force_refresh=False):
        hashes = str(list(lora_patches.keys()))

        if hashes == self.loaded_hash and not force_refresh:
            return

        # Merge Patches

        all_patches = {}

        for (_, _, _, online_mode), patches in lora_patches.items():
            for key, current_patches in patches.items():
                all_patches[(key, online_mode)] = all_patches.get((key, online_mode), []) + current_patches

        # Initialize

        memory_management.signal_empty_cache = True

        parameter_devices = get_parameter_devices(self.model)

        # Restore

        for m in set(self.online_backup):
            del m.forge_online_loras

        self.online_backup = []

        for k, w in self.backup.items():
            if not isinstance(w, torch.nn.Parameter):
                # In very few cases
                w = torch.nn.Parameter(w, requires_grad=False)

            utils.set_attr_raw(self.model, k, w)

        self.backup = {}

        set_parameter_devices(self.model, parameter_devices=parameter_devices)

        # Patch
        n_online = 0
        n_mp_quant = 0  # convert_weight → merge → set_weight (ComfyUI MixedPrecision)
        n_legacy = 0  # float / BnB / GGUF offline merge
        sample_mp = None
        import comfy.utils as comfy_utils

        for (key, online_mode), current_patches in all_patches.items():
            try:
                parent_layer, child_key, weight = utils.get_attr_with_parent(self.model, key)
                assert isinstance(weight, torch.nn.Parameter)
            except:
                raise ValueError(f"Wrong LoRA Key: {key}")

            if online_mode:
                if not hasattr(parent_layer, "forge_online_loras"):
                    parent_layer.forge_online_loras = {}

                parent_layer.forge_online_loras[child_key] = current_patches
                self.online_backup.append(parent_layer)
                n_online += 1
                continue

            if key not in self.backup:
                self.backup[key] = weight.to(device=offload_device)

            # MixedPrecision QuantizedTensor only (Krea2 comfy_quant / similar).
            # Plain float Linear / BnB / GGUF keep the legacy path below — no behavior change.
            convert_func = getattr(parent_layer, "convert_{}".format(child_key), None)
            set_func = getattr(parent_layer, "set_{}".format(child_key), None)
            use_mp_quant_lora = False
            layout_type = getattr(parent_layer, "layout_type", None)
            weight_kind = type(weight.data if hasattr(weight, "data") else weight).__name__
            if layout_type is not None and convert_func is not None and set_func is not None:
                use_mp_quant_lora = True
            else:
                try:
                    from comfy.quant_ops import QuantizedTensor

                    raw = weight.data if hasattr(weight, "data") else weight
                    if isinstance(weight, QuantizedTensor) or isinstance(raw, QuantizedTensor):
                        if convert_func is not None and set_func is not None:
                            use_mp_quant_lora = True
                            weight_kind = type(raw).__name__
                except Exception:
                    pass

            if use_mp_quant_lora:
                # ComfyUI-master: dequantize → merge → requantize via set_weight.
                # Offline .to(float32) on QuantizedTensor does not apply LoRA correctly.
                weight_f = convert_func(weight)
                if sample_mp is None:
                    sample_mp = (
                        key,
                        weight_kind,
                        layout_type,
                        type(weight_f).__name__,
                        getattr(weight_f, "dtype", None),
                        tuple(weight_f.shape) if hasattr(weight_f, "shape") else None,
                    )
                    print(
                        "[LORA] MixedPrecision path (Comfy convert→merge→set): "
                        f"key={key} storage={weight_kind} layout={layout_type} "
                        f"dequant={type(weight_f).__name__}/{getattr(weight_f, 'dtype', None)} "
                        f"shape={getattr(weight_f, 'shape', None)}"
                    )
                try:
                    weight_f = merge_lora_to_weight(current_patches, weight_f, key, computation_dtype=torch.float32)
                except Exception:
                    print("Patching LoRA weights out of memory. Retrying by offloading models.")
                    set_parameter_devices(self.model, parameter_devices={k: offload_device for k in parameter_devices.keys()})
                    memory_management.soft_empty_cache()
                    weight_f = merge_lora_to_weight(current_patches, weight_f, key, computation_dtype=torch.float32)
                # ComfyUI-master model_patcher.patch_weight_to_device: seed must be int.
                # set_weight(..., seed=None) → stochastic_rounding=None → TypeError on `> 0`.
                set_func(
                    weight_f,
                    inplace_update=False,
                    seed=comfy_utils.string_to_seed(key),
                )
                n_mp_quant += 1
                continue

            bnb_layer = None

            if hasattr(weight, "bnb_quantized") and operations.bnb_available:
                bnb_layer = parent_layer
                from backend.operations_bnb import functional_dequantize_4bit

                weight = functional_dequantize_4bit(weight)

            gguf_cls = getattr(weight, "gguf_cls", None)
            gguf_parameter = None

            if gguf_cls is not None:
                gguf_parameter = weight
                from backend.operations_gguf import dequantize_tensor

                weight = dequantize_tensor(weight)

            try:
                weight = merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)
            except:
                print("Patching LoRA weights out of memory. Retrying by offloading models.")
                set_parameter_devices(self.model, parameter_devices={k: offload_device for k in parameter_devices.keys()})
                memory_management.soft_empty_cache()
                weight = merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)

            if bnb_layer is not None:
                bnb_layer.reload_weight(weight)
                n_legacy += 1
                continue

            if gguf_cls is not None:
                gguf_cls.quantize_pytorch(weight, gguf_parameter)
                n_legacy += 1
                continue

            utils.set_attr_raw(self.model, key, torch.nn.Parameter(weight, requires_grad=False))
            n_legacy += 1

        # End
        total = n_online + n_mp_quant + n_legacy
        if total > 0:
            print(
                "[LORA] refresh apply summary: "
                f"online={n_online} "
                f"mixed_precision_convert_merge_set={n_mp_quant} "
                f"legacy_offline_merge={n_legacy} "
                f"(total={total})"
            )
            if n_mp_quant > 0:
                print(
                    "[LORA] flow: QuantizedTensor → convert_weight(dequant) → "
                    "merge_lora_to_weight(fp32) → set_weight(requantize)  "
                    f"[keys={n_mp_quant}]"
                )
            if n_legacy > 0 and n_mp_quant == 0:
                print("[LORA] flow: Parameter → merge_lora_to_weight(fp32) → write back (legacy)")

        set_parameter_devices(self.model, parameter_devices=parameter_devices)
        self.loaded_hash = hashes
        return
