# Fix: Float8 + fp16 LoRA crash when switching model across categories

## 1. Error

### Message

```
NotImplementedError: "addmm_cuda" not implemented for 'Float8_e4m3fn'
"addmm_cuda" not implemented for 'Float8_e4m3fn'
```

### Location (stack trace)

- **File**: `backend/nn/_zimage_lora.py`
- **Function**: `_apply_lora_to_module`
- **Line**: Around `delta = torch.mm(B_gpu, A_gpu)` (LoRA B @ A computation)

The log may show `[Z-Image] Composing 1 LoRA(s)...` while the model actually in use is **Flux1** or another category.

### Reproduction

- Does **not** occur on first load or when keeping the same model.
- Does **not** occur when changing only the checkpoint within the same category (e.g. Flux1 A → Flux1 B).
- **Only** when switching to a **different category** (e.g. Z-Image → Flux1, Flux1 → SDXL).
- When "Diffusion in Low Bits" is set to **float8-e4m3fn (fp16 LoRA)** (Float8 + fp16 LoRA mode).

---

## 2. Cause

### Direct cause

- LoRA application cast A/B to `module.weight.dtype` and computed the product with `torch.mm(B_gpu, A_gpu)`.
- After a category switch, diffusion weights can be loaded in **Float8** (e.g. `torch.float8_e4m3fn`).
- CUDA’s `addmm` (and thus `torch.mm`) does **not** support Float8_e4m3fn, so a `NotImplementedError` is raised.

### Why only when switching category

- On **first load** or **same model**, weights are often still fp16 (or not yet quantized) when LoRA is applied, so `torch.mm` runs in fp16 and does not fail.
- When changing **within the same category**, structure and quantization timing are similar, so computation often stays in fp16.
- When switching **to another category**, the new model may load in Float8 or get quantized earlier, so **at LoRA application time `module.weight.dtype` is already Float8**. Using that dtype for A/B and calling `torch.mm` then triggers the unsupported Float8 path.

So the intended "fp8e4m3 + fp16 LoRA" behavior (**LoRA computed in fp16**) and the old implementation (A/B cast to weight dtype, then mm) only diverged after a category switch, when weights were already Float8.

---

## 3. Changed file

| Item | Value |
|------|--------|
| File | `backend/nn/_zimage_lora.py` |
| Function | `_apply_lora_to_module` |
| Scope | "Standard ZIT" branch (path that adds LoRA directly to Linear-like modules with `weight` / `in_features` / `out_features`) |

This path is used not only for Z-Image but also for other categories (e.g. Flux1) that share the same LoRA application logic, which is why the error appeared on Flux1 after a category switch.

---

## 4. Code change

### Before (excerpt)

```python
        # Move A and B to same device/dtype as module.weight for calculation
        A_gpu = A.to(dtype=module.weight.dtype, device=module.weight.device)
        B_gpu = B.to(dtype=module.weight.dtype, device=module.weight.device)
        
        # Calculate B @ A: [out_features, rank] @ [rank, in_features] = [out_features, in_features]
        delta = torch.mm(B_gpu, A_gpu)
        # ...
        with torch.no_grad():
            module.weight.data.add_(delta)
```

### After (excerpt)

```python
        # fp8e4m3+fp16lora: CUDA addmm is not implemented for Float8 -> compute LoRA in fp16
        weight_dtype = module.weight.dtype
        weight_device = module.weight.device
        is_float8 = weight_dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e4m3fn_u", None),
            getattr(torch, "float8_e5m2_u", None),
        ) or "float8" in str(weight_dtype).lower()
        compute_dtype = torch.float16 if is_float8 else weight_dtype

        A_gpu = A.to(dtype=compute_dtype, device=weight_device)
        B_gpu = B.to(dtype=compute_dtype, device=weight_device)

        # B @ A: [out_features, rank] @ [rank, in_features] = [out_features, in_features]
        delta = torch.mm(B_gpu, A_gpu)
        delta *= scale
        # ... shape checks unchanged ...

        # Add delta: if weight is float8, add in fp16 then cast back (fp8+fp16 LoRA mode)
        with torch.no_grad():
            if is_float8:
                w = module.weight.data.float()
                w.add_(delta)
                module.weight.data = w.to(weight_dtype)
            else:
                module.weight.data.add_(delta.to(weight_dtype))
```

---

## 5. Explanation of the fix

### 5.1 Float8 detection

- Read current weight `dtype` and `device`.
- Set `is_float8` if the dtype is a PyTorch Float8 type (e4m3fn, e5m2, etc.) or if `"float8"` appears in `str(weight_dtype)`.
- `getattr(..., None)` avoids errors on environments where some Float8 dtypes are missing.

### 5.2 Compute dtype and casting A/B

- **When Float8**: LoRA matrix multiply runs in **fp16** ("fp8e4m3 + fp16 LoRA"), so CUDA addmm is supported.
- **Otherwise**: Keep using the weight dtype (fp16/bf16, etc.) as before.

### 5.3 Add and cast back

- **When Float8**: Promote weight to float32, add fp16 `delta`, then cast the result back to the original Float8 and assign to `module.weight.data`. This correctly implements "Diffusion in Float8, LoRA computed in fp16 and then applied."
- **Otherwise**: Cast `delta` to the weight dtype and add as before.

---

## 6. Summary

| Item | Description |
|------|-------------|
| **Symptom** | After switching model to a different category, using Float8 + fp16 LoRA caused `"addmm_cuda" not implemented for 'Float8_e4m3fn'`. |
| **Cause** | LoRA B@A was computed in the weight dtype; after a category switch weights were Float8, so unsupported Float8 ops were used. |
| **Approach** | When weights are Float8, compute LoRA in fp16 and perform the add in float, then cast back to Float8. |
| **File** | `backend/nn/_zimage_lora.py`, `_apply_lora_to_module` (Standard ZIT branch). |
| **Result** | Float8 + fp16 LoRA works after switching category (e.g. Z-Image ↔ Flux1) as well as on first load and within-category use. |

With this change, using "Diffusion in Low Bits" with float8-e4m3fn (fp16 LoRA) and switching between categories (Z-Image, Flux1, etc.) no longer triggers the error and remains stable.
