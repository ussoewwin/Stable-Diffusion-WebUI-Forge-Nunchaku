<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/zhmd/v1.7.4.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## 1. Symptoms and Root Causes

| Item | Details |
|------|---------|
| **Symptoms** | With Pony-style SDXL LoRAs (e.g. `cureflora.pony.safetensors`): **only the UNet seems to work / CLIP side has no effect**, or the console shows `[LORA] LoRA mismatch` and the **entire LoRA is rejected** |
| **Direct cause** | **`model_lora_keys_clip()`**, which maps LoRA file CLIP keys (`lora_te1_*`) to model weight keys, **did not match the actual CLIP `state_dict` key layout** |
| **Structural cause** | Forge SDXL loads CLIP via **IntegratedCLIP**, so keys become **`clip_l.transformer.encoder.layers.*`**. The old code only looked for **`clip_l.transformer.text_model.encoder.layers.*`** |

---

## 2. Why Forge CLIP Keys “Change”

### 2.1 IntegratedCLIP Structure

For SDXL, `backend/loader.py` wraps `CLIPTextModel` as **`IntegratedCLIP`** (`backend/nn/clip.py`). It uses `self.transformer = cls(config)` only—there is no HuggingFace-style `text_model` wrapper.

### 2.2 Key Normalization in the Loader

When loading a checkpoint (near `load_huggingface_component`), keys like

```
text_model.encoder.layers.0....
```

have their prefix stripped and only **`transformer.{clean}`** is prepended. Runtime keys therefore look like:

```
clip_l.transformer.encoder.layers.0.self_attn.k_proj.weight
```

They **do not** become `clip_l.transformer.text_model.encoder.layers.*`.

### 2.3 What `model_lora_keys_clip` Assumed Before the Fix

Before the fix, only this path was searched in `state_dict`:

```
clip_l.transformer.text_model.encoder.layers.{block}.{layer}.weight
```

With IntegratedCLIP and the normalization above, **those keys do not exist** → almost no CLIP encoder entries in `key_map` → Pony LoRA `lora_te1_*` keys do not match.

---

## 3. LoRA Application Flow (Where the Fix Matters)

```
[LoRA .safetensors]
        ↓
extensions-builtin/sd_forge_lora/networks.py
  load_lora_for_models()
        ↓
model_lora_keys_unet()  → UNet key_map
model_lora_keys_clip()  → CLIP key_map   ← ★ fix location
        ↓
load_lora(lora_dict, key_map)  (modules_forge/packages/comfy/lora.py)
  Bind LoRA keys ↔ model weight keys; build patches
        ↓
Unmatched keys > **50%** of LoRA keys → reject entire file ([LORA] LoRA mismatch)
        ↓
model.add_patches() / clip.add_patches()
```

### 3.1 What `key_map` Means

`model_lora_keys_clip()` builds a dictionary like this:

| key (logical name on LoRA file side) | value (actual key in model `state_dict`) |
|--------------------------------------|------------------------------------------|
| `lora_te1_text_model_encoder_layers_0_self_attn_k_proj` | `clip_l.transformer.encoder.layers.0.self_attn.k_proj.weight` |
| `text_encoder.text_model.encoder.layers.0.self_attn.k_proj` (diffusers style) | same as above |

`load_lora()` uses **keys** (left) as prefixes for `*.lora_up` / `*.lora_down` in the LoRA file, and applies patches to **values** (right).

### 3.2 50% Gate (`networks.py`, lines ~79–86)

```python
_unmatches = len(lora_unmatch)

if _unmatches / len(lora) > 0.5:
    print(f"[LORA] LoRA mismatch for {model_flag}: {filename}")
    return model, clip

if _unmatches > 0:
    print(f"[LORA] Loading {filename} for {model_flag} with {_unmatches} unmatched keys")
```

If almost all CLIP keys fail to match, even when UNet keys would match, **the whole file is skipped**. That can look like “LoRA does nothing” rather than “only UNet works.”

---

## 4. The Fix (Current Implementation on `main`)

**File:** `modules_forge/packages/comfy/lora.py`  
**Function:** `model_lora_keys_clip()` (approx. lines 107–174)  
**Origin:** `3a8ea0c` / `3d1950d` / `08e6e70`

### 4.1 `LORA_CLIP_MAP` (unchanged)

Maps layer names inside the LoRA file (underscore-separated) to attribute names in the model:

| Model side (inside `encoder.layers`) | LoRA suffix |
|--------------------------------------|-------------|
| `self_attn.k_proj` | `self_attn_k_proj` |
| `self_attn.q_proj` | `self_attn_q_proj` |
| `self_attn.v_proj` | `self_attn_v_proj` |
| `self_attn.out_proj` | `self_attn_out_proj` |
| `mlp.fc1` | `mlp_fc1` |
| `mlp.fc2` | `mlp_fc2` |

### 4.2 Core Fix: `clip_layer_paths` Loop

For each of `clip_h`, `clip_l`, and `clip_g`, the code tries **two relative path patterns** in order (the implementation prepends `clip_l.` etc. and searches `sdk`):

```python
# transformers 4.x–5.5: clip_*.transformer.text_model.encoder.layers.*
# transformers 5.6+ (IntegratedCLIP + f1c299e loader): clip_*.transformer.encoder.layers.*
clip_layer_paths = (
    "transformer.text_model.encoder.layers.{}.{}.weight",
    "transformer.encoder.layers.{}.{}.weight",
)
```

For each `(block, layer)`:

1. Prepend `clip_h.` / `clip_l.` / `clip_g.` to the paths above and check whether they exist in `sdk` (`model.state_dict().keys()`).
2. If found, register multiple LoRA key forms in `key_map` for that model key:
   - `lora_te1_text_model_encoder_layers_{block}_{layer}` (SDXL / Pony standard)
   - `text_encoder.text_model.encoder.layers.{block}.{layer}` (diffusers format)
   - `lora_te2_*` (for `clip_g` when SDXL dual text encoders with `clip_l` are present)

Paths **not** in `sdk` are skipped. On IntegratedCLIP, the first pattern usually misses and the second hits.

### 4.3 Intent Documented in Comments (lines ~114–118)

- **TF 4.x–5.5:** `transformer.text_model.encoder.layers.*`
- **TF 5.6+ / IntegratedCLIP:** `transformer.encoder.layers.*` (no `text_model`)

Pony / Forge Classic Neo SDXL pipelines use the latter.

---

## 5. Difference vs. Alternate Implementation (`45a0101`)

Local commit `45a0101` used helper `_resolve_clip_encoder_layer_weight_key()` with the **opposite search order**:

| Implementation | Search order |
|----------------|--------------|
| **`45a0101`** | ① `transformer.encoder.layers` → ② `transformer.text_model.encoder.layers` |
| **current `main`** | ① `text_model.encoder` → ② `encoder.layers` |

With **IntegratedCLIP only**, real keys are under `encoder.layers`, so **both approaches should reach the same keys and Pony LoRA should work**.

They differ only in the rare case where **both key layouts coexist** in `state_dict`:

- current `main` version: prefers `text_model` paths  
- `45a0101`: prefers `encoder` paths  

**Current `main` `lora.py` uses the `clip_layer_paths` implementation** (`origin/main` was chosen at merge). The `45a0101` helper is **not in the current file** (it remains only as a historical commit).

---

## 6. Position in Git History

```
08e6e70  Merge origin/main into main for Pony SDXL LoRA CLIP key fix  ← current HEAD
45a0101  fix: resolve Pony SDXL LoRA CLIP keys for TF 5.6+ encoder layout  ← alternate (not merged)
3d1950d  Merge cursor/lora-clip-paths-fc1b5
3a8ea0c  Refactor LORA key mapping for CLIP models in lora.py  ← implementation
```

- **The functional change is only `model_lora_keys_clip` in `lora.py`** (no Pony-specific UNet or loader changes)

---

## 7. Pony LoRA File Keys (Reference)

Typical SDXL Pony LoRAs contain:

- **UNet:** `lora_unet_*` → usually matches via `model_lora_keys_unet`  
- **CLIP-L:** `lora_te1_text_model_encoder_layers_{N}_{self_attn_k_proj, ...}` → **target of this fix**  
- **CLIP-G (SDXL):** `lora_te2_*` → for `clip_g` (when both `clip_l` and `clip_g` exist)

After the fix, `lora_te1_*` binds to `clip_l.transformer.encoder.layers.*`, so **text-prompt LoRA effects** apply.

---

## 8. Verification Script (Uncommitted)

`_tmp_pony_lora_probe.py` is a temporary script to reproduce, without starting the WebUI:

- Match rate: old approach (`text_model` only) vs. current `model_lora_keys_clip`  
- The same 50% gate as `networks.py`  

It failed with system Python due to missing `psutil`, but you can get numeric results using the **Forge bundled venv Python**.

---

## 9. Summary

| Question | Answer |
|----------|--------|
| What was fixed? | CLIP LoRA key mapping did not support **TF 5.6+ / IntegratedCLIP `transformer.encoder.layers.*`** |
| Where? | `model_lora_keys_clip()` in `modules_forge/packages/comfy/lora.py` |
| What is on current `main`? | **`clip_layer_paths` two-path lookup** (merged in `08e6e70`) |
| What changes for Pony? | `lora_te1_*` binds to CLIP-L; **prompt-side LoRA works** / full-file rejection is reduced |

---

## Related Files

| Path | Role |
|------|------|
| `modules_forge/packages/comfy/lora.py` | `model_lora_keys_clip()`, `load_lora()` |
| `extensions-builtin/sd_forge_lora/networks.py` | LoRA loading, 50% match gate |
| `backend/nn/clip.py` | `IntegratedCLIP` |
| `backend/loader.py` | CLIP key normalization (`transformer.{clean}`) |
| `_tmp_pony_lora_probe.py` | Match-rate verification (temporary script) |
