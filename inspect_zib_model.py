"""Inspect redcraftFeb1926Latest_zibDistilledDX3Lucis.safetensors structure."""
import safetensors.torch

path = r"D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftFeb1926Latest_zibDistilledDX3Lucis.safetensors"
sd = safetensors.torch.load_file(path, device="cpu")
keys = list(sd.keys())
print("Total keys:", len(keys))

# Top-level prefix
prefixes = {}
for k in keys:
    p = k.split(".")[0] if "." in k else k
    prefixes[p] = prefixes.get(p, 0) + 1
print("Key prefixes (top):", prefixes)

print("\nSample keys (first 30):")
for k in keys[:30]:
    print(" ", k, sd[k].shape)

# cap_embedder
cap = [k for k in keys if "cap_embedder" in k]
print("\ncap_embedder keys:", cap[:10])
for k in cap:
    if ".weight" in k or ".bias" in k:
        print(" ", k, "=", sd[k].shape)

# feed_forward - exact .weight (not weight_scale_2)
ff_w1 = [k for k in keys if "feed_forward" in k and k.endswith("w1.weight")]
print("\nfeed_forward.w1.weight (exact) count:", len(ff_w1), "first 3:", ff_w1[:3])
for k in ff_w1[:3]:
    print(" ", k, "=", sd[k].shape)
if not ff_w1:
    # try layers.0.feed_forward
    any_ff = [k for k in keys if "layers.0.feed_forward" in k]
    print("  layers.0.feed_forward keys:", any_ff[:15])

# n_layers
w1_keys = [k for k in keys if "layers." in k and "feed_forward.w1.weight" in k]
layer_ids = set()
for k in w1_keys:
    parts = k.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_ids.add(int(parts[i + 1]))
            break
print("\nn_layers (main):", len(layer_ids), "indices:", sorted(layer_ids)[:5], "...", sorted(layer_ids)[-3:])

# x_embedder (in_channels, dim)
xe = [k for k in keys if "x_embedder" in k and "weight" in k]
if xe:
    print("\nx_embedder.weight:", sd[xe[0]].shape, "-> in_channels from patch, dim from out")

# t_embedder
te = [k for k in keys if "t_embedder" in k]
print("\nt_embedder keys (sample):", te[:5])

# Keys that contain "transformer" or not
has_transformer_prefix = sum(1 for k in keys if k.startswith("transformer."))
print("\nKeys starting with 'transformer.':", has_transformer_prefix)
if has_transformer_prefix:
    print("  Example:", [k for k in keys if k.startswith("transformer.")][:3])
