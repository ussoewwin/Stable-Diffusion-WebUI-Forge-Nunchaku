
# region: SDXL

from backend.nn.unet import IntegratedUNet2DConditionModel


class SVDQUNet2DConditionModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        nunchaku = config.pop("nunchaku", False)
        filename = config.pop("filename", None)
        precision = config.pop("precision", "int4")
        rank = config.pop("rank", 32)
        
        # Load standard model architecture
        self.model = IntegratedUNet2DConditionModel.from_config(config)

        # Patch with SVDQ linear layers
        self.patch_nunchaku_sdxl(self.model, precision=precision, rank=rank)

    def patch_nunchaku_sdxl(self, model, precision="int4", rank=32):
        kwargs = {"precision": precision, "rank": rank}

        def replace_linear(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    try:
                        new_layer = SVDQW4A4Linear.from_linear(child, **kwargs)
                        setattr(module, name, new_layer)
                    except Exception as e:
                        print(f"Failed to patch {name}: {e}")
                else:
                    replace_linear(child)

        def recursive_find_transformer(module):
            for name, child in module.named_children():
                if child.__class__.__name__ in ["Transformer2DModel", "BasicTransformerBlock"]:
                    replace_linear(child)
                else:
                    recursive_find_transformer(child)
        
        # Apply to transformer blocks only
        recursive_find_transformer(model)

    def forward(self, x, timestep, context, y=None, **kwargs):
        return self.model(x, timestep, context, y=y, **kwargs)
