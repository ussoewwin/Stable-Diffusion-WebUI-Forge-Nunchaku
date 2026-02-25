
import sys
import os
import torch
from PIL import Image
import numpy as np

# Add the directory containing nodes.py to sys.path
sys.path.insert(0, os.path.join(os.getcwd(), "comfyui-florence2"))

# Mock comfy modules to allow importing nodes.py without full ComfyUI environment
import types
sys.modules["comfy"] = types.ModuleType("comfy")
sys.modules["comfy.model_management"] = types.ModuleType("model_management")
sys.modules["comfy.utils"] = types.ModuleType("utils")
sys.modules["folder_paths"] = types.ModuleType("folder_paths")

# Mock functions needed by nodes.py
def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def unet_offload_device():
    return torch.device("cpu")
def load_torch_file(path):
    return torch.load(path, map_location="cpu")
def ProgressBar(total):
    class MockParams:
        def update(self, n): pass
    return MockParams()

sys.modules["comfy.model_management"].get_torch_device = get_torch_device
sys.modules["comfy.model_management"].unet_offload_device = unet_offload_device
sys.modules["comfy.model_management"].soft_empty_cache = lambda: None
sys.modules["comfy.utils"].ProgressBar = ProgressBar
sys.modules["comfy.utils"].load_torch_file = load_torch_file
sys.modules["folder_paths"].models_dir = "models"
sys.modules["folder_paths"].add_model_folder_path = lambda x, y: None

# Now import the function to test
from nodes import load_model, Florence2Run

def test_inference(model_path):
    print(f"Testing with model at: {model_path}")
    
    device = get_torch_device()
    offload_device = torch.device("cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"Loading model (Device: {device}, Dtype: {dtype})...")
    try:
        model, processor = load_model(model_path, "sdpa", dtype, offload_device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Prepare dummy inputs
    print("Preparing dummy input image...")
    # Create a dummy PIL Image and convert it to a tensor
    pil_image = Image.new("RGB", (768, 768), color="white")
    dummy_image = torch.from_numpy(np.array(pil_image)).float().unsqueeze(0) / 255.0 # (Batch, H, W, C)
    text_input = ""
    task = "caption" # <CAPTION>
    
    florence2_model = {
        'model': model,
        'processor': processor,
        'dtype': dtype
    }
    
    print("Running inference (Florence2Run.encode)...")
    try:
        runner = Florence2Run()
        # encode(self, image, text_input, florence2_model, task, fill_mask, keep_model_loaded=False, ...)
        # Note: image arg expects (Batch, H, W, C)
        out_tensor, out_mask_tensor, out_results, out_data = runner.encode(
            image=dummy_image,
            text_input=text_input,
            florence2_model=florence2_model,
            task=task,
            fill_mask=True,
            keep_model_loaded=True
        )
        
        print("Inference successful!")
        print(f"Output Results: {out_results}")
        
    except Exception as e:
        print(f"FAILED during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    target_path = r"D:\USERFILES\GitHub\sd-webui-forge-classic-neo\Florence-2-large-PromptGen-v2.0"
    if not os.path.exists(target_path):
        print(f"Path not found: {target_path}")
    else:
        test_inference(target_path)
