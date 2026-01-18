import os
import sys
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir, cwd  # noqa: F401

# Ensure ComfyUI-master is in sys.path before script_path
# This ensures we use ComfyUI-master/comfy instead of the project root comfy directory
comfyui_master_path = os.path.join(script_path, "ComfyUI-master")
comfyui_master_path = os.path.normpath(comfyui_master_path)

if os.path.exists(comfyui_master_path):
    if comfyui_master_path not in sys.path:
        sys.path.insert(0, comfyui_master_path)
    # Remove project root comfy directory from sys.path if it exists
    project_comfy_path = os.path.join(script_path, "comfy")
    project_comfy_path = os.path.normpath(project_comfy_path)
    if project_comfy_path in sys.path:
        sys.path.remove(project_comfy_path)

sys.path.insert(0, script_path)

sd_path = os.path.dirname(__file__)
