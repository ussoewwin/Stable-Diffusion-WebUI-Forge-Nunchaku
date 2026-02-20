import os
import sys
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir, cwd, COMFYUI_FOLDER_NAME  # noqa: F401

# Ensure ComfyUI is in sys.path before script_path (use COMFYUI_FOLDER_NAME so updating ComfyUI is easy)
comfyui_master_path = os.path.normpath(os.path.join(script_path, COMFYUI_FOLDER_NAME))

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
