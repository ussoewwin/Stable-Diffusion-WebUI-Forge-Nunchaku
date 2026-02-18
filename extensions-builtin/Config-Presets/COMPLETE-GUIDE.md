# Config-Presets: Complete Guide

This document explains the **Config-Presets** feature added to sd-webui-forge-classic-neo, including the added files, folders, code, and their purposes.

---

## 1. Overview

Config-Presets is an extension that allows you to save and switch UI settings for the txt2img and img2img tabs as presets. Originally published as [Zyin055/Config-Presets](https://github.com/Zyin055/Config-Presets), it is compatible with stable-diffusion-webui-forge. This project integrates it as a **built-in extension**.

### Key Features
- **Preset Selection**: Select an existing preset from the dropdown to apply its settings in bulk
- **Preset Create/Delete**: Save current UI values as a new preset, or delete existing presets
- **Custom Fields**: Track components from other extensions via `*custom-tracked-components.txt`
- **Manual Editing**: Edit JSON config files directly and reload them with the refresh button

---

## 2. File and Folder Structure

```
extensions-builtin/Config-Presets/
├── preload.py                              # Registers command-line args at startup
├── scripts/
│   └── config_presets.py                   # Main script (UI & logic)
├── javascript/
│   └── config_presets.js                   # UI helpers (tooltips, etc.)
├── style.css                               # Styles
├── config-txt2img.json                     # txt2img preset definitions (default)
├── config-img2img.json                     # img2img preset definitions (default)
├── config-txt2img-custom-tracked-components.txt   # txt2img custom tracked components
├── config-img2img-custom-tracked-components.txt   # img2img custom tracked components
├── README.md                               # Extension description
└── LICENSE                                 # License
```

### .gitignore Entries

The following paths are added to the root `.gitignore` (user-specific config files):

```
/extensions-builtin/Config-Presets/scripts/__pycache__
/extensions-builtin/Config-Presets/config.json
/extensions-builtin/Config-Presets/config-img2img.json
/extensions-builtin/Config-Presets/config-img2img-custom-tracked-components.txt
/extensions-builtin/Config-Presets/config-txt2img.json
/extensions-builtin/Config-Presets/config-txt2img-custom-tracked-components.txt
```

- `config*.json` / `*custom-tracked-components.txt`: User-editable config files
- `scripts/__pycache__`: Python cache directory

---

## 3. Role and Processing of Each File

### 3.1 `preload.py`

**Role**: Registers command-line arguments at startup.

```python
def preload(parser):
    parser.add_argument("--configpresets-dir", type=Path, ...)
```

- `--configpresets-dir`: Option to specify the directory for config files
- If not specified, the extension directory is used
- Useful for sharing settings across multiple installations

**Load Timing**: Executed early at startup via `script_loading.preload_extensions(extensions_builtin_dir, parser)` in `modules/shared_cmd_options.py`.

---

### 3.2 `scripts/config_presets.py` (Main Script)

An extension script inheriting from A1111's `scripts.Script`.

#### Main Constants and Variables

| Name | Meaning |
|------|---------|
| `BASEDIR` | Extension base path (`scripts.basedir()`) |
| `config_folder` | Config file directory. Uses `--configpresets-dir` path when specified, otherwise `BASEDIR` |
| `CONFIG_TXT2IMG_FILE_NAME` | `config-txt2img.json` |
| `CONFIG_IMG2IMG_FILE_NAME` | `config-img2img.json` |
| `CONFIG_*_CUSTOM_TRACKED_COMPONENTS_FILE_NAME` | Custom tracked component definition files |

#### Key Functions

- **`load_txt2img_custom_tracked_component_ids()` / `load_img2img_custom_tracked_component_ids()`**  
  Load `*custom-tracked-components.txt` and return the list of component IDs to track. Creates a default template if the file does not exist.

- **`load_txt2img_config_file()` / `load_img2img_config_file()`**  
  Load `config-txt2img.json` / `config-img2img.json` and return the preset dictionary. Initializes with default presets if the file does not exist.

- **`dict_synonyms(d, lsyn)`**  
  Maps synonymous component IDs to handle version differences across WebUI versions.

- **`save_config(config_presets, component_map, config_file_name)`**  
  Returns a closure that saves the current UI values as a preset.

#### `Script` Class

- **`__init__`**  
  Loads custom tracked components, initializes `txt2img_component_ids` / `img2img_component_ids`, and loads config files.

- **`title()`**  
  Returns the script name `"Config Presets"`.

- **`show(is_img2img)`**  
  Returns `scripts.AlwaysVisible`; the script is always shown in the UI but not in the Scripts dropdown.

- **`after_component(component, **kwargs)`**  
  Called by Gradio's `after_component` callback after each component is created. When processing `txt2img_generation_info_button` / `img2img_generation_info_button`, it builds the Config Presets dropdown and related buttons for that tab.

#### Component ID Handling

- **Required IDs**: Sampler, steps, image size, batch count, CFG, Hires fix-related, etc.
- **Optional IDs**: IDs that exist only in specific WebUIs or extensions (e.g., `txt2img_distilled_cfg_scale`)
- **Custom IDs**: IDs listed in `*custom-tracked-components.txt`  
  These are combined to determine which fields are saved and applied in presets.

---

### 3.3 `javascript/config_presets.js`

**Role**: Adds tooltips (`title` attribute) to UI elements.

- Runs on DOM update via `onUiUpdate`
- Targets: refresh button, save button, trash button, open config file button, etc.

---

### 3.4 `style.css`

**Role**: Adjusts spacing around the Config Presets dropdown.

```css
#config_preset_wrapper_txt2img,
#config_preset_wrapper_img2img {
    margin-top: 10px;
}
```

---

### 3.5 `config-txt2img.json` / `config-img2img.json`

**Format**: JSON with preset names as keys and component ID-to-value maps as values.

Example (txt2img):

```json
{
  "None": {},
  "SD1.5 - 512x512": {
    "txt2img_width": 512,
    "txt2img_height": 512
  },
  "High res --- [Hires fix ...]": {
    "txt2img_enable_hr": true,
    "txt2img_hr_scale": 2,
    ...
  }
}
```

- This JSON is updated when the user saves a preset
- Excluded from the repo via `.gitignore` (user-specific config)

---

### 3.6 `config-*-custom-tracked-components.txt`

**Role**: Defines component IDs for including other extensions' UI components in presets.

- Lines starting with `#` are comments
- One ID per line
- Contains commented examples for ControlNet, ADetailer, Tiled Diffusion, etc.

---

## 4. Integration with the Main Application

### 4.1 Extension Loading

1. **Preload Phase**  
   In `modules/shared_cmd_options.py`:
   ```python
   script_loading.preload_extensions(extensions_builtin_dir, parser)
   ```
   Each extension's `preload.py` in `extensions-builtin` is executed, registering Config-Presets' `--configpresets-dir` option.

2. **Extension Listing**  
   `modules/extensions.py` scans `extensions_builtin_dir` and `extensions_dir`, loading scripts under `scripts/`.

3. **Script Execution**  
   The `Script` class in `scripts/config_presets.py` is registered as a `scripts.Script`, and `after_component` adds the Config Presets dropdown to the txt2img/img2img UI.

### 4.2 Path Definition

`modules/paths_internal.py`:

```python
extensions_builtin_dir = os.path.join(script_path, "extensions-builtin")
```

`script_path` points to the project root (parent of `modules`), so Config-Presets' actual path is:

```
{project_root}/extensions-builtin/Config-Presets/
```

---

## 5. Data Flow

1. **At Startup**
   - `preload.py` registers `--configpresets-dir`
   - `config_presets.py` loads config files and custom tracked IDs
   - `after_component` maps UI components and builds the dropdown

2. **When Selecting a Preset**
   - The dropdown's `change` event runs `config_preset_dropdown_change`
   - Selected preset values update each Gradio component

3. **When Saving a Preset**
   - User opens edit UI with "🖌️", enters a preset name, and saves
   - `save_config` updates `config_*.json` with current UI values

4. **When Deleting a Preset**
   - Clicking "🗑️" runs `delete_selected_preset` and removes the entry from the JSON

---

## 6. Summary

| Item | Description |
|------|-------------|
| Source | Zyin055/Config-Presets |
| Location | `extensions-builtin/Config-Presets/` |
| Main App Changes | None (auto-loaded as built-in extension via `preload_extensions`) |
| Config Storage | Extension directory, or path specified by `--configpresets-dir` |
| .gitignore | Excludes config JSON, custom tracked component definitions, and `__pycache__` |

Config-Presets enables one-click switching of model resolutions, quality presets, Hires fix combinations, and more.
