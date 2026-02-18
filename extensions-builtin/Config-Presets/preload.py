from pathlib import Path

def preload(parser):
    parser.add_argument("--configpresets-dir", type=Path, help="[Extension][Config-Presets] Path to directory with Config Presets configuration files (use forward slashes or double blackslashes). Default config files will be created if none exist.", default=None)
    