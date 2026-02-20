""" Preload module for DeepDanbooru or onnxtagger. """
from argparse import ArgumentParser

_BUILTIN_ARGS = (
    ('--deepdanbooru-projects-path', str, 'Path to directory with DeepDanbooru project(s).'),
    ('--onnxtagger-path', str, 'Path to directory with Onnyx project(s).'),
    ('--additional-device-ids', str, 'Device ID to use. cpu:0, gpu:0 or gpu:1, etc.'),
)


def preload(parser: ArgumentParser):
    """ Preload module for DeepDanbooru or onnxtagger. """
    # default deepdanbooru use different paths:
    # models/deepbooru and models/torch_deepdanbooru
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/c81d440d876dfd2ab3560410f37442ef56fc6632
    # Skip if already added (e.g. by user copy of this extension) to avoid ArgumentError.
    existing = {opt for action in parser._optionals._actions for opt in (action.option_strings or [])}
    for opt, typ, help_text in _BUILTIN_ARGS:
        if opt in existing:
            continue
        parser.add_argument(opt, type=typ, help=help_text)
