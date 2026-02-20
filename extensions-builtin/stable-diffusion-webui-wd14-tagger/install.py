"""WD14-tagger (builtin). Dependencies are in the main requirements.txt."""
# No separate install; skip when launch.args.skip_install is set.
try:
    import launch  # pylint: disable=import-error
    if launch.args.skip_install:
        pass
except Exception:
    pass
