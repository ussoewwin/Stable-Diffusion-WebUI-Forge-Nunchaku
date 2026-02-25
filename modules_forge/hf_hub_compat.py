"""
Compatibility shim for Gradio with huggingface_hub >= 1.0.
Gradio's oauth.py does "from huggingface_hub import HfFolder, whoami".
HfFolder was removed in huggingface_hub 1.0; we inject a thin wrapper so Gradio does not need to be changed.
"""


def apply():
    import huggingface_hub

    if hasattr(huggingface_hub, "HfFolder"):
        return

    get_token = getattr(huggingface_hub, "get_token", None)
    login = getattr(huggingface_hub, "login", None)
    logout = getattr(huggingface_hub, "logout", None)

    class HfFolder:
        @classmethod
        def get_token(cls):
            return get_token() if get_token else None

        @classmethod
        def save_token(cls, token):
            if login and token:
                login(token=token)

        @classmethod
        def delete_token(cls):
            if logout:
                logout()

    huggingface_hub.HfFolder = HfFolder
