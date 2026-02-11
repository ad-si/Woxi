"""Woxi kernel for JupyterLite."""


def _jupyter_labextension_paths():
    return [{"src": "labextension", "dest": "@woxi/jupyterlite-woxi-kernel"}]
