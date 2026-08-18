---
icon: lucide/notebook
---

# Jupyter

You can also use Woxi in Jupyter notebooks.
Install the kernel with:

```sh
woxi install-kernel
```

The kernelspec is embedded in the `woxi` binary,
so this works from any directory and with any installation method.
It is registered with `jupyter kernelspec install` when the `jupyter` command
is available, and written to Jupyter's kernels directory directly otherwise.
Use `--system` to install it for all users instead of only the current one.

Then start JupyterLab:

```sh
cd examples && jupyter lab
```

Or use Woxi as a terminal REPL via `jupyter console`:

```sh
uv tool install jupyter-console
jupyter console --kernel=woxi
```

!!! tip

    **Try it out yourself in our
    [JupyterLite instance](/jupyterlite/lab/index.html?path=showcase.ipynb)!**
