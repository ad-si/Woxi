# Jupyter

You can also use Woxi in Jupyter notebooks.
Install the kernel with:

```sh
woxi install-kernel
```

Then start JupyterLab:

```sh
cd examples && jupyter lab
```

Or use Woxi as a terminal REPL via `jupyter console`:

```sh
uv tool install jupyter-console
jupyter console --kernel=woxi
```

> [!TIP]
> **Try it out yourself in our
> [JupyterLite instance](https://woxi.ad-si.com/jupyterlite/lab/index.html?path=showcase.ipynb)!**
