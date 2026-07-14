# Woxi for Python

Python bindings for [Woxi](https://github.com/ad-si/Woxi),
an interpreter for a subset of the Wolfram Language
implemented in Rust.
Woxi is a computer algebra system (CAS),
so computations are solved symbolically.

```sh
pip install woxi
```


## Python API

```python
import woxi

woxi.interpret("Plus[1, 2]")
# => '3'

woxi.interpret("Integrate[x^2, x]")
# => 'x^3/3'

# Session state persists across calls (per thread):
woxi.interpret("x = 42;")
woxi.interpret("x + 1")
# => '43'
woxi.clear_state()

# Capture stdout, graphics, sound, and warnings:
res = woxi.evaluate('Print["hi"]; Plot[Sin[x], {x, 0, 10}]')
res.stdout    # => 'hi\n'
res.graphics  # => '<svg …'  (SVG markup)

# Reproducible random numbers:
woxi.seed_rng(1)

# Errors raise woxi.WolframError:
try:
    woxi.interpret("1 +")
except woxi.WolframError as err:
    print(err)
```


## Command line

The package also installs a `woxi` command:

```sh
woxi eval 'Plus[1, 2]'     # Evaluate an expression
woxi run script.wls        # Run a Wolfram Language file
woxi repl                  # Interactive session
```

For the full-featured native CLI (including the Jupyter kernel),
install the Rust binary instead: `cargo install woxi`.


## Building from source

The package is built with [maturin](https://maturin.rs)
and requires a Rust toolchain:

```sh
cd woxi-py
pip install maturin
maturin develop  # Build and install into the active virtualenv
```

Run the tests with:

```sh
python -m pytest tests/
```


## License

AGPL-3.0-or-later
