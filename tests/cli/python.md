---
icon: lucide/braces
---

# Python

Woxi is available [on PyPI](https://pypi.org/project/woxi/) as a
Python package with pre-built wheels for Linux, macOS, and Windows
(CPython 3.10+):

```sh
pip install woxi
```


## API

`woxi.interpret` evaluates Wolfram Language code and returns the
result formatted as Wolfram Language text.
`Print` output goes to the process stdout,
exactly like `woxi eval` on the command line:

```python
import woxi

woxi.interpret("Plus[1, 2]")
# => '3'

woxi.interpret("Integrate[x^2, x]")
# => 'x^3/3'
```

Session state (variable definitions, function definitions, the RNG
seed, …) persists across calls within a thread:

```python
woxi.interpret("x = 42;")
woxi.interpret("x + 1")
# => '43'

woxi.clear_state()  # Reset the session
```

`woxi.evaluate` captures everything an evaluation produces —
stdout, graphics (as SVG), sound, and warnings —
instead of letting it escape to the terminal:

```python
res = woxi.evaluate('Print["hi"]; Plot[Sin[x], {x, 0, 10}]')
res.result    # Final expression value as text
res.stdout    # => 'hi\n'
res.graphics  # => '<svg …'
res.warnings  # => []
```

Failures raise `woxi.WolframError`:

```python
try:
    woxi.interpret("1 +")
except woxi.WolframError as err:
    print(err)  # Parse error: …
```

Random numbers can be made reproducible:

```python
woxi.seed_rng(1)
a = woxi.interpret("RandomInteger[100]")
woxi.seed_rng(1)
assert woxi.interpret("RandomInteger[100]") == a
woxi.unseed_rng()
```


## Command line

The package also installs a `woxi` command mirroring the core
subcommands of the native CLI:

```sh
woxi eval 'Plus[1, 2]'     # Evaluate an expression
woxi run script.wls        # Run a Wolfram Language file
woxi repl                  # Interactive session
```

For the full-featured native CLI (including the Jupyter kernel),
install the Rust binary instead: `cargo install woxi`.
