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


## Expression trees

Results are also available as a structured tree,
so you don't have to parse Wolfram output text to use them:

```python
import woxi

woxi.evaluate("1/3 + 1/6").expr
# => Expr(Symbol("Rational"), [1, 2])

woxi.evaluate("Solve[x^2 == 2, x]").expr
# => [[Expr(Symbol("Rule"), [Symbol("x"), …])], …]
```

The tree mirrors Wolfram's `FullForm`,
so `x + y` is `Plus[x, y]` and `a/b` is `Times[a, Power[b, -1]]`
no matter how they were written.
Only exact, total mappings use native Python types:

| Wolfram                     | Python                                |
| --------------------------- | ------------------------------------- |
| `Integer`, `BigInteger`     | `int`                                 |
| `Real`                      | `float`                               |
| `String`                    | `str`                                 |
| `List[…]`                   | `list`                                |
| symbol or constant          | `woxi.Symbol`                         |
| arbitrary-precision real    | `woxi.BigReal`                        |
| everything else             | `woxi.Expr(head, args)`               |

`woxi.to_python` converts the rest
when convenience matters more than exactness —
`Rational` to `fractions.Fraction`, `Complex` to `complex`,
`Association` to `dict`, and `True`/`False`/`Null`
to `True`/`False`/`None`:

```python
woxi.to_python(woxi.evaluate("1/3 + 1/6").expr)
# => Fraction(1, 2)

woxi.to_python(woxi.evaluate('<|"a" -> 1|>').expr)
# => {'a': 1}
```

Trees also go the other way,
so Python values reach an evaluation without string concatenation:

```python
from woxi import wl

woxi.evaluate_expr(wl.Integrate(wl.Power(wl.x, 2), wl.x)).result
# => 'x^3/3'

woxi.evaluate_expr(wl.Total([1, 2, 3, 4])).result
# => '10'

# A str argument is data here, never code:
woxi.evaluate_expr(wl.StringLength("1 + 1")).result
# => '5'
```

`wl.<name>` is the symbol of that name and calling it applies it,
so any Wolfram expression can be built without quoting.
`woxi.parse_expr` parses source into a tree *without* evaluating it:

```python
woxi.parse_expr("1 + 1")
# => Expr(Symbol("Plus"), [1, 1])
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
