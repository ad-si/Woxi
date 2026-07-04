# `Control`

Represents a standalone interactive control bound to a variable. In a
Jupyter notebook or the [playground](/), `Control` renders as an interactive
widget (a slider, popup menu, 2D slider, or interval slider). Because the
widget cannot be compared textually, the examples below use `Head[...]` to
verify that the expression is held as a `Control`.

A continuous slider:

```scrut
$ wo 'Head[Control[{x, 0, 1}]]'
Control
```

A popup menu over a discrete list of values:

```scrut
$ wo 'Head[Control[{x, {a, b, c}}]]'
Control
```

A 2D slider, given either as a pair of corner points or via
`ControlType -> Slider2D`:

```scrut
$ wo 'Head[Control[{x, {0, 0}, {1, 1}}]]'
Control
```

```scrut
$ wo 'Head[Control[{xy, 0, 1, ControlType -> Slider2D}]]'
Control
```

An interval slider:

```scrut
$ wo 'Head[Control[{int, 0, 1, ControlType -> IntervalSlider}]]'
Control
```

An initial value and label may be given in the variable slot:

```scrut
$ wo 'Head[Control[{{x, 0.5, "variable"}, 0, 1}]]'
Control
```
