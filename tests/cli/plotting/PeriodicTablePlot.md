# `PeriodicTablePlot`

Renders the periodic table of the elements as a `Graphics` object, with
each element shown in its standard group/period position and colored by
its electronic block (s, p, d, f).

```scrut
$ wo 'Head[PeriodicTablePlot[]]'
Graphics
```

A list of `Entity["Element", …]` specifications highlights just those
elements; the remaining cells are faded out.

```scrut
$ wo 'Head[PeriodicTablePlot[{Entity["Element", "Iron"], Entity["Element", "Gold"]}]]'
Graphics
```

A single entity is also accepted.

```scrut
$ wo 'Head[PeriodicTablePlot[Entity["Element", "Carbon"]]]'
Graphics
```

`ElementData` resolves element names, symbols, and atomic numbers to
their entity forms.

```scrut
$ wo 'Head[PeriodicTablePlot[{ElementData["Fe"], ElementData[8]}]]'
Graphics
```

### Options

- **`ImageSize`** — overall width in pixels (or `{width, height}`, or a
  symbolic size such as `Small`, `Medium`, `Large`).
