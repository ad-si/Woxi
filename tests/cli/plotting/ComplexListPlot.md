# `ComplexListPlot`

Plots a list of complex numbers as points at `(Re[z], Im[z])`
in the complex plane.

```scrut
$ wo 'Head[ComplexListPlot[{1 + I, 2 - I, -1 + 2 I}]]'
Graphics
```

Accepts the same core options as `ListPlot`, e.g. `Joined`:

```scrut
$ wo 'Head[ComplexListPlot[{1 + I, 2 - I, -1 + 2 I}, Joined -> True]]'
Graphics
```
