# `FilledPolarCurve`

Graphics primitive representing the filled region enclosed by a
[`PolarCurve`](PolarCurve.md): `FilledPolarCurve[PolarCurve[r, {t, t0, t1}]]`.

On its own it stays symbolic:

```scrut
$ wo 'FilledPolarCurve[PolarCurve[Sin[2 t], {t, 0, 2 Pi}]]'
FilledPolarCurve[PolarCurve[Sin[2*t], {t, 0, 2*Pi}]]
```

Inside `Graphics` it is rendered as a filled region:

```scrut
$ wo 'Head[Graphics[FilledPolarCurve[PolarCurve[Sin[2 t], {t, 0, 2 Pi}]]]]'
Graphics
```

Color and `Opacity` directives apply as for `Polygon`:

```wolfram
Graphics[{Blue, Opacity[0.5],
  FilledPolarCurve[PolarCurve[1 + Cos[t], {t, 0, 2 Pi}]]}]
```
