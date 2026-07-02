# `FilledPolarCurve`

Graphics primitive representing the filled region enclosed by a
[`PolarCurve`](PolarCurve.md): `FilledPolarCurve[PolarCurve[r, {t, t0, t1}]]`.
`FilledPolarCurve[r, t]` (bare variable) fills the region enclosed over the
full period `{t, 0, 2 Pi}`.

In the CLI it stays symbolic on its own (graphical front ends like the
playground and Woxi Studio render it as a graphic, like Wolfram notebooks):

```scrut
$ wo 'FilledPolarCurve[PolarCurve[Sin[2 t], {t, 0, 2 Pi}]]'
FilledPolarCurve[PolarCurve[Sin[2*t], {t, 0, 2*Pi}]]
```

```scrut
$ wo 'FilledPolarCurve[1 - Cos[t], t]'
FilledPolarCurve[1 - Cos[t], t]
```

Inside `Graphics` it is rendered as a filled region:

```scrut
$ wo 'Head[Graphics[FilledPolarCurve[PolarCurve[Sin[2 t], {t, 0, 2 Pi}]]]]'
Graphics
```

```scrut
$ wo 'Head[Graphics[FilledPolarCurve[1 - Cos[t], t]]]'
Graphics
```

Color and `Opacity` directives apply as for `Polygon`:

```wolfram
Graphics[{Blue, Opacity[0.5],
  FilledPolarCurve[PolarCurve[1 + Cos[t], {t, 0, 2 Pi}]]}]
```
