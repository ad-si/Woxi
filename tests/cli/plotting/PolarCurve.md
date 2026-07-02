# `PolarCurve`

Graphics primitive representing the polar curve with radius `r` as a
function of the angle `t` over the range `t0` to `t1`:
`PolarCurve[r, {t, t0, t1}]`.

In the CLI it stays symbolic on its own (graphical front ends like the
playground and Woxi Studio render it as a graphic, like Wolfram notebooks):

```scrut
$ wo 'PolarCurve[1 + Cos[t], {t, 0, 2 Pi}]'
PolarCurve[1 + Cos[t], {t, 0, 2*Pi}]
```

Inside `Graphics` it is rendered as a curve:

```scrut
$ wo 'Head[Graphics[PolarCurve[1 + Cos[t], {t, 0, 2 Pi}]]]'
Graphics
```

Style directives like colors and `Thickness` apply as for `Line`:

```wolfram
Graphics[{Red, Thickness[0.02], PolarCurve[Sin[3 t], {t, 0, Pi}]}]
```

Use [`FilledPolarCurve`](FilledPolarCurve.md) to fill the region
enclosed by the curve.
