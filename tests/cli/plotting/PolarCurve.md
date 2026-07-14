# `PolarCurve`

Graphics primitive representing the polar curve with radius `r` as a
function of the angle `t` over the range `t0` to `t1`:
`PolarCurve[r, {t, t0, t1}]`.

In the CLI Woxi keeps it symbolic on its own so the graphical front ends (the
playground and Woxi Studio) can render it as a graphic, like Wolfram notebooks
do. (`wolframscript` instead lowers it to a `Region[ParametricRegion[…]]`, so
this bare form is documentation only and not part of the conformance sweep; it
is covered by the `polar_curves_stay_symbolic_in_script_mode` unit test.)

```wolfram
PolarCurve[1 + Cos[t], {t, 0, 2 Pi}]
(* Woxi: PolarCurve[1 + Cos[t], {t, 0, 2*Pi}] *)
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
