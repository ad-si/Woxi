# `FilledPolarCurve`

Graphics primitive representing the filled region enclosed by a
[`PolarCurve`](PolarCurve.md): `FilledPolarCurve[PolarCurve[r, {t, t0, t1}]]`.
`FilledPolarCurve[r, t]` (bare variable) fills the region enclosed over the
full period `{t, 0, 2 Pi}`.

In the CLI Woxi keeps it symbolic on its own so the graphical front ends (the
playground and Woxi Studio) can render it as a graphic, like Wolfram notebooks
do. (`wolframscript` instead lowers it to a `Region[ParametricRegion[…]]` — and
warns `FilledPolarCurve::argr` on the one-argument `PolarCurve` form, which it
does not support — so these bare forms are documentation only and not part of
the conformance sweep; they are covered by the
`polar_curves_stay_symbolic_in_script_mode` unit test.)

```wolfram
FilledPolarCurve[PolarCurve[Sin[2 t], {t, 0, 2 Pi}]]
(* Woxi: FilledPolarCurve[PolarCurve[Sin[2*t], {t, 0, 2*Pi}]] *)

FilledPolarCurve[1 - Cos[t], t]
(* Woxi: FilledPolarCurve[1 - Cos[t], t] *)
```

Inside `Graphics` it is rendered as a filled region. The two-argument form
matches `wolframscript` (which wraps its own region in `Graphics`):

```scrut
$ wo 'Head[Graphics[FilledPolarCurve[1 - Cos[t], t]]]'
Graphics
```

Color and `Opacity` directives apply as for `Polygon`:

```wolfram
Graphics[{Blue, Opacity[0.5],
  FilledPolarCurve[PolarCurve[1 + Cos[t], {t, 0, 2 Pi}]]}]
```
