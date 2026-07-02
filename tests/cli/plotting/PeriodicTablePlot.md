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

The `"Phase"` property (equivalently
`EntityProperty["Element", "Phase"]`) colors each element by its phase at
standard temperature and pressure and attaches a swatch legend below the
table.

```scrut
$ wo 'PeriodicTablePlot["Phase"]'
Legended[-Graphics-, Placed[SwatchLegend[{RGBColor[0.493332, 0.733333, 0.866667], RGBColor[0.96666, 0.7513329, 0.4283329], RGBColor[0.636667, 0.799999, 0.473333], GrayLevel[0.9]}, {gas, solid, liquid, Missing[NotAvailable]}, LegendLayout -> Row, LegendLabel -> EntityProperty[Element, Phase]], Below]]
```

### Options

- **`ImageSize`** — overall width in pixels (or `{width, height}`, or a
  symbolic size such as `Small`, `Medium`, `Large`).
