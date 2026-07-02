# `GeoHistogram`

Geographic density histogram — bins locations into tiles of equal visual
size over a map and shades every non-empty tile by the number of locations
that fell in it (darker = denser). The result is a `GeoGraphics` object
drawn like [`GeoGraphics`](GeoGraphics.md): tiles are rendered over the
equirectangular basemap and the view auto-zooms to fit them.

A location is `GeoPosition[{lat, lon}]`, a bare `{lat, lon}` pair, or a
geographic `Entity[…]`.

```scrut
$ wo 'Head[GeoHistogram[{GeoPosition[{40, -100}], GeoPosition[{41, -101}], GeoPosition[{34, -118}]}]]'
GeoGraphics
```

The second argument controls the binning: `Automatic` (the default) uses
hexagonal tiles, a number gives approximately that many tiles across the
data, `"Hexagon"`/`"Rectangle"`/`"Triangle"` select the tile shape, a
`Quantity` length sets the tile diameter, and `{shape, size}` combines the
two.

```scrut
$ wo 'Head[GeoHistogram[{{40, -100}, {41, -101}, {34, -118}}, "Rectangle"]]'
GeoGraphics
```

```scrut
$ wo 'Head[GeoHistogram[{{40, -100}, {41, -101}}, Quantity[100, "Kilometers"]]]'
GeoGraphics
```

Weights come from an association `location -> weight` or from
`WeightedData[locations, weights]`:

```scrut
$ wo 'Head[GeoHistogram[<|GeoPosition[{40, -100}] -> 10, GeoPosition[{34, -118}] -> 1|>]]'
GeoGraphics
```

Options: the `GeoGraphics` options (`ImageSize`, `GeoRange`,
`GeoProjection`, `GeoGridLines`) plus `PlotLegends` (`Automatic` adds a
color-scale bar; the default is no legend). A third positional argument
selects how bin values are computed (`"Count"`, `"Probability"`,
`"Intensity"`, or `"PDF"`).
