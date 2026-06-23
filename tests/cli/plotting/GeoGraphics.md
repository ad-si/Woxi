# `GeoGraphics`

Renders a geographic map. Woxi draws an equirectangular map using the
embedded lowest-resolution (1:110m) [Natural Earth](https://www.naturalearthdata.com)
country data, then overlays geographic primitives. The view auto-zooms to
fit the data.

Supported primitives inside the content list:

- `GeoMarker[pos]` — a map pin at `pos`
- `Point[pos]` — a filled dot at `pos`

together with color directives (`Red`, `RGBColor[...]`, …) and `PointSize`.
A position is `GeoPosition[{lat, lon}]` or a bare `{lat, lon}` pair
(latitude first).

Like `Graphics`, the result is a `Graphics` object that renders as an SVG
image in the playground, Jupyter, or Woxi Studio.

```scrut
$ wo 'Head[GeoGraphics[{Red, PointSize[Large], GeoMarker[GeoPosition[{-26.2041, 28.0473}]]}]]'
Graphics
```

`GeoMarker` on its own stays symbolic until placed inside `GeoGraphics`:

```scrut
$ wo 'GeoMarker[GeoPosition[{-26.2041, 28.0473}]]'
GeoMarker[GeoPosition[{-26.2041, 28.0473}]]
```

### Options

- **`ImageSize`** — map width in pixels (the height follows the data aspect
  ratio).
- **`GeoRange`** — `"World"` for the whole globe, `All`/`Automatic` to fit the
  data (the default), or an explicit `{{latmin, latmax}, {lonmin, lonmax}}`
  window.

### Notes

Auto-zoom fits a single point to a square regional window and several points to
their bounding box with padding. The basemap is fixed at 1:110m resolution, so
very tight zooms show coarse coastlines.
