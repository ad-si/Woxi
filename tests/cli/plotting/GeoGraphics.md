# `GeoGraphics`

Renders a geographic map. Woxi draws an equirectangular map using the
embedded lowest-resolution (1:110m) [Natural Earth](https://www.naturalearthdata.com)
country data, then overlays geographic primitives. The view auto-zooms to
fit the data.

Supported primitives inside the content list:

- `GeoMarker[pos]` — a map pin at `pos`
- `Point[pos]` — a filled dot at `pos`

together with color directives (`Red`, `RGBColor[...]`, …) and `PointSize`.
A position is `GeoPosition[{lat, lon}]`, a bare `{lat, lon}` pair (latitude
first), or a geographic `Entity[…]` (see below).

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

### Drawing primitives

Besides `GeoMarker` and `Point`, `GeoGraphics` draws connecting paths, filled
regions and geodesic circles. Each renders as part of the `Graphics` image:

```scrut
$ wo 'Head[GeoGraphics[GeoPath[{{40, -100}, {34, -118}, {47, -122}}]]]'
Graphics
```

```scrut
$ wo 'Head[GeoGraphics[{Red, GeoDisk[{40, -100}, Quantity[500, "Kilometers"]]}]]'
Graphics
```

On their own (outside `GeoGraphics`) these primitives stay symbolic:

```scrut
$ wo 'GeoPath[{{1, 2}, {3, 4}}]'
GeoPath[{{1, 2}, {3, 4}}]
```

A named country is highlighted over the basemap:

```scrut
$ wo 'Head[GeoGraphics[Entity["Country", "France"]]]'
Graphics
```

### Geographic entities

A position may be given as an `Entity[…]`, resolved to coordinates through the
[keshvar](https://crates.io/crates/keshvar) gazetteer. `Entity["Country", name]`
resolves to the country's center, and `Entity["City", {city, region, country}]`
to the center of its administrative subdivision (the gazetteer has no
city-level coordinates, so the `region` — e.g. a state or province — gives the
finest available placement):

```scrut
$ wo 'Head[GeoGraphics[{Red, GeoMarker[Entity["City", {"Munich", "Bavaria", "Germany"}]]}, GeoRange -> Quantity[50, "Kilometers"]]]'
Graphics
```

### Options

- **`ImageSize`** — map width in pixels (the height follows the data aspect
  ratio).
- **`GeoRange`** — `"World"` for the whole globe, `All`/`Automatic` to fit the
  data (the default), a length `Quantity` (e.g. `Quantity[50, "Kilometers"]`)
  for a disk of that radius around the data, or an explicit
  `{{latmin, latmax}, {lonmin, lonmax}}` window.
- **`GeoProjection`** — `"Equirectangular"` (the default) or `"Mercator"`.
- **`GeoGridLines`** — `Automatic` draws a latitude/longitude graticule;
  `None` (the default) omits it.

### Geodesy functions

Distances, bearings and destinations use Karney's geodesic on the GRS80
ellipsoid (the Wolfram `ITRF00` model). Values agree with the Wolfram Language
to about twelve significant figures.

```scrut
$ wo 'GeoDirection[{0, 0}, {0, 10}]'
Quantity[90., AngularDegrees]
```

```scrut
$ wo 'GeoBounds[{GeoPosition[{40, -100}], GeoPosition[{34, -118}]}]'
{{34., 40.}, {-118., -100.}}
```

`GeoNearest["Country", pos]` returns the country containing a position:

```scrut
$ wo 'GeoNearest["Country", GeoPosition[{48.85, 2.35}]]'
{Entity[Country, France]}
```

### Notes

Auto-zoom fits a single point to a square regional window and several points to
their bounding box with padding. The basemap is fixed at 1:110m resolution, so
very tight zooms show coarse coastlines.

`GeoDistance`, `GeoDirection` and `GeoDestination` differ from the Wolfram
Language in the last few displayed digits because Wolfram uses its own geodesic
implementation. `GeoRegionValuePlot` shades each country by its value and draws
a color-scale legend on the right; it returns a single `Graphics` object (the
Wolfram Language instead wraps a separate legend with `Legended`).
