# `ImagePad`

Pads an image on each side.
A single extent pads all four sides, with black as the default fill:

```scrut
$ wo 'ImageData[ImagePad[Image[{{0.5}}], 1]]'
{{0., 0., 0.}, {0., 0.5, 0.}, {0., 0., 0.}}
```

Per-side extents are given as `{{left, right}, {bottom, top}}`:

```scrut
$ wo 'ImageDimensions[ImagePad[Image[{{0.1, 0.2}, {0.3, 0.4}}], {{1, 0}, {0, 2}}]]'
{3, 4}
```

A third argument gives the fill: a constant value,
or one of the edge-extension methods
`"Fixed"`, `"Periodic"` and `"Reflected"`.

```scrut
$ wo 'Round[ImageData[ImagePad[Image[{{0.2, 0.4}}], 1, "Fixed"]], 0.001]'
{{0.2, 0.2, 0.4, 0.4}, {0.2, 0.2, 0.4, 0.4}, {0.2, 0.2, 0.4, 0.4}}
```

A negative extent trims that side instead of growing it.
