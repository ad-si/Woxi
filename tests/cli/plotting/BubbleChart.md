# `BubbleChart`

Bubble chart — each point is drawn as a disk whose size encodes a
third coordinate.

```scrut
$ wo 'Head[BubbleChart[{{1, 2, 3}, {4, 5, 6}}]]'
Graphics
```

Each bubble is a `Disk` whose semi-axes are a fraction of the data span
along each axis — a tenth of it for the largest `z`, with the area
proportional to `z`:

```scrut
$ wo 'Cases[BubbleChart[{{1, 5, 3}, {4, 6, 9}}], Disk[_, r_] :> Round[100 r], Infinity]'
{{30, 10}, {17, 6}}
```

`BubbleSizes -> {smin, smax}` sets the smallest and largest diameters as
fractions of the span:

```scrut
$ wo 'Cases[BubbleChart[{{1, 2, 3}, {2, 3, 4}, {4, 1, 1}}, BubbleSizes -> {0.05, 0.2}], Disk[_, r_] :> Round[1000 r], Infinity]'
{{300, 200}, {240, 160}, {75, 50}}
```

Options: `ChartLabels`, `ChartLegends`, `ChartStyle`, `BubbleSizes`,
`ImageSize`.
