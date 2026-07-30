# `EdgeDetect`

Detects edges by thinning the derivative-of-Gaussian gradient to single-pixel
ridges, then keeping the ridges that clear a threshold.

The gradient of a single bright pixel peaks immediately either side of it, so
those are the pixels marked:

```scrut
$ wo 'ImageData[EdgeDetect[Image[{{0., 0., 0., 1., 0., 0., 0.}}]]]'
{{0, 0, 1, 0, 1, 0, 0}}
```

```scrut
$ wo 'ImageData[EdgeDetect[Image[{{0., 0., 0., 0., 0.}, {0., 1., 1., 1., 0.}, {0., 1., 1., 1., 0.}, {0., 1., 1., 1., 0.}, {0., 0., 0., 0., 0.}}]]]'
{{0, 0, 1, 0, 0}, {0, 1, 1, 1, 0}, {1, 1, 0, 1, 1}, {0, 1, 1, 1, 0}, {0, 0, 1, 0, 0}}
```

The second argument is the range over which the gradient is taken. A wider
range smooths further first, so the ridge moves outwards:

```scrut
$ wo 'ImageData[EdgeDetect[Image[{{0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.}}], 4]]'
{{0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0}}
```

Without a threshold the split is chosen from the gradient magnitudes, which
here keeps the strong bar and drops the faint one. An explicit threshold is
compared against the magnitude directly:

```scrut
$ wo 'ImageData[EdgeDetect[Image[{{0., 0., 0., 0.3, 0.3, 0.3, 0., 0., 0., 1., 1., 1., 0., 0., 0.}}]]]'
{{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0}}
```

```scrut
$ wo 'ImageData[EdgeDetect[Image[{{0., 0., 0., 0.3, 0.3, 0.3, 0., 0., 0., 1., 1., 1., 0., 0., 0.}}], 2, 0.1]]'
{{0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0}}
```

A range that is not a non-negative number is reported:

```scrut
$ wo 'EdgeDetect[Image[{{0., 1.}}], -1]'

EdgeDetect::bdrad: The specified radius -1 should be either a non-negative number or a list of 2 non-negative numbers.
EdgeDetect[-Image-, -1]
```
