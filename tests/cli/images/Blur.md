# `Blur`

Applies Gaussian blur to an image.

`Blur[image, r]` is `GaussianFilter[image, r]`: a discrete Gaussian of
half-width `Ceiling[r]` and standard deviation `r/2`, read off here from a
single bright pixel.

```scrut
$ wo 'Round[1000 ImageData[Blur[Image[{{0., 0., 0., 1., 0., 0., 0.}}], 1]]]'
{{0, 0, 99, 801, 99, 0, 0}}
```

A pair of radii blurs each axis by a different amount, rows first:

```scrut
$ wo 'Round[1000 ImageData[Blur[Image[{{0., 0., 0.}, {0., 1., 0.}, {0., 0., 0.}}], {0, 1}]]]'
{{0, 0, 0}, {99, 801, 99}, {0, 0, 0}}
```

Blur never reaches further than half way across the image, so along an axis
of `n` pixels the radius is capped at `Ceiling[n/2]`:

```scrut
$ wo 'ImageData[Blur[Image[{{1., 0., 0., 0., 0., 0.}}], 10]] == ImageData[Blur[Image[{{1., 0., 0., 0., 0., 0.}}], 3]]'
True
```

A radius that is not a non-negative number is reported:

```scrut
$ wo 'Blur[Image[{{0.1, 0.2}}], -1]'

Blur::bdrad: The specified radius -1 should be either a non-negative number or a list of 2 non-negative numbers.
Blur[-Image-, -1]
```
