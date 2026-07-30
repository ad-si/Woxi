# `ImageResize`

Resizes an image to specified dimensions.

```scrut
$ wo 'ImageDimensions[ImageResize[Image[{{0., 1.}, {1., 0.}}], {4, 6}]]'
{4, 6}
```

One side may be left to `Automatic`, and the aspect ratio decides the other:

```scrut
$ wo 'ImageDimensions[ImageResize[Image[{{0., 1.}, {1., 0.}}], {Automatic, 4}]]'
{4, 4}
```

A fractional size rounds, and never falls below a single pixel:

```scrut
$ wo 'ImageDimensions[ImageResize[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}], 0.5]]'
{1, 1}
```

A size that is not positive is reported and the call left alone:

```scrut
$ wo 'ImageResize[Image[{{0.}}], 0]'

ImageResize::imgrssz: The size 0 is not a valid image size specification.
ImageResize[-Image-, 0]
```
