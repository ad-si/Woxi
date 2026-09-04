# `ImageCorrelate`

Correlates an image with a kernel.

Correlation multiplies the kernel with the neighborhood *in the same
orientation*, so filtering a single lit pixel replays the kernel backwards:

```scrut
$ wo 'ImageData[ImageCorrelate[Image[{{0., 0., 1., 0., 0.}}], {{1, 10, 100}}]]'
{{0., 100., 10., 1., 0.}}
```

`ImageConvolve` reflects the kernel first, which is the one thing that tells
the two apart — its impulse response is the kernel the right way round:

```scrut
$ wo 'ImageData[ImageConvolve[Image[{{0., 0., 1., 0., 0.}}], {{1, 10, 100}}]]'
{{0., 1., 10., 100., 0.}}
```

The kernel entries are reflected however they are written, so a kernel of
rationals behaves like any other:

```scrut
$ wo 'Round[111 ImageData[ImageCorrelate[Image[{{0., 0., 1., 0., 0.}}], {{1, 10, 100}}/111]]]'
{{0, 100, 10, 1, 0}}
```

A kernel that is its own reflection makes them agree, so the usual
`GaussianMatrix` / `BoxMatrix` / `DiskMatrix` kernels can be used with either:

```scrut
$ wo 'ImageData[ImageCorrelate[Image[{{0., 1., 0.}}], {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}]] == ImageData[ImageConvolve[Image[{{0., 1., 0.}}], {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}]]'
True
```

The boundary is extended by repeating the edge pixels, and the image keeps its
dimensions and channel count:

```scrut
$ wo 'ImageDimensions[ImageCorrelate[Image[{{{1., 0., 0.}, {0., 1., 0.}}}], BoxMatrix[1]/9]]'
{2, 1}
```

A first argument that is not an image is reported:

```scrut
$ wo 'ImageCorrelate[5, {{1}}]'

ImageCorrelate::imginv: Expecting an image or graphics instead of 5.
ImageCorrelate[5, {{1}}]
```
