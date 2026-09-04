# `HistogramTransform`

Equalizes an image's histogram.

`HistogramTransform[image]` replaces each pixel value by its position in the
image's cumulative distribution, spreading the values over the whole range —
four dark pixels come back as an even ramp:

```scrut
$ wo 'ImageData[HistogramTransform[Image[{{0., 0.1, 0.2, 0.3}}]]]'
{{0.12352941185235977, 0.3745098114013672, 0.6254901885986328, 0.8764705657958984}}
```

A histogram that is already flat is left alone, so an image that already uses
every one of the 256 display levels once comes back unchanged:

```scrut
$ wo 'Max[Abs[ImageData[HistogramTransform[Image[{N[Range[0, 255]/255]}]]] - {N[Range[0, 255]/255]}]] < 0.001'
True
```

Each channel of a multichannel image is equalized on its own, so the same
value can land differently in different channels:

```scrut
$ wo 'ImageData[HistogramTransform[Image[{{{0., 0.5, 1.}, {0.5, 1., 0.}}}]]]'
{{{0.24901960790157318, 0.24901960790157318, 0.7509803771972656}, {0.7509803771972656, 0.7509803771972656, 0.24901960790157318}}}
```

Every pixel of one value lands in the middle of the band the equalization
gives that value, so repeated values stay together and share the space they
take up in the histogram:

```scrut
$ wo 'ImageData[HistogramTransform[Image[{{0., 0., 0., 1.}}]]]'
{{0.3745098114013672, 0.3745098114013672, 0.3745098114013672, 0.8764705657958984}}
```

A first argument that is not an image is reported:

```scrut
$ wo 'HistogramTransform[5]'

HistogramTransform::imginv: 5 should be an image, a dataset or a list of datasets.
HistogramTransform[5]
```
