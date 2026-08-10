# `HistogramTransform`

Equalizes an image's histogram.

`HistogramTransform[image]` replaces each pixel value by its position in the
image's cumulative distribution, spreading the values over the whole range —
four dark pixels come back as an even ramp:

```scrut
$ wo 'ImageData[HistogramTransform[Image[{{0., 0.1, 0.2, 0.3}}]]]'
{{0., 0.3333333432674408, 0.6666666865348816, 1.}}
```

A histogram that is already flat is left alone:

```scrut
$ wo 'ImageData[HistogramTransform[Image[{{0., 0.25, 0.5, 0.75, 1.}}]]]'
{{0., 0.25, 0.5, 0.75, 1.}}
```

Each channel of a multichannel image is equalized on its own, so the same
value can land differently in different channels:

```scrut
$ wo 'ImageData[HistogramTransform[Image[{{{0., 0.5, 1.}, {0.5, 1., 0.}}}]]]'
{{{0., 0., 1.}, {1., 1., 0.}}}
```

A constant channel has nothing to spread out and passes through untouched:

```scrut
$ wo 'ImageData[HistogramTransform[Image[{{0.4, 0.4}}]]]'
{{0.4000000059604645, 0.4000000059604645}}
```

A first argument that is not an image is reported:

```scrut
$ wo 'HistogramTransform[5]'

HistogramTransform::imginv: Expecting an image or graphics instead of 5.
HistogramTransform[5]
```
