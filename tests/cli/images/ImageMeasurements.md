# `ImageMeasurements`

`ImageMeasurements[image, property]` gives a named measurement of an image.
The statistics are taken over the samples, with the sample (n − 1) divisor
for the deviation:

```scrut
$ wo 'ImageMeasurements[Image[{{0., 1.}, {0.5, 0.25}}], "Mean"]'
0.4375
```

A colour image is measured channel by channel:

```scrut
$ wo 'ImageMeasurements[Image[{{{1., 0., 0.}, {0., 1., 0.}}}], "Mean"]'
{0.5, 0.5, 0.}
```

`Entropy` and `Energy` are taken over the histogram of distinct *pixels*, so a
colour pixel counts once however many channels it carries — two distinct
pixels give `Log[2]`:

```scrut
$ wo 'ImageMeasurements[Image[{{{1., 0., 0.}, {0., 1., 0.}}}], "Entropy"]'
0.6931471805599453
```

A list of names gives a list of measurements, and `"Properties"` names the
ones it knows:

```scrut
$ wo 'ImageMeasurements[Image[{{0., 1.}, {0.5, 0.25}}], {"Mean", "Total"}]'
{0.4375, 1.75}
```

```scrut
$ wo 'Take[ImageMeasurements[Image[{{0., 1.}}], "Properties"], 3]'
{AspectRatio, Channels, ColorSpace}
```
