# `RandomImage`

Generates a random image with given pixel range and dimensions.

```scrut
$ wo 'ImageDimensions[RandomImage[{0.2, 0.8}, {2, 3}]]'
{2, 3}
```

A range with nothing in it has no pixel values to draw,
so it is reported rather than sampled:

```scrut
$ wo 'RandomImage[0, {2, 2}]'

RandomImage::bddist: The specified random distribution UniformDistribution\[\{0, 0\}\] should generate a real number or a list of real numbers\.\s? (regex)
RandomImage[0, {2, 2}]
```
