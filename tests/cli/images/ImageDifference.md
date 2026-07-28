# `ImageDifference`

`ImageDifference[a, b]` gives the absolute difference of two images of the
same size, sample by sample.

```scrut
$ wo 'ImageData[ImageDifference[Image[{{0., 1.}}], Image[{{1., 0.}}]]]'
{{1., 1.}}
```

The samples an image actually holds are what get differenced, so the answer
carries their `Real32` precision rather than that of the literals written:

```scrut
$ wo 'ImageData[ImageDifference[Image[{{0.1, 0.7}}], Image[{{0.3, 0.2}}]]]'
{{0.20000001788139343, 0.5}}
```

Images of different shapes are refused:

```scrut
$ wo 'ImageDifference[Image[{{0., 1.}}], Image[{{1., 0.}, {0., 1.}}]]'

ImageDifference::imginvd: Expecting images of the same size in the input.
ImageDifference[-Image-, -Image-]
```
