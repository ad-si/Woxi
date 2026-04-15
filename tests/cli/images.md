# Images

Woxi provides a minimal set of image-construction and -introspection
functions from the Wolfram Language. Most examples below use a 2×2
matrix of greyscale values as a stand-in for a real image file.


## `Image`

Wraps a list of numeric values as an image expression.

```scrut
$ wo 'Head[Image[{{0, 1}, {1, 0}}]]'
Image
```


## `ImageQ`

Tests whether an expression is an `Image`.

```scrut
$ wo 'ImageQ[Image[{{0, 1}, {1, 0}}]]'
True
```

```scrut
$ wo 'ImageQ[5]'
False
```


## `ImageData`

Returns the underlying pixel data.

```scrut
$ wo 'ImageData[Image[{{0, 1}, {1, 0}}]]'
{{0., 1.}, {1., 0.}}
```


## `ImageDimensions`

Returns the `{width, height}` of an image in pixels.

```scrut
$ wo 'ImageDimensions[Image[{{0, 1}, {1, 0}}]]'
{2, 2}
```

## Other implemented image functions

Woxi recognizes the following additional image functions — consult
[`functions.csv`](https://github.com/ad-si/Woxi/blob/main/functions.csv)
for the current status and
[the Wolfram Language reference](https://reference.wolfram.com/language/)
for usage details:

`ImageAdd`, `ImageAdjust`, `ImageApply`, `ImageAssemble`,
`ImageChannels`, `ImageColorSpace`, `ImageCompose`, `ImageConvolve`,
`ImageCorrelate`, `ImageCrop`, `ImageFilter`, `ImageHistogram`,
`ImageMultiply`, `ImagePad`, `ImageResize`, `ImageRotate`,
`ImageSubtract`, `ImageType`, `Binarize`, `Blur`, `ColorNegate`,
`Dilation`, `Erosion`, `GaussianFilter`, `MaxFilter`, `MeanFilter`,
`MedianFilter`, `MinFilter`, `Opening`, `Closing`, `Sharpen`,
`Thinning`.
