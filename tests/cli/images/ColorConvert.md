# `ColorConvert`

Convert between color spaces.

```scrut
$ wo 'ColorConvert[RGBColor[1, 0, 0], "CMYK"]'
CMYKColor[0., 1., 1., 0.]
```

```scrut
$ wo 'ColorConvert[RGBColor[1, 0, 0], "HSB"]'
Hue[0., 1., 1.]
```

`Hue` and `CMYKColor` are accepted as inputs too.

```scrut
$ wo 'ColorConvert[CMYKColor[0, 1, 1, 0], "RGB"]'
RGBColor[1., 0., 0.]
```

A converted image records the space it was converted to, where one that was
never converted has none:

```scrut
$ wo 'ImageColorSpace[ColorConvert[Image[{{0., 1.}}], "RGB"]]'
RGB
```

```scrut
$ wo 'ImageColorSpace[Image[{{0., 1.}}]]'
Automatic
```

Images convert between the same spaces the color directives do, pixel by
pixel. In `"HSB"` red is hue 0 and green is hue 1/3, both fully saturated:

```scrut
$ wo 'ImageData[ColorConvert[Image[{{{1., 0., 0.}, {0., 1., 0.}}}], "HSB"]]'
{{{0., 1., 1.}, {0.3333333432674408, 1., 1.}}}
```

```scrut
$ wo 'ImageData[ColorConvert[Image[{{{1., 0., 0.}}}], "CMYK"]]'
{{{0., 1., 1., 0.}}}
```

A grayscale image is read as the gray RGB it stands for, so it gains the extra
channels:

```scrut
$ wo 'ImageChannels[ColorConvert[Image[{{0.5}}], "CMYK"]]'
4
```

An alpha channel is not a color channel and rides along unchanged:

```scrut
$ wo 'ImageData[ColorConvert[Image[{{{1., 0., 0., 0.25}}}], "HSB"]]'
{{{0., 1., 1., 0.25}}}
```
