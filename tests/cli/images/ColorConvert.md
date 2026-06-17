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
