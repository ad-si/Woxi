# `ColorNegate`

Negates colors in an image.

```scrut
$ wo 'ColorNegate[Yellow] == Blue'
True
```

Negation stays in the input's color space.

```scrut
$ wo 'ColorNegate[Hue[0.5]]'
Hue[0., 1., 1.]
```

```scrut
$ wo 'ColorNegate[CMYKColor[0, 1, 1, 0]]'
CMYKColor[1., 0., 0., 0.]
```
