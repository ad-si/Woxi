# `Haloing`

Graphics directive that draws a contrasting outline (halo) behind
subsequent primitives so they stay visible against any background.

`Haloing[]` uses a white halo, `Haloing[color]` sets the halo color, and
`Haloing[color, r]` gives it a pixel radius `r`. As a directive it is inert
outside of a graphics context:

```scrut
$ wo 'Haloing[Red]'
Haloing[RGBColor[1, 0, 0]]
```
