# `ColorBalance`

White-balances an image against a reference color.

`ColorBalance[image, col]` scales the three cone responses of every pixel so
that the color cast `col` stands for is taken out of the picture. It is the
*chromaticity* of `col` that is divided away, not `col` itself, so the
picture keeps its brightness and `col` lands on the neutral gray of its own
luminance rather than on white:

```scrut
$ wo 'Round[100 ImageData[ColorBalance[Image[{{{0., 1., 0.}}}], Green]]]'
{{{86, 86, 86}}}
```

Balancing against white is therefore the identity:

```scrut
$ wo 'Round[1000 ImageData[ColorBalance[Image[{{{0.2, 0.4, 0.6}}}], White]]]'
{{{200, 400, 600}}}
```

Correcting for a green cast pulls the green channel down relative to the other
two, so a neutral gray comes back magenta:

```scrut
$ wo 'Round[100 ImageData[ColorBalance[Image[{{{0.5, 0.5, 0.5}}}], Green]]]'
{{{66, 36, 100}}}
```

A rule sends the reference somewhere other than white:

```scrut
$ wo 'Round[10 ImageData[ColorBalance[Image[{{{0., 1., 0.}}}], Green -> Red]]]'
{{{10, 0, 0}}}
```

A single-channel image has no color channels to rebalance, but the balancing
gives it some — it is taken up to RGB first:

```scrut
$ wo 'Round[100 ImageData[ColorBalance[Image[{{0.3, 0.7}}], Green]]]'
{{{40, 21, 86}, {92, 51, 100}}}
```

A first argument that is not an image is reported:

```scrut
$ wo 'ColorBalance[5, Green]'

ColorBalance::imginv: Expecting an image or graphics instead of 5.
ColorBalance[5, RGBColor[0, 1, 0]]
```
