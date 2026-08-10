# `ColorBalance`

White-balances an image against a reference color.

`ColorBalance[image, col]` scales the three cone responses of every pixel so
that `col` lands exactly on white — the point of white balancing:

```scrut
$ wo 'ImageData[ColorBalance[Image[{{{0., 1., 0.}}}], Green]]'
{{{1., 1., 1.}}}
```

Balancing against white is therefore the identity:

```scrut
$ wo 'ImageData[ColorBalance[Image[{{{0.2, 0.4, 0.6}}}], White]]'
{{{0.20000000298023224, 0.4000000059604645, 0.6000000238418579}}}
```

Correcting for a green cast pulls the green channel down relative to the other
two, so a neutral gray comes back magenta:

```scrut
$ wo 'Round[100 ImageData[ColorBalance[Image[{{{0.5, 0.5, 0.5}}}], Green]]]'
{{{77, 42, 100}}}
```

A rule sends the reference somewhere other than white:

```scrut
$ wo 'Round[10 ImageData[ColorBalance[Image[{{{0., 1., 0.}}}], Green -> Red]]]'
{{{10, 0, 0}}}
```

A single-channel image carries no color to rebalance and comes back as it went
in:

```scrut
$ wo 'ImageData[ColorBalance[Image[{{0.3, 0.7}}], Green]]'
{{0.30000001192092896, 0.699999988079071}}
```

A first argument that is not an image is reported:

```scrut
$ wo 'ColorBalance[5, Green]'

ColorBalance::imginv: Expecting an image or graphics instead of 5.
ColorBalance[5, RGBColor[0, 1, 0]]
```
