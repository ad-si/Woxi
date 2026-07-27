# `ColorData`

Named color data. Without arguments it lists the categories:

```scrut
$ wo 'ColorData[]'
{Gradients, Indexed, Named, Physical}
```

`ColorData[n, k]` gives the k-th color of indexed scheme `n`. Scheme 1 is
generated rather than tabulated — its hues advance by an irrational step, so
it never repeats and takes any index:

```scrut
$ wo 'Head[ColorData[1, 1]]'
RGBColor
```

```scrut
$ wo 'Round[List @@ ColorData[1, 4], 0.000001]'
{0.24, 0.6, 0.33692}
```

The tabulated schemes start over past their last color:

```scrut
$ wo 'List @@ ColorData[2, 1]'
{0.8588235294117647, 0.00784313725490196, 0.00784313725490196}
```

```scrut
$ wo 'List @@ ColorData[2, 10]'
{0.8588235294117647, 0.00784313725490196, 0.00784313725490196}
```

`ColorData[n, "ColorList"]` gives the scheme's colors:

```scrut
$ wo 'Length[ColorData[3, "ColorList"]]'
10
```
