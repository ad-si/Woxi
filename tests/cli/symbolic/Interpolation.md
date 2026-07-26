# `Interpolation`

Constructs a piecewise interpolating function from data points.

```scrut
$ wo 'Head[Interpolation[{{0, 1}, {1, 4}, {2, 9}, {3, 16}}]]'
InterpolatingFunction
```

An exact query point over exact data gives an exact value; machine data keeps
giving machine values:

```scrut
$ wo '{Interpolation[{1, 4, 9}][5/2], Interpolation[{1., 4., 9.}][3/2]}'
{25/4, 2.25}
```

Outside the data range the boundary piece is extended, with a warning:

```scrut {output_stream: combined}
$ wo 'Interpolation[{{0, 0}, {1, 1}, {2, 4}}][3]'

Interpolation::inhr: Requested order is too high; order has been reduced to {2}.

InterpolatingFunction::dmval: Input value {3} lies outside the range of data in the interpolating function. Extrapolation will be used.
9
```

`f'` differentiates the local polynomial piece and is itself an interpolating
function, so the prime may be applied directly to the call:

```scrut
$ wo "f = Interpolation[{{0, 0}, {1, 1}, {2, 4}}]; {f'[1], f''[1], Head[f']}"
{2, 2, InterpolatingFunction}
```
