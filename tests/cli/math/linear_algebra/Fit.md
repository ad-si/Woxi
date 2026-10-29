# `Fit`

Performs a least-squares fit of data to a linear combination of basis functions.

```scrut
$ wo 'Fit[{2, 4, 6}, {x}, x]'
2\.(0+\d*)?\*x (regex)
```

When the design matrix does not have full column rank there is no unique
least-squares solution, and the minimum-norm one is returned (the norm taken
after scaling each basis function's column of values to unit length). Fitting a line
through a single point gives the shortest coefficient vector among the
infinitely many lines through it:

```scrut
$ wo 'Fit[{{1, 1}}, {1, x}, x]'
0\.(5(0+\d*)?|4999999999\d*) \+ 0\.(5(0+\d*)?|4999999999\d*)\*x (regex)
```

Repeated abscissae are the same situation — the fit passes through their mean:

```scrut
$ wo 'Fit[{{1, 1}, {1, 2}}, {1, x}, x]'
0\.75(0+\d*)? \+ 0\.7(5(0+\d*)?|4999999999\d*)\*x (regex)
```

```scrut
$ wo 'Fit[{{1, 1}, {1, 2}, {1, 3}}, {1, x}, x]'
(1\.(0+\d*)?|0\.9999999999999\d*) \+ (1\.(0+\d*)?|0\.9999999999999\d*)\*x (regex)
```

The scaling shows when the basis columns differ in length: at `x = 2` the
column of `x` is twice that of `1`, so the fit is `1 + x/2` rather than the
unscaled shortest `2/5 + 4/5 x`:

```scrut
$ wo 'Fit[{{2, 1}, {2, 2}, {2, 3}}, {1, x}, x]'
(1\.(0+\d*)?|0\.9999999999999\d*) \+ (0\.5(0+\d*)?|0\.4999999999999\d*)\*x (regex)
```
