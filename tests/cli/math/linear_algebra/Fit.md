# `Fit`

Performs a least-squares fit of data to a linear combination of basis functions.

```scrut
$ wo 'Fit[{2, 4, 6}, {x}, x]'
2\.(0+\d*)?\*x (regex)
```

When the design matrix does not have full column rank there is no unique
least-squares solution, and the minimum-norm one is returned. Fitting a line
through a single point gives the shortest coefficient vector among the
infinitely many lines through it:

```scrut
$ wo 'Fit[{{1, 1}}, {1, x}, x]'
0.5 + 0.5*x
```

Repeated abscissae are the same situation — the fit passes through their mean:

```scrut
$ wo 'Fit[{{1, 1}, {1, 2}}, {1, x}, x]'
0.75 + 0.75*x
```

```scrut
$ wo 'Fit[{{1, 1}, {1, 2}, {1, 3}}, {1, x}, x]'
1. + 1.*x
```
