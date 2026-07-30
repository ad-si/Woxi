# `Sum`

Symbolic summation.

```scrut
$ wo 'Sum[k, {k, 1, 10}]'
55
```

```scrut
$ wo 'Sum[k^2, {k, 1, n}]'
(n*(1 + n)*(1 + 2*n))/6
```

```scrut
$ wo 'Sum[1/k^2, {k, 1, Infinity}]'
Pi^2/6
```

```scrut
$ wo 'Sum[(-1)^n x^(2n+1)/Factorial[2n+1], {n, 0, Infinity}]'
Sin[x]
```

A trailing option is not an iterator, so it leaves the sum alone:

```scrut
$ wo 'Sum[n, {n, 1, 10}, Method -> Automatic]'
55
```

`Regularization` assigns a value to a divergent sum. `"Dirichlet"` sums `n^k` as
the Dirichlet series it continues, giving `Zeta[-k]`:

```scrut
$ wo 'Sum[n, {n, 1, Infinity}, Regularization -> "Dirichlet"]'
-1/12
```

```scrut
$ wo 'Sum[n^3, {n, 1, Infinity}, Regularization -> "Dirichlet"]'
1/120
```

`"Abel"` applies to an alternating summand:

```scrut
$ wo 'Sum[(-1)^n, {n, 1, Infinity}, Regularization -> "Abel"]'
-1/2
```

```scrut
$ wo 'Sum[(-1)^n n, {n, 1, Infinity}, Regularization -> "Abel"]'
-1/4
```

A summand neither scheme reaches keeps the call — `1/n` lands on the pole of
`Zeta` at 1:

```scrut
$ wo 'Sum[1/n, {n, 1, Infinity}, Regularization -> "Dirichlet"]'
Sum[n^(-1), {n, 1, Infinity}, Regularization -> Dirichlet]
```
