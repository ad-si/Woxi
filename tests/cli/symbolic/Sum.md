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
