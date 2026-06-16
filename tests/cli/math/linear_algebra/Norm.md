# `Norm`

Vector or matrix norm (Euclidean by default).

```scrut
$ wo 'Norm[{3, 4}]'
5
```

```scrut
$ wo 'Norm[{1, 2, 2}]'
3
```

A second argument gives the p-norm `(Sum Abs[x]^p)^(1/p)`.

```scrut
$ wo 'Norm[{3, 4}, 4]'
337^(1/4)
```

It works for symbolic entries and a symbolic exponent too.

```scrut
$ wo 'Norm[{a, b, c}, p]'
(Abs[a]^p + Abs[b]^p + Abs[c]^p)^p^(-1)
```
