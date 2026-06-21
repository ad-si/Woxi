# `CoprimeQ`

Tests if two integers are coprime.

```scrut
$ wo 'CoprimeQ[3, 5]'
True
```

```scrut
$ wo 'CoprimeQ[6, 9]'
False
```

```scrut
$ wo 'CoprimeQ[14, 15]'
True
```

Gaussian integers are coprime when their gcd over `Z[i]` is a unit.

```scrut
$ wo 'CoprimeQ[1 + 2 I, 2 + I]'
True
```
