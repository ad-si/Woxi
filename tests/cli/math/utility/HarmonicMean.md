# `HarmonicMean`

Returns the harmonic mean of a list.

```scrut
$ wo 'HarmonicMean[{5, 5, 5}]'
5
```

Symbolic lists are returned as `n / (x1^-1 + x2^-1 + ... + xn^-1)`:

```scrut
$ wo 'HarmonicMean[{a, b, c, d}]'
4/(a^(-1) + b^(-1) + c^(-1) + d^(-1))
```

For a list-of-lists (matrix), the harmonic mean is computed column-wise:

```scrut
$ wo 'HarmonicMean[{{1, 2}, {5, 10}, {5, 2}, {4, 8}}]'
{80/33, 160/49}
```
