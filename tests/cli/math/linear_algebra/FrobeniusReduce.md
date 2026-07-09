# `FrobeniusReduce`

Returns the Frobenius (rational canonical) form of a square matrix: the block
diagonal of companion matrices of the invariant factors of `xI - m`, computed
exactly without factoring the characteristic polynomial.

```scrut
$ wo 'FrobeniusReduce[{{1, 1}, {0, 1}}]'
{{0, -1}, {1, 2}}
```

Distinct eigenvalues merge into a single companion block of the
characteristic polynomial:

```scrut
$ wo 'FrobeniusReduce[{{2, 0}, {0, 3}}]'
{{0, -6}, {1, 5}}
```

With several invariant factors, the smaller (dividing) factor comes first:

```scrut
$ wo 'FrobeniusReduce[{{2, 0, 0}, {0, 2, 0}, {0, 0, 3}}]'
{{2, 0, 0}, {0, 0, -6}, {0, 1, 5}}
```

Rational and complex-rational matrices stay exact:

```scrut
$ wo 'FrobeniusReduce[{{1/2, 1/3}, {1/4, 1/5}}]'
{{0, -1/60}, {1, 7/10}}
```

```scrut
$ wo 'FrobeniusReduce[{{I, 0}, {0, 2}}]'
{{0, -2*I}, {1, 2 + I}}
```

With `Modulus -> p` the reduction happens over the prime field GF(p):

```scrut
$ wo 'FrobeniusReduce[{{1, 2}, {3, 4}}, Modulus -> 5]'
{{0, 2}, {1, 0}}
```
