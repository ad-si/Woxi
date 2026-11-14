# `FactorInteger`

Returns the prime factorization as {prime, exponent} pairs.

```scrut
$ wo 'FactorInteger[60]'
{{2, 2}, {3, 1}, {5, 1}}
```

```scrut
$ wo 'FactorInteger[100]'
{{2, 2}, {5, 2}}
```

```scrut
$ wo 'FactorInteger[17]'
{{17, 1}}
```

```scrut
$ wo 'FactorInteger[2^128 - 1]'
{{3, 1}, {5, 1}, {17, 1}, {257, 1}, {641, 1}, {65537, 1}, {274177, 1}, {6700417, 1}, {67280421310721, 1}}
```

With `GaussianIntegers -> True` the factorization is over the Gaussian
integers, even for large numbers:

```scrut
$ wo 'FactorInteger[10^89 + 81, GaussianIntegers -> True]'
{{-I, 1}, {132030265880273952462448610462422176060076884 + 287346496222912326279069848536662728000002975*I, 1}, {287346496222912326279069848536662728000002975 + 132030265880273952462448610462422176060076884*I, 1}}
```
