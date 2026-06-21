# `PrimeQ`

Tests whether an integer is prime.

```scrut
$ wo 'PrimeQ[7]'
True
```

```scrut
$ wo 'PrimeQ[8]'
False
```

A complex argument is tested for Gaussian primality, and
`GaussianIntegers -> True` applies the Gaussian test to a real integer.

```scrut
$ wo 'PrimeQ[1 + I]'
True
```

```scrut
$ wo 'PrimeQ[5, GaussianIntegers -> True]'
False
```
