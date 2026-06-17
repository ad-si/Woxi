# `HypergeometricDistribution`

Hypergeometric probability distribution: `n` draws without replacement from a
population of `nt` items containing `ns` successes.

```scrut
$ wo 'HypergeometricDistribution[n, ns, nt]'
HypergeometricDistribution[n, ns, nt]
```

The mean is `(n*ns)/nt`:

```scrut
$ wo 'Mean[HypergeometricDistribution[n, ns, nt]]'
(n*ns)/nt
```

Concrete parameters collapse to an exact number:

```scrut
$ wo 'Mean[HypergeometricDistribution[5, 10, 50]]'
1
```
