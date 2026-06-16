# `Entropy`

Returns the Shannon information entropy of the categorical distribution of a
list's elements. Without a base the entropy is measured in nats (natural log).

```scrut
$ wo 'Entropy[{1, 2, 3, 4}]'
Log[4]
```

`Entropy[b, list]` measures the entropy in base `b`. When the category counts
make the rebased logarithms land on exact powers of the base, the result
collapses to an integer.

```scrut
$ wo 'Entropy[2, {1, 1, 2, 2}]'
1
```

```scrut
$ wo 'Entropy[2, {a, b, c, d}]'
2
```

Otherwise it stays as a logarithm ratio.

```scrut
$ wo 'Entropy[2, {a, a, b, b, c, c}]'
-1 + Log[6]/Log[2]
```
