# `PermutationProduct`

Composes permutations, applying them left to right. With permutation lists
(image lists) the result is a permutation list; with `Cycles` objects the
result is in `Cycles` form.

```scrut
$ wo 'PermutationProduct[{2, 1, 3}, {1, 3, 2}]'
{3, 1, 2}
```

```scrut
$ wo 'PermutationProduct[{2, 3, 1}, {2, 3, 1}, {2, 3, 1}]'
{1, 2, 3}
```

A shorter permutation list fixes points beyond its length:

```scrut
$ wo 'PermutationProduct[{2, 1}, {2, 3, 1}]'
{3, 2, 1}
```

`Cycles` arguments compose to a `Cycles` object:

```scrut
$ wo 'PermutationProduct[Cycles[{{1, 2}}], Cycles[{{2, 3}}]]'
Cycles[{{1, 3, 2}}]
```
