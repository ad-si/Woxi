# `PermutationPower`

Power of a permutation.

```scrut
$ wo 'PermutationPower[{2, 3, 1}, 3]'
{1, 2, 3}
```

Also accepts a permutation in disjoint cycle form. The result is returned
in canonical `Cycles[...]` form (each cycle rotated so its smallest
element comes first, cycles sorted by smallest element, fixed points
dropped). Negative exponents apply the inverse permutation.

```scrut
$ wo 'PermutationPower[Cycles[{{4, 2, 5}, {6, 3, 1, 7}}], 6]'
Cycles[{{1, 6}, {3, 7}}]
```

```scrut
$ wo 'PermutationPower[Cycles[{{4, 2, 5}, {6, 3, 1, 7}}], -2]'
Cycles[{{1, 6}, {2, 5, 4}, {3, 7}}]
```
