# `InversePermutation`

Returns the inverse of a permutation given in list form.

```scrut
$ wo 'InversePermutation[{2, 3, 1}]'
{3, 1, 2}
```

Also accepts a permutation in disjoint cycle form. Each cycle is reversed
and rotated so its smallest element comes first; cycles are then sorted by
their smallest element.

```scrut
$ wo 'InversePermutation[Cycles[{{3, 2, 5, 1}, {4, 7}}]]'
Cycles[{{1, 5, 2, 3}, {4, 7}}]
```

Involutions are their own inverse:

```scrut
$ wo 'InversePermutation[Cycles[{{1, 2}, {3, 4}}]]'
Cycles[{{1, 2}, {3, 4}}]
```

An entry appearing twice would leave a hole in the inverse, so it is reported
instead:

```scrut
$ wo 'InversePermutation[{1, 1}]'

InversePermutation::permlist: Invalid permutation list {1, 1}.
InversePermutation[{1, 1}]
```
