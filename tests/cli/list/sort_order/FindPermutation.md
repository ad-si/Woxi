# `FindPermutation`

Returns the permutation in `Cycles` form that maps one list to another.

```scrut
$ wo 'FindPermutation[{a, b, c}, {c, a, b}]'
Cycles[{{1, 2, 3}}]
```

Repeated elements are paired off one at a time, so a list of equal elements
needs no rearranging at all:

```scrut
$ wo 'FindPermutation[{1, 1}]'
Cycles[{}]
```
