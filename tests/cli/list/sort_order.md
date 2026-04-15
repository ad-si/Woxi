# Sorting and Ordering

Functions that rearrange or query the order of list elements.

## `FindPermutation`

Returns the permutation in `Cycles` form that maps one list to another.

```scrut
$ wo 'FindPermutation[{a, b, c}, {c, a, b}]'
Cycles[{{1, 2, 3}}]
```




## `InversePermutation`

Returns the inverse of a permutation given in list form.

```scrut
$ wo 'InversePermutation[{2, 3, 1}]'
{3, 1, 2}
```




## `MaximalBy`

Returns the element(s) maximising a key function.

```scrut
$ wo 'MaximalBy[{1, -3, 2, -4}, Abs]'
{-4}
```




## `MinimalBy`

Returns the element(s) minimising a key function.

```scrut
$ wo 'MinimalBy[{-1, 2, 3, -2}, Abs]'
{-1}
```




## `OrderedQ`

Tests whether a list is already in canonical order.

```scrut
$ wo 'OrderedQ[{1, 2, 3}]'
True
```

```scrut
$ wo 'OrderedQ[{3, 1, 2}]'
False
```




## `Ordering`

Returns the permutation that sorts a list.

```scrut
$ wo 'Ordering[{30, 10, 20}]'
{2, 3, 1}
```




## `PermutationListQ`

Tests whether a list represents a permutation of `{1, …, n}`.

```scrut
$ wo 'PermutationListQ[{3, 1, 2}]'
True
```

```scrut
$ wo 'PermutationListQ[{1, 2, 2}]'
False
```




## `Permutations`

Generates all permutations of a list.

```scrut
$ wo 'Permutations[{a, b, c}]'
{{a, b, c}, {a, c, b}, {b, a, c}, {b, c, a}, {c, a, b}, {c, b, a}}
```

```scrut
$ wo 'Permutations[{1, 2, 3}, {2}]'
{{1, 2}, {1, 3}, {2, 1}, {2, 3}, {3, 1}, {3, 2}}
```

```scrut
$ wo 'Permutations[{1, 2}, {1}]'
{{1}, {2}}
```

```scrut
$ wo 'Permutations[{a}]'
{{a}}
```

```scrut
$ wo 'Permutations[{}]'
{{}}
```

```scrut
$ wo 'Length[Permutations[Range[4]]]'
24
```




## `ReverseSort`

Sorts a list in descending order.

```scrut
$ wo 'ReverseSort[{3, 1, 4, 1, 5, 9}]'
{9, 5, 4, 3, 1, 1}
```




## `ReverseSortBy`

Sorts a list in descending order by a key function.

```scrut
$ wo 'ReverseSortBy[{1, -3, 2, -4}, Abs]'
{-4, -3, 2, 1}
```




## `Signature`

Returns `1` / `-1` / `0` for the signature of a permutation.

```scrut
$ wo 'Signature[{1, 2, 3}]'
1
```

```scrut
$ wo 'Signature[{2, 1, 3}]'
-1
```




## `SortBy`

Sorts elements of a list according to a function.

```scrut
$ wo 'SortBy[{3, 1, 2}, # &]'
{1, 2, 3}
```




