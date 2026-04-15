# Grouping and Partitioning

Group list elements into blocks, bins, or runs.

## `BinCounts`

Counts how many elements fall into evenly-spaced bins.

```scrut
$ wo 'BinCounts[{1, 2, 2, 3, 4, 4, 4, 5}, 1]'
{0, 1, 2, 1, 3, 1}
```




## `Downsample`

Keeps every `n`-th element of a list.

```scrut
$ wo 'Downsample[Range[10], 2]'
{1, 3, 5, 7, 9}
```




## `Gather`

Groups identical elements together, maintaining order of first appearance.

```scrut
$ wo 'Gather[{1, 1, 2, 2, 1}]'
{{1, 1, 1}, {2, 2}}
```

```scrut
$ wo 'Gather[{a, b, a, c, b}]'
{{a, a}, {b, b}, {c}}
```




## `GatherBy`

Groups elements by applying a function, maintaining order of first appearance.

```scrut
$ wo 'GatherBy[{1, 2, 3, 4, 5}, EvenQ]'
{{1, 3, 5}, {2, 4}}
```

```scrut
$ wo 'GatherBy[{-2, -1, 0, 1, 2}, Sign]'
{{-2, -1}, {0}, {1, 2}}
```




## `GroupBy`

Groups elements of a list according to a function.

```scrut
$ wo 'GroupBy[{{a, b}, {a, c}, {b, c}}, First]'
<|a -> {{a, b}, {a, c}}, b -> {{b, c}}|>
```




## `Partition`


Breaks a list into smaller sublists.

```scrut
$ wo 'Partition[{1, 2, 3, 4}, 2]'
{{1, 2}, {3, 4}}
```




## `Split`

Splits list at boundaries where consecutive elements differ.

```scrut
$ wo 'Split[{1, 1, 2, 2, 3}]'
{{1, 1}, {2, 2}, {3}}
```

```scrut
$ wo 'Split[{a, a, b, c, c, c}]'
{{a, a}, {b}, {c, c, c}}
```




## `SplitBy`

Splits list at boundaries where a function changes value.

```scrut
$ wo 'SplitBy[{1, 2, 3, 4, 5}, EvenQ]'
{{1}, {2}, {3}, {4}, {5}}
```




## `TakeDrop`

Splits a list at a given position, returning `{Take[l, n], Drop[l, n]}`.

```scrut
$ wo 'TakeDrop[{a, b, c, d, e}, 2]'
{{a, b}, {c, d, e}}
```




## `TakeList`

Splits a list into chunks of given lengths.

```scrut
$ wo 'TakeList[{a, b, c, d, e}, {2, 3}]'
{{a, b}, {c, d, e}}
```




