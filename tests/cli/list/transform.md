# Transforming Lists

Higher-order functions for applying other functions to list elements, and bulk reshaping operations.

## `Apply`

Applies a function to list elements as arguments.

```scrut
$ wo 'Apply[Plus, {1, 2, 3}]'
6
```

```scrut
$ wo 'Apply[Times, {2, 3, 4}]'
24
```




## `ArrayFlatten`

Flattens a nested "block matrix" into a plain matrix.

```scrut
$ wo 'ArrayFlatten[{{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}}]'
{{1, 2, 5, 6}, {3, 4, 7, 8}}
```




## `ArrayPad`

Pads a list on both sides with a filler value (`0` by default).

```scrut
$ wo 'ArrayPad[{1, 2, 3}, 2]'
{0, 0, 1, 2, 3, 0, 0}
```

```scrut
$ wo 'ArrayPad[{1, 2, 3}, 2, 0]'
{0, 0, 1, 2, 3, 0, 0}
```




## `ArrayReshape`

Reshapes a flat list of values into a multi-dimensional array.

```scrut
$ wo 'ArrayReshape[Range[12], {3, 4}]'
{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}
```




## `ArrayRules`

Returns a list of rules describing the non-default entries of a sparse array.

```scrut
$ wo 'ArrayRules[SparseArray[{1 -> a, 3 -> c}, 4]]'
{{1} -> a, {3} -> c, {_} -> 0}
```




## `BlockMap`

Applies a function to consecutive blocks of a list.

```scrut
$ wo 'BlockMap[f, Range[6], 2]'
{f[{1, 2}], f[{3, 4}], f[{5, 6}]}
```




## `ComposeList`

Returns the list `{x, f[x], g[f[x]], h[g[f[x]]], …}` for a list of functions.

```scrut
$ wo 'ComposeList[{f, g, h}, x]'
{x, f[x], g[f[x]], h[g[f[x]]]}
```




## `Composition`

Composes functions — `Composition[f, g][x] == f[g[x]]`.

```scrut
$ wo 'Composition[f, g][x]'
f[g[x]]
```




## `FlattenAt`

Flattens a single sublist at a specified position.

```scrut
$ wo 'FlattenAt[{1, {2, 3}, 4, {5, 6}}, 2]'
{1, 2, 3, 4, {5, 6}}
```




## `Flatten`

Flattens nested lists.

```scrut
$ wo 'Flatten[{{1}, {2, 3}}]'
{1, 2, 3}
```




## `FoldPairList`

Like `FoldList`, but the combining function returns a pair
`{value_to_output, new_state}`.

```scrut
$ wo 'FoldPairList[{#1, #1 + #2} &, 0, {1, 2, 3, 4}]'
{0, 1, 3, 6}
```




## `Identity`

Returns its argument unchanged.

```scrut
$ wo 'Identity[5]'
5
```

```scrut
$ wo 'Identity[{1, 2, 3}]'
{1, 2, 3}
```




## `Inner`

Generalized inner product (like dot product).

```scrut
$ wo 'Inner[Times, {1, 2, 3}, {4, 5, 6}, Plus]'
32
```

```scrut
$ wo 'Inner[Plus, {1, 2}, {3, 4}, Times]'
24
```




## `Map`

Applies a function to each element of a list.

```scrut
$ wo 'Map[Sign, {-6, 0, 2, 5}]'
{-1, 0, 1, 1}
```




## `MapAll`

Applies a function to every subexpression at every level, like `Map[f, expr, Infinity]`.

```scrut
$ wo 'MapAll[f, {{a, b}, {c, d}}]'
f[{f[{f[a], f[b]}], f[{f[c], f[d]}]}]
```




## `MapApply`

Applies the head `f` to the parts of each sublist
(`MapApply[f, list] == Apply[f, #]& /@ list`).

```scrut
$ wo 'MapApply[f, {{1, 2}, {3, 4}}]'
{f[1, 2], f[3, 4]}
```




## `MapAt`

Applies a function to the element at a given position.

```scrut
$ wo 'MapAt[f, {a, b, c}, 2]'
{a, f[b], c}
```




## `MapIndexed`

Applies a function to each element and its index.

```scrut
$ wo 'MapIndexed[f, {a, b, c}]'
{f[a, {1}], f[b, {2}], f[c, {3}]}
```




## `MapThread`

Applies a function to corresponding elements in several lists.

```scrut
$ wo 'MapThread[Plus, {{1, 2}, {3, 4}}]'
{4, 6}
```




## `Normal`

Converts a sparse array (or other special form) to an explicit nested list.

```scrut
$ wo 'Normal[SparseArray[{1 -> a, 3 -> c}, 4]]'
{a, 0, c, 0}
```




## `Outer`

Generalized outer product - applies function to all pairs.

```scrut
$ wo 'Outer[Times, {1, 2}, {3, 4}]'
{{3, 4}, {6, 8}}
```

```scrut
$ wo 'Outer[Plus, {1, 2}, {10, 20}]'
{{11, 21}, {12, 22}}
```




## `PadLeft`

Pads a list on the left to a specified length.

```scrut
$ wo 'PadLeft[{1, 2, 3}, 5]'
{0, 0, 1, 2, 3}
```

```scrut
$ wo 'PadLeft[{a, b}, 4, x]'
{x, x, a, b}
```

```scrut
$ wo 'PadLeft[{1, 2, 3, 4, 5}, 3]'
{3, 4, 5}
```




## `PadRight`

Pads a list on the right to a specified length.

```scrut
$ wo 'PadRight[{1, 2, 3}, 5]'
{1, 2, 3, 0, 0}
```

```scrut
$ wo 'PadRight[{a, b}, 4, x]'
{a, b, x, x}
```

```scrut
$ wo 'PadRight[{1, 2, 3, 4, 5}, 3]'
{1, 2, 3}
```




## `Pick`

Picks the elements of the first list for which the second list is `True`.

```scrut
$ wo 'Pick[{1, 2, 3, 4}, {True, False, True, False}]'
{1, 3}
```




## `ReplacePart`

Replaces element at a specific position.

```scrut
$ wo 'ReplacePart[{a, b, c}, 2 -> x]'
{a, x, c}
```

```scrut
$ wo 'ReplacePart[{1, 2, 3, 4}, 1 -> 0]'
{0, 2, 3, 4}
```

```scrut
$ wo 'ReplacePart[{a, b, c}, -1 -> z]'
{a, b, z}
```




## `RightComposition`

Composes functions from left to right —
`RightComposition[f, g][x] == g[f[x]]`.

```scrut
$ wo 'RightComposition[f, g][x]'
g[f[x]]
```




## `RotateLeft`

Rotates list elements to the left.

```scrut
$ wo 'RotateLeft[{1, 2, 3, 4, 5}]'
{2, 3, 4, 5, 1}
```

```scrut
$ wo 'RotateLeft[{a, b, c, d}, 2]'
{c, d, a, b}
```

```scrut
$ wo 'RotateLeft[{1, 2, 3}, 0]'
{1, 2, 3}
```




## `RotateRight`

Rotates list elements to the right.

```scrut
$ wo 'RotateRight[{1, 2, 3, 4, 5}]'
{5, 1, 2, 3, 4}
```

```scrut
$ wo 'RotateRight[{a, b, c, d}, 2]'
{c, d, a, b}
```

```scrut
$ wo 'RotateRight[{1, 2, 3}, 0]'
{1, 2, 3}
```




## `Scan`

Applies a function to each element for its side effects only;
returns `Null`.

```scrut
$ wo 'Scan[Print, {1, 2, 3}]'
1
2
3
Null
```




## `SparseArray`

Creates a sparse array from position-value rules, a dense list, or with an
explicit default fill value. All forms normalize to the canonical
`SparseArray[Automatic, dims, default, rules]` representation.

```scrut
$ wo 'Normal[SparseArray[{{1, 2} -> "Q", {3, 1} -> "Q"}, {3, 3}, "."]]'
{{., Q, .}, {., ., .}, {Q, ., .}}
```

```scrut
$ wo 'Normal[SparseArray[{{1, 2} -> "Q"}, {2, 2}, "."]]'
{{., Q}, {., .}}
```

Dimensions are inferred from the maximum position when omitted:

```scrut
$ wo 'SparseArray[{{1, 1} -> 1, {2, 2} -> 2, {3, 3} -> 3, {1, 3} -> 4}]'
SparseArray[Automatic, {3, 3}, 0, {1, {{0, 2, 3, 4}, {{1}, {3}, {2}, {3}}}, {1, 4, 2, 3}}]
```

```scrut
$ wo 'Normal[SparseArray[{{1, 1} -> 1, {2, 2} -> 2, {3, 3} -> 3, {1, 3} -> 4}]]'
{{1, 0, 4}, {0, 2, 0}, {0, 0, 3}}
```

A dense nested list is converted by recording its non-default entries:

```scrut
$ wo 'SparseArray[{{0, a}, {b, 0}}]'
SparseArray[Automatic, {2, 2}, 0, {1, {{0, 1, 2}, {{2}, {1}}}, {a, b}}]
```




## `Thread`

Threads a function over corresponding list elements.

```scrut
$ wo 'Thread[f[{a, b}, {x, y}]]'
{f[a, x], f[b, y]}
```

```scrut
$ wo 'Thread[Plus[{1, 2}, {3, 4}]]'
{4, 6}
```

```scrut
$ wo 'Thread[g[{a, b, c}, {1, 2, 3}]]'
{g[a, 1], g[b, 2], g[c, 3]}
```




## `Transpose`

Transposes a matrix (list of lists).

```scrut
$ wo 'Transpose[{{1, 2}, {3, 4}}]'
{{1, 3}, {2, 4}}
```

```scrut
$ wo 'Transpose[{{a, b, c}, {d, e, f}}]'
{{a, d}, {b, e}, {c, f}}
```

```scrut
$ wo 'Transpose[{{1, 2, 3}}]'
{{1}, {2}, {3}}
```




