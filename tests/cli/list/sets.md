# Set Operations

Union, intersection, containment, and related relational queries.

## `Complement`

Returns elements in the first list but not in any of the other lists.

```scrut
$ wo 'Complement[{1, 2, 3, 4, 5}, {2, 4}]'
{1, 3, 5}
```

```scrut
$ wo 'Complement[{a, b, c, d}, {b, d}]'
{a, c}
```

```scrut
$ wo 'Complement[{1, 2, 3}, {1, 2, 3}]'
{}
```

```scrut
$ wo 'Complement[{1, 2, 3, 4}, {2}, {4}]'
{1, 3}
```




## `ContainsAll`

Tests whether a list contains every element of another list.

```scrut
$ wo 'ContainsAll[{1, 2, 3, 4}, {2, 3}]'
True
```

```scrut
$ wo 'ContainsAll[{1, 2, 3}, {2, 4}]'
False
```




## `ContainsAny`

Tests whether a list contains any element of another list.

```scrut
$ wo 'ContainsAny[{1, 2, 3}, {4, 2}]'
True
```

```scrut
$ wo 'ContainsAny[{1, 2, 3}, {5, 6}]'
False
```




## `ContainsNone`

Tests whether a list contains none of the elements of another list.

```scrut
$ wo 'ContainsNone[{1, 2, 3}, {4, 5}]'
True
```

```scrut
$ wo 'ContainsNone[{1, 2, 3}, {2, 4}]'
False
```




## `ContainsOnly`

Tests whether every element of a list is contained in another list.

```scrut
$ wo 'ContainsOnly[{1, 2, 1}, {1, 2, 3}]'
True
```

```scrut
$ wo 'ContainsOnly[{1, 2, 4}, {1, 2, 3}]'
False
```




## `DisjointQ`

Tests whether two lists have no elements in common.

```scrut
$ wo 'DisjointQ[{1, 2}, {3, 4}]'
True
```

```scrut
$ wo 'DisjointQ[{1, 2}, {2, 3}]'
False
```




## `IntersectingQ`

Tests whether two lists share at least one element.

```scrut
$ wo 'IntersectingQ[{1, 2, 3}, {3, 4}]'
True
```

```scrut
$ wo 'IntersectingQ[{1, 2}, {3, 4}]'
False
```




## `Intersection`

Returns the sorted intersection of lists.

```scrut
$ wo 'Intersection[{1, 2, 3, 4}, {2, 4, 6}]'
{2, 4}
```

```scrut
$ wo 'Intersection[{a, b, c}, {b, c, d}]'
{b, c}
```

```scrut
$ wo 'Intersection[{1, 2, 3}, {4, 5, 6}]'
{}
```

```scrut
$ wo 'Intersection[{1, 2, 3}, {1, 2, 3}]'
{1, 2, 3}
```

```scrut
$ wo 'Intersection[{1, 2, 3}, {2, 3}, {3}]'
{3}
```




## `MinMax`

Returns the minimum and maximum of a list.

```scrut
$ wo 'MinMax[{3, 1, 4, 1, 5, 9}]'
{1, 9}
```

```scrut
$ wo 'MinMax[{-5, 0, 5}]'
{-5, 5}
```

```scrut
$ wo 'MinMax[{42}]'
{42, 42}
```

`MinMax[list, d]` expands the returned interval on both sides by `d`.

```scrut
$ wo 'MinMax[{3, 1, 4, 1, 5, 9}, 1]'
{0, 10}
```

With an exact rational expansion, the result stays exact.

```scrut
$ wo 'MinMax[{3, 1, 4, 1, 5, 9}, 1/2]'
{1/2, 19/2}
```

`MinMax[list, {dMin, dMax}]` expands asymmetrically.

```scrut
$ wo 'MinMax[{3, 1, 4, 1, 5, 9}, {1, 2}]'
{0, 11}
```




## `SquareMatrixQ`

Tests whether an expression is a square matrix.

```scrut
$ wo 'SquareMatrixQ[{{1, 2}, {3, 4}}]'
True
```

```scrut
$ wo 'SquareMatrixQ[{{1, 2, 3}, {4, 5, 6}}]'
False
```




## `Subsequences`

Returns all contiguous subsequences of a list (including the empty one).

```scrut
$ wo 'Subsequences[{a, b, c}]'
{{}, {a}, {b}, {c}, {a, b}, {b, c}, {a, b, c}}
```




## `Subsets`

Generates subsets (combinations) of a list.

```scrut
$ wo 'Subsets[{a, b, c}]'
{{}, {a}, {b}, {c}, {a, b}, {a, c}, {b, c}, {a, b, c}}
```

```scrut
$ wo 'Subsets[{a, b, c}, {2}]'
{{a, b}, {a, c}, {b, c}}
```

```scrut
$ wo 'Subsets[{a, b, c}, {0}]'
{{}}
```

```scrut
$ wo 'Subsets[{1, 2, 3, 4}, {3}]'
{{1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}}
```




## `Union`

Returns the sorted union of lists (removes duplicates).

```scrut
$ wo 'Union[{1, 2, 3}, {2, 3, 4}]'
{1, 2, 3, 4}
```

```scrut
$ wo 'Union[{a, b}, {b, c}]'
{a, b, c}
```

```scrut
$ wo 'Union[{3, 1, 2}]'
{1, 2, 3}
```

```scrut
$ wo 'Union[{1, 1, 2, 2}]'
{1, 2}
```

```scrut
$ wo 'Union[{1, 2}, {3, 4}, {5, 6}]'
{1, 2, 3, 4, 5, 6}
```




