# Searching and Filtering

Find, count, select, and remove matching elements.

## `AnyTrue`

Checks if any element satisfies a predicate.

```scrut
$ wo 'AnyTrue[{1, 2, 3, 4}, EvenQ]'
True
```

```scrut
$ wo 'AnyTrue[{1, 3, 5}, EvenQ]'
False
```

```scrut
$ wo 'AnyTrue[{1, 2, 3}, # > 2 &]'
True
```




## `Cases`

Extracts elements from an expression that match a pattern.

```scrut
$ wo 'Cases[{a, b, a}, a]'
{a, a}
```


### Cases with Except pattern

```scrut
$ wo 'Cases[{1, 2, 3, 4, 5}, Except[3]]'
{1, 2, 4, 5}
```

### Cases with Alternatives

```scrut
$ wo 'Cases[{1, 2, 3, 4, 5}, Except[2 | 4]]'
{1, 3, 5}
```

### Cases with level specification

```scrut
$ wo 'Cases[{{1, 2}, {3, 4}}, _Integer, {2}]'
{1, 2, 3, 4}
```




## `Commonest`

Returns the most frequent elements of a list.

```scrut
$ wo 'Commonest[{1, 2, 2, 3, 3, 3}]'
{3}
```




## `Count`

Counts the number of occurrences of a pattern in a list.

```scrut
$ wo 'Count[{a, b, a, c, a}, a]'
3
```

```scrut
$ wo 'Count[{1, 2, 3, 2, 1}, 2]'
2
```

```scrut
$ wo 'Count[{1, 2, 3}, 4]'
0
```

```scrut
$ wo 'Count[{x, x, x, y}, x]'
3
```




## `CountDistinct`

Counts the number of distinct elements in a list.

```scrut
$ wo 'CountDistinct[{1, 2, 2, 3, 3, 3}]'
3
```




## `Counts`

Returns an association mapping each element to its number of occurrences.

```scrut
$ wo 'Counts[{a, b, a, c, b, a}]'
<|a -> 3, b -> 2, c -> 1|>
```




## `CountsBy`

Like `Counts`, but groups by the value of a classifier function.

```scrut
$ wo 'CountsBy[{1, 2, 3, 4, 5, 6}, OddQ]'
<|True -> 3, False -> 3|>
```




## `DeleteAdjacentDuplicates`

Removes consecutive duplicate elements.

```scrut
$ wo 'DeleteAdjacentDuplicates[{1, 1, 2, 2, 3, 1, 1}]'
{1, 2, 3, 1}
```




## `DeleteCases`

Removes elements from an expression that match a pattern.

```scrut
$ wo 'DeleteCases[{a, b, a}, a]'
{b}
```




## `DeleteDuplicates`

Removes duplicate elements from a list.

```scrut
$ wo 'DeleteDuplicates[{a, b, a, c, b}]'
{a, b, c}
```

```scrut
$ wo 'DeleteDuplicates[{1, 2, 1, 3, 2}]'
{1, 2, 3}
```

```scrut
$ wo 'DeleteDuplicates[{1, 1, 1}]'
{1}
```

```scrut
$ wo 'DeleteDuplicates[{a}]'
{a}
```

```scrut
$ wo 'DeleteDuplicates[{}]'
{}
```




## `DeleteDuplicatesBy`

Like `DeleteDuplicates`, but compares elements by applying a function.

```scrut
$ wo 'DeleteDuplicatesBy[{1, -1, 2, -2, 3}, Abs]'
{1, 2, 3}
```




## `DuplicateFreeQ`

Tests whether a list has no duplicate elements.

```scrut
$ wo 'DuplicateFreeQ[{1, 2, 3}]'
True
```

```scrut
$ wo 'DuplicateFreeQ[{1, 2, 2}]'
False
```




## `FirstCase`

Returns the first element that matches a pattern.

```scrut
$ wo 'FirstCase[{1, 2, 3, 4}, x_ /; x > 2]'
3
```




## `FirstPosition`

Returns the position of the first occurrence of an expression.

```scrut
$ wo 'FirstPosition[{1, 2, 3, 2, 1}, 2]'
{2}
```




## `FreeQ`

Tests if expression is free of a specified form.

```scrut
$ wo 'FreeQ[{1, 2, 3}, 4]'
True
```

```scrut
$ wo 'FreeQ[{1, 2, 3}, 2]'
False
```




## `LengthWhile`

Returns the length of the longest prefix for which a predicate holds.

```scrut
$ wo 'LengthWhile[{2, 4, 6, 7, 8}, EvenQ]'
3
```




## `NoneTrue`

Checks if no element satisfies a predicate.

```scrut
$ wo 'NoneTrue[{1, 3, 5}, EvenQ]'
True
```

```scrut
$ wo 'NoneTrue[{1, 2, 3}, EvenQ]'
False
```

```scrut
$ wo 'NoneTrue[{1, 2, 3}, # > 5 &]'
True
```




## `Position`

Finds all positions of a pattern in a list.

```scrut
$ wo 'Position[{a, b, a, c, a}, a]'
{{1}, {3}, {5}}
```

```scrut
$ wo 'Position[{1, 2, 3, 2, 1}, 2]'
{{2}, {4}}
```

```scrut
$ wo 'Position[{1, 2, 3}, 4]'
{}
```

```scrut
$ wo 'Position[{x, y, x, z}, x]'
{{1}, {3}}
```




## `PositionIndex`

Returns an association mapping each element to the list of its positions.

```scrut
$ wo 'PositionIndex[{a, b, a, c, b}]'
<|a -> {1, 3}, b -> {2, 5}, c -> {4}|>
```




## `Select`

Picks elements of a list that satisfy a criterion.

```scrut
$ wo 'Select[{1, 2, 3, 4}, EvenQ]'
{2, 4}
```




## `SelectFirst`

Returns the first element satisfying a predicate.

```scrut
$ wo 'SelectFirst[{1, 2, 3, 4}, # > 2 &]'
3
```




## `SequenceCount`

Counts non-overlapping occurrences of a sub-sequence inside a list.

```scrut
$ wo 'SequenceCount[{1, 2, 3, 1, 2, 3}, {1, 2}]'
2
```




## `Tally`

Counts occurrences of each distinct element.

```scrut
$ wo 'Tally[{a, b, a, c, b, a}]'
{{a, 3}, {b, 2}, {c, 1}}
```

```scrut
$ wo 'Tally[{1, 2, 1, 3}]'
{{1, 2}, {2, 1}, {3, 1}}
```

```scrut
$ wo 'Tally[{x, x, x}]'
{{x, 3}}
```

```scrut
$ wo 'Tally[{}]'
{}
```




## `TakeLargestBy`

Returns the `n` largest elements of a list by a key function.

```scrut
$ wo 'TakeLargestBy[{1, -3, 2, -4, 5}, Abs, 2]'
{5, -4}
```




## `TakeSmallestBy`

Returns the `n` smallest elements of a list by a key function.

```scrut
$ wo 'TakeSmallestBy[{1, -3, 2, -4, 5}, Abs, 2]'
{1, 2}
```


## `TakeWhile`

Takes elements from the start while a predicate is true.

```scrut
$ wo 'TakeWhile[{1, 2, 3, 4, 5}, # < 4 &]'
{1, 2, 3}
```

```scrut
$ wo 'TakeWhile[{2, 4, 6, 7, 8}, EvenQ]'
{2, 4, 6}
```




