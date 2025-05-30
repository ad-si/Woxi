# List operations tests

## `Length`

Returns the number of elements in a list.

```scrut
$ wo 'Length[{7, 2, 4}]'
3
```


## `First`

Returns the first element of a list.

```scrut
$ wo 'First[{7, 2, 4}]'
7
```


## `Last`

Returns the last element of a list.

```scrut
$ wo 'Last[{7, 2, 4}]'
4
```


## `Rest`

Returns the list without its first element.

```scrut
$ wo 'Rest[{7, 2, 4}]'
{2, 4}
```


## `Most`

Returns the list without its last element.

```scrut
$ wo 'Most[{7, 2, 4}]'
{7, 2}
```


## `Take`

Returns the first n elements of a list.

```scrut
$ wo 'Take[{7, 2, 4}, 2]'
{7, 2}
```


## `Part`

Returns the nth element of a list.

```scrut
$ wo 'Part[{7, 6, 4}, 2]'
6
```


## `Drop`

Returns the list without its first n elements.

```scrut
$ wo 'Drop[{7, 2, 4}, 2]'
{4}
```


## `Append`

Adds an element to the end of a list.

```scrut
$ wo 'Append[{7, 2, 4}, 5]'
{7, 2, 4, 5}
```


## `Prepend`

Adds an element to the beginning of a list.

```scrut
$ wo 'Prepend[{7, 2, 4}, 5]'
{5, 7, 2, 4}
```


## `Map`

Applies a function to each element of a list.

```scrut
$ wo 'Map[Sign, {-6, 0, 2, 5}]'
{-1, 0, 1, 1}
```


## `Select`

Picks elements of a list that satisfy a criterion.

```scrut
$ wo 'Select[{1, 2, 3, 4}, EvenQ]'
{2, 4}
```


## `Flatten`

Flattens nested lists.

```scrut
$ wo 'Flatten[{{1}, {2, 3}}]'
{1, 2, 3}
```


## `Total`

Sums the elements of a list.

```scrut
$ wo 'Total[{1, 2, 3}]'
6
```


## `Cases`

Extracts elements from an expression that match a pattern.

```todo
$ wo 'Cases[{a, b, a}, a]'
{a, a}
```


## `DeleteCases`

Removes elements from an expression that match a pattern.

```todo
$ wo 'DeleteCases[{a, b, a}, a]'
{b}
```


## `MapThread`

Applies a function to corresponding elements in several lists.

```todo
$ wo 'MapThread[Plus, {{1, 2}, {3, 4}}]'
{4, 6}
```


## `Partition`


Breaks a list into smaller sublists.

```todo
$ wo 'Partition[{1, 2, 3, 4}, 2]'
{{1, 2}, {3, 4}}
```


## `SortBy`

Sorts elements of a list according to a function.

```todo
$ wo 'SortBy[{3, 1, 2}, # &]'
{1, 2, 3}
```


## `GroupBy`

Groups elements of a list according to a function.

```todo
$ wo 'GroupBy[{{a, b}, {a, c}, {b, c}}, First]'
> /<|a -> {{a, b}, {a, c}}, b -> {{b, c}}|>/
```


## `Accumulate`

Returns the cumulative sums of a list.

```todo
$ wo 'Accumulate[{1, 2, 3}]'
{1, 3, 6}
```


## `Array`

Constructs an array using a function to generate elements.

```todo
$ wo 'Array[#^2 &, 3]'
{1, 4, 9}
```
