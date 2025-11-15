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


## `Reverse`

Returns a list with elements in reverse order.

```scrut
$ wo 'Reverse[{1, 2, 3}]'
{3, 2, 1}
```

```scrut
$ wo 'Reverse[{7, 2, 4}]'
{4, 2, 7}
```

```scrut
$ wo 'Reverse[{1}]'
{1}
```

```scrut
$ wo 'Reverse[{}]'
{}
```

```scrut
$ wo 'Reverse[{a, b, c, d}]'
{d, c, b, a}
```

```scrut
$ wo 'Reverse[{-5, 3.14, 0, 42}]'
{42, 0, 3.14, -5}
```


## `Rest`

Returns the list without its first element.

```scrut
$ wo 'Rest[{1, 2, 3}]'
{2, 3}
```

```scrut
$ wo 'Rest[{5, 10, 15, 20}]'
{10, 15, 20}
```

```scrut
$ wo 'Rest[{a, b, c}]'
{b, c}
```

```scrut
$ wo 'Rest[{42}]'
{}
```

```scrut
$ wo 'Rest[{-5, 0, 5}]'
{0, 5}
```

```scrut
$ wo 'Rest[{1.5, 2.5, 3.5}]'
{2.5, 3.5}
```


## `Range`

Generates a sequence of numbers.

### Range[n]

Generates {1, 2, ..., n}.

```scrut
$ wo 'Range[5]'
{1, 2, 3, 4, 5}
```

```scrut
$ wo 'Range[1]'
{1}
```

```scrut
$ wo 'Range[0]'
{}
```

```scrut
$ wo 'Range[10]'
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
```

### Range[min, max]

Generates {min, min+1, ..., max}.

```scrut
$ wo 'Range[3, 7]'
{3, 4, 5, 6, 7}
```

```scrut
$ wo 'Range[0, 5]'
{0, 1, 2, 3, 4, 5}
```

```scrut
$ wo 'Range[-3, 2]'
{-3, -2, -1, 0, 1, 2}
```

```scrut
$ wo 'Range[5, 5]'
{5}
```

### Range[min, max, step]

Generates {min, min+step, ..., max}.

```scrut
$ wo 'Range[1, 10, 2]'
{1, 3, 5, 7, 9}
```

```scrut
$ wo 'Range[0, 20, 5]'
{0, 5, 10, 15, 20}
```

```scrut
$ wo 'Range[10, 1, -1]'
{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
```

```scrut
$ wo 'Range[5, -5, -2]'
{5, 3, 1, -1, -3, -5}
```

```scrut
$ wo 'Range[1, 10, 3]'
{1, 4, 7, 10}
```


## `Join`

Concatenates multiple lists together.

```scrut
$ wo 'Join[{a, b}, {c, d}]'
{a, b, c, d}
```

```scrut
$ wo 'Join[{1, 2}, {3, 4}, {5, 6}]'
{1, 2, 3, 4, 5, 6}
```

```scrut
$ wo 'Join[{1}]'
{1}
```

```scrut
$ wo 'Join[{}]'
{}
```

```scrut
$ wo 'Join[{}, {1, 2}]'
{1, 2}
```

```scrut
$ wo 'Join[{1, 2}, {}]'
{1, 2}
```

```scrut
$ wo 'Join[{a}, {b}, {c}, {d}]'
{a, b, c, d}
```

```scrut
$ wo 'Join[{1, 2}, {3.14, 2.71}]'
{1, 2, 3.14, 2.71}
```


## `Sort`

Sorts a list in ascending order.

```scrut
$ wo 'Sort[{3, 1, 4, 1, 5, 9, 2, 6}]'
{1, 1, 2, 3, 4, 5, 6, 9}
```

```scrut
$ wo 'Sort[{5, 2, 8, 1, 9}]'
{1, 2, 5, 8, 9}
```

```scrut
$ wo 'Sort[{1}]'
{1}
```

```scrut
$ wo 'Sort[{}]'
{}
```

```scrut
$ wo 'Sort[{-5, 3, 0, -2, 7}]'
{-5, -2, 0, 3, 7}
```

```scrut
$ wo 'Sort[{3.14, 2.71, 1.41, 2.23}]'
{1.41, 2.23, 2.71, 3.14}
```

```scrut
$ wo 'Sort[{10, 5, 15, 5, 20}]'
{5, 5, 10, 15, 20}
```

```scrut
$ wo 'Sort[{-10, -5, -15, -20}]'
{-20, -15, -10, -5}
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


## `Mean`

Calculates the arithmetic mean (average) of a list.

```scrut
$ wo 'Mean[{1, 2, 3, 4, 5}]'
3
```

```scrut
$ wo 'Mean[{10, 20, 30}]'
20
```

```scrut
$ wo 'Mean[{1, 2, 3}]'
2
```

```scrut
$ wo 'Mean[{5}]'
5
```

```scrut
$ wo 'Mean[{-5, 5}]'
0
```

```scrut
$ wo 'Mean[{1.5, 2.5, 3.5}]'
2.5
```

```scrut
$ wo 'Mean[{0, 0, 0, 10}]'
2.5
```

```scrut
$ wo 'Mean[{-10, -5, 0, 5, 10}]'
0
```


## `Median`

Returns the median value of a list.

```scrut
$ wo 'Median[{1, 2, 3, 4, 5}]'
3
```

```scrut
$ wo 'Median[{1, 2, 3, 4}]'
2.5
```

```scrut
$ wo 'Median[{5, 1, 3, 2, 4}]'
3
```

```scrut
$ wo 'Median[{10}]'
10
```

```scrut
$ wo 'Median[{1, 10}]'
5.5
```

```scrut
$ wo 'Median[{1, 3, 5, 7, 9}]'
5
```

```scrut
$ wo 'Median[{-5, 0, 5}]'
0
```

```scrut
$ wo 'Median[{1.5, 2.5, 3.5, 4.5}]'
3
```

```scrut
$ wo 'Median[{100, 1, 50}]'
50
```


## `Product`

Multiplies all the elements of a list together.

```scrut
$ wo 'Product[{1, 2, 3}]'
6
```

```scrut
$ wo 'Product[{2, 3, 4}]'
24
```

```scrut
$ wo 'Product[{1, 2, 3, 4, 5}]'
120
```

```scrut
$ wo 'Product[{5}]'
5
```

```scrut
$ wo 'Product[{}]'
1
```

```scrut
$ wo 'Product[{-2, 3}]'
-6
```

```scrut
$ wo 'Product[{-2, -3}]'
6
```

```scrut
$ wo 'Product[{2, 0, 5}]'
0
```

```scrut
$ wo 'Product[{1.5, 2}]'
3
```

```scrut
$ wo 'Product[{2, 2.5}]'
5
```


## `Accumulate`

Returns the cumulative sums of a list.

```scrut
$ wo 'Accumulate[{1, 2, 3}]'
{1, 3, 6}
```

```scrut
$ wo 'Accumulate[{1, 2, 3, 4, 5}]'
{1, 3, 6, 10, 15}
```

```scrut
$ wo 'Accumulate[{5}]'
{5}
```

```scrut
$ wo 'Accumulate[{}]'
{}
```

```scrut
$ wo 'Accumulate[{-1, 2, -3}]'
{-1, 1, -2}
```

```scrut
$ wo 'Accumulate[{10, -5, 3}]'
{10, 5, 8}
```

```scrut
$ wo 'Accumulate[{1.5, 2.5}]'
{1.5, 4}
```

```scrut
$ wo 'Accumulate[{0, 0, 1}]'
{0, 0, 1}
```

```scrut
$ wo 'Accumulate[{-5, -10, -15}]'
{-5, -15, -30}
```


## `Differences`

Returns successive differences between consecutive elements.

```scrut
$ wo 'Differences[{1, 3, 6, 10}]'
{2, 3, 4}
```

```scrut
$ wo 'Differences[{1, 2, 3, 4, 5}]'
{1, 1, 1, 1}
```

```scrut
$ wo 'Differences[{10, 5, 3}]'
{-5, -2}
```

```scrut
$ wo 'Differences[{5}]'
{}
```

```scrut
$ wo 'Differences[{}]'
{}
```

```scrut
$ wo 'Differences[{0, 1, 0, 1}]'
{1, -1, 1}
```

```scrut
$ wo 'Differences[{1.5, 3, 5.5}]'
{1.5, 2.5}
```

```scrut
$ wo 'Differences[{-5, -3, 0, 5}]'
{2, 3, 5}
```

```scrut
$ wo 'Differences[{100, 90, 80}]'
{-10, -10}
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
