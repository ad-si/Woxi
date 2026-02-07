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

### Part extraction using `[[]]` notation

The `[[n]]` notation is equivalent to `Part[..., n]`.

```scrut
$ wo '{1, 2, 3}[[2]]'
2
```

```scrut
$ wo '{a, b, c, d}[[3]]'
c
```

### Part extraction from variables

```scrut
$ wo 'x = {10, 20, 30}; x[[2]]'
20
```

```scrut
$ wo 'list = {a, b, c}; list[[1]]'
a
```

### Out of bounds error

Attempting to access an index beyond the list length returns an error.

```scrut {output_stream: combined}
$ wo '{1, 2, 3}[[5]]'

Part::partw: Part 5 of {1, 2, 3} does not exist.
{1, 2, 3}[[5]]
```

```scrut {output_stream: combined}
$ wo 'x = {1, 2}; x[[10]]'

Part::partw: Part 10 of {1, 2} does not exist.
{1, 2}[[10]]
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


## `Take`

Returns the first n elements of a list.

```scrut
$ wo 'Take[{1, 2, 3, 4, 5}, 3]'
{1, 2, 3}
```

```scrut
$ wo 'Take[{1, 2, 3, 4, 5}, 1]'
{1}
```

```scrut {output_stream: combined}
$ wo 'Take[{1, 2, 3}, 5]'

Take::take: Cannot take positions 1 through 5 in {1, 2, 3}.
Take[{1, 2, 3}, 5]
```

```scrut
$ wo 'Take[{a, b, c, d}, 2]'
{a, b}
```

```scrut
$ wo 'Take[{1.5, 2.5, 3.5, 4.5}, 2]'
{1.5, 2.5}
```

```scrut
$ wo 'Take[{10, 20, 30, 40}, 4]'
{10, 20, 30, 40}
```

```scrut
$ wo 'Take[{1, 2, 3, 4, 5}, 5]'
{1, 2, 3, 4, 5}
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
5/2
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
5/2
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
11/2
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
3.
```

```scrut
$ wo 'Median[{100, 1, 50}]'
50
```


## `Product`

Evaluates the product.

```scrut
$ wo 'Product[i^2, {i, 1, 6}]'
518400
```

```scrut
$ wo 'Product[i^2, {i, 1, n}]'
n!^2
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
{1.5, 4.}
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


## `DeleteCases`

Removes elements from an expression that match a pattern.

```scrut
$ wo 'DeleteCases[{a, b, a}, a]'
{b}
```


## `MapThread`

Applies a function to corresponding elements in several lists.

```scrut
$ wo 'MapThread[Plus, {{1, 2}, {3, 4}}]'
{4, 6}
```


## `Partition`


Breaks a list into smaller sublists.

```scrut
$ wo 'Partition[{1, 2, 3, 4}, 2]'
{{1, 2}, {3, 4}}
```


## `SortBy`

Sorts elements of a list according to a function.

```scrut
$ wo 'SortBy[{3, 1, 2}, # &]'
{1, 2, 3}
```


## `GroupBy`

Groups elements of a list according to a function.

```scrut
$ wo 'GroupBy[{{a, b}, {a, c}, {b, c}}, First]'
<|a -> {{a, b}, {a, c}}, b -> {{b, c}}|>
```


## `Array`

Constructs an array using a function to generate elements.

```scrut
$ wo 'Array[#^2 &, 3]'
{1, 4, 9}
```


## `Table`

Generates a list by evaluating an expression for different values of a variable.

### Table[expr, n]

Generates a list of n copies of expr.

```scrut
$ wo 'Table[x, 3]'
{x, x, x}
```

```scrut
$ wo 'Table[1, 5]'
{1, 1, 1, 1, 1}
```

### Table[expr, {i, max}]

Generates a list where i goes from 1 to max.

```scrut
$ wo 'Table[i^2, {i, 5}]'
{1, 4, 9, 16, 25}
```

```scrut
$ wo 'Table[i, {i, 4}]'
{1, 2, 3, 4}
```

```scrut
$ wo 'Table[2*i, {i, 3}]'
{2, 4, 6}
```

### Table[expr, {i, min, max}]

Generates a list where i goes from min to max.

```scrut
$ wo 'Table[i, {i, 3, 7}]'
{3, 4, 5, 6, 7}
```

```scrut
$ wo 'Table[i^2, {i, 0, 4}]'
{0, 1, 4, 9, 16}
```

```scrut
$ wo 'Table[i, {i, -2, 2}]'
{-2, -1, 0, 1, 2}
```

### Table[expr, {i, min, max, step}]

Generates a list where i goes from min to max in steps.

```scrut
$ wo 'Table[i, {i, 1, 10, 2}]'
{1, 3, 5, 7, 9}
```

```scrut
$ wo 'Table[i, {i, 10, 1, -2}]'
{10, 8, 6, 4, 2}
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


## `ConstantArray`

Creates an array of repeated elements.

```scrut
$ wo 'ConstantArray[x, 3]'
{x, x, x}
```

```scrut
$ wo 'ConstantArray[0, 5]'
{0, 0, 0, 0, 0}
```

```scrut
$ wo 'ConstantArray[1, 0]'
{}
```

```scrut
$ wo 'ConstantArray[a, {2, 3}]'
{{a, a, a}, {a, a, a}}
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


## `Riffle`

Inserts an element between each pair of elements.

```scrut
$ wo 'Riffle[{a, b, c}, x]'
{a, x, b, x, c}
```

```scrut
$ wo 'Riffle[{1, 2, 3, 4}, 0]'
{1, 0, 2, 0, 3, 0, 4}
```

```scrut
$ wo 'Riffle[{a}, x]'
{a}
```

```scrut
$ wo 'Riffle[{}, x]'
{}
```

Element-wise interleaving with a second list:

```scrut
$ wo 'Riffle[{a, b, c, d}, {1, 2, 3, 4}]'
{a, 1, b, 2, c, 3, d, 4}
```

```scrut
$ wo 'Riffle[{a, b, c}, {1, 2}]'
{a, 1, b, 2, c}
```


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


## `Extract`

Extracts parts at specified positions.

```scrut
$ wo 'Extract[{a, b, c, d}, 2]'
b
```

```scrut
$ wo 'Extract[{a, {b1, b2, b3}, c, d}, {2, 3}]'
b3
```


## `Catenate`

Flattens one level of lists.

```scrut
$ wo 'Catenate[{{1, 2}, {3, 4}}]'
{1, 2, 3, 4}
```

```scrut
$ wo 'Catenate[{{a, b}, {c}, {d, e, f}}]'
{a, b, c, d, e, f}
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


## `SparseArray`

Creates a matrix from position-value rules with a default fill value.

```scrut
$ wo 'Normal[SparseArray[{{1, 2} -> "Q", {3, 1} -> "Q"}, {3, 3}, "."]]'
{{., Q, .}, {., ., .}, {Q, ., .}}
```

```scrut
$ wo 'Normal[SparseArray[{{1, 2} -> "Q"}, {2, 2}, "."]]'
{{., Q}, {., .}}
```


## `Table` (multi-dimensional)

Table supports multiple iterator specifications for multi-dimensional arrays.

```scrut
$ wo 'Table[i + j, {i, 1, 2}, {j, 1, 3}]'
{{2, 3, 4}, {3, 4, 5}}
```

```scrut
$ wo 'Table["x", {3}]'
{x, x, x}
```

```scrut
$ wo 'Table[0, {3}, {2}]'
{{0, 0}, {0, 0}, {0, 0}}
```
