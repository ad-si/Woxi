# List Basics

Core functions for accessing list elements and building simple lists.

## `Append`

Adds an element to the end of a list.

```scrut
$ wo 'Append[{7, 2, 4}, 5]'
{7, 2, 4, 5}
```




## `Array`

Constructs an array using a function to generate elements.

```scrut
$ wo 'Array[#^2 &, 3]'
{1, 4, 9}
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




## `Delete`

Removes the element at a given position.

```scrut
$ wo 'Delete[{1, 2, 3, 4}, 2]'
{1, 3, 4}
```




## `Dimensions`

Returns the dimensions of a nested list.

```scrut
$ wo 'Dimensions[{{1, 2, 3}, {4, 5, 6}}]'
{2, 3}
```




## `Drop`

Returns the list without its first n elements.

```scrut
$ wo 'Drop[{7, 2, 4}, 2]'
{4}
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




## `First`

Returns the first element of a list.

```scrut
$ wo 'First[{7, 2, 4}]'
7
```




## `Insert`

Inserts an element at a given 1-based position.

```scrut
$ wo 'Insert[{a, b, c}, x, 2]'
{a, x, b, c}
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




## `Last`

Returns the last element of a list.

```scrut
$ wo 'Last[{7, 2, 4}]'
4
```




## `Length`

Returns the number of elements in a list.

```scrut
$ wo 'Length[{7, 2, 4}]'
3
```




## `Most`

Returns the list without its last element.

```scrut
$ wo 'Most[{7, 2, 4}]'
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




## `Prepend`

Adds an element to the beginning of a list.

```scrut
$ wo 'Prepend[{7, 2, 4}, 5]'
{5, 7, 2, 4}
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
.* (regex*)
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




