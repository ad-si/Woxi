# Statistics and Summaries

Summary statistics and running-reduction functions.

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




## `Ratios`

Returns consecutive ratios of a list.

```scrut
$ wo 'Ratios[{1, 2, 4, 8}]'
{2, 2, 2}
```




## `Total`

Sums the elements of a list.

```scrut
$ wo 'Total[{1, 2, 3}]'
6
```




