# `Median`

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

Rational lists keep an exact result.

```scrut
$ wo 'Median[{1/2, 1/3, 1/4}]'
1/3
```

```scrut
$ wo 'Median[{1/2, 1/3, 1/4, 1/5}]'
7/24
```

Symbolic-but-real elements (Pi, E, ...) are ordered by value and the exact
result is preserved.

```scrut
$ wo 'Median[{Pi, E, 1}]'
E
```

```scrut
$ wo 'Median[{Pi, E, 1, 2}]'
(2 + E)/2
```

A single inexact element does not force the selected exact element to a real.

```scrut
$ wo 'Median[{1.5, 1/2, 1/3}]'
1/2
```

On an association, the values are used.

```scrut
$ wo 'Median[<|a -> 1, b -> 2, c -> 3|>]'
2
```
