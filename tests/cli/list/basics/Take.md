# `Take`

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

A span `i;;j` takes elements `i` through `j`.

```scrut
$ wo 'Take[{1, 2, 3, 4, 5}, 2;;4]'
{2, 3, 4}
```

```scrut
$ wo 'Take[{1, 2, 3, 4, 5}, 1;;-1;;2]'
{1, 3, 5}
```
