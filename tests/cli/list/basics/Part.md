# `Part`

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
