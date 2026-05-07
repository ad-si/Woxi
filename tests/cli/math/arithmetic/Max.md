# `Max`

Returns the maximum value from a set of arguments or a list.

### Multiple arguments

```scrut
$ wo 'Max[1, 5, 3]'
5
```

```scrut
$ wo 'Max[1, 2]'
2
```

```scrut
$ wo 'Max[-5, -2, -8]'
-2
```

```scrut
$ wo 'Max[3.14, 2.71, 3.5]'
3.5
```

```scrut
$ wo 'Max[1, 2.5, 3]'
3
```

### Single list argument

```scrut
$ wo 'Max[{1, 5, 3}]'
5
```

```scrut
$ wo 'Max[{-10, -5, -20}]'
-5
```

```scrut
$ wo 'Max[{3.14, 2.71, 3.5}]'
3.5
```

### Single value

```scrut
$ wo 'Max[42]'
42
```

```scrut
$ wo 'Max[-7]'
-7
```

### Empty list

```scrut
$ wo 'Max[{}]'
-Infinity
```

### With expressions

```scrut
$ wo 'Max[2 + 3, 4 * 2, 10 - 1]'
9
```

```scrut
$ wo 'Max[{1 + 1, 2 * 2, 3 - 1}]'
4
```
