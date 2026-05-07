# `Min`

Returns the minimum value from a set of arguments or a list.

### Multiple arguments

```scrut
$ wo 'Min[1, 5, 3]'
1
```

```scrut
$ wo 'Min[1, 2]'
1
```

```scrut
$ wo 'Min[-5, -2, -8]'
-8
```

```scrut
$ wo 'Min[3.14, 2.71, 3.5]'
2.71
```

```scrut
$ wo 'Min[1, 2.5, 3]'
1
```

### Single list argument

```scrut
$ wo 'Min[{1, 5, 3}]'
1
```

```scrut
$ wo 'Min[{-10, -5, -20}]'
-20
```

```scrut
$ wo 'Min[{3.14, 2.71, 3.5}]'
2.71
```

### Single value

```scrut
$ wo 'Min[42]'
42
```

```scrut
$ wo 'Min[-7]'
-7
```

### Empty list

```scrut
$ wo 'Min[{}]'
Infinity
```

### With expressions

```scrut
$ wo 'Min[2 + 3, 4 * 2, 10 - 1]'
5
```

```scrut
$ wo 'Min[{1 + 1, 2 * 2, 3 - 1}]'
2
```
