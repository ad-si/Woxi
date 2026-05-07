# `Table`

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


## Multi-dimensional

`Table` supports multiple iterator specifications for multi-dimensional arrays.

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
