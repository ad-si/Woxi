# Comparison Operators

Relational comparisons: equality, inequality, ordering, and multi-way chains.

## `Equal`

Compare values for equality.

```scrut
$ wo 'Equal[1, 1]'
True
```

```scrut
$ wo 'Equal[1, 2]'
False
```

```scrut
$ wo 'Equal[1, 1, 1]'
True
```


### `==`

Check if values are equal to each other.

```scrut
$ wo '2 == 2'
True
```

```scrut
$ wo 'x = 2; x == 2'
True
```

```scrut
$ wo '2 == 3'
False
```




## `Unequal`

Compare values for inequality.

```scrut
$ wo 'Unequal[1, 1]'
False
```

```scrut
$ wo 'Unequal[1, 2]'
True
```

```scrut
$ wo 'Unequal[1, 1, 1]'
False
```


### `!=`

Check if values are not equal to each other.

```scrut
$ wo '2 != 2'
False
```

```scrut
$ wo 'x = 2; x != 2'
False
```

```scrut
$ wo '2 != 3'
True
```




## `Greater`

Check if values are greater than each other.

```scrut
$ wo 'Greater[2, 1]'
True
```

```scrut
$ wo 'Greater[1, 2]'
False
```

```scrut
$ wo 'Greater[1, 1]'
False
```

```scrut
$ wo 'Greater[1, 2, 3]'
False
```

```scrut
$ wo 'Greater[3, 2, 1]'
True
```


### `>`

Check if values are greater than each other.

```scrut
$ wo '2 > 1'
True
```

```scrut
$ wo '1 > 2'
False
```




## `GreaterEqual`

Check if values are greater than or equal to each other.

```scrut
$ wo 'GreaterEqual[2, 1]'
True
```

```scrut
$ wo 'GreaterEqual[1, 2]'
False
```

```scrut
$ wo 'GreaterEqual[1, 1]'
True
```

```scrut
$ wo 'GreaterEqual[1, 2, 3]'
False
```

```scrut
$ wo 'GreaterEqual[3, 2, 1]'
True
```


### `>=`

```scrut
$ wo '2 >= 1'
True
```

```scrut
$ wo '1 >= 2'
False
```




## `Less`

Check if values are less than each other.

```scrut
$ wo 'Less[1, 2]'
True
```

```scrut
$ wo 'Less[2, 1]'
False
```

```scrut
$ wo 'Less[1, 1]'
False
```

```scrut
$ wo 'Less[1, 2, 3]'
True
```

```scrut
$ wo 'Less[3, 2, 1]'
False
```


### `<`

Check if values are greater than each other.

```scrut
$ wo '1 < 2'
True
```

```scrut
$ wo '2 < 1'
False
```




## `LessEqual`

Check if values are less than or equal to each other.

```scrut
$ wo 'LessEqual[1, 2]'
True
```

```scrut
$ wo 'LessEqual[2, 1]'
False
```

```scrut
$ wo 'LessEqual[1, 1]'
True
```

```scrut
$ wo 'LessEqual[1, 2, 3]'
True
```

```scrut
$ wo 'LessEqual[3, 2, 1]'
False
```


### `<=`

```scrut
$ wo '1 <= 2'
True
```

```scrut
$ wo '2 <= 2'
True
```

```scrut
$ wo '2 <= 1'
False
```




# Multiple comparisons

```scrut
$ wo 'x = 1; 0 <= x <= 2'
True
```

```scrut
$ wo 'x = 1; 0 < x < 2'
True
```

```scrut
$ wo 'x = 1; 0 > x < 2'
False
```

```scrut
$ wo 'x = 1; 0 < x > 2'
False
```




