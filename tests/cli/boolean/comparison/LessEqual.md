# `LessEqual`

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
