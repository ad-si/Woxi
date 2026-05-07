# `Less`

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
