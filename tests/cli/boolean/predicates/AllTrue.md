# `AllTrue`

Check if all elements in a list satisfy a condition.

```scrut
$ wo 'AllTrue[{2, 4, 6}, EvenQ]'
True
```

```scrut
$ wo 'AllTrue[{2, 3, 4}, EvenQ]'
False
```

```scrut
$ wo 'AllTrue[{1, 3, 6}, 1 <= # <= 6 &]'
True
```
