# `ArrayPad`

Pads a list on both sides with a filler value (`0` by default).

```scrut
$ wo 'ArrayPad[{1, 2, 3}, 2]'
{0, 0, 1, 2, 3, 0, 0}
```

```scrut
$ wo 'ArrayPad[{1, 2, 3}, 2, 0]'
{0, 0, 1, 2, 3, 0, 0}
```
