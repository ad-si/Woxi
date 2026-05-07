# `MinMax`

Returns the minimum and maximum of a list.

```scrut
$ wo 'MinMax[{3, 1, 4, 1, 5, 9}]'
{1, 9}
```

```scrut
$ wo 'MinMax[{-5, 0, 5}]'
{-5, 5}
```

```scrut
$ wo 'MinMax[{42}]'
{42, 42}
```

`MinMax[list, d]` expands the returned interval on both sides by `d`.

```scrut
$ wo 'MinMax[{3, 1, 4, 1, 5, 9}, 1]'
{0, 10}
```

With an exact rational expansion, the result stays exact.

```scrut
$ wo 'MinMax[{3, 1, 4, 1, 5, 9}, 1/2]'
{1/2, 19/2}
```

`MinMax[list, {dMin, dMax}]` expands asymmetrically.

```scrut
$ wo 'MinMax[{3, 1, 4, 1, 5, 9}, {1, 2}]'
{0, 11}
```
