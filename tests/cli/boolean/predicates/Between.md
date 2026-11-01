# `Between`

Tests whether a number lies within an inclusive range.

```scrut
$ wo 'Between[5, {1, 10}]'
True
```

```scrut
$ wo 'Between[15, {1, 10}]'
False
```

The range can also be an `Interval`, which also tests whole sub-intervals.

```scrut
$ wo 'Between[Interval[{3, 4}], Interval[{2, 5}]]'
True
```
