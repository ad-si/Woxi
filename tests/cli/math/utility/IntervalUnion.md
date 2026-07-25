# `IntervalUnion`

Computes the union of intervals.

```scrut
$ wo 'IntervalUnion[Interval[{1, 4}], Interval[{3, 7}]]'
Interval[{1, 7}]
```

The union of nothing, and of empty intervals, is the empty interval:

```scrut
$ wo 'IntervalUnion[]'
Interval[]
```

An empty interval contributes nothing to a union:

```scrut
$ wo 'IntervalUnion[Interval[{1, 2}], Interval[]]'
Interval[{1, 2}]
```
