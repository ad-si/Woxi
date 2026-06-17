# `ArcLength`

Compute the arc length of a curve.

```scrut
$ wo 'ArcLength[Disk[]]'
Undefined
```

A parameterized curve `ArcLength[curve, {t, a, b}]` integrates the speed
`Sqrt[Sum of squared derivatives]` over the range.

```scrut
$ wo 'ArcLength[{Sin[t], Cos[t]}, {t, 0, 2 Pi}]'
2*Pi
```

```scrut
$ wo 'ArcLength[{Cos[t], Sin[t], t}, {t, 0, 2 Pi}]'
2*Sqrt[2]*Pi
```

Unbounded regions have infinite length.

```scrut
$ wo 'ArcLength[HalfLine[{0, 0}, {1, 1}]]'
Infinity
```
