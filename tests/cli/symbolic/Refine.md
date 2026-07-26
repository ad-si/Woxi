# `Refine`

Simplifies expressions using assumptions.

```scrut
$ wo 'Refine[x]'
x
```

A finite factor with a known sign collapses `factor * Infinity` to the
correctly-signed infinity:

```scrut
$ wo 'Refine[a*Infinity, a > 0]'
Infinity
```

```scrut
$ wo 'Refine[a*Infinity, a < 0]'
-Infinity
```

A condition the assumption settles collapses the head around it:

```scrut
$ wo 'Refine[Boole[x > 0], x > 0]'
1
```

```scrut
$ wo 'Refine[If[x > 0, a, b], x < 0]'
b
```

Step functions resolve once the assumption fixes the sign of the argument:

```scrut
$ wo 'Refine[UnitStep[x], x < 0]'
0
```

A chained inequality settles the rounding functions whenever the range cannot
straddle a boundary:

```scrut
$ wo '{Refine[Floor[x], 0 < x < 1], Refine[Ceiling[x], 0 < x < 1], Refine[IntegerPart[x], -1 < x < 0]}'
{0, 1, 0}
```

A range that can straddle one is left alone — `x` may be 1 below, and `Round`
turns over at 1/2:

```scrut
$ wo '{Refine[Floor[x], 0 < x <= 1], Refine[Round[x], 0 < x < 1]}'
{Floor[x], Round[x]}
```

The sign such a range implies reaches the other refinements too:

```scrut
$ wo '{Refine[Sign[x], 0 < x < 1], Refine[Abs[x], 0 < x < 1]}'
{1, x}
```
