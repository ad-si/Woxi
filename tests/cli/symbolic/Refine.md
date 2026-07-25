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
