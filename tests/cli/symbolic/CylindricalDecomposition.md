# `CylindricalDecomposition`

Describes the real solution set of a condition cylindrically: the first
variable has constant bounds, and each later variable is bounded only by
the variables listed before it.

```scrut
$ wo 'CylindricalDecomposition[x + y > 1 && x < 2, {x, y}]'
x < 2 && y > 1 - x
```

An equation fixes a variable in terms of the earlier ones:

```scrut
$ wo 'CylindricalDecomposition[x == 2 y && y > 1, {x, y}]'
x > 2 && y == x/2
```
