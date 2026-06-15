# `SubtractSides`

Subtract an expression from both sides of an equation.

```scrut
$ wo 'SubtractSides[x + 3 == 5, 3]'
x == 2
```

When the second argument is itself an equation, the matching sides are paired.

```scrut
$ wo 'SubtractSides[a == b, c == d]'
a - c == b - d
```
