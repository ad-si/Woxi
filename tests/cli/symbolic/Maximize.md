# `Maximize`

Global symbolic maximization of a function over variables with optional constraints.

```scrut
$ wo 'Maximize[x^2 - 4*x + 5, x]'

Maximize::natt: The maximum is not attained at any point satisfying the given constraints\.\s? (regex)
{Infinity, {x -> -Infinity}}
```

Constraints that leave the objective unbounded report the same way.

```scrut
$ wo 'Maximize[{x^2, x > 1}, x]'

Maximize::natt: The maximum is not attained at any point satisfying the given constraints\.\s? (regex)
{Infinity, {x -> Infinity}}
```

A bounded region gives an exact optimum,
including for a periodic objective whose critical points form a family.

```scrut
$ wo 'Maximize[{Sin[x], 0 < x < 2*Pi}, x]'
{1, {x -> Pi/2}}
```
