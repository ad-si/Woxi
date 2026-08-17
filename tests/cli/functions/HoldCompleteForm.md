# `HoldCompleteForm`

Holds an expression completely unevaluated for display.

```scrut
$ wo 'Attributes[HoldCompleteForm]'
{HoldAllComplete, Protected}
```

```scrut
$ wo 'HoldCompleteForm[2 + 3]'
HoldCompleteForm[2 + 3]
```

Unlike `HoldForm`, an inner `Evaluate` does not break through the hold:

```scrut
$ wo 'HoldCompleteForm[Evaluate[1 + 2]]'
HoldCompleteForm[Evaluate[1 + 2]]
```
