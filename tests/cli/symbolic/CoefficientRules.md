# `CoefficientRules`

Coefficient rules for polynomials.

```scrut
$ wo 'CoefficientRules[0, x]'
{}
```

With one argument the variables are detected automatically.

```scrut
$ wo 'CoefficientRules[(x + y)^3]'
{{3, 0} -> 1, {2, 1} -> 3, {1, 2} -> 3, {0, 3} -> 1}
```
