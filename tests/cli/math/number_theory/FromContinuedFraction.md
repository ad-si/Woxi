# `FromContinuedFraction`

Converts a continued fraction to a number.

```scrut
$ wo 'FromContinuedFraction[{}]'
Infinity
```

Symbolic terms are combined into a single fraction.

```scrut
$ wo 'FromContinuedFraction[{a, b, c}]'
(a + c + a*b*c)/(1 + b*c)
```
