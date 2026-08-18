# `PatternTest`

Tests whether a pattern matches using a predicate function (pattern?test).

```scrut
$ wo 'PatternTest[x_, IntegerQ]'
(x_)?IntegerQ
```

The left side of `?` may be any self-delimiting expression,
not just a blank pattern:

```scrut
$ wo 'Cases[{-3, 0, 5, 7, 12, "text", 7.5}, x : Except[7]?Positive]'
{5, 12, 7.5}
```

```scrut
$ wo 'f[x : Except[0]?NumericQ] := 1/x; f[4]'
1/4
```
