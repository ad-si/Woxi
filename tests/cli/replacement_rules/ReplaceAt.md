# `ReplaceAt`

Applies transformation rules to the parts of an expression at the given
position(s). The position is specified the same way as `Position` output, and
each targeted part is transformed with the first matching rule (parts that
match no rule are left unchanged).

Replace a single part:

```scrut
$ wo 'ReplaceAt[{a, a, a, a}, a -> xx, 2]'
{a, xx, a, a}
```

Replace several parts at once:

```scrut
$ wo 'ReplaceAt[{a, a, a, a}, a -> xx, {{1}, {4}}]'
{xx, a, a, xx}
```

Address a nested part:

```scrut
$ wo 'ReplaceAt[{{a, a}, {a, a}}, a -> xx, {2, 1}]'
{{a, a}, {xx, a}}
```

Delayed rules evaluate per match:

```scrut
$ wo 'ReplaceAt[{1, 2, 3, 4}, x_ :> 2 x - 1, {{2}, {4}}]'
{1, 3, 3, 7}
```

A list of rules uses the first matching rule at each position:

```scrut
$ wo 'ReplaceAt[{a, b, c, d}, {a -> xx, _ -> yy}, {{1}, {2}, {4}}]'
{xx, yy, c, yy}
```

The operator form `ReplaceAt[rules, pos]` is a function:

```scrut
$ wo 'ReplaceAt[a -> xx, 2][{a, a, a, a}]'
{a, xx, a, a}
```
