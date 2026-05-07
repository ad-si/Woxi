# `Replace`

Replaces the entire expression (or a specific level) using a rule.
Unlike `ReplaceAll`, the rule applies to the whole expression by default.

```scrut
$ wo 'Replace[x + y, x -> a]'
x + y
```

```scrut
$ wo 'Replace[{1, 2, 3}, x_ -> x^2, {1}]'
{1, 4, 9}
```
