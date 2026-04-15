# Replacement rules tests

## `ReplaceAll` (`/.`)

Replaces parts of an expression according to rules.

```scrut
$ wo '{a, b} /. a -> x'
{x, b}
```


## `Replace`

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


## More `ReplaceAll` examples

```scrut
$ wo '{x^2, y^2} /. x -> 2'
{4, y^2}
```


## `ReplaceRepeated` (`//.`)

Repeatedly applies transformation rules until no more changes occur.

```scrut
$ wo 'f[f[f[2]]] //. f[x_] -> x'
2
```

```scrut
$ wo 'f[f[f[f[2]]]] //. f[2] -> 2'
2
```

```scrut
$ wo 'f[f[f[2]]] //. f[x_] -> x + 1'
5
```


## `Rule` and `RuleDelayed`

`Rule[a, b]` creates an immediate replacement rule (`a -> b`).
`RuleDelayed[a, b]` creates a delayed rule (`a :> b`) whose right-hand
side is re-evaluated every time the rule fires.

```scrut
$ wo 'Rule[a, b]'
a -> b
```

```scrut
$ wo 'RuleDelayed[x, Random[]]'
x :> Random[]
```
