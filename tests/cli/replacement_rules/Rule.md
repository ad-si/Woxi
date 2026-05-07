# `Rule`

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
