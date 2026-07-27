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

`//`, `/.` and `//.` all bind looser than `->`, so a rule written as an
argument or a list item can be handed to a postfix function or replaced in:

```scrut
$ wo 'ToString[f[a -> 5 // Head], InputForm]'
f[Rule]
```

```scrut
$ wo 'ToString[{a -> 5 /. 5 -> 6}, InputForm]'
{a -> 6}
```

A rule answers to a head replacement the way any other expression does:

```scrut
$ wo 'ToString[(1 -> 2) /. Rule -> List, InputForm]'
{1, 2}
```

```scrut
$ wo 'ToString[{a -> 1} /. Rule -> ff, InputForm]'
{ff[a, 1]}
```
