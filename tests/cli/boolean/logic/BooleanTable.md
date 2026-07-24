# `BooleanTable`

Truth table for a boolean expression over all variable combinations.

```scrut
$ wo 'BooleanTable[Not[p], {p}]'
{False, True}
```

Without a variable list the expression's own variables are used.

```scrut
$ wo 'BooleanTable[Implies[p, q]]'
{True, False, True, True}
```

Several variable lists give one nesting level each,
with the outermost level varying the first group.

```scrut
$ wo 'BooleanTable[p || q, {p}, {q}]'
{{True, True}, {True, False}}
```
