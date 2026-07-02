# `Highlighted`

Display wrapper that draws a colored background behind its argument.
Without a front-end it prints verbatim, matching `wolframscript`, while its
argument is still evaluated.

```scrut
$ wo 'Highlighted[5]'
Highlighted[5]
```

The argument is evaluated before being wrapped.

```scrut
$ wo 'Highlighted[1 + 2]'
Highlighted[3]
```

Like other display wrappers, `Highlighted` nests.

```scrut
$ wo 'NestList[Highlighted, x, 2]'
{x, Highlighted[x], Highlighted[Highlighted[x]]}
```
