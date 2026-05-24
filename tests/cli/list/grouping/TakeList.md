# `TakeList`

Splits a list into chunks of given lengths.

```scrut
$ wo 'TakeList[{a, b, c, d, e}, {2, 3}]'
{{a, b}, {c, d, e}}
```

Negative counts take from the end of the remaining slice:

```scrut
$ wo 'TakeList[{a, b, c, d, e, f, g, h}, {-2, -3, -1}]'
{{g, h}, {d, e, f}, {c}}
```

`All` and `UpTo[n]` are also accepted as sequence specifications, and any
head is preserved on each resulting chunk:

```scrut
$ wo 'TakeList[h[a, b, c, d, e, f, g], {2, 3, 1}]'
{h[a, b], h[c, d, e], h[f]}
```
