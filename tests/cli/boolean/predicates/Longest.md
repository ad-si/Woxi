# `Longest`

`Longest[p]` tries the longest split of a sequence pattern first; the default
is the shortest, which is what `Shortest[p]` asks for explicitly.

```scrut
$ wo '{1, 2, 3} /. {Longest[x__], y__} :> {{x}, {y}}'
{{1, 2}, {3}}
```

```scrut
$ wo '{1, 2, 3} /. {Shortest[x__], y__} :> {{x}, {y}}'
{{1}, {2, 3}}
```

Around anything that is not a sequence the wrapper is transparent:

```scrut
$ wo '{{1, 2, 3} /. {Longest[x_], ___} :> x, MatchQ[{1, 2}, {Longest[__], _}]}'
{1, True}
```
