# `TableForm`

Display wrapper that returns unevaluated in text/CLI mode, matching
`wolframscript` behavior. Arguments are still evaluated.

```scrut
$ wo 'TableForm[{a, b, c}]'
TableForm[{a, b, c}]
```

```scrut
$ wo 'TableForm[{{1, 2, 3}, {4, 5, 6}}]'
TableForm[{{1, 2, 3}, {4, 5, 6}}]
```

```scrut
$ wo 'TableForm[Table[{i, i^2}, {i, 3}]]'
TableForm[{{1, 1}, {2, 4}, {3, 9}}]
```
