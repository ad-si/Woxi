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

Under `ToString`, the wrapper renders as an aligned text grid: columns are
left-aligned and padded to the widest cell, separated by three spaces.

```scrut
$ wo 'ToString[TableForm[{{1, 2, 3}}]]'
1   2   3
```
