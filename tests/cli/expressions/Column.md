# `Column`

Displays elements in a vertical column. Without a front-end it prints verbatim,
matching `wolframscript`.

```scrut
$ wo 'Column[{1, 2, 3}]'
Column[{1, 2, 3}]
```

Under `ToString` the elements are stacked one per line.

```scrut
$ wo 'ToString[Column[{1, 2, 3}]]'
1
2
3
```
