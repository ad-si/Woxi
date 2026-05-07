# `Gather`

Groups identical elements together, maintaining order of first appearance.

```scrut
$ wo 'Gather[{1, 1, 2, 2, 1}]'
{{1, 1, 1}, {2, 2}}
```

```scrut
$ wo 'Gather[{a, b, a, c, b}]'
{{a, a}, {b, b}, {c}}
```
