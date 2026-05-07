# `Subsets`

Generates subsets (combinations) of a list.

```scrut
$ wo 'Subsets[{a, b, c}]'
{{}, {a}, {b}, {c}, {a, b}, {a, c}, {b, c}, {a, b, c}}
```

```scrut
$ wo 'Subsets[{a, b, c}, {2}]'
{{a, b}, {a, c}, {b, c}}
```

```scrut
$ wo 'Subsets[{a, b, c}, {0}]'
{{}}
```

```scrut
$ wo 'Subsets[{1, 2, 3, 4}, {3}]'
{{1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}}
```
