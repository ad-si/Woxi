# `Permutations`

Generates all permutations of a list.

```scrut
$ wo 'Permutations[{a, b, c}]'
{{a, b, c}, {a, c, b}, {b, a, c}, {b, c, a}, {c, a, b}, {c, b, a}}
```

```scrut
$ wo 'Permutations[{1, 2, 3}, {2}]'
{{1, 2}, {1, 3}, {2, 1}, {2, 3}, {3, 1}, {3, 2}}
```

```scrut
$ wo 'Permutations[{1, 2}, {1}]'
{{1}, {2}}
```

```scrut
$ wo 'Permutations[{a}]'
{{a}}
```

```scrut
$ wo 'Permutations[{}]'
{{}}
```

```scrut
$ wo 'Length[Permutations[Range[4]]]'
24
```
