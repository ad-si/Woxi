# `Tally`

Counts occurrences of each distinct element.

```scrut
$ wo 'Tally[{a, b, a, c, b, a}]'
{{a, 3}, {b, 2}, {c, 1}}
```

```scrut
$ wo 'Tally[{1, 2, 1, 3}]'
{{1, 2}, {2, 1}, {3, 1}}
```

```scrut
$ wo 'Tally[{x, x, x}]'
{{x, 3}}
```

```scrut
$ wo 'Tally[{}]'
{}
```

On an association, the values are tallied.

```scrut
$ wo 'Tally[<|a -> 1, b -> 1, c -> 2|>]'
{{1, 2}, {2, 1}}
```
