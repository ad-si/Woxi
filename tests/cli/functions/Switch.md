# `Switch`

Multi-way branching — the first pattern that matches wins.

```scrut
$ wo 'Switch[2, 1, "one", 2, "two", _, "other"]'
two
```

Each candidate is evaluated as it is tried, so an expression works as a
pattern:

```scrut
$ wo 'Switch[3, 1 + 1, "two", 2 + 1, "three"]'
three
```
