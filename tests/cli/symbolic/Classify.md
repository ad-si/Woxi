# `Classify`

Trains a classifier on labeled examples. The trained `ClassifierFunction`
is applied to new examples to decide their class.

```scrut
$ wo 'c = Classify[<|"small" -> {1, 2, 3}, "large" -> {10, 11, 12}|>]; {c[1.5], c[11]}'
{small, large}
```

A list of examples is classified element by element.

```scrut
$ wo 'c = Classify[{{0, 0}, {1, 1}, {9, 9}, {10, 10}} -> {"a", "a", "b", "b"}, Method -> "NearestNeighbors"]; c[{{0, 1}, {9, 10}}]'
{a, b}
```
