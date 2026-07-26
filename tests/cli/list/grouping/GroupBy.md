# `GroupBy`

Groups elements of a list according to a function.

```scrut
$ wo 'GroupBy[{{a, b}, {a, c}, {b, c}}, First]'
<|a -> {{a, b}, {a, c}}, b -> {{b, c}}|>
```

A list of classifiers sub-groups level by level:

```scrut
$ wo 'GroupBy[{{1, a, x}, {1, b, y}, {2, a, z}}, {First, #[[2]] &}]'
<|1 -> <|a -> {{1, a, x}}, b -> {{1, b, y}}|>, 2 -> <|a -> {{2, a, z}}|>|>
```

A third argument reduces the innermost groups:

```scrut
$ wo 'GroupBy[{{1, a, x}, {1, b, y}, {2, a, z}, {1, a, w}}, {First, #[[2]] &}, Length]'
<|1 -> <|a -> 2, b -> 1|>, 2 -> <|a -> 1|>|>
```
