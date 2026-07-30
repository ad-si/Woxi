# `AssociationThread`

```scrut
$ wo 'AssociationThread[{a, b, c}, {1, 2, 3}]'
<|a -> 1, b -> 2, c -> 3|>
```

A single value is shared by every key:

```scrut
$ wo 'AssociationThread[{1, 2}, 3]'
<|1 -> 3, 2 -> 3|>
```

Two lists have to be the same length:

```scrut
$ wo 'AssociationThread[{1, 2}, {3}]'

AssociationThread::idim: {1, 2} and {3} must have the same length.
AssociationThread[{1, 2}, {3}]
```
