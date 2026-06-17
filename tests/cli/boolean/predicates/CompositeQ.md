# `CompositeQ`

Tests if a number is composite.

```scrut
$ wo 'CompositeQ[2^128]'
True
```

It threads over a list.

```scrut
$ wo 'CompositeQ[{4, 5, 6}]'
{True, False, True}
```
