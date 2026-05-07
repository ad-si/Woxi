# `AtomQ`

Tests whether an expression is atomic — i.e. has no parts.
Strings and numbers are atoms, lists and function calls are not.

```scrut
$ wo 'AtomQ[5]'
True
```

```scrut
$ wo 'AtomQ["hello"]'
True
```

```scrut
$ wo 'AtomQ[{1, 2}]'
False
```
