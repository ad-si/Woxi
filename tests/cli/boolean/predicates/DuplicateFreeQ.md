# `DuplicateFreeQ`

Tests whether a list has no duplicated elements.

```scrut
$ wo 'DuplicateFreeQ[{1, 7, 8, 4, 3, 4, 1, 9, 9, 2}]'
False
```

No duplicates:

```scrut
$ wo 'DuplicateFreeQ[{1, 2, 3, 4}]'
True
```

An empty list is duplicate-free:

```scrut
$ wo 'DuplicateFreeQ[{}]'
True
```

A custom test decides which elements count as duplicates:

```scrut
$ wo 'DuplicateFreeQ[{1, -1, 2, -3}, (Abs[#1] == Abs[#2]) &]'
False
```
