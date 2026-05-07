# `ExactNumberQ`

Tests whether a number is exact (integer, rational, etc. but not a machine float).

```scrut
$ wo 'ExactNumberQ[3]'
True
```

```scrut
$ wo 'ExactNumberQ[1/2]'
True
```

```scrut
$ wo 'ExactNumberQ[3.5]'
False
```
