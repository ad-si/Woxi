# `ChineseRemainder`

Solves a system of modular congruences.

```scrut
$ wo 'ChineseRemainder[{1,2,3},{3,5,7}]'
52
```

A third argument returns the smallest solution greater than or equal to it.

```scrut
$ wo 'ChineseRemainder[{1, 2}, {3, 5}, 10]'
22
```
