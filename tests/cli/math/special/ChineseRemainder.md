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

Without congruences there is nothing to solve, so the call is left unevaluated.

```scrut
$ wo 'ChineseRemainder[{}, {}]'
ChineseRemainder[{}, {}]
```

Lists of differing length are reported as an argument error.

```scrut
$ wo 'ChineseRemainder[{1, 2}, {3}]'
ChineseRemainder::pilist: The arguments to ChineseRemainder must be two lists of integers of identical length, with the second list containing only positive integers.
ChineseRemainder[{1, 2}, {3}]
```
