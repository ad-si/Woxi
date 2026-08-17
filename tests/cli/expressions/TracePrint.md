# `TracePrint`

Print every sub-expression used while evaluating an expression and
return the result of the evaluation.
Each line is indented by one space per level of the evaluation,
so nested computations are visibly deeper than their enclosing ones.

```scrut
$ wo 'TracePrint[2 + 3]'
 2 + 3
  Plus
  2
  3
 5
5
```

```scrut
$ wo 'TracePrint[2^3 + 5]'
 2^3 + 5
  Plus
  2^3
   Power
   2
   3
  8
  5
 8 + 5
 13
13
```

A second argument restricts the printing to sub-expressions
matching the given pattern:

```scrut
$ wo 'TracePrint[2^3 + 5, _Integer]'
   2
   3
  8
  5
 13
13
```
