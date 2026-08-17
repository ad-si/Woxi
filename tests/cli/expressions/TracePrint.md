# `TracePrint`

Print every sub-expression used while evaluating an expression and
return the result of the evaluation.
Each step is wrapped in `HoldCompleteForm` and indented
by one space per level of the evaluation,
so nested computations are visibly deeper than their enclosing ones.

```scrut
$ wo 'TracePrint[2 + 3]'
 HoldCompleteForm[2 + 3]
  HoldCompleteForm[Plus]
  HoldCompleteForm[2]
  HoldCompleteForm[3]
 HoldCompleteForm[5]
5
```

```scrut
$ wo 'TracePrint[2^3 + 5]'
 HoldCompleteForm[2^3 + 5]
  HoldCompleteForm[Plus]
  HoldCompleteForm[2^3]
   HoldCompleteForm[Power]
   HoldCompleteForm[2]
   HoldCompleteForm[3]
  HoldCompleteForm[8]
  HoldCompleteForm[5]
 HoldCompleteForm[8 + 5]
 HoldCompleteForm[13]
13
```

A second argument restricts the printing to sub-expressions
matching the given pattern:

```scrut
$ wo 'TracePrint[2^3 + 5, _Integer]'
   HoldCompleteForm[2]
   HoldCompleteForm[3]
  HoldCompleteForm[8]
  HoldCompleteForm[5]
 HoldCompleteForm[13]
13
```
