# `BooleanConvert`

Convert Boolean expressions.

```scrut
$ wo 'BooleanConvert[x]'
x
```

Distributing a sum of products into a product of sums leaves no repeated
literals and no clause that a shorter one already covers:

```scrut
$ wo 'BooleanConvert[Xor[a, b, c], "CNF"]'
( !a ||  !b || c) && ( !a || b ||  !c) && (a ||  !b ||  !c) && (a || b || c)
```

The literals of a clause follow the variables they mention, whether or not
they are negated:

```scrut
$ wo 'BooleanConvert[Equivalent[a, b], "CNF"]'
( !a || b) && (a ||  !b)
```
