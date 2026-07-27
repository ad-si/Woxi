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

`"SOP"` and `"POS"` are other names for those two. Beyond them a form can ask
for one connective only — a sum of products is a `Nand` of `Nand`s, a product
of sums a `Nor` of `Nor`s:

```scrut
$ wo 'BooleanConvert[Xor[a, b], "NAND"]'
Nand[Nand[a,  !b], Nand[ !a, b]]
```

```scrut
$ wo 'BooleanConvert[(a || b) && c, "NOR"]'
Nor[Nor[a, b],  !c]
```

`"ANF"` gives the algebraic normal form: an `Xor` of conjunctions of plain
variables, shorter conjunctions first.

```scrut
$ wo 'BooleanConvert[a || b, "ANF"]'
Xor[a, b, a && b]
```

`"IF"` gives a decision tree over the variables, skipping any that both
branches agree on — here the `a` branch never looks at `b`:

```scrut
$ wo 'BooleanConvert[(a || b) && c, "IF"]'
If[a, If[c, True, False], If[b, If[c, True, False], False]]
```

`Majority` holds when more than half its arguments do:

```scrut
$ wo 'BooleanConvert[Majority[a, b, c], "DNF"]'
(a && b) || (a && c) || (b && c)
```

An expression with only one value has no normal form to write out:

```scrut
$ wo 'BooleanConvert[a || !a, "CNF"]'
True
```
