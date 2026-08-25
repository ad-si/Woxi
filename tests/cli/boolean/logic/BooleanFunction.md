# `BooleanFunction`

`BooleanFunction[k, n]` is the `k`-th Boolean function of `n` variables. The
index is read as a truth table: bit `v` of `k` is the value for the assignment
`v`, written as a binary number with the first variable most significant and
True as 1. Index 7 of two variables is therefore `Nand`, index 1 is `Nor` and
index 6 is `Xor`.

The bare form is an opaque object carrying the function's reduced ordered
binary decision diagram:

```scrut
$ wo 'BooleanFunction[7, 2]'
BooleanFunction[BDD -> {-2, 0, 2, -1, 1, 1, -1}]
```

The leading `±n` is the variable count, negative when the root edge is
complemented, and each following triple is a node `{var, then, else}` in the
order a then-branch-first walk from the root first reaches them. A branch
refers to the `|v|`-th such node, negated when `v` is negative, except that
`±1` is the True/False terminal. A constant function is just the signed count:

```scrut
$ wo 'BooleanFunction[0, 2]'
BooleanFunction[BDD -> {-2}]
```

Applied to Boolean arguments it answers directly, and integer `1`/`0` count as
True/False:

```scrut
$ wo 'BooleanFunction[7, 2][True, False]'
True
```

```scrut
$ wo 'BooleanFunction[7, 2][1, 1]'
False
```

Some arguments literal and some symbolic restricts the function to the rest —
`Nand[a, True]` is `Not[a]`, the one-variable function of index 1:

```scrut
$ wo 'BooleanFunction[7, 2][a, True]'
BooleanFunction[BDD -> {-1, 0, 1, -1}][a]
```

Symbolic arguments alone leave the application as it stands; `BooleanConvert`
writes it out:

```scrut
$ wo 'BooleanConvert[BooleanFunction[6, 2][a, b]]'
(a &&  !b) || ( !a && b)
```

Given a list of variables instead of a count, `BooleanFunction` writes the
expression out directly, in minimal sum-of-products form:

```scrut
$ wo 'BooleanFunction[7, 2, {a, b}]'
 !a ||  !b
```

```scrut
$ wo 'BooleanFunction[2, 3, {a, b, c}]'
 !a &&  !b && c
```

The index is taken two's-complement, so index `-1` is the constant True:

```scrut
$ wo 'BooleanFunction[-1, 2, {a, b}]'
True
```
