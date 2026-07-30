# `LinearProgramming`

Finds a vector `x` minimizing `c . x` subject to `m . x >= b` and `x >= 0`,
solved with an exact simplex so the answer stays rational:

```scrut
$ wo 'LinearProgramming[{1, 1}, {{1, 2}, {3, 1}}, {3, 3}]'
{3/5, 6/5}
```

A `{value, sign}` pair in place of a bare right-hand side picks the relation:
`1` for `>=`, `0` for `==`, `-1` for `<=`:

```scrut
$ wo 'LinearProgramming[{2, 3}, {{1, 1}}, {{10, 0}}]'
{10, 0}
```

```scrut
$ wo 'LinearProgramming[{-2, -3}, {{1, 1}, {2, 1}}, {{4, -1}, {6, -1}}]'
{0, 4}
```

A fourth argument replaces the default `x >= 0`. It may be a single lower bound
shared by every variable, a vector of lower bounds, or a matrix of
`{lower, upper}` pairs:

```scrut
$ wo 'LinearProgramming[{1, 1}, {{1, 2}, {3, 1}}, {3, 4}, {{2, 3}, {2, 3}}]'
{2, 2}
```

```scrut
$ wo 'LinearProgramming[{-1, -2, -3}, {{1, 1, 1}}, {{6, -1}}, {{0, 2}, {0, 2}, {0, 2}}]'
{2, 2, 2}
```

A bound may be `Infinity` or `-Infinity`, which leaves the variable free in that
direction:

```scrut
$ wo 'LinearProgramming[{1, -1}, {{1, 1}}, {{2, 0}}, {{-5, 5}, {-5, 5}}]'
{-3, 5}
```

```scrut
$ wo 'LinearProgramming[{1, 1}, {{1, 2}, {3, 1}}, {3, 4}, -Infinity]'
{1, 1}
```

Bounds that leave nothing feasible are reported like any other infeasible
problem:

```scrut
$ wo 'LinearProgramming[{1, 1}, {{1, 2}, {3, 1}}, {3, 4}, {{5, 0}, {5, 0}}]'

LinearProgramming::lpsnf: No solution can be found that satisfies the constraints.
LinearProgramming[{1, 1}, {{1, 2}, {3, 1}}, {3, 4}, {{5, 0}, {5, 0}}]
```

A bound specification of the wrong shape is refused:

```scrut
$ wo 'LinearProgramming[{1, 1}, {{1, 2}, {3, 1}}, {3, 4}, {{0, 5}, 2}]'

LinearProgramming::lprank012: {{0, 5}, 2} must be a scalar, a vector or a matrix with 2 columns.
LinearProgramming[{1, 1}, {{1, 2}, {3, 1}}, {3, 4}, {{0, 5}, 2}]
```
