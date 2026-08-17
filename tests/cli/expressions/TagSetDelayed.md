# `TagSetDelayed`

Defines a delayed upvalue assignment associated with a tag symbol. The
`tag /: lhs := rhs` shorthand is the usual way to write it.

```scrut
$ wo 'TagSetDelayed[g, f[g[x_]], 1 + 2]; f[g[5]]'
3
```

Upvalues attached to `Plus`, `Times` and `Power` also apply to the arithmetic
shorthands that expand to those heads: `a - b` is `Plus[a, Times[-1, b]]`,
`a / b` is `Times[a, Power[b, -1]]` and `-a` is `Times[-1, a]`.

```scrut
$ wo 'mytag /: mytag[a_] + mytag[b_] := mytag[a + b]; mytag /: i_Integer mytag[a_] := mytag[a i]; mytag[3] - mytag[3]'
mytag[0]
```

```scrut
$ wo 'mytag /: i_Integer mytag[a_] := mytag[a i]; -mytag[3]'
mytag[-3]
```

```scrut
$ wo 'q /: q[a_] q[b_] := q[a b]; q /: q[a_]^n_Integer := q[a^n]; q[6] / q[3]'
q[2]
```

Together they implement modular arithmetic, where `m[a, n]` stands for
`a` modulo `n`:

```scrut
$ wo 'm /: m[a_, n_] + m[b_, n_] := m[Mod[a + b, n], n]; m /: c_Integer m[a_, n_] := m[Mod[c a, n], n]; m[3, 7] - m[6, 7]'
m[4, 7]
```
