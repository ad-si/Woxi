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

The tag has to sit where the stored rule can be found again, but the wrappers
that only name a pattern or restrict it are transparent — which is how an
object system writes its formatting rule:

```scrut
$ wo 'obj /: fmt[o : obj[_Symbol]] := "boxed"; fmt[obj[x]]'
boxed
```

```scrut
$ wo 'obj /: fmt[obj[_]?q] := 1; Length[UpValues[obj]]'
1
```

A tag buried deeper than that has nowhere to hang the rule, and nothing is
defined:

```scrut
$ wo 'obj /: fmt[wrap[obj[_]]] := 1; Length[UpValues[obj]]'

TagSetDelayed::tagpos: Tag obj in fmt[wrap[obj[_]]] is too deep for an assigned rule to be found.
0
```

An assignment also consults the upvalues of the symbols on its *right*, so a
symbol can give `sym := f[…]` a meaning of its own:

```scrut
$ wo 'wrapper /: SetDelayed[s_, wrapper[a_]] := (s := held[a]); tpl := wrapper["x"]; tpl'
held[x]
```

An upvalue on `Set` sees the right-hand side's *value*, which `Set` works out
once. Here the constructor runs once however many times the body names it:

```scrut
$ wo 'n = 0; T[o___Rule] := (n++; T[Unique["t$"]]); T /: Set[name_Symbol, object_T] := (object; object; object; name); p = T["x" -> 1]; n'
1
```
