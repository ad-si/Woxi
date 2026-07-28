# Asymptotic comparisons

`AsymptoticLess`, `AsymptoticLessEqual`, `AsymptoticGreater`,
`AsymptoticGreaterEqual`, `AsymptoticEqual`, and `AsymptoticEquivalent` compare
the growth of two expressions as a variable approaches a limit point. They are
Landau notation: `o`, `O`, `ω`, `Ω`, `Θ`, and `~` respectively.

`AsymptoticLess[f, g, x -> a]` asks whether `f` is negligible next to `g`, i.e.
whether `f ∈ o(g)`:

```scrut
$ wo 'AsymptoticLess[x, x^2, x -> Infinity]'
True
```

```scrut
$ wo 'AsymptoticLess[Log[x], x, x -> Infinity]'
True
```

The comparison depends on where the limit is taken — near 0 the higher power is
the smaller one:

```scrut
$ wo 'AsymptoticLess[x, x^2, x -> 0]'
False
```

```scrut
$ wo 'AsymptoticLess[x^2, x, x -> 0]'
True
```

`AsymptoticLessEqual` is the `O` relation, so unlike `AsymptoticLess` it holds
between functions of the same order:

```scrut
$ wo 'AsymptoticLessEqual[x, x, x -> Infinity]'
True
```

```scrut
$ wo 'AsymptoticLess[x, x, x -> Infinity]'
False
```

It only needs the ratio to stay bounded, not to converge, so a bounded
oscillation qualifies:

```scrut
$ wo 'AsymptoticLessEqual[Sin[x], 1, x -> Infinity]'
True
```

`AsymptoticGreater` and `AsymptoticGreaterEqual` are the same relations with
the arguments exchanged:

```scrut
$ wo 'AsymptoticGreater[Exp[x], x^3, x -> Infinity]'
True
```

`AsymptoticEqual` is the `Θ` relation — bounded in both directions — so it
ignores constant factors and lower-order terms:

```scrut
$ wo 'AsymptoticEqual[3 x + 1, x, x -> Infinity]'
True
```

`AsymptoticEquivalent` is stricter: it requires the ratio to reach exactly 1,
which a constant factor destroys:

```scrut
$ wo 'AsymptoticEquivalent[x + 1, x, x -> Infinity]'
True
```

```scrut
$ wo 'AsymptoticEquivalent[2 x, x, x -> Infinity]'
False
```
