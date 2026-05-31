# `MittagLefflerE`

Mittag-Leffler function.

Two-argument form `MittagLefflerE[a, z]` is
`Sum[z^k / Gamma[a*k + 1], {k, 0, Infinity}]`; the three-argument form
`MittagLefflerE[a, b, z]` is `Sum[z^k / Gamma[a*k + b], {k, 0, Infinity}]`.

For exact integer `a` in `{0, 1, 2}` the two-argument form has closed forms:

```scrut
$ wo 'MittagLefflerE[0, z]'
(1 - z)^(-1)
```

```scrut
$ wo 'MittagLefflerE[1, x]'
E^x
```

```scrut
$ wo 'MittagLefflerE[2, z]'
Cosh[Sqrt[z]]
```

```scrut
$ wo 'MittagLefflerE[2, 1]'
Cosh[1]
```

The three-argument form with `b == 1` reduces to the two-argument form:

```scrut
$ wo 'MittagLefflerE[2, 1, z]'
Cosh[Sqrt[z]]
```

Arbitrary-precision evaluation works through the closed form:

```scrut
$ wo 'N[MittagLefflerE[2, 1], 20]'
1.5430806348152437784779056207570616826`20.
```

At `z == 0` only the `k == 0` term survives, so the two-argument form is `1`
and the three-argument form is `1/Gamma[b]`:

```scrut
$ wo 'MittagLefflerE[a, 0]'
1
```

```scrut
$ wo 'MittagLefflerE[a, b, 0]'
Gamma[b]^(-1)
```

The function threads over lists (it is `Listable`):

```scrut
$ wo 'MittagLefflerE[2, {1.0, 2.0}]'
{1.5430806348152437, 2.178183556608571}
```

Symbolic arguments without a closed form stay unevaluated:

```scrut
$ wo 'MittagLefflerE[a, z]'
MittagLefflerE[a, z]
```
