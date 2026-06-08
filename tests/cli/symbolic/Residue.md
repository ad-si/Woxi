# `Residue`

Computes the residue of an expression at a given point — the coefficient of
`(z - z0)^(-1)` in its Laurent expansion.

```scrut
$ wo 'Residue[1/z, {z, 0}]'
1
```

```scrut
$ wo 'Residue[1/(z^2 - 1), {z, 1}]'
1/2
```

```scrut
$ wo 'Residue[Exp[z]/z^3, {z, 0}]'
1/2
```

```scrut
$ wo 'Residue[1/z^2, {z, 0}]'
0
```

```scrut
$ wo 'Residue[1/(z^2 + 1), {z, I}]'
-1/2*I
```

```scrut
$ wo 'Residue[Cot[z], {z, 0}]'
1
```

```scrut
$ wo 'Residue[1/(z - a), {z, a}]'
1
```
