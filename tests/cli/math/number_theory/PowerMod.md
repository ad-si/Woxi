# `PowerMod`

Fast modular exponentiation — `PowerMod[b, e, m] == Mod[b^e, m]`.

```scrut
$ wo 'PowerMod[2, 10, 100]'
24
```

A unit-fraction exponent `1/n` requests a modular `n`-th root.

```scrut
$ wo 'PowerMod[5, 1/3, 11]'
3
```
