# `FortranForm`

Format expression as Fortran code.

```scrut
$ wo 'FortranForm[Exp[x]]'
FortranForm[E^x]
```

`**` binds tighter than everything else, so compound bases and exponents are
parenthesized:

```scrut
$ wo 'ToString[FortranForm[(a + b)^2]]'
(a + b)**2
```

The comparison and logical operators are spelled between dots:

```scrut
$ wo 'ToString[FortranForm[x < y && ! z]]'
x.lt.y.and..not.z
```
