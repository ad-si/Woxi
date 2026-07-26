# `CForm`

Format expression as C code.

```scrut
$ wo 'CForm[E]'
CForm[E]
```

Sums subtract, and the logical operators are C's:

```scrut
$ wo 'ToString[CForm[3 - 2 x]]'
3 - 2*x
```

```scrut
$ wo 'ToString[CForm[Mod[a, b] && c]]'
a % b && c
```
