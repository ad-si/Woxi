# `Refine`

Simplifies expressions using assumptions.

```scrut
$ wo 'Refine[x]'
x
```

A finite factor with a known sign collapses `factor * Infinity` to the
correctly-signed infinity:

```scrut
$ wo 'Refine[a*Infinity, a > 0]'
Infinity
```

```scrut
$ wo 'Refine[a*Infinity, a < 0]'
-Infinity
```
