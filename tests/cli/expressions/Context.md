# `Context`

Returns the context a symbol belongs to. `Context[]` gives the current one.

```scrut
$ wo 'Context[]'
Global`
```

```scrut
$ wo 'Context[Plus]'
System`
```

`Context` holds its argument, so it reports the context of the symbol rather
than of its value:

```scrut
$ wo 'x = 1; Context[x]'
Global`
```

A symbol written with a context belongs to it. This is what a package's
`Begin["`Private`"]` builds on: names read inside it become symbols of the
private context, so they mean nothing outside the package.

```scrut
$ wo 'A`b = 5; {A`b, Context[A`b]}'
{5, A`}
```

A name given as a string is resolved the same way, and one that names no
symbol is reported:

```scrut
$ wo 'Context["b"]'

Context::notfound: Symbol b not found.
Context[b]
```
