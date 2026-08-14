# `Names`

Lists symbol names matching a string pattern with wildcards.

```scrut
$ wo 'Names["List"]'
{List}
```

A pattern's context part selects the context and its name part the symbol.
A pattern without a context searches the contexts on `$ContextPath`, which
is why the plain form above finds the built-ins — they are `System`` symbols.

```scrut
$ wo 'A`b = 5; A`c = 6; Names["A`*"]'
{A`b, A`c}
```

A context that is not on `$ContextPath` is listed under its full name, and
`Names` never reaches across a backtick: `A`*` does not match `A`Sub`x`.

```scrut
$ wo 'A`b = 5; A`Sub`x = 1; {Names["A`*"], Names["A`Sub`*"]}'
{{A`b}, {A`Sub`x}}
```

Names are listed case-insensitively, with `$` sorting after the letters:

```scrut
$ wo 'zz$a::usage = "u"; zzb::usage = "u"; zz1::usage = "u"; zzA::usage = "u"; Names["zz*"]'
{zz1, zzA, zzb, zz$a}
```
