# `Not`

Logical NOT operation.

```scrut
$ wo 'Not[True]'
False
```

```scrut
$ wo 'Not[False]'
True
```

```scrut
$ wo 'Not[True, True]'

Not::argx: Not called with 2 arguments; 1 argument is expected.
.* (regex*)
Not[True, True]
```

An unevaluated negation is written with the operator, and bracketed wherever
the operator would otherwise reach past what it negates — arithmetic,
comparison and application all bind tighter than `!`:

```scrut
$ wo 'ToString[Hold[Not[a]^2], InputForm]'
Hold[( !a)^2]
```

```scrut
$ wo 'ToString[Hold[Not[a] && b], InputForm]'
Hold[ !a && b]
```

A function body is written the same way:

```scrut
$ wo 'ToString[!#1 || !#2 &, InputForm]'
 !#1 ||  !#2 & 
```
