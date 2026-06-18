# `FullForm` - Prints the full form of the expression

In `wolframscript`'s script (`-code`) mode the result is printed in InputForm,
so `FullForm[expr]` echoes with its wrapper as `FullForm[<expr in InputForm>]`
rather than as the bare `Head[...]` tree. Woxi matches this exactly.

```scrut
$ wo 'FullForm[x+y+z]'
FullForm[x + y + z]
```

```scrut
$ wo 'FullForm[x y z]'
FullForm[x*y*z]
```

```scrut
$ wo 'FullForm[5*x]'
FullForm[5*x]
```

```scrut
$ wo 'FullForm[x^n]'
FullForm[x^n]
```

```scrut
$ wo 'FullForm[{a,b,c}]'
FullForm[{a, b, c}]
```

```scrut
$ wo 'FullForm[a->b]'
FullForm[a -> b]
```

<!-- `a = b` evaluates to `b` before FullForm sees it. -->
```scrut
$ wo 'FullForm[a=b]'
FullForm[b]
```

```scrut
$ wo 'FullForm[a b + c]'
FullForm[a*b + c]
```

```scrut
$ wo 'FullForm[a (b + c)]'
FullForm[a*(b + c)]
```
