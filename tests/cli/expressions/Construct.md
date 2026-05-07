# `Construct` - Constructs an expression from its head and arguments

Apply a function to an argument:

```scrut
$ wo 'Construct[f, x]'
f[x]
```

Apply a function to several arguments:

```scrut
$ wo 'Construct[f, x, y, z]'
f[x, y, z]
```

Build a curried function:

```scrut
$ wo 'Fold[Construct, f, {a, b, c}]'
f[a][b][c]
```
