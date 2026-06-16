# `ToString`

Converts an expression to a string.

```scrut
$ wo 'ToString[123]'
123
```

```scrut
$ wo 'ToString[{1, 2, 3}]'
{1, 2, 3}
```

```scrut
$ wo 'ToString[1 + 2]'
3
```

`NumberForm` renders a number to a given number of significant figures.

```scrut
$ wo 'ToString[NumberForm[3.14159, 3]]'
3.14
```

With a `{n, f}` specification it shows exactly `f` digits after the decimal
point (zero-padded).

```scrut
$ wo 'ToString[NumberForm[3.0, {5, 2}]]'
3.00
```
