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
