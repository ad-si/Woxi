# `Return`

Inside a function body, exits early returning a value.

```scrut
$ wo 'f[x_] := Return[x + 1]; f[5]'
6
```

Only a definition body, `Do` and `Scan` take a `Return`. Anywhere a value is
being collected it stands as the expression it names, so a `Table` still
gives back a list of the same length:

```scrut
$ wo 'ToString[Table[Return[1], {2}], InputForm]'
{Return[1], Return[1]}
```

which means a `Return` inside a `Table` does not leave the function around
it:

```scrut
$ wo 'f[] := (Table[Return[1], {2}]; 9); f[]'
9
```

```scrut
$ wo 'Do[Return[5], {2}]'
5
```
