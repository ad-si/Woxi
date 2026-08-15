# `Get`

Reads and evaluates a file returning the last result.

A file name starting with `!` evaluates the code an external command writes
to its standard output:

```scrut
$ wo 'Get["!echo 1 + 2"]'
3
```

`<<` is the same function in operator form:

```scrut
$ wo '<< "!echo 6*7"'
42
```
