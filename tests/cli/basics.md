# Basics

## Comments

```scrut
$ wo '(* This comment is ignored *) 5'
5
```

```scrut
$ wo '5 (* This comment is ignored *)'
5
```


## Semicolon

```scrut
$ wo 'x = 2; x'
2
```

```scrut
$ wo 'x = 2; x = x + 5'
7
```


## `Set`

Assign a value to a variable.

```scrut
$ wo 'Set[x, 5]'
5
```

```scrut
$ wo 'Set[x, 5]; x + 3'
8
```


## `Print`

Print values to the console.

```scrut
$ wo 'Print[]'

Null
```

```scrut
$ wo 'Print[5]'
5
Null
```

Multiple arguments are concatenated:

```scrut
$ wo 'Print["a", "b", "c"]'
abc
Null
```

```scrut
$ wo 'Print[1, " + ", 2, " = ", 3]'
1 + 2 = 3
Null
```
