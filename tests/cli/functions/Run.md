# `Run`

Executes a system command and returns the exit code.

```scrut
$ wo 'Run["echo hello"]'
hello
0
```

```scrut
$ wo 'Run["exit 0"]'
0
```

```scrut
$ wo 'Run["exit 1"]'
256
```


## Pattern Definitions with `SetDelayed`

Define a function using pattern matching.

```scrut
$ wo 'f[x_] := x^2; f[3]'
9
```

```scrut
$ wo 'f[x_] := x^2; f[5]'
25
```

```scrut
$ wo 'f[x_] := x + 1; f[10]'
11
```


## Optional Pattern Arguments

Optional arguments use the `x_:default` syntax to provide default values.

```scrut
$ wo 'f[x_:0] := x + 1; f[]'
1
```

```scrut
$ wo 'f[x_:0] := x + 1; f[5]'
6
```

### Optional with head constraint

```scrut
$ wo 'g[q_List: {}, n_] := {q, n}; g[5]'
{{}, 5}
```

```scrut
$ wo 'g[q_List: {}, n_] := {q, n}; g[{1, 2}, 3]'
{{1, 2}, 3}
```

### Multiple optional arguments

```scrut
$ wo 'f[a_:0, b_, c_:99] := {a, b, c}; f[5]'
{0, 5, 99}
```

```scrut
$ wo 'f[a_:0, b_, c_:99] := {a, b, c}; f[1, 5]'
{1, 5, 99}
```

```scrut
$ wo 'f[a_:0, b_, c_:99] := {a, b, c}; f[1, 5, 10]'
{1, 5, 10}
```
