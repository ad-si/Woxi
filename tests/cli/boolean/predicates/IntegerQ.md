# `IntegerQ`

Check if a value is an integer.

```scrut
$ wo 'IntegerQ[5]'
True
```

```scrut
$ wo 'IntegerQ[0]'
True
```

```scrut
$ wo 'IntegerQ[-7]'
True
```

```scrut
$ wo 'IntegerQ[3.0]'
False
```

```scrut
$ wo 'IntegerQ[3.5]'
False
```

```scrut
$ wo 'IntegerQ[1.2]'
False
```

```scrut
$ wo 'IntegerQ[-0.5]'
False
```

```scrut
$ wo 'IntegerQ[0.0]'
False
```

```scrut
$ wo 'IntegerQ[a]'
False
```
