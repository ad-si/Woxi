# `Head` - Returns the head of an expression

```scrut
$ wo 'Head[f[x, y]]'
f
```

```scrut
$ wo 'Head[a + b + c]'
Plus
```

```scrut
$ wo 'Head[{a, b, c}]'
List
```

```scrut
$ wo 'Head[23432]'
Integer
```

```scrut
$ wo 'Head[345.6]'
Real
```
