# `NonCommutativeMultiply`

Non-commutative multiplication operator.

```scrut
$ wo 'NonCommutativeMultiply[x, y]'
x**y
```

The infix operator `**` is the shorthand form.

```scrut
$ wo 'Head[x**y]'
NonCommutativeMultiply
```

```scrut
$ wo 'ToString[FullForm[a**b**c]]'
NonCommutativeMultiply[a, b, c]
```
