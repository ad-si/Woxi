# `NumericQ`

Like `NumberQ` but also recognizes numeric constants such as `Pi`.

```scrut
$ wo 'NumericQ[Pi]'
True
```

```scrut
$ wo 'NumericQ["abc"]'
False
```

A constant is numeric wherever it sits in a product:

```scrut
$ wo 'NumericQ[Pi 2]'
True
```

```scrut
$ wo 'NumericQ[Pi I]'
True
```

A symbol with no numeric value is not, and neither is `Infinity`:

```scrut
$ wo 'NumericQ[Pi x]'
False
```

```scrut
$ wo 'NumericQ[Infinity]'
False
```
