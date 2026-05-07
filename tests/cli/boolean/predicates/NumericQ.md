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
