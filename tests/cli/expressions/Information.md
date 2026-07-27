# `Information`

Returns descriptive information about a symbol or function.

```scrut
$ wo 'StringContainsQ[ToString[Information[Sin]], "Sin"]'
True
```

The result is an `InformationData` record, not plain text.

```scrut
$ wo 'Head[Information[Sin]]'
InformationData
```
