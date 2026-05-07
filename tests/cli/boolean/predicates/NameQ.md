# `NameQ`

Tests whether a string is the name of a known (built-in or user-defined)
symbol.

```scrut
$ wo 'NameQ["Plus"]'
True
```

```scrut
$ wo 'NameQ["xyz_woxi_nosuch"]'
False
```
