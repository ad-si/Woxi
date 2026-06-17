# `Pochhammer`

Returns the Pochhammer symbol (rising factorial).

```scrut
$ wo 'Pochhammer[3, 0]'
1
```

It threads over either argument.

```scrut
$ wo 'Pochhammer[3, {1, 2}]'
{3, 12}
```
