# `Attributes`

Returns the attributes of a symbol.

```scrut
$ wo 'Attributes[E]'
{Constant, Protected, ReadProtected}
```

`Attributes` is listable, so a list of symbols gives one list of attributes per
symbol:

```scrut
$ wo 'Attributes[{Hold, Sequence}]'
{{HoldAll, Protected}, {Protected}}
```
