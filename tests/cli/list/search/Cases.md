# `Cases`

Extracts elements from an expression that match a pattern.

```scrut
$ wo 'Cases[{a, b, a}, a]'
{a, a}
```


### Cases with Except pattern

```scrut
$ wo 'Cases[{1, 2, 3, 4, 5}, Except[3]]'
{1, 2, 4, 5}
```

### Cases with Alternatives

```scrut
$ wo 'Cases[{1, 2, 3, 4, 5}, Except[2 | 4]]'
{1, 3, 5}
```

### Cases with level specification

```scrut
$ wo 'Cases[{{1, 2}, {3, 4}}, _Integer, {2}]'
{1, 2, 3, 4}
```
